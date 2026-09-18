"""针对图像审核模型的对抗攻击。

实现三类攻击（均在本地沙箱内、针对本项目自己部署的检测器，用于评估其鲁棒性）：

  白盒
    FGSM     单步梯度符号法（Goodfellow et al., 2015）
    PGD      多步投影梯度下降（Madry et al., 2018），L∞ 约束
  黑盒
    SPSA     同时扰动随机逼近，仅需查询模型输出概率，不需要梯度
    迁移攻击  在 A 模型上生成扰动，迁移到 B 模型（评估跨模型可迁移性）
  非梯度（真实攻击者更常用）
    压缩/缩放/加噪/旋转/贴片 等"自然扰动"，成本极低且不需要模型访问权

扰动预算 epsilon 以 0–1 像素尺度表示，eps=8/255 是文献常用设置。
"""
from __future__ import annotations
import functools
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...config import device


# ---------------------------------------------------------------- 可微封装
class DifferentiableClassifier:
    """把 HF 图像分类器包成"输入 0-1 像素张量 → 类别概率"的可微函数。

    关键点：processor 内部的 resize / normalize 必须用张量算子重写，
    否则梯度无法回传到原始像素，攻击就失效了。
    """

    def __init__(self, model, processor, target_index: int):
        self.model = model.eval()
        self.dev = next(model.parameters()).device
        self.target_index = target_index
        ip = processor.image_processor if hasattr(processor, "image_processor") else processor
        size = getattr(ip, "size", {"height": 224, "width": 224})
        self.hw = (size.get("height", size.get("shortest_edge", 224)),
                   size.get("width", size.get("shortest_edge", 224)))
        self.mean = torch.tensor(getattr(ip, "image_mean", [0.5, 0.5, 0.5])).view(1, 3, 1, 1).to(self.dev)
        self.std = torch.tensor(getattr(ip, "image_std", [0.5, 0.5, 0.5])).view(1, 3, 1, 1).to(self.dev)

    def logits(self, x01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x01, size=self.hw, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.model(pixel_values=x).logits

    def prob(self, x01: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x01).float(), -1)[:, self.target_index]


def to_tensor01(img, dev) -> torch.Tensor:
    if isinstance(img, Image.Image):
        a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    else:
        a = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)


def to_pil(x01: torch.Tensor) -> Image.Image:
    a = (x01.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


# ---------------------------------------------------------------- 白盒攻击
def fgsm(clf: DifferentiableClassifier, x: torch.Tensor, eps: float = 8 / 255,
         minimize: bool = True) -> torch.Tensor:
    """单步 FGSM。minimize=True 表示压低目标类概率（让违规内容"变正常"）。"""
    x = x.clone().detach().requires_grad_(True)
    p = clf.prob(x)
    loss = p.sum() if minimize else -p.sum()
    grad = torch.autograd.grad(loss, x)[0]
    return (x - eps * grad.sign()).clamp(0, 1).detach()


def pgd(clf: DifferentiableClassifier, x: torch.Tensor, eps: float = 8 / 255,
        alpha: float | None = None, steps: int = 20, minimize: bool = True,
        random_start: bool = True) -> torch.Tensor:
    """L∞ PGD。每步沿梯度方向走 alpha，再投影回 eps 球内。"""
    alpha = alpha if alpha is not None else max(eps / 6, 1 / 255)
    x0 = x.clone().detach()
    adv = x0.clone()
    if random_start:
        adv = (adv + torch.empty_like(adv).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        adv.requires_grad_(True)
        p = clf.prob(adv)
        loss = p.sum() if minimize else -p.sum()
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() - alpha * grad.sign()
        adv = (x0 + (adv - x0).clamp(-eps, eps)).clamp(0, 1).detach()
    return adv


# ---------------------------------------------------------------- 黑盒攻击
def spsa(clf: DifferentiableClassifier, x: torch.Tensor, eps: float = 8 / 255,
         steps: int = 40, lr: float = 2 / 255, samples: int = 16, delta: float = 0.01,
         minimize: bool = True) -> tuple[torch.Tensor, int]:
    """SPSA 黑盒攻击：只用模型输出概率估计梯度方向。返回 (对抗样本, 查询次数)。"""
    x0 = x.clone().detach()
    adv = x0.clone()
    queries = 0
    for _ in range(steps):
        g = torch.zeros_like(adv)
        for _ in range(samples // 2):
            v = torch.randint(0, 2, adv.shape, device=adv.device, dtype=adv.dtype) * 2 - 1
            with torch.no_grad():
                p_plus = clf.prob((adv + delta * v).clamp(0, 1))
                p_minus = clf.prob((adv - delta * v).clamp(0, 1))
            queries += 2
            g += ((p_plus - p_minus).view(-1, 1, 1, 1) / (2 * delta)) * v
        g /= max(1, samples // 2)
        adv = adv - lr * g.sign() if minimize else adv + lr * g.sign()
        adv = (x0 + (adv - x0).clamp(-eps, eps)).clamp(0, 1).detach()
    return adv, queries


# ---------------------------------------------------------------- 非梯度"自然扰动"
def natural_perturbations(img: Image.Image) -> dict[str, Image.Image]:
    """真实攻击者最常用、成本最低的规避手段：重压缩、缩放、加噪、旋转、镜像、贴边框。"""
    import io
    import cv2
    out = {}
    for q in (40, 20, 10):
        b = io.BytesIO(); img.save(b, "JPEG", quality=q); b.seek(0)
        out[f"jpeg_q{q}"] = Image.open(b).convert("RGB")
    w, h = img.size
    for s in (0.5, 0.25):
        out[f"downscale_{s}"] = img.resize((max(8, int(w * s)), max(8, int(h * s)))).resize((w, h))
    a = np.asarray(img).astype(np.int16)
    for sigma in (8, 20):
        n = np.random.default_rng(0).normal(0, sigma, a.shape)
        out[f"gauss_noise_{sigma}"] = Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))
    out["hflip"] = img.transpose(Image.FLIP_LEFT_RIGHT)
    for ang in (5, 15):
        out[f"rotate_{ang}"] = img.rotate(ang, expand=False, fillcolor=(128, 128, 128))
    arr = np.asarray(img).copy()
    bw = max(4, min(w, h) // 12)
    arr[:bw, :] = 255; arr[-bw:, :] = 255; arr[:, :bw] = 255; arr[:, -bw:] = 255
    out["white_border"] = Image.fromarray(arr)
    blur = cv2.GaussianBlur(np.asarray(img), (7, 7), 0)
    out["blur"] = Image.fromarray(blur)
    return out


# ---------------------------------------------------------------- 模型加载
@functools.lru_cache(maxsize=4)
def load_target(which: str = "nsfw"):
    """加载被攻击的审核模型，返回可微封装。"""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from ...config import HF_NSFW, HF_DEEPFAKE
    mid = {"nsfw": HF_NSFW, "deepfake": HF_DEEPFAKE}[which]
    proc = AutoImageProcessor.from_pretrained(mid)
    mdl = AutoModelForImageClassification.from_pretrained(mid).eval()
    dev = device()
    try:
        mdl = mdl.to(dev)
    except Exception:
        mdl = mdl.to("cpu")
    want = {"nsfw": ("nsfw",), "deepfake": ("fake", "fakes")}[which]
    idx = [i for i, l in mdl.config.id2label.items() if l.lower() in want]
    return DifferentiableClassifier(mdl, proc, idx[0] if idx else 1)


def linf(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
