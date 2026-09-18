"""针对对抗图像的防御。

三类，按"是否需要重训练"划分：

  A. 输入变换（免训练，可直接部署在推理前）
       JPEG 压缩、比特深度缩减、中值/高斯平滑、随机缩放padding、TVM（总变差最小化）
  B. 随机化 + 集成（免训练）
       随机变换多次推理取中位数（近似随机平滑），显著提高攻击所需查询量
  C. 检测（免训练）
       特征压缩 (Feature Squeezing, Xu et al., NDSS 2018)：
       比较原图与压缩图的预测差异，差异大则判为对抗样本

重要声明：A/B 属于梯度混淆类防御，在**自适应攻击**（攻击者已知防御、用 BPDA 等技术
绕过不可微变换）下会显著削弱。本模块在评估时会同时给出静态攻击与自适应攻击两组结果，
避免得出虚假的安全结论（参考 Athalye et al., ICML 2018 对混淆梯度的批评）。
"""
from __future__ import annotations
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------- A. 输入变换
def jpeg_defense(x01: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """JPEG 重编码：破坏高频对抗扰动。不可微，需用 BPDA 才能自适应攻击。"""
    out = []
    for i in range(x01.shape[0]):
        a = (x01[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=quality); b.seek(0)
        c = np.asarray(Image.open(b).convert("RGB"), dtype=np.float32) / 255.0
        out.append(torch.from_numpy(c).permute(2, 0, 1))
    return torch.stack(out).to(x01.device)


def bit_depth(x01: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """比特深度缩减：把像素量化到 2^bits 级，抹掉小幅扰动。"""
    levels = 2 ** bits - 1
    return (x01 * levels).round() / levels


def median_filter(x01: torch.Tensor, k: int = 3) -> torch.Tensor:
    """k×k 中值滤波。偶数 k 的反射填充无法保持尺寸，这里强制使用奇数窗口。"""
    if k % 2 == 0:
        k += 1
    pad = k // 2
    x = F.pad(x01, (pad, pad, pad, pad), mode="reflect")
    patches = x.unfold(2, k, 1).unfold(3, k, 1)          # B,C,H,W,k,k
    b, c, h, w = x01.shape
    return patches.contiguous().view(b, c, h, w, k * k).median(dim=-1).values


def gaussian_blur(x01: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    xs = torch.arange(-r, r + 1, dtype=x01.dtype, device=x01.device)
    k1 = torch.exp(-xs ** 2 / (2 * sigma ** 2)); k1 = k1 / k1.sum()
    k2 = (k1[:, None] @ k1[None, :]).expand(x01.shape[1], 1, -1, -1)
    return F.conv2d(F.pad(x01, (r, r, r, r), mode="reflect"), k2, groups=x01.shape[1])


def random_resize_pad(x01: torch.Tensor, max_scale: float = 1.12, seed: int | None = None) -> torch.Tensor:
    """随机缩放 + 随机填充（Xie et al., ICLR 2018）：引入随机性破坏固定扰动。"""
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    h, w = x01.shape[-2:]
    s = 1.0 + float(torch.rand(1, generator=g)) * (max_scale - 1.0)
    nh, nw = int(h * s), int(w * s)
    y = F.interpolate(x01, size=(nh, nw), mode="bilinear", align_corners=False)
    ph, pw = nh - h, nw - w
    top = int(torch.randint(0, ph + 1, (1,), generator=g)) if ph > 0 else 0
    left = int(torch.randint(0, pw + 1, (1,), generator=g)) if pw > 0 else 0
    return y[..., top:top + h, left:left + w]


DEFENSES = {
    "none": lambda x: x,
    "jpeg75": lambda x: jpeg_defense(x, 75),
    "jpeg50": lambda x: jpeg_defense(x, 50),
    "bit4": lambda x: bit_depth(x, 4),
    "bit3": lambda x: bit_depth(x, 3),
    "median3": lambda x: median_filter(x, 3),
    "blur1.0": lambda x: gaussian_blur(x, 1.0),
    "randresize": lambda x: random_resize_pad(x),
}


# ---------------------------------------------------------------- B. 随机化集成
def randomized_predict(clf, x01: torch.Tensor, n: int = 7) -> float:
    """多次随机变换后推理取中位数，降低单次确定性攻击的成功率。"""
    vals = []
    for i in range(n):
        y = random_resize_pad(x01, seed=i)
        y = bit_depth(y, 5)
        with torch.no_grad():
            vals.append(float(clf.prob(y)))
    return float(np.median(vals))


def ensemble_predict(clfs: list, x01: torch.Tensor, mode: str = "max") -> float:
    """多专家集成：任一专家判定高风险即高风险（max），提高攻击者需同时绕过的模型数。"""
    vals = []
    for c in clfs:
        with torch.no_grad():
            vals.append(float(c.prob(x01)))
    return float(max(vals) if mode == "max" else np.mean(vals))


# ---------------------------------------------------------------- C. 对抗样本检测
def feature_squeezing_score(clf, x01: torch.Tensor,
                            squeezers=(("bit4", lambda x: bit_depth(x, 4)),
                                       ("median3", lambda x: median_filter(x, 3)),
                                       ("blur", lambda x: gaussian_blur(x, 1.0)))) -> dict:
    """Feature Squeezing：原图与"压缩图"的预测差异越大，越可能是对抗样本。

    直觉：正常图像在轻微压缩下预测稳定；对抗样本依赖精心构造的高频细节，
    一旦被压缩掉，预测就会剧烈变化。
    """
    with torch.no_grad():
        p0 = float(clf.prob(x01))
    diffs = {}
    for name, fn in squeezers:
        with torch.no_grad():
            diffs[name] = abs(float(clf.prob(fn(x01))) - p0)
    return {"p_original": p0, "diffs": diffs, "max_diff": max(diffs.values()) if diffs else 0.0}


def detect_adversarial(clf, x01: torch.Tensor, threshold: float = 0.3) -> dict:
    r = feature_squeezing_score(clf, x01)
    return {"is_adversarial": r["max_diff"] >= threshold, "score": r["max_diff"], "detail": r}


# ---------------------------------------------------------------- BPDA：自适应攻击
def bpda_pgd(clf, defense_fn, x: torch.Tensor, eps: float = 8 / 255, steps: int = 20,
             alpha: float | None = None, minimize: bool = True) -> torch.Tensor:
    """BPDA（Backward Pass Differentiable Approximation）：

    前向走真实的不可微防御，反向用恒等函数近似其梯度。
    这是评估"输入变换类防御"的正确方式——只报告静态攻击下的鲁棒性会严重高估安全性。
    """
    alpha = alpha if alpha is not None else max(eps / 6, 1 / 255)
    x0 = x.clone().detach()
    adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        adv_var = adv.clone().detach().requires_grad_(True)
        # 恒等近似：前向经过真实防御，反向让梯度直接绕过它
        with torch.no_grad():
            purified = defense_fn(adv_var.detach())
        z = adv_var + (purified - adv_var).detach()
        p = clf.prob(z)
        loss = p.sum() if minimize else -p.sum()
        grad = torch.autograd.grad(loss, adv_var)[0]
        adv = adv.detach() - alpha * grad.sign()
        adv = (x0 + (adv - x0).clamp(-eps, eps)).clamp(0, 1).detach()
    return adv
