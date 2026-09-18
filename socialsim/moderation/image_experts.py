"""图像帧级专家：NSFW 分类、CLIP 零样本风险分类、人脸检测 + 深伪检测。

参考 DFscan 的多专家范式：同一帧交给若干互补的专家，各自输出置信度，再在上层做加权融合。
"""
from __future__ import annotations
import functools
import math
from pathlib import Path
import numpy as np
from PIL import Image

# ---------------- CLIP 零样本风险类目（prompt ensemble） ----------------
RISK_PROMPTS = {
    "violence": ["a photo of people fighting violently", "a violent assault with weapons",
                 "a scene of physical violence and beating"],
    "gore": ["a bloody gory image with blood and wounds", "a graphic injury with a lot of blood"],
    "weapon": ["a photo of a gun or firearm", "a person holding a knife as a weapon"],
    "porn": ["a pornographic photo of a naked person", "explicit sexual content"],
    "smoking_drug": ["a person smoking a cigarette", "illegal drugs and syringes"],
    "protest_crowd": ["a large crowd protest with banners", "a riot in the street"],
    "fire_disaster": ["a building on fire with heavy smoke", "an explosion disaster scene"],
    "horror": ["a scary horror image of a monster", "a disturbing creepy scene"],
}
SAFE_PROMPTS = ["a normal everyday photo", "a photo of scenery or landscape", "a photo of food on a table",
                "a photo of a person talking to the camera", "a cartoon animation frame",
                "a screenshot of text or a document", "a photo of a pet animal", "a photo of a product",
                "an empty room or plain background", "a person sitting and talking indoors"]

# 画面域判别：卡通/游戏画面里的"打斗、血、怪物"与真实世界风险等级完全不同，
# 不做域判别会导致动画片被大量误拦（实测 Big Buck Bunny 误判为 block）。
DOMAIN_PROMPTS = {
    "photo": ["a real photograph taken by a camera", "a real-world photo of people"],
    "cartoon": ["a cartoon animation frame", "a 3D animated movie scene", "a drawing or illustration"],
    "game": ["a video game screenshot", "a computer generated game scene"],
    "text_screen": ["a screenshot of text or a document", "a slide with large text on solid background"],
}
# 非真实照片域的风险折减。暴力/恐怖/武器在动画中与真实世界不同级，折减最多；
# 色情类即使在动画中仍属违规（如成人动画），只做轻度折减。
DOMAIN_DISCOUNT = {"cartoon": 0.3, "game": 0.35, "text_screen": 0.6, "photo": 1.0}
PORN_DOMAIN_DISCOUNT = {"cartoon": 0.6, "game": 0.6, "text_screen": 0.5, "photo": 1.0}
DISCOUNTABLE = {"violence", "gore", "horror", "weapon", "fire_disaster", "protest_crowd", "smoking_drug"}
RISK_WEIGHT = {"violence": 0.85, "gore": 0.9, "weapon": 0.75, "porn": 0.95,
               "smoking_drug": 0.5, "protest_crowd": 0.45, "fire_disaster": 0.5, "horror": 0.4}


@functools.lru_cache(maxsize=1)
def _nsfw():
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from ..config import HF_NSFW, device
    proc = AutoImageProcessor.from_pretrained(HF_NSFW)
    mdl = AutoModelForImageClassification.from_pretrained(HF_NSFW).eval()
    dev = device()
    try:
        mdl = mdl.to(dev)
    except Exception:
        dev = "cpu"
    nsfw_idx = [i for i, l in mdl.config.id2label.items() if l.lower() == "nsfw"][0]
    return proc, mdl, dev, nsfw_idx


@functools.lru_cache(maxsize=1)
def _clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from ..config import HF_CLIP, device
    proc = CLIPProcessor.from_pretrained(HF_CLIP)
    mdl = CLIPModel.from_pretrained(HF_CLIP).eval()
    dev = device()
    try:
        mdl = mdl.to(dev)
    except Exception:
        dev = "cpu"
    prompts, owners = [], []
    for k, ps in RISK_PROMPTS.items():
        prompts += ps; owners += [k] * len(ps)
    prompts += SAFE_PROMPTS; owners += ["safe"] * len(SAFE_PROMPTS)
    dprompts, downers = [], []
    for k, ps in DOMAIN_PROMPTS.items():
        dprompts += ps; downers += [k] * len(ps)
    with torch.no_grad():
        tf = proc(text=prompts, return_tensors="pt", padding=True)
        tf = {k: v.to(dev) for k, v in tf.items()}
        temb = mdl.get_text_features(**tf); temb = temb / temb.norm(dim=-1, keepdim=True)
        df = proc(text=dprompts, return_tensors="pt", padding=True)
        df = {k: v.to(dev) for k, v in df.items()}
        demb = mdl.get_text_features(**df); demb = demb / demb.norm(dim=-1, keepdim=True)
    return proc, mdl, dev, temb, owners, demb, downers


@functools.lru_cache(maxsize=1)
def _deepfake():
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from ..config import HF_DEEPFAKE, device
    proc = AutoImageProcessor.from_pretrained(HF_DEEPFAKE)
    mdl = AutoModelForImageClassification.from_pretrained(HF_DEEPFAKE).eval()
    dev = device()
    try:
        mdl = mdl.to(dev)
    except Exception:
        dev = "cpu"
    fake_idx = [i for i, l in mdl.config.id2label.items() if l.lower() in ("fake", "fakes", "label_1")][0]
    return proc, mdl, dev, fake_idx


@functools.lru_cache(maxsize=1)
def _face_detector():
    import cv2
    from ..config import ROOT
    path = ROOT / "models" / "cv" / "yunet.onnx"
    if not path.exists():
        return None
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.7, 0.3, 500)


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, (str, Path)):
        return Image.open(img).convert("RGB")
    import cv2
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def nsfw_batch(images: list) -> list[float]:
    import torch
    proc, mdl, dev, idx = _nsfw()
    out = []
    for i in range(0, len(images), 16):
        pil = [_to_pil(x) for x in images[i:i + 16]]
        inp = proc(images=pil, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            p = torch.softmax(mdl(**inp).logits.float(), -1)[:, idx]
        out += p.cpu().tolist()
    return out


def clip_risk_batch(images: list) -> list[dict]:
    """CLIP 零样本：在"风险类目 + 安全类目"上做 softmax，返回各风险类目概率。
    加入安全类目做对照，可显著降低普通画面的误报。"""
    import torch
    proc, mdl, dev, temb, owners, demb, downers = _clip()
    owners, downers = np.array(owners), np.array(downers)
    res = []
    for i in range(0, len(images), 16):
        pil = [_to_pil(x) for x in images[i:i + 16]]
        inp = proc(images=pil, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            iemb = mdl.get_image_features(**inp)
            iemb = iemb / iemb.norm(dim=-1, keepdim=True)
            scale = mdl.logit_scale.exp()
            probs = torch.softmax((iemb @ temb.T) * scale, -1).float().cpu().numpy()
            dprobs = torch.softmax((iemb @ demb.T) * scale, -1).float().cpu().numpy()
        for row, drow in zip(probs, dprobs):
            d = {}
            for k in RISK_PROMPTS:
                d[k] = float(row[owners == k].sum())      # 同类目多 prompt 概率求和
            d["safe"] = float(row[owners == "safe"].sum())
            # 风险质量总和：所有风险类目的概率之和（与 safe 类目在同一个 softmax 中竞争）。
            # 实测这是比"各类目取最大"强得多的连续信号——
            # 取最大会丢掉"多个类目都有点像"这一重要证据（XD-Violence 上 AUC 0.56 → 0.95）。
            d["_risk_sum"] = float(row[owners != "safe"].sum())
            dom = {k: float(drow[downers == k].sum()) for k in DOMAIN_PROMPTS}
            d["_domain"] = max(dom, key=dom.get)
            d["_domain_probs"] = dom
            res.append(d)
    return res


def detect_faces(frame_bgr) -> list[tuple[int, int, int, int]]:
    det = _face_detector()
    if det is None:
        return []
    import cv2
    h, w = frame_bgr.shape[:2]
    det.setInputSize((w, h))
    try:
        _, faces = det.detect(frame_bgr)
    except Exception:
        return []
    if faces is None:
        return []
    out = []
    for f in faces:
        x, y, fw, fh = [int(v) for v in f[:4]]
        x, y = max(0, x), max(0, y)
        if fw > 24 and fh > 24:
            out.append((x, y, min(fw, w - x), min(fh, h - y)))
    return out


def deepfake_batch(face_images: list) -> list[float]:
    import torch
    proc, mdl, dev, idx = _deepfake()
    out = []
    for i in range(0, len(face_images), 16):
        pil = [_to_pil(x) for x in face_images[i:i + 16]]
        inp = proc(images=pil, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            p = torch.softmax(mdl(**inp).logits.float(), -1)[:, idx]
        out += p.cpu().tolist()
    return out


@functools.lru_cache(maxsize=1)
def calibration() -> dict:
    """加载在正常素材上估计的阈值；缺失时回退到保守默认值。"""
    import json
    from ..config import MODEL_DIR
    p = MODEL_DIR / "calibration.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _tail_risk(p: float, q: dict) -> float:
    """把零样本置信度映射为连续的风险值，依据其在**正常素材分布**中的位置。

    此前用硬阈值门控，导致所有刚过阈值的帧拿到几乎相同的分数，
    视频级排序能力被完全破坏（实测 XD-Violence 上 AUC 仅 0.55）。
    改为连续映射：
        r = (p - q90) / (q99 - q90)        在 q90 处为 0，在 q99 处为 1
        risk = 1 - exp(-r)                 单调、连续、对超出分布的极端值不饱和到常数
    这样"略高于正常"和"远高于正常"能被区分开，才谈得上排序。
    """
    q90, q99 = q.get("p90"), q.get("p99")
    if q90 is None or q99 is None:
        q90, q99 = q.get("p50", 0.05), q.get("thresh", 0.25)
    span = max(1e-4, q99 - q90)
    r = (p - q90) / span
    if r <= 0:
        return 0.0
    return float(1.0 - math.exp(-r))


def _logit(x: float, eps: float = 1e-6) -> float:
    x = min(1 - eps, max(eps, x))
    return math.log(x / (1 - x))


def calibrated_risk_sum(rs: float, q: dict) -> float:
    """把"风险类目概率总和"映射到 0–1 的可解释风险分。

    做法：以**正常帧分布的中位数**为参照中心，在 logit 空间做标准化后过 sigmoid，
    温度由正常帧的 p95 决定，使正常帧的 p95 大致落在 0.88。
    这样"0.5 = 与典型正常帧同等风险"，跨画面域可比。

    为什么用总和而不是各类目取最大：网格搜索（experiments/exp_prompt_tuning.py）显示
    在 XD-Violence 上前者 AUC 0.955、后者 0.56——取最大会丢掉"多个风险类目同时被激活"
    这一关键证据，而这正是真实暴力场景的典型特征。
    """
    if not q:
        return max(0.0, min(1.0, (rs - 0.6) / 0.4))
    c = _logit(q.get("p50", 0.6))
    hi = _logit(q.get("p95", max(0.95, q.get("p50", 0.6) + 0.1)))
    T = max(0.3, (hi - c) / 2.0)
    return float(1.0 / (1.0 + math.exp(-(_logit(rs) - c) / T)))


def fuse_frame(nsfw: float, clip: dict, deepfake: float | None = None, use_calibration: bool = True) -> dict:
    """帧级多专家融合，输出连续风险分。

    四点关键设计（均由实测误报/漏报驱动）：
      1. 分布校准：零样本置信度在不同画面域上尺度差异极大，用正常素材分布的分位数做映射，
         而不是固定阈值（对应 DFscan 用 EER 定阈值的思路）。
      2. 连续映射：用分位数尾部函数而非硬门控，保留排序能力。
      3. 相对优势：风险类目需显著超过 safe 类目，避免"什么都不像"时被随机类目占据。
      4. 画面域折减：卡通/游戏中的打斗与真实暴力不同级，否则动画片会被整片误拦。
    """
    cal = calibration() if use_calibration else {}
    ccal = cal.get("clip", {})
    labels, score = [], 0.0
    parts = {}
    domain = clip.get("_domain", "photo")
    discount = DOMAIN_DISCOUNT.get(domain, 1.0)
    parts["domain"] = domain
    safe = clip.get("safe", 0.0)

    # 主分数：风险质量总和，经"以正常帧分布为参照"的 logit 校准。
    rs = clip.get("_risk_sum", 0.0)
    parts["risk_sum"] = rs
    base = calibrated_risk_sum(rs, ccal.get("_risk_sum", {}))
    score = max(score, base * discount)

    ncal = cal.get("nsfw", {})
    nsfw_th = ncal.get("thresh", 0.6)
    if nsfw is not None:
        parts["nsfw"] = nsfw
        dom_f = 1.0 if domain == "photo" else 0.5
        s_n = 0.95 * _tail_risk(nsfw, ncal) * dom_f if ncal else (
            (0.6 + 0.4 * nsfw) * dom_f if nsfw >= nsfw_th else 0.0)
        if s_n > 0:
            score = max(score, s_n)
            if nsfw >= nsfw_th:
                labels.append("nsfw" if domain == "photo" else "nsfw_cartoon")

    for k, w in RISK_WEIGHT.items():
        p = clip.get(k, 0.0)
        parts[f"clip_{k}"] = p
        q = ccal.get(k, {})
        # 相对优势门限仍然保留：显著低于 safe 的类目不计入
        if p <= safe * 0.5:
            continue
        s_k = w * _tail_risk(p, q)
        if k in DISCOUNTABLE:
            s_k *= discount
        elif k == "porn":
            s_k *= PORN_DOMAIN_DISCOUNT.get(domain, 1.0)
        if s_k > score:
            score = s_k          # 单类目极端确信时可以抬高总分
        # 标签按阈值给出（标签是离散概念，分数是连续的）
        if p >= q.get("thresh", 0.25) and p > safe * 0.8:
            labels.append(k if domain == "photo" else f"{k}_{domain}")

    if nsfw is not None and nsfw >= nsfw_th and clip.get("porn", 0) >= ccal.get("porn", {}).get("thresh", 0.2):
        score = min(1.0, score + 0.1)       # 双专家一致

    if deepfake is not None:
        parts["deepfake"] = deepfake
        dcal = cal.get("deepfake", {})
        df_th = dcal.get("thresh", 0.9)
        # 深伪专家只在真实照片域上有意义，且必须显著高于正常人脸的分布
        if domain == "photo":
            s_d = 0.65 * _tail_risk(deepfake, dcal) if dcal else 0.0
            if deepfake >= df_th:
                labels.append("deepfake_suspect")
                s_d = max(s_d, 0.35 + 0.3 * deepfake)
            score = max(score, s_d)
    return {"score": float(min(1.0, score)), "labels": sorted(set(labels)), "parts": parts}


def analyze_image(path: str, with_deepfake: bool = True) -> dict:
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return {"score": 0.0, "labels": ["unreadable"], "experts": {}}
    pil = _to_pil(img)
    nsfw = nsfw_batch([pil])[0]
    clip = clip_risk_batch([pil])[0]
    df, faces = None, detect_faces(img)
    if with_deepfake and faces:
        crops = [pil.crop((x, y, x + w, y + h)) for x, y, w, h in faces[:4]]
        df = float(max(deepfake_batch(crops)))
    fused = fuse_frame(nsfw, clip, df)
    # OCR：图片内嵌文字（广告引流常见）
    ocr_text, ocr_risk = _ocr_text(img), None
    if ocr_text:
        from .text_filter import analyze_text
        ocr_risk = analyze_text(ocr_text, noisy_source=True)
        if ocr_risk["score"] > fused["score"]:
            fused["score"] = ocr_risk["score"]
        fused["labels"] = sorted(set(fused["labels"]) | {f"ocr_{l}" for l in ocr_risk["labels"]})
    return {"score": fused["score"], "labels": fused["labels"],
            "experts": {"frame": fused, "faces": len(faces), "ocr": {"text": ocr_text[:300], "risk": ocr_risk}}}


@functools.lru_cache(maxsize=1)
def _ocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except Exception:
        return None


def _ocr_text(frame_bgr) -> str:
    ocr = _ocr()
    if ocr is None:
        return ""
    try:
        res, _ = ocr(frame_bgr)
    except Exception:
        return ""
    if not res:
        return ""
    # OCR 置信度门限提高到 0.75：低置信度识别结果噪声大，是误报的主要来源
    return " ".join(r[1] for r in res if len(r) > 2 and r[2] > 0.75)
