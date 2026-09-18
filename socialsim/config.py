"""全局配置：路径、模型名、阈值。"""
from __future__ import annotations
import os
from pathlib import Path

# HuggingFace 镜像（国内网络）；如已能直连可在环境变量中覆盖
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
UPLOAD_DIR = DATA_DIR / "uploads"
VIDEO_DIR = DATA_DIR / "videos"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
DB_PATH = DATA_DIR / "socialsim.db"
for d in (DATA_DIR, RAW_DIR, UPLOAD_DIR, VIDEO_DIR, MODEL_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- 预训练模型（HF hub id）----
HF_NSFW = "Falconsai/nsfw_image_detection"
HF_CLIP = "openai/clip-vit-base-patch32"
HF_COLD = "thu-coai/roberta-base-cold"           # 中文冒犯性语言检测
HF_DEEPFAKE = "dima806/deepfake_vs_real_image_detection"
HF_EMBED = "BAAI/bge-small-zh-v1.5"
HF_WHISPER = os.environ.get("SOCIALSIM_WHISPER", "openai/whisper-small")

# ---- 审核阈值（默认值；实际运行时优先使用校准得到的自适应阈值）----
RISK_BLOCK = 0.70      # >= 直接拦截
RISK_REVIEW = 0.40     # >= 转人工复核


def thresholds() -> tuple[float, float]:
    """返回 (复核阈值, 拦截阈值)。

    零样本专家的分数尺度高度依赖画面域，固定阈值在换域后完全失效
    （实测：在动画上校准的阈值用于影视素材时，正常视频平均分就已超过拦截线）。
    因此阈值由**正常素材的视频级分数分布**决定：
        复核 = 正常视频分数的 p75，拦截 = p95
    对应目标误报率约 25% / 5%。这与 DFscan 用 EER 确定判定阈值的做法同源。
    """
    import json
    p = MODEL_DIR / "calibration.json"
    if p.exists():
        try:
            v = json.loads(p.read_text(encoding="utf-8")).get("video_level")
            if v and v.get("p75") is not None:
                return float(v["p75"]), float(v["p95"])
        except Exception:
            pass
    return RISK_REVIEW, RISK_BLOCK
SHORT_VIDEO_SEC = 60   # <=60s 视为短视频

# ---- 视频解析 ----
KEYFRAME_MAX = 48          # 每个视频最多抽取关键帧数
KEYFRAME_MIN_GAP = 0.8     # 关键帧最小间隔(秒)
SCENE_DIFF_THRESH = 0.35   # 直方图差异阈值(场景切换)
FAST_SAMPLE_FPS = 2        # Fast 通道采样帧率

# ---- 模拟 ----
SIM_DAYS = 30
RANDOM_SEED = 2026

def device():
    """推理设备。可用 SOCIALSIM_DEVICE=cpu 强制回退——
    多个进程同时使用 MPS 会互相阻塞，批量实验时建议其中一个走 CPU。"""
    forced = os.environ.get("SOCIALSIM_DEVICE")
    if forced:
        return forced
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
