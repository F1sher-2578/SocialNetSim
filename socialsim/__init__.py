"""SocialNetSim —— 社交网络拟真沙箱：水军模拟/检测 + 多模态不良内容审核 + AI 安全攻防。

注意：HuggingFace 镜像必须在导入 transformers / huggingface_hub 之前设置，
因为 huggingface_hub 在模块导入时就固化了 ENDPOINT 常量。
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# 模型已全部预下载到本地缓存；离线模式避免每次加载都去网络校验 etag（国内网络会卡住）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

__all__ = ["config", "db", "models"]
