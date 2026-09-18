import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
from huggingface_hub import snapshot_download
PATTERNS = {
    "thu-coai/roberta-base-cold": ["*.json", "*.txt", "model.safetensors"],
    "dima806/deepfake_vs_real_image_detection": ["config.json", "preprocessor_config.json", "model.safetensors"],
    "BAAI/bge-small-zh-v1.5": ["*.json", "*.txt", "model.safetensors"],
    "openai/whisper-base": ["*.json", "*.txt", "model.safetensors"],
    "openai/whisper-small": ["*.json", "*.txt", "model.safetensors"],
    "protectai/deberta-v3-base-prompt-injection-v2": ["*.json", "*.txt", "*.model", "model.safetensors"],
    "laion/clap-htsat-unfused": ["*.json", "*.txt", "pytorch_model.bin"],
    "MIT/ast-finetuned-audioset-10-10-0.4593": ["*.json", "model.safetensors"],
}
repo = sys.argv[1]
for attempt in range(8):
    try:
        t = time.time()
        p = snapshot_download(repo_id=repo, allow_patterns=PATTERNS[repo], max_workers=2)
        print(f"OK {repo} -> {p} ({time.time()-t:.0f}s)", flush=True); break
    except Exception as e:
        print(f"RETRY {repo} attempt {attempt}: {str(e)[:150]}", flush=True); time.sleep(8)
