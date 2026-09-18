"""从 HuggingFace 镜像分层抽样下载 XD-Violence 官方测试集子集。

XD-Violence（Wu et al., ECCV 2020）是视频暴力检测的标准基准：4754 段未剪辑视频、217 小时，
含视觉+音频双模态，测试集 800 段带**帧级**标注，可同时评估视频级判别与时序定位。
全量 85GB，本项目按类目分层抽样若干段作为真实基准。

标签含义（文件名中的 label_X）：
  A  = Normal（正常）     B1 = Fighting   B2 = Shooting   B4 = Riot
  B5 = Abuse              B6 = Car accident               G  = Explosion
"""
from __future__ import annotations
import argparse, json, random, re, subprocess, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "videos" / "xdviolence"
REPO = "https://hf-mirror.com/datasets/jherng/xd-violence/resolve/main/"
LABEL_NAME = {"A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
              "B5": "abuse", "B6": "car_accident", "G": "explosion"}


def list_test_videos() -> list[str]:
    import urllib.request
    req = urllib.request.Request("https://hf-mirror.com/api/datasets/jherng/xd-violence",
                                 headers={"User-Agent": "SocialNetSim/1.0"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return [s["rfilename"] for s in d.get("siblings", [])
            if s["rfilename"].endswith(".mp4") and "/test_videos/" in s["rfilename"]]


def parse_labels(fname: str) -> tuple[list[str], bool]:
    m = re.search(r"label_([A-Z0-9\-]+)\.mp4$", fname)
    if not m:
        return [], False
    raw = m.group(1)
    codes = [c for c in raw.split("-") if c and c != "0"]
    names = [LABEL_NAME.get(c, c) for c in codes]
    violent = any(c != "A" for c in codes)
    return names, violent


def load_frame_annotations() -> dict:
    """官方 annotations.txt：每行 `视频id 起帧 止帧 [起帧 止帧 …]`，仅暴力视频有条目。"""
    p = ROOT / "data" / "raw" / "xdv_annotations.txt"
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        vid = parts[0]
        nums = [int(x) for x in parts[1:] if x.isdigit()]
        out[vid] = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    return out


def download(fname: str, dest: Path, timeout: int = 900) -> bool:
    if dest.exists() and dest.stat().st_size > 10000:
        return True
    url = REPO + urllib.parse.quote(fname)
    r = subprocess.run(["curl", "-sL", "--max-time", str(timeout), "--retry", "2", "-o", str(dest), url],
                       capture_output=True)
    ok = dest.exists() and dest.stat().st_size > 10000
    if not ok:
        dest.unlink(missing_ok=True)
    return ok


def main(n_violent: int = 40, n_normal: int = 40, seed: int = 0, max_mb: int = 60):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    files = list_test_videos()
    print(f"官方测试集共 {len(files)} 段")
    ann = load_frame_annotations()
    by_cat: dict[str, list[str]] = {}
    normals = []
    for f in files:
        names, violent = parse_labels(f)
        if violent:
            by_cat.setdefault(names[0], []).append(f)
        else:
            normals.append(f)
    print("类目分布:", {k: len(v) for k, v in sorted(by_cat.items())}, "normal:", len(normals))

    # 各暴力类目均匀抽取
    cats = sorted(by_cat)
    per = max(1, n_violent // len(cats))
    picked = []
    for c in cats:
        picked += [(f, c, True) for f in rng.sample(by_cat[c], min(per, len(by_cat[c])))]
    picked += [(f, "normal", False) for f in rng.sample(normals, min(n_normal, len(normals)))]
    rng.shuffle(picked)

    meta = {}
    ok_n = 0
    for i, (f, cat, violent) in enumerate(picked, 1):
        name = Path(f).name.replace("#", "")
        dest = OUT / name
        got = download(f, dest)
        size_mb = dest.stat().st_size / 1e6 if dest.exists() else 0
        if got and size_mb > max_mb:            # 跳过过大的样本，控制评测耗时
            dest.unlink(missing_ok=True); got = False
        if got:
            vid = Path(f).stem
            names, _ = parse_labels(f)
            meta[name] = {"violent": violent, "categories": names, "source": "XD-Violence test",
                          "orig": vid, "size_mb": round(size_mb, 1),
                          "frame_annotations": ann.get(vid, [])}
            ok_n += 1
        print(f"  [{i}/{len(picked)}] {'OK ' if got else 'SKIP'} {cat:14s} {size_mb:6.1f}MB {name[:58]}", flush=True)
    (OUT / "labels.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    nv = sum(1 for v in meta.values() if v["violent"])
    print(f"\n成功下载 {ok_n} 段：暴力 {nv} / 正常 {ok_n - nv}")
    print(f"其中 {sum(1 for v in meta.values() if v['frame_annotations'])} 段带帧级标注（可做时序定位评估）")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--violent", type=int, default=42)
    ap.add_argument("--normal", type=int, default=42)
    ap.add_argument("--max-mb", type=int, default=60)
    a = ap.parse_args()
    main(a.violent, a.normal, max_mb=a.max_mb)
