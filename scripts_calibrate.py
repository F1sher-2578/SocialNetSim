"""阈值校准：在"确定正常"的素材上估计各专家分数分布，按目标误报率设定阈值。

对应 DFscan 用 EER 确定判定阈值的做法。零样本专家（CLIP）的绝对置信度不可直接当概率用，
必须在负样本分布上校准，否则卡通、夜景、特写等正常画面会被大量误判。
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np
import cv2

from socialsim.moderation import image_experts as ie
from socialsim.config import MODEL_DIR, ROOT

BENIGN_VIDEOS = ["data/videos/bbb_360p.webm", "data/videos/bench/benign_synthetic_short.mp4",
                 "data/videos/bench/benign_photos_short.mp4"]
# 同域校准：从 XD-Violence 的**正常**视频中取一半作为校准集，另一半留作测试。
# 零样本专家的置信度尺度高度依赖画面域，跨域校准是前一轮 AUC 仅 0.55 的直接原因。
XDV_CALIB_SPLIT = 0.5
BENIGN_IMAGE_DIRS = ["data/images/bench"]
BENIGN_IMAGE_PREFIX = ("normal_",)


def sample_video_frames(path: str, n: int = 60) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1e-6, cap.get(cv2.CAP_PROP_FPS))
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_MSEC, dur * 1000 * (i + 0.5) / n)
        ok, f = cap.read()
        if ok:
            out.append(f)
    cap.release()
    return out


def collect_xdv_calib_videos() -> list[str]:
    """取 XD-Violence 正常视频中按文件名排序的前一半作为校准集。

    划分写死为"排序后前 50%"，保证可复现，且测试时必须排除这些文件。
    """
    import json as _json
    d = ROOT / "data" / "videos" / "xdviolence"
    lab = d / "labels.json"
    if not lab.exists():
        return []
    labels = _json.loads(lab.read_text(encoding="utf-8"))
    normals = sorted(k for k, v in labels.items() if not v.get("violent"))
    return [str(d / k) for k in normals[:int(len(normals) * XDV_CALIB_SPLIT)]]


def collect_benign(include_xdv: bool = True) -> list:
    frames = []
    if include_xdv:
        for v in collect_xdv_calib_videos():
            got = sample_video_frames(v, 12)
            frames += got
        print(f"  XD-Violence 正常视频（校准划分）: {len(frames)} 帧")
    for v in BENIGN_VIDEOS:
        p = ROOT / v
        if p.exists():
            got = sample_video_frames(str(p), 60 if "bbb" in v else 15)
            frames += got
            print(f"  {v}: {len(got)} 帧")
    for d in BENIGN_IMAGE_DIRS:
        for p in sorted((ROOT / d).glob("*.jpg")):
            if p.name.startswith(BENIGN_IMAGE_PREFIX):
                im = cv2.imread(str(p))
                if im is not None:
                    frames.append(im)
    print(f"  合计负样本帧: {len(frames)}")
    return frames


def main(target_fpr: float = 0.01, include_xdv: bool = True, video_level: bool = True):
    print("采样正常素材 …")
    frames = collect_benign(include_xdv=include_xdv)
    pil = [ie._to_pil(f) for f in frames]
    print("计算 CLIP 类目分布 …")
    clip = ie.clip_risk_batch(pil)
    print("计算 NSFW 分布 …")
    nsfw = np.array(ie.nsfw_batch(pil))
    print("计算深伪分布（仅用真实人像照片，卡通人脸不参与）…")
    df_scores = []
    for p_img in sorted((ROOT / "data" / "images" / "bench").glob("normal_face_*.jpg")):
        im = cv2.imread(str(p_img))
        if im is None:
            continue
        for (x, y, w, h) in ie.detect_faces(im)[:2]:
            pad = int(0.15 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(im.shape[1], x + w + pad), min(im.shape[0], y + h + pad)
            df_scores.append(ie.deepfake_batch([ie._to_pil(im[y0:y1, x0:x1])])[0])
    print(f"  真实人脸样本: {len(df_scores)}")
    q = 100 * (1 - target_fpr)
    cal = {"target_fpr": target_fpr, "n_benign_frames": len(frames), "clip": {}, }
    for k in ie.RISK_PROMPTS:
        vals = np.array([c[k] for c in clip])
        cal["clip"][k] = {"p50": float(np.percentile(vals, 50)), "p90": float(np.percentile(vals, 90)),
                          "p95": float(np.percentile(vals, 95)), "p99": float(np.percentile(vals, 99)),
                          "thresh": float(max(np.percentile(vals, q), 0.12)), "max": float(vals.max())}
    from collections import Counter
    cal_domains = Counter(c.get("_domain", "?") for c in clip)
    rsum = np.array([c.get("_risk_sum", 0.0) for c in clip])
    cal["clip"]["_risk_sum"] = {"p50": float(np.percentile(rsum, 50)), "p90": float(np.percentile(rsum, 90)),
                                "p95": float(np.percentile(rsum, 95)), "p99": float(np.percentile(rsum, 99)),
                                "thresh": float(np.percentile(rsum, q)), "max": float(rsum.max())}
    safe = np.array([c["safe"] for c in clip])
    cal["domains"] = dict(cal_domains)
    cal["clip_safe"] = {"p5": float(np.percentile(safe, 5)), "p50": float(np.percentile(safe, 50))}
    cal["nsfw"] = {"p50": float(np.percentile(nsfw, 50)), "p90": float(np.percentile(nsfw, 90)),
                   "p95": float(np.percentile(nsfw, 95)), "p99": float(np.percentile(nsfw, 99)),
                   "thresh": float(max(np.percentile(nsfw, q), 0.5)), "max": float(nsfw.max())}
    if df_scores:
        d = np.array(df_scores)
        cal["deepfake"] = {"n": len(d), "p50": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                           "p95": float(np.percentile(d, 95)), "p99": float(np.percentile(d, 99)),
                           "thresh": float(max(np.percentile(d, q), 0.7)), "max": float(d.max())}
    # ---- 视频级分数分布：用于自适应判定阈值 ----
    (MODEL_DIR / "calibration.json").write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
    if video_level:
        vids = collect_xdv_calib_videos()
        if vids:
            from socialsim.moderation.video_pipeline import analyze_video
            print(f"\n在 {len(vids)} 段正常视频上估计视频级分数分布 …")
            scores = []
            for i, v in enumerate(vids, 1):
                try:
                    r = analyze_video(v, budget=24, save_report=False, do_audio=False, do_ocr=False)
                    scores.append(r["visual_score"])
                    print(f"  [{i}/{len(vids)}] {r['visual_score']:.4f} {Path(v).name[:48]}", flush=True)
                except Exception as e:
                    print(f"  [{i}/{len(vids)}] 失败: {str(e)[:50]}")
            if scores:
                a = np.array(scores)
                cal["video_level"] = {"n": len(a), "p50": float(np.percentile(a, 50)),
                                      "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
                                      "p95": float(np.percentile(a, 95)), "max": float(a.max())}
                (MODEL_DIR / "calibration.json").write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  自适应阈值：复核={cal['video_level']['p75']:.4f}  拦截={cal['video_level']['p95']:.4f}")
    print("\n校准结果（阈值 = 正常素材分布的 %.1f 分位）:" % q)
    for k, v in cal["clip"].items():
        print(f"  clip.{k:16s} p50={v['p50']:.3f} p95={v['p95']:.3f} max={v['max']:.3f} -> thresh={v['thresh']:.3f}")
    print(f"  nsfw             p95={cal['nsfw']['p95']:.3f} max={cal['nsfw']['max']:.3f} -> thresh={cal['nsfw']['thresh']:.3f}")
    if "deepfake" in cal:
        print(f"  deepfake(n={cal['deepfake']['n']}) p50={cal['deepfake']['p50']:.3f} max={cal['deepfake']['max']:.3f} -> thresh={cal['deepfake']['thresh']:.3f}")
    print(f"  safe 类目 p5={cal['clip_safe']['p5']:.3f} p50={cal['clip_safe']['p50']:.3f}")
    return cal


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--no-xdv", action="store_true", help="不使用 XD-Violence 正常视频做同域校准")
    ap.add_argument("--no-video-level", action="store_true", help="跳过视频级阈值估计（较慢）")
    a = ap.parse_args()
    main(a.fpr, not a.no_xdv, not a.no_video_level)
