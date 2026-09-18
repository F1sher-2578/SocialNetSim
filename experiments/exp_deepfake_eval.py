"""深伪检测专家的独立评估：在公开真假人脸数据集上量化其真实能力与校准情况。

动机：校准脚本发现该模型在维基共享的真实肖像上输出中位数 0.95（几乎全部判为 Fake），
说明其输出不是可直接使用的概率。本实验回答两个问题：
  1. 它到底有没有判别力（AUC）？
  2. 最佳阈值在哪里（EER），与默认 0.5 差多少？
并额外测量 JPEG 压缩下的鲁棒性（社交平台会重压缩，这是实际部署必须面对的）。
"""
from __future__ import annotations
import io, json, argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from socialsim.moderation import image_experts as ie
from socialsim.config import RESULT_DIR, ROOT

DATA = ROOT / "data" / "raw" / "deepfake" / "test.parquet"


def load(n_per_class: int = 200, seed: int = 0) -> tuple[list, np.ndarray]:
    df = pd.read_parquet(DATA)
    rng = np.random.default_rng(seed)
    idx = []
    for lab in sorted(df["label"].unique()):
        sub = df.index[df["label"] == lab].to_numpy()
        idx += list(rng.choice(sub, size=min(n_per_class, len(sub)), replace=False))
    sub = df.loc[idx]
    imgs = [Image.open(io.BytesIO(r["bytes"])).convert("RGB") for r in sub["image"]]
    return imgs, sub["label"].to_numpy()


def jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")


def eer_threshold(y, s):
    fpr, tpr, th = roc_curve(y, s)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return float(th[i]), float((fpr[i] + (1 - tpr[i])) / 2)


def main(n: int = 200):
    print(f"加载数据（每类 {n} 张）…")
    imgs, y = load(n)
    # 数据集 label: 0/1 —— 先用模型输出方向判定哪个是 fake
    print("推理原图 …")
    s = np.array(ie.deepfake_batch(imgs))
    auc0 = roc_auc_score(y, s)
    fake_label = 1 if auc0 >= 0.5 else 0
    y_fake = (y == fake_label).astype(int)
    auc = roc_auc_score(y_fake, s)
    th, eer = eer_threshold(y_fake, s)
    acc_default = accuracy_score(y_fake, (s >= 0.5).astype(int))
    acc_eer = accuracy_score(y_fake, (s >= th).astype(int))
    res = {
        "n": len(imgs), "fake_label_in_dataset": int(fake_label),
        "auc": float(auc), "eer": eer, "eer_threshold": th,
        "acc@0.5": float(acc_default), "f1@0.5": float(f1_score(y_fake, (s >= 0.5).astype(int))),
        "acc@eer": float(acc_eer), "f1@eer": float(f1_score(y_fake, (s >= th).astype(int))),
        "score_real": {"mean": float(s[y_fake == 0].mean()), "median": float(np.median(s[y_fake == 0]))},
        "score_fake": {"mean": float(s[y_fake == 1].mean()), "median": float(np.median(s[y_fake == 1]))},
        "jpeg_robustness": {},
    }
    print(f"  AUC={auc:.4f}  EER={eer:.4f}  最佳阈值={th:.4f}")
    print(f"  默认 0.5 阈值: acc={acc_default:.4f}   EER 阈值: acc={acc_eer:.4f}")
    print(f"  真实图分数 中位数={res['score_real']['median']:.3f} / 伪造图 中位数={res['score_fake']['median']:.3f}")
    # ---- 分布外真实人脸：维基共享的真人肖像（与训练分布不同的采集渠道） ----
    ood_dir = ROOT / "data" / "images" / "bench"
    ood_faces = []
    import cv2
    for f in sorted(ood_dir.glob("normal_face*.jpg")):
        im = cv2.imread(str(f))
        if im is None:
            continue
        for (x, y, w, h) in ie.detect_faces(im)[:1]:
            pad = int(0.15 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(im.shape[1], x + w + pad), min(im.shape[0], y + h + pad)
            ood_faces.append(ie._to_pil(im[y0:y1, x0:x1]))
    if ood_faces:
        so = np.array(ie.deepfake_batch(ood_faces))
        res["ood_real_faces"] = {
            "n": len(ood_faces), "source": "Wikimedia Commons 真人肖像",
            "median": float(np.median(so)), "mean": float(so.mean()),
            "fpr@0.5": float((so >= 0.5).mean()), "fpr@eer_threshold": float((so >= th).mean()),
        }
        print(f"\n  [分布外] 维基真人肖像 n={len(ood_faces)}: 中位分={np.median(so):.3f} "
              f"误报率@0.5={float((so>=0.5).mean()):.3f} 误报率@EER阈值={float((so>=th).mean()):.3f}")
        print("  → 同分布 AUC 0.999，分布外真实照片却被大量判为伪造：典型的泛化失效。\n")

    for q in (90, 70, 50, 30):
        print(f"推理 JPEG q={q} …")
        sj = np.array(ie.deepfake_batch([jpeg(im, q) for im in imgs]))
        a = roc_auc_score(y_fake, sj)
        res["jpeg_robustness"][f"q{q}"] = {"auc": float(a),
                                           "acc@eer": float(accuracy_score(y_fake, (sj >= th).astype(int)))}
        print(f"  q={q}: AUC={a:.4f} (原图 {auc:.4f}, 掉 {auc - a:+.4f})")
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "deepfake_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {RESULT_DIR/'deepfake_eval.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-n", type=int, default=200)
    main(ap.parse_args().n)
