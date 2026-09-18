"""图像审核模型的对抗攻防实验。

被攻击对象：本项目部署的深伪检测专家（dima806/deepfake_vs_real_image_detection）。
选它而不是 NSFW 模型，是因为我们有带标签的公开真假人脸数据，攻击成功率可以被严格定义为
"原本被正确判为伪造、攻击后被判为真实"的比例；NSFW 模型没有可用的违规样本集。

实验矩阵：
  1. 白盒 FGSM / PGD，扫描扰动预算 eps
  2. 黑盒 SPSA（只查询概率）与查询开销
  3. 非梯度自然扰动（JPEG/缩放/噪声/旋转）——真实攻击者的低成本手段
  4. 防御：输入变换、随机化、集成、特征压缩检测
  5. 自适应攻击（BPDA）——检验防御是否只是梯度混淆
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from socialsim.config import RESULT_DIR, ROOT, device
from socialsim.security.attacks import image_attack as A
from socialsim.security.defenses import image_defense as D

DATA = ROOT / "data" / "raw" / "deepfake" / "test.parquet"


def load_fake_images(n: int, seed: int = 0) -> list[Image.Image]:
    """取"伪造"类样本：攻击目标是让它们被判为真实（逃逸检测）。"""
    df = pd.read_parquet(DATA)
    # 先探测哪个 label 对应 fake
    rng = np.random.default_rng(seed)
    probe_idx = rng.choice(len(df), size=min(60, len(df)), replace=False)
    probe = df.iloc[probe_idx]
    imgs = [Image.open(io.BytesIO(r["bytes"])).convert("RGB") for r in probe["image"]]
    clf = A.load_target("deepfake")
    dev = clf.dev
    scores = np.array([float(clf.prob(A.to_tensor01(im, dev))) for im in imgs])
    means = {int(l): scores[probe["label"].to_numpy() == l].mean() for l in probe["label"].unique()}
    fake_label = max(means, key=means.get)
    sub = df[df["label"] == fake_label]
    idx = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
    return [Image.open(io.BytesIO(r["bytes"])).convert("RGB") for r in sub.iloc[idx]["image"]], fake_label


def main(n: int = 30, steps: int = 20):
    clf = A.load_target("deepfake")
    dev = clf.dev
    print(f"设备: {dev}")
    imgs, fake_label = load_fake_images(n)
    print(f"取 {len(imgs)} 张伪造样本（数据集 label={fake_label}）")

    xs = [A.to_tensor01(im, dev) for im in imgs]
    with torch.no_grad():
        p_clean = np.array([float(clf.prob(x)) for x in xs])
    keep = p_clean >= 0.5          # 只攻击本来就被正确检出的样本
    xs = [x for x, k in zip(xs, keep) if k]
    imgs = [im for im, k in zip(imgs, keep) if k]
    print(f"其中被正确检出为伪造的 {len(xs)} 张（干净准确率 {keep.mean():.3f}），以下攻击均在这些样本上进行\n")

    res = {"n_attacked": len(xs), "clean_detection_rate": float(keep.mean()),
           "clean_mean_prob": float(p_clean[keep].mean()), "device": str(dev)}

    # ---------- 1. 白盒 ----------
    print("== 白盒攻击（攻击成功 = 伪造样本被判为真实）==")
    res["whitebox"] = {}
    for eps_n in (1, 2, 4, 8, 16):
        eps = eps_n / 255
        t0 = time.time()
        p_f, p_p, linfs, psnrs = [], [], [], []
        for x in xs:
            xf = A.fgsm(clf, x, eps)
            xp = A.pgd(clf, x, eps, steps=steps)
            with torch.no_grad():
                p_f.append(float(clf.prob(xf))); p_p.append(float(clf.prob(xp)))
            linfs.append(A.linf(xp, x)); psnrs.append(A.psnr(xp, x))
        res["whitebox"][f"eps_{eps_n}/255"] = {
            "fgsm_asr": float(np.mean(np.array(p_f) < 0.5)), "fgsm_mean_prob": float(np.mean(p_f)),
            "pgd_asr": float(np.mean(np.array(p_p) < 0.5)), "pgd_mean_prob": float(np.mean(p_p)),
            "pgd_linf": float(np.mean(linfs)), "pgd_psnr_db": float(np.mean(psnrs)),
            "time_s": round(time.time() - t0, 1)}
        r = res["whitebox"][f"eps_{eps_n}/255"]
        print(f"  eps={eps_n:2d}/255  FGSM 成功率={r['fgsm_asr']:.3f}  PGD 成功率={r['pgd_asr']:.3f}  "
              f"PGD 后均值={r['pgd_mean_prob']:.3f}  PSNR={r['pgd_psnr_db']:.1f}dB")

    # ---------- 2. 黑盒 SPSA ----------
    print("\n== 黑盒攻击 SPSA（仅查询概率，不用梯度）==")
    sub = xs[:min(8, len(xs))]
    res["blackbox_spsa"] = {}
    for eps_n in (8, 16):
        eps = eps_n / 255
        t0 = time.time(); succ, qs = 0, 0
        for x in sub:
            adv, q = A.spsa(clf, x, eps, steps=30, samples=12)
            qs += q
            with torch.no_grad():
                succ += float(clf.prob(adv)) < 0.5
        res["blackbox_spsa"][f"eps_{eps_n}/255"] = {
            "asr": succ / len(sub), "avg_queries": qs / len(sub), "time_s": round(time.time() - t0, 1)}
        r = res["blackbox_spsa"][f"eps_{eps_n}/255"]
        print(f"  eps={eps_n}/255 成功率={r['asr']:.3f} 平均查询次数={r['avg_queries']:.0f} 耗时={r['time_s']}s")

    # ---------- 3. 非梯度自然扰动 ----------
    print("\n== 非梯度自然扰动（零模型访问成本）==")
    agg: dict[str, list[float]] = {}
    for im in imgs[:min(20, len(imgs))]:
        for name, pim in A.natural_perturbations(im).items():
            with torch.no_grad():
                agg.setdefault(name, []).append(float(clf.prob(A.to_tensor01(pim, dev))))
    res["natural"] = {k: {"asr": float(np.mean(np.array(v) < 0.5)), "mean_prob": float(np.mean(v))}
                      for k, v in agg.items()}
    for k, v in sorted(res["natural"].items(), key=lambda kv: -kv[1]["asr"]):
        print(f"  {k:16s} 成功率={v['asr']:.3f} 均值={v['mean_prob']:.3f}")

    # ---------- 4. 防御（在多个扰动预算下评估）----------
    # 只报告某一个 eps 会得出片面结论：强攻击下所有轻量防御都失效，
    # 弱攻击下部分防御有效。两者都要给出，才能说明防御的适用边界。
    cleans = xs[:min(20, len(xs))]
    res["defense_by_eps"] = {}
    for eps_n in (1, 2, 4, 8):
        eps = eps_n / 255
        advs = [A.pgd(clf, x, eps, steps=steps) for x in cleans]
        print(f"\n== 防御效果（对 eps={eps_n}/255 的 PGD 对抗样本）==")
        block = {}
        for name, fn in D.DEFENSES.items():
            with torch.no_grad():
                p_adv = [float(clf.prob(fn(a))) for a in advs]
                p_cln = [float(clf.prob(fn(c))) for c in cleans]
            block[name] = {"recovered_rate": float(np.mean(np.array(p_adv) >= 0.5)),
                           "clean_accuracy": float(np.mean(np.array(p_cln) >= 0.5)),
                           "adv_mean_prob": float(np.mean(p_adv))}
            print(f"  {name:12s} 对抗样本恢复率={block[name]['recovered_rate']:.3f}  "
                  f"干净样本准确率={block[name]['clean_accuracy']:.3f}")
        p_rand_adv = [D.randomized_predict(clf, a) for a in advs]
        p_rand_cln = [D.randomized_predict(clf, c) for c in cleans]
        block["randomized_ensemble"] = {
            "recovered_rate": float(np.mean(np.array(p_rand_adv) >= 0.5)),
            "clean_accuracy": float(np.mean(np.array(p_rand_cln) >= 0.5)),
            "adv_mean_prob": float(np.mean(p_rand_adv))}
        print(f"  {'随机化集成':12s} 对抗样本恢复率={block['randomized_ensemble']['recovered_rate']:.3f}  "
              f"干净样本准确率={block['randomized_ensemble']['clean_accuracy']:.3f}")
        res["defense_by_eps"][f"eps_{eps_n}/255"] = block
        if eps_n == 8:
            res["defense"] = block
            advs8 = advs

    # ---------- 5. 对抗样本检测（特征压缩）----------
    print("\n== 对抗样本检测（Feature Squeezing）==")
    advs = advs8
    d_adv = [D.feature_squeezing_score(clf, a)["max_diff"] for a in advs]
    d_cln = [D.feature_squeezing_score(clf, c)["max_diff"] for c in cleans]
    from sklearn.metrics import roc_auc_score
    y = np.array([1] * len(d_adv) + [0] * len(d_cln))
    s = np.array(d_adv + d_cln)
    best = max(((float(np.mean(np.array(d_adv) >= t)), float(np.mean(np.array(d_cln) >= t)), t)
                for t in np.arange(0.05, 0.95, 0.05)), key=lambda z: z[0] - z[1])
    res["feature_squeezing"] = {"auc": float(roc_auc_score(y, s)), "adv_mean_diff": float(np.mean(d_adv)),
                                "clean_mean_diff": float(np.mean(d_cln)),
                                "best_threshold": best[2], "tpr": best[0], "fpr": best[1]}
    print(f"  AUC={res['feature_squeezing']['auc']:.3f}  对抗样本平均差异={np.mean(d_adv):.3f} "
          f"干净={np.mean(d_cln):.3f}  最佳阈值={best[2]:.2f} 检出率={best[0]:.2f} 误报={best[1]:.2f}")

    # 弱扰动下的检测能力（此时对抗样本尚未把概率压到 0，压缩更可能改变预测）
    advs2 = [A.pgd(clf, x, 2 / 255, steps=steps) for x in cleans]
    d_adv2 = [D.feature_squeezing_score(clf, a)["max_diff"] for a in advs2]
    y2 = np.array([1] * len(d_adv2) + [0] * len(d_cln))
    res["feature_squeezing_eps2"] = {"auc": float(roc_auc_score(y2, np.array(d_adv2 + d_cln))),
                                     "adv_mean_diff": float(np.mean(d_adv2))}
    print(f"  eps=2/255 时 AUC={res['feature_squeezing_eps2']['auc']:.3f} "
          f"（对抗样本平均差异={np.mean(d_adv2):.3f}）")

    # ---------- 6. 自适应攻击 BPDA ----------
    # 关键对照必须在**防御确实有效的扰动预算**下做：
    # 在 eps=8/255 下静态攻击已经打穿一切，自适应攻击的增量无从体现。
    print("\n== 自适应攻击 BPDA（攻击者已知防御的存在）==")
    res["adaptive_bpda_by_eps"] = {}
    for eps_n in (1, 2, 8):
        e = eps_n / 255
        blk = {}
        for name in ("jpeg75", "bit3", "bit4", "median3"):
            fn = D.DEFENSES[name]
            n_ = min(10, len(xs))
            succ = 0
            for x in xs[:n_]:
                adv = D.bpda_pgd(clf, fn, x, e, steps=steps)
                with torch.no_grad():
                    succ += float(clf.prob(fn(adv))) < 0.5
            static = res["defense_by_eps"][f"eps_{eps_n}/255"][name]["recovered_rate"]
            blk[name] = {"asr": succ / n_, "adaptive_recovered": 1 - succ / n_,
                         "static_recovered": static}
            print(f"  eps={eps_n}/255 {name:10s} 静态攻击下恢复率={static:.3f} → "
                  f"自适应攻击下恢复率={1 - succ / n_:.3f}（攻击成功率 {succ/n_:.3f}）")
        res["adaptive_bpda_by_eps"][f"eps_{eps_n}/255"] = blk
        if eps_n == 8:
            res["adaptive_bpda"] = {k: {"asr": v["asr"], "static_recovered": v["static_recovered"]}
                                    for k, v in blk.items()}

    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "adv_image.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {RESULT_DIR/'adv_image.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    main(a.n, a.steps)
