"""零样本提示集的快速迭代：先把关键帧的 CLIP 图像嵌入缓存下来，再离线比较不同提示集。

动机：整条视频流水线跑一遍 62 段要十几分钟，而提示集的调整只影响文本侧。
把图像嵌入缓存后，换一套提示只需要几百毫秒的矩阵乘法，可以快速做提示工程与消融。

这也是 2025–2026 免训练视频异常检测工作的通行做法：
性能主要来自"正常侧"和"异常侧"提示的精细设计，而不是更换视觉骨干。
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from socialsim.config import RESULT_DIR, ROOT, HF_CLIP, device
from socialsim.moderation import video_pipeline as VP
from socialsim.moderation import image_experts as IE

XDV = ROOT / "data" / "videos" / "xdviolence"
CACHE = RESULT_DIR / "clip_cache.npz"


# ---------------------------------------------------------------- 提示集
PROMPTS_V1 = {   # 现用版本：通用、笼统
    "risk": {
        "violence": ["a photo of people fighting violently", "a violent assault with weapons",
                     "a scene of physical violence and beating"],
        "gore": ["a bloody gory image with blood and wounds", "a graphic injury with a lot of blood"],
        "weapon": ["a photo of a gun or firearm", "a person holding a knife as a weapon"],
        "porn": ["a pornographic photo of a naked person", "explicit sexual content"],
        "smoking_drug": ["a person smoking a cigarette", "illegal drugs and syringes"],
        "protest_crowd": ["a large crowd protest with banners", "a riot in the street"],
        "fire_disaster": ["a building on fire with heavy smoke", "an explosion disaster scene"],
        "horror": ["a scary horror image of a monster", "a disturbing creepy scene"],
    },
    "safe": ["a normal everyday photo", "a photo of scenery or landscape", "a photo of food on a table",
             "a photo of a person talking to the camera", "a cartoon animation frame",
             "a screenshot of text or a document", "a photo of a pet animal", "a photo of a product",
             "an empty room or plain background", "a person sitting and talking indoors"],
}

# V2：按 XD-Violence 的实际类目重写异常侧，并大幅加厚"正常侧"
# （正常侧建模是免训练方法的性能关键：正常帧必须有强锚点，否则任何画面都"有点像"异常）
PROMPTS_V2 = {
    "risk": {
        "fighting": ["two people punching and kicking each other in a fight",
                     "a brutal street fight with people brawling",
                     "a man beating another person on the ground"],
        "shooting": ["a person firing a gun with muzzle flash",
                     "an armed man shooting a pistol at someone",
                     "a shootout with people taking cover from gunfire"],
        "riot": ["a violent riot with a crowd clashing with police",
                 "rioters throwing objects and burning things in the street",
                 "a mob fighting with riot police in tear gas"],
        "abuse": ["an adult violently abusing and hitting a victim",
                  "a person being physically assaulted and dragged"],
        "car_accident": ["a severe car crash with wrecked vehicles",
                         "a traffic accident with a car flipping over",
                         "a vehicle collision with debris on the road"],
        "explosion": ["a huge explosion with a fireball and smoke",
                      "a building exploding with flames bursting out",
                      "a bomb blast destroying a structure"],
        "gore": ["a bloody wounded body with visible blood",
                 "a graphic injury with blood splatter on skin"],
        "weapon": ["a close-up of a gun held by a person",
                   "a person brandishing a knife threateningly"],
        "porn": ["a pornographic photo of a naked person", "explicit sexual content"],
    },
    "safe": [
        # 影视素材的常见正常画面（XD-Violence 大量来自电影）
        "a movie scene of two people having a conversation",
        "a close-up portrait of a person's face talking",
        "an indoor scene of people sitting in a room",
        "people walking calmly on a street",
        "a car driving normally on a road",
        "a wide landscape or city skyline view",
        "a dark scene with low lighting and no action",
        "a person standing still and looking at something",
        "a group of people gathered peacefully",
        "opening or closing credits with text on screen",
        "a sports game or performance on a stage",
        "a news anchor or interview in a studio",
        # 通用正常
        "a normal everyday photo", "a photo of food on a table", "a photo of a pet animal",
        "a cartoon animation frame", "an empty room or plain background",
    ],
}

# V3：在 V2 基础上，正常侧再加入"看起来紧张但并不暴力"的困难负样本
PROMPTS_V3 = {
    "risk": PROMPTS_V2["risk"],
    "safe": PROMPTS_V2["safe"] + [
        "an action movie scene with running but no violence",
        "a person shouting or arguing without physical contact",
        "a dramatic scene with tense facial expressions",
        "a dark alley at night with nobody around",
        "a crowded street with many people walking",
        "a hand holding an ordinary object like a phone or tool",
        "red colored lighting or a red object in the scene",
        "smoke or fog in a scene without fire",
    ],
}

PROMPT_SETS = {"V1_通用": PROMPTS_V1, "V2_类目对齐": PROMPTS_V2, "V3_加困难负样本": PROMPTS_V3}


# ---------------------------------------------------------------- 特征缓存
def build_cache(budget: int = 24, limit: int | None = None, force: bool = False) -> dict:
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=True)
        print(f"复用缓存 {CACHE}（{len(z['files'])} 段视频，{z['emb'].shape[0]} 帧）")
        return {"emb": z["emb"], "vid": z["vid"], "files": list(z["files"]), "truth": z["truth"]}

    import scripts_calibrate as CAL
    labels = json.loads((XDV / "labels.json").read_text(encoding="utf-8"))
    calib = {Path(x).name for x in CAL.collect_xdv_calib_videos()}
    items = sorted((k, v) for k, v in labels.items() if (XDV / k).exists() and k not in calib)
    if limit:
        items = items[:limit]
    print(f"提取 {len(items)} 段视频的关键帧 CLIP 嵌入 …")

    proc, mdl, dev, *_ = IE._clip()
    embs, vids, files, truths = [], [], [], []
    for i, (name, lab) in enumerate(items, 1):
        p = XDV / name
        t0 = time.time()
        try:
            meta = VP.probe(p)
            cands = VP.fast_scan(p, meta, budget)
            times = VP.select_keyframes(cands, budget)
            frames = VP.grab_frames(p, times)
            if not frames:
                continue
            pil = [IE._to_pil(f) for _, f in frames]
            vs = []
            for j in range(0, len(pil), 16):
                inp = proc(images=pil[j:j + 16], return_tensors="pt")
                inp = {k: v.to(dev) for k, v in inp.items()}
                with torch.no_grad():
                    e = mdl.get_image_features(**inp)
                    e = e / e.norm(dim=-1, keepdim=True)
                vs.append(e.float().cpu().numpy())
            e = np.concatenate(vs)
            embs.append(e); vids += [len(files)] * len(e)
            files.append(name); truths.append(int(bool(lab.get("violent"))))
            print(f"  [{i}/{len(items)}] {len(e):2d} 帧 {time.time()-t0:5.1f}s {name[:50]}", flush=True)
        except Exception as ex:
            print(f"  [{i}/{len(items)}] 失败 {name[:40]}: {str(ex)[:50]}")
    emb = np.concatenate(embs).astype(np.float32)
    out = {"emb": emb, "vid": np.array(vids), "files": files, "truth": np.array(truths)}
    RESULT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, emb=emb, vid=out["vid"], files=np.array(files), truth=out["truth"])
    print(f"缓存已保存：{emb.shape[0]} 帧 / {len(files)} 段视频")
    return out


# ---------------------------------------------------------------- 评估
def encode_prompts(pset: dict):
    proc, mdl, dev, *_ = IE._clip()
    prompts, owners = [], []
    for k, ps in pset["risk"].items():
        prompts += ps; owners += [k] * len(ps)
    prompts += pset["safe"]; owners += ["safe"] * len(pset["safe"])
    with torch.no_grad():
        tf = proc(text=prompts, return_tensors="pt", padding=True)
        tf = {k: v.to(dev) for k, v in tf.items()}
        t = mdl.get_text_features(**tf)
        t = t / t.norm(dim=-1, keepdim=True)
    return t.float().cpu().numpy(), np.array(owners)


def frame_risk(emb: np.ndarray, temb: np.ndarray, owners: np.ndarray, logit_scale: float = 100.0,
               agg: str = "sum") -> np.ndarray:
    """帧级风险 = 各风险类目概率之和（或最大值）。safe 类目参与 softmax 竞争。"""
    logits = emb @ temb.T * logit_scale
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    risk_mask = owners != "safe"
    cat_p = []
    for k in [c for c in dict.fromkeys(owners) if c != "safe"]:
        cat_p.append(p[:, owners == k].sum(axis=1))
    cat_p = np.stack(cat_p, axis=1)
    return cat_p.max(axis=1) if agg == "max" else p[:, risk_mask].sum(axis=1)


def video_score(frame_scores: np.ndarray, vid: np.ndarray, n_vid: int, mode: str = "top15") -> np.ndarray:
    out = np.zeros(n_vid)
    for i in range(n_vid):
        s = frame_scores[vid == i]
        if len(s) == 0:
            continue
        if mode == "max":
            out[i] = s.max()
        elif mode == "mean":
            out[i] = s.mean()
        elif mode.startswith("top"):
            k = max(1, int(len(s) * int(mode[3:]) / 100))
            out[i] = np.sort(s)[-k:].mean()
    return out


def main(budget: int = 24, limit: int | None = None, force: bool = False):
    from sklearn.metrics import roc_auc_score, average_precision_score
    c = build_cache(budget, limit, force)
    emb, vid, files, truth = c["emb"], c["vid"], c["files"], c["truth"]
    n = len(files)
    print(f"\n评估：{n} 段视频（暴力 {int(truth.sum())}），共 {emb.shape[0]} 帧\n")

    res = {"n_videos": n, "n_pos": int(truth.sum()), "n_frames": int(emb.shape[0]), "grid": {}}
    best = None
    for pname, pset in PROMPT_SETS.items():
        temb, owners = encode_prompts(pset)
        for agg in ("sum", "max"):
            fr = frame_risk(emb, temb, owners, agg=agg)
            for mode in ("max", "top10", "top15", "top30", "mean"):
                vs = video_score(fr, vid, n, mode)
                auc = float(roc_auc_score(truth, vs))
                ap = float(average_precision_score(truth, vs))
                key = f"{pname}|类目聚合={agg}|时序聚合={mode}"
                res["grid"][key] = {"auc": round(auc, 4), "ap": round(ap, 4)}
                if best is None or auc > best[1]:
                    best = (key, auc, ap, pname, agg, mode)
    print("== 提示集 × 聚合方式 网格搜索（按 AUC 排序，前 15）==")
    for k, v in sorted(res["grid"].items(), key=lambda kv: -kv[1]["auc"])[:15]:
        print(f"  AUC={v['auc']:.4f} AP={v['ap']:.4f}  {k}")
    print(f"\n最佳：{best[0]}  AUC={best[1]:.4f} AP={best[2]:.4f}")
    res["best"] = {"key": best[0], "auc": round(best[1], 4), "ap": round(best[2], 4),
                   "prompt_set": best[3], "cat_agg": best[4], "temporal_agg": best[5]}

    # 各提示集的最好成绩对比
    print("\n== 各提示集的最好成绩 ==")
    per = {}
    for k, v in res["grid"].items():
        pn = k.split("|")[0]
        if pn not in per or v["auc"] > per[pn]["auc"]:
            per[pn] = {**v, "config": k}
    for pn, v in per.items():
        print(f"  {pn:16s} AUC={v['auc']:.4f} AP={v['ap']:.4f}  ({v['config'].split('|',1)[1]})")
    res["per_prompt_set"] = per

    (RESULT_DIR / "prompt_tuning.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {RESULT_DIR/'prompt_tuning.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.budget, a.limit, a.force)
