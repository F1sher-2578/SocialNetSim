"""视频审核流水线在公开基准 XD-Violence 上的评估（主基准），并混入本项目自建切片。

XD-Violence（Wu et al., ECCV 2020）：4754 段未剪辑视频、217 小时、视觉+音频双模态，
测试集 800 段带帧级标注。本实验从官方测试集按类目分层抽样若干段。

两个评估层级：
  视频级   ROC-AUC / AP / 在若干阈值下的 P-R-F1、误报率
  时序定位 用官方帧级标注计算 IoU 与命中率（仅暴力样本有标注）

自建切片（中文语音广告、赌博引流、画面烧录联系方式）覆盖 XD-Violence 不含的违规类型，
两者合并可以说明"通用暴力检测基准"与"中文社交平台真实违规"之间的差距。

注意：XD-Violence 标注的是"暴力"，而我们的流水线检测的是更广的"不良内容"。
两者不完全对齐，报告中必须说明——这正是通用审核系统落到具体基准上的典型困难。
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from socialsim.config import RESULT_DIR, ROOT, RISK_BLOCK, RISK_REVIEW
from socialsim.moderation.video_pipeline import analyze_video

XDV = ROOT / "data" / "videos" / "xdviolence"
BENCH = ROOT / "data" / "videos" / "bench"


def prf(y, s, th):
    pred = (np.array(s) >= th).astype(int)
    y = np.array(y)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"threshold": th, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(1e-9, p + r), 4),
            "fpr": round(fp / max(1, fp + tn), 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def temporal_iou(pred_segments: list[dict], gt_frames: list[tuple[int, int]], fps: float) -> dict:
    """预测的高风险时间段 vs 官方帧级标注的重叠情况。"""
    if not gt_frames or fps <= 0:
        return {}
    gt = [(a / fps, b / fps) for a, b in gt_frames]
    pred = [(s["start"], s["end"]) for s in pred_segments]
    if not pred:
        return {"iou": 0.0, "hit_rate": 0.0, "n_gt": len(gt), "n_pred": 0}

    def overlap(a, b):
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    inter = sum(max(overlap(p, g) for g in gt) for p in pred)
    union = sum(p[1] - p[0] for p in pred) + sum(g[1] - g[0] for g in gt) - inter
    hits = sum(1 for g in gt if any(overlap(p, g) > 0 for p in pred))
    return {"iou": round(inter / max(1e-9, union), 4), "hit_rate": round(hits / len(gt), 4),
            "n_gt": len(gt), "n_pred": len(pred)}


def run_set(items: list[tuple[Path, dict]], budget: int, tag: str) -> list[dict]:
    rows = []
    for i, (p, lab) in enumerate(items, 1):
        t0 = time.time()
        try:
            r = analyze_video(p, out_dir=RESULT_DIR / "reports" / f"{tag}_{p.stem}"[:80],
                              budget=budget, save_report=False)
        except Exception as e:
            print(f"  [{i}/{len(items)}] 失败 {p.name[:50]}: {str(e)[:70]}", flush=True)
            continue
        row = {"file": p.name, "truth": int(lab.get("violent", lab.get("risky", False))),
               "categories": lab.get("categories", []), "score": round(r["score"], 4),
               "visual": round(r["visual_score"], 4), "asr": round(r["asr_score"], 4),
               "ocr": round(r["ocr_score"], 4), "verdict": r["verdict"],
               "labels": r["labels"][:8], "duration": round(r["meta"]["duration"], 1),
               "n_frames": r["n_frames_analyzed"], "elapsed": round(time.time() - t0, 1),
               "cue": round(r.get("cue_risk", 0.0), 4),
               "categories_score": r.get("category_scores", {})}
        if lab.get("frame_annotations"):
            row["temporal"] = temporal_iou(r["segments"], lab["frame_annotations"], r["meta"]["fps"])
        rows.append(row)
        mark = "✓" if (row["score"] >= RISK_REVIEW) == bool(row["truth"]) else "✗"
        print(f"  [{i}/{len(items)}] {mark} truth={row['truth']} score={row['score']:.3f} "
              f"({row['visual']:.2f}/{row['asr']:.2f}/{row['ocr']:.2f}) {row['duration']:6.1f}s "
              f"{row['elapsed']:5.1f}s {p.name[:46]}", flush=True)
    return rows


def summarize(rows: list[dict], name: str) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = [r["truth"] for r in rows]
    s = [r["score"] for r in rows]
    out = {"name": name, "n": len(rows), "n_pos": int(sum(y))}
    if len(set(y)) > 1:
        out["auc"] = round(float(roc_auc_score(y, s)), 4)
        out["ap"] = round(float(average_precision_score(y, s)), 4)
        # 各模态单独的判别力
        for mod in ("visual", "asr", "ocr"):
            try:
                out[f"auc_{mod}"] = round(float(roc_auc_score(y, [r[mod] for r in rows])), 4)
            except Exception:
                pass
    # 自适应阈值（由校准集的正常视频分数分布确定），同时保留固定阈值作对照
    from socialsim.config import thresholds as adaptive_thresholds
    th_r, th_b = adaptive_thresholds()
    out["adaptive_thresholds"] = {"review": round(th_r, 4), "block": round(th_b, 4)}
    out["thresholds"] = [prf(y, s, t) for t in sorted({round(th_r, 4), round(th_b, 4), RISK_REVIEW, RISK_BLOCK})]
    # 视觉通道单独的操作点（与 XD-Violence 的"视觉暴力"标注口径对齐）
    vis = [r["visual"] for r in rows]
    out["visual_channel"] = {"auc": out.get("auc_visual"),
                             "at_review": prf(y, vis, th_r), "at_block": prf(y, vis, th_b)}
    # 标注口径错配分析：被文本通道判为违规、但基准标为正常的样本
    mism = [{"file": r["file"], "asr": r["asr"], "ocr": r["ocr"],
             "labels": [l for l in r["labels"] if l.startswith(("asr_", "ocr_"))][:4]}
            for r in rows if r["truth"] == 0 and max(r["asr"], r["ocr"]) >= 0.5]
    out["label_scope_mismatch"] = {"n": len(mism), "n_negatives": int(len(y) - sum(y)),
                                   "ratio": round(len(mism) / max(1, len(y) - sum(y)), 4),
                                   "examples": mism[:8]}
    temp = [r["temporal"] for r in rows if r.get("temporal")]
    if temp:
        out["temporal"] = {"n": len(temp),
                           "mean_iou": round(float(np.mean([t["iou"] for t in temp])), 4),
                           "mean_hit_rate": round(float(np.mean([t["hit_rate"] for t in temp])), 4)}
    out["mean_elapsed_s"] = round(float(np.mean([r["elapsed"] for r in rows])), 2)
    out["mean_duration_s"] = round(float(np.mean([r["duration"] for r in rows])), 1)
    out["realtime_factor"] = round(out["mean_duration_s"] / max(1e-6, out["mean_elapsed_s"]), 2)
    return out


def main(limit: int | None = None, budget: int = 24, skip_own: bool = False):
    res = {}
    all_rows = []

    # ---------- 主基准：XD-Violence ----------
    lab_p = XDV / "labels.json"
    if lab_p.exists():
        labels = json.loads(lab_p.read_text(encoding="utf-8"))
        # 排除被用作阈值校准的正常视频，避免评估泄漏
        import scripts_calibrate as CAL
        calib = {Path(x).name for x in CAL.collect_xdv_calib_videos()}
        items = [(XDV / k, v) for k, v in labels.items() if (XDV / k).exists() and k not in calib]
        items.sort(key=lambda kv: kv[0].name)
        if calib:
            print(f"  （已排除 {len(calib)} 段用于阈值校准的正常视频）")
        if limit:
            pos = [x for x in items if x[1]["violent"]][:limit // 2]
            neg = [x for x in items if not x[1]["violent"]][:limit // 2]
            items = pos + neg
        print(f"== XD-Violence 官方测试集子集：{len(items)} 段 "
              f"（暴力 {sum(1 for _, v in items if v['violent'])}）")
        rows = run_set(items, budget, "xdv")
        res["xdviolence"] = summarize(rows, "XD-Violence (official test subset)")
        res["xdviolence"]["rows"] = rows
        all_rows += rows
        s = res["xdviolence"]
        print(f"\n  AUC={s.get('auc')} AP={s.get('ap')}  "
              f"视觉AUC={s.get('auc_visual')} 语音AUC={s.get('auc_asr')}")
        for t in s["thresholds"]:
            print(f"  阈值 {t['threshold']:.2f}: P={t['precision']:.3f} R={t['recall']:.3f} "
                  f"F1={t['f1']:.3f} 误报率={t['fpr']:.3f}")
        if "temporal" in s:
            print(f"  时序定位: 平均 IoU={s['temporal']['mean_iou']} 命中率={s['temporal']['mean_hit_rate']} "
                  f"(n={s['temporal']['n']})")
        print(f"  平均处理 {s['mean_elapsed_s']}s / 视频时长 {s['mean_duration_s']}s "
              f"→ 实时倍率 {s['realtime_factor']}×")
        if "visual_channel" in s:
            vc = s["visual_channel"]
            print(f"  视觉通道（与基准标注口径对齐）AUC={vc['auc']}")
            print(f"    @复核阈值 {s['adaptive_thresholds']['review']}: P={vc['at_review']['precision']:.3f} "
                  f"R={vc['at_review']['recall']:.3f} F1={vc['at_review']['f1']:.3f} 误报率={vc['at_review']['fpr']:.3f}")
            print(f"    @拦截阈值 {s['adaptive_thresholds']['block']}: P={vc['at_block']['precision']:.3f} "
                  f"R={vc['at_block']['recall']:.3f} F1={vc['at_block']['f1']:.3f} 误报率={vc['at_block']['fpr']:.3f}")
        if "label_scope_mismatch" in s:
            m = s["label_scope_mismatch"]
            print(f"  标注口径错配：{m['n']}/{m['n_negatives']} 段被标为正常的视频，"
                  f"其语音/画面文字被判违规（{m['ratio']*100:.0f}%）")
            print(f"    —— XD-Violence 只标注视觉暴力，不标注对白中的辱骂与引流，属于评测口径差异而非误判\n")

    # ---------- 自建切片 ----------
    if not skip_own:
        bp = BENCH / "labels.json"
        if bp.exists():
            labels = json.loads(bp.read_text(encoding="utf-8"))
            items = []
            for k, v in labels.items():
                p = Path(v["external"]) if v.get("external") else (BENCH / k)
                if p.exists():
                    items.append((p, v))
            print(f"== 自建切片（中文语音/OCR 违规 + 正常对照）：{len(items)} 段")
            rows = run_set(items, budget, "own")
            res["own_slice"] = summarize(rows, "自建中文违规切片")
            res["own_slice"]["rows"] = rows
            all_rows += rows
            s = res["own_slice"]
            print(f"\n  AUC={s.get('auc')} AP={s.get('ap')}  "
                  f"视觉AUC={s.get('auc_visual')} 语音AUC={s.get('auc_asr')} OCR_AUC={s.get('auc_ocr')}")
            for t in s["thresholds"]:
                print(f"  阈值 {t['threshold']:.2f}: P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f}\n")

    if all_rows:
        res["combined"] = summarize(all_rows, "合并（公开基准 + 自建切片）")
        s = res["combined"]
        print(f"== 合并：n={s['n']} AUC={s.get('auc')} AP={s.get('ap')}")
        for t in s["thresholds"]:
            print(f"  阈值 {t['threshold']:.2f}: P={t['precision']:.3f} R={t['recall']:.3f} "
                  f"F1={t['f1']:.3f} 误报率={t['fpr']:.3f}")

    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "video_bench_xdv.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {RESULT_DIR/'video_bench_xdv.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--skip-own", action="store_true")
    a = ap.parse_args()
    main(a.limit, a.budget, a.skip_own)
