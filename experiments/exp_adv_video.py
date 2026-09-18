"""视频审核流水线的攻击与防御实验。

视频场景独有的攻击面是**采样策略**：审核系统不看每一帧，攻击者只要让违规内容
避开被采样的时刻，或只攻破被采样的少数帧，就能整段逃逸。

实验内容：
  A1 时序稀疏攻击   只对被采样的关键帧加对抗扰动（扰动帧占比通常 <5%）
  A2 关键帧躲避     把违规片段压缩到关键帧最小间隔以下
  A3 镜头切换灌水   插入大量闪帧，占满关键帧预算
  A4 重压缩 / 缩放 / 画中画

  D1 随机化采样     采样时刻不可预测，攻击者无法针对固定帧构造扰动
  D2 提高采样预算   直接代价是处理时间
  D3 切换率异常检测 识别 A3 这类灌水攻击
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from socialsim.config import RESULT_DIR, ROOT, RISK_REVIEW
from socialsim.moderation import video_pipeline as VP
from socialsim.security.attacks import video_attack as VA

XDV = ROOT / "data" / "videos" / "xdviolence"
BENCH = ROOT / "data" / "videos" / "bench"


def pick_targets(n: int = 6) -> list[Path]:
    """选取原本能被正确检出为高风险的视频作为攻击目标。"""
    out = []
    lab = XDV / "labels.json"
    if lab.exists():
        labels = json.loads(lab.read_text(encoding="utf-8"))
        vio = sorted(k for k, v in labels.items() if v.get("violent"))
        for k in vio:
            p = XDV / k
            if p.exists() and p.stat().st_size < 25e6:
                out.append(p)
    own = BENCH / "labels.json"
    if own.exists():
        labels = json.loads(own.read_text(encoding="utf-8"))
        for k, v in labels.items():
            if v.get("risky") and (BENCH / k).exists():
                out.append(BENCH / k)
    return out[:n * 3]


def score_of(path: Path, budget: int = 24, randomize: bool = False) -> dict:
    r = VP.analyze_video(path, out_dir=RESULT_DIR / "reports" / "advvid", budget=budget,
                         save_report=False, slow_pass=not randomize)
    return {"score": r["score"], "verdict": r["verdict"], "visual": r["visual_score"],
            "asr": r["asr_score"], "ocr": r["ocr_score"], "cuts": r["n_scene_cuts"],
            "duration": r["meta"]["duration"], "frames": r["n_frames_analyzed"],
            "hot_times": [f["t"] for f in sorted(r["frames"], key=lambda x: -x["score"])[:8]]}


def main(n: int = 4, budget: int = 24, skip_pgd: bool = False):
    from socialsim.security.attacks import image_attack as IA
    cands = pick_targets(n)
    print(f"候选目标 {len(cands)} 段，先筛出原本就被判为高风险的\n")

    targets = []
    for p in cands:
        if len(targets) >= n:
            break
        try:
            base = score_of(p, budget)
        except Exception as e:
            print(f"  跳过 {p.name[:40]}: {str(e)[:50]}")
            continue
        if base["score"] >= RISK_REVIEW:
            targets.append((p, base))
            print(f"  选中 {p.name[:46]:48s} 原始分={base['score']:.3f} ({base['verdict']}) "
                  f"时长={base['duration']:.0f}s 切换={base['cuts']}")
    if not targets:
        print("没有可用目标"); return {}

    res = {"n_targets": len(targets), "budget": budget, "attacks": {}, "defenses": {}}
    tmp = Path(tempfile.mkdtemp())
    clf = IA.load_target("nsfw")

    def record(name, before, after, extra=None):
        d = res["attacks"].setdefault(name, {"before": [], "after": [], "escaped": 0, "n": 0, "extra": []})
        d["before"].append(before["score"]); d["after"].append(after["score"])
        d["escaped"] += after["score"] < RISK_REVIEW
        d["n"] += 1
        if extra:
            d["extra"].append(extra)

    for p, base in targets:
        print(f"\n== {p.name[:52]}  原始分 {base['score']:.3f}")

        # A2 关键帧躲避
        try:
            out = tmp / f"kf_{p.stem[:30]}.mp4"
            dur = base["duration"]
            hot = base["hot_times"][:1] or [dur / 2]
            info = VA.keyframe_evasion(p, out, (max(0, hot[0] - 2), min(dur, hot[0] + 2)), target_len=0.4)
            if out.exists() and out.stat().st_size > 10000:
                a = score_of(out, budget)
                record("A2_关键帧躲避", base, a, info)
                print(f"   A2 关键帧躲避       {base['score']:.3f} → {a['score']:.3f} ({a['verdict']})")
        except Exception as e:
            print(f"   A2 失败: {str(e)[:60]}")

        # A3 镜头切换灌水
        try:
            out = tmp / f"fl_{p.stem[:30]}.mp4"
            info = VA.scene_flood(p, out, n_flashes=80)
            if out.exists() and out.stat().st_size > 10000:
                a = score_of(out, budget)
                det = VA.detect_scene_flood(a["cuts"], a["duration"])
                record("A3_镜头切换灌水", base, a, {**info, "detect": det})
                print(f"   A3 镜头切换灌水     {base['score']:.3f} → {a['score']:.3f} "
                      f"（切换 {base['cuts']}→{a['cuts']}，灌水检测：{'命中' if det['suspicious'] else '未命中'}）")
        except Exception as e:
            print(f"   A3 失败: {str(e)[:60]}")

        # A4 重压缩 / 画中画
        for name, fn in (("A4a_重压缩crf40", lambda s, d: VA.recompress(s, d, crf=40)),
                         ("A4b_降分辨率0.4", lambda s, d: VA.recompress(s, d, crf=28, scale=0.4)),
                         ("A4c_画中画0.35", lambda s, d: VA.picture_in_picture(s, d, 0.35))):
            try:
                out = tmp / f"{name[:4]}_{p.stem[:26]}.mp4"
                info = fn(p, out)
                if out.exists() and out.stat().st_size > 10000:
                    a = score_of(out, budget)
                    record(name, base, a, info)
                    print(f"   {name:18s} {base['score']:.3f} → {a['score']:.3f} ({a['verdict']})")
            except Exception as e:
                print(f"   {name} 失败: {str(e)[:60]}")

        # A1 时序稀疏对抗攻击（最贵，默认只对短视频做）
        if not skip_pgd and base["duration"] <= 120:
            try:
                out = tmp / f"sp_{p.stem[:30]}.mp4"
                t0 = time.time()
                info = VA.sparse_frame_attack(p, out, base["hot_times"], clf, eps=8 / 255, steps=8)
                if out.exists() and out.stat().st_size > 10000:
                    a = score_of(out, budget)
                    record("A1_时序稀疏对抗", base, a, {**info, "time_s": round(time.time() - t0, 1)})
                    print(f"   A1 时序稀疏对抗     {base['score']:.3f} → {a['score']:.3f} "
                          f"（扰动 {info['n_perturbed']}/{info['n_frames_total']} 帧 = "
                          f"{info['perturb_ratio']*100:.1f}%，耗时 {time.time()-t0:.0f}s）")
            except Exception as e:
                print(f"   A1 失败: {str(e)[:70]}")

    # ---------- 防御：提高采样预算 ----------
    print("\n== 防御 D2：提高采样预算 ==")
    for bud in (24, 48, 80):
        scores, times = [], []
        for p, _ in targets[:2]:
            t0 = time.time()
            s = score_of(p, bud)
            scores.append(s["score"]); times.append(time.time() - t0)
        res["defenses"][f"budget_{bud}"] = {"mean_score": round(float(np.mean(scores)), 4),
                                            "mean_time_s": round(float(np.mean(times)), 1)}
        print(f"  预算 {bud:3d} 帧: 平均分={np.mean(scores):.3f} 平均耗时={np.mean(times):.1f}s")

    # ---------- 汇总 ----------
    print("\n== 攻击效果汇总 ==")
    summary = {}
    for name, d in res["attacks"].items():
        if not d["n"]:
            continue
        b, a = np.array(d["before"]), np.array(d["after"])
        summary[name] = {"n": d["n"], "mean_before": round(float(b.mean()), 4),
                         "mean_after": round(float(a.mean()), 4),
                         "mean_drop": round(float((b - a).mean()), 4),
                         "escape_rate": round(d["escaped"] / d["n"], 4)}
        s = summary[name]
        print(f"  {name:20s} {s['mean_before']:.3f} → {s['mean_after']:.3f} "
              f"(降 {s['mean_drop']:+.3f})  逃逸率={s['escape_rate']:.2f} (n={s['n']})")
    res["summary"] = dict(sorted(summary.items(), key=lambda kv: -kv[1]["escape_rate"]))

    shutil.rmtree(tmp, ignore_errors=True)
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "adv_video.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {RESULT_DIR/'adv_video.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--skip-pgd", action="store_true")
    a = ap.parse_args()
    main(a.n, a.budget, a.skip_pgd)
