"""语音审核链路的对抗攻防实验。

威胁模型：违规视频把风险信息放在语音里（广告引流话术、辱骂、赌博推广）。
平台的审核链路是  音频 → Whisper 转写 → 文本审核。
攻击者只要让转写失真，下游文本审核就拿不到任何风险信号，内容即可放行。

因此本实验的"攻击成功"定义为**端到端的审核逃逸**：
    原始音频 → 文本审核判定为高风险（block/review）
    攻击后音频 → 文本审核判定为通过（pass）
而不是只看字符错误率。这一点很关键：转写有些错字但风险词还在，审核依然成立，不算成功。

同时记录攻击带来的信噪比代价——攻击者不能把音频毁到人也听不懂，否则内容本身就失效了。
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from socialsim.config import RESULT_DIR, ROOT, RISK_REVIEW
from socialsim.security.attacks import audio_attack as AA
from socialsim.moderation.text_filter import analyze_text
from socialsim.moderation import video_pipeline as VP

SR = 16000
# 带风险语音的测试视频（由 scripts_make_testset.py 生成，语音内容已知）
RISKY_VIDEOS = ["ad_scam_short.mp4", "insult_speech_short.mp4", "gambling_short.mp4", "long_mixed_with_ad.mp4"]


def load_audio_from_video(path: Path) -> np.ndarray:
    import librosa
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                        "-ar", str(SR), str(wav)], capture_output=True, timeout=300)
        if not wav.exists():
            return np.zeros(SR, dtype=np.float32)
        y, _ = librosa.load(str(wav), sr=SR, mono=True)
    return y.astype(np.float32)


def transcribe_array(wav: np.ndarray) -> str:
    """直接对 numpy 波形转写，复用流水线里的 ASR 后端（SenseVoice / Whisper）。"""
    import soundfile as sf
    from socialsim.moderation.asr import transcribe as asr_transcribe
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.wav"
        sf.write(str(p), np.clip(wav, -1, 1), SR)
        segs = asr_transcribe(p)
    return " ".join(s["text"] for s in segs if s.get("text"))


def audit(text: str) -> dict:
    r = analyze_text(text)
    r["verdict"] = "block" if r["score"] >= 0.7 else ("review" if r["score"] >= RISK_REVIEW else "pass")
    return r


def main(n_attacks: int | None = None):
    bench = ROOT / "data" / "videos" / "bench"
    vids = [bench / v for v in RISKY_VIDEOS if (bench / v).exists()]
    print(f"测试素材 {len(vids)} 段带风险语音的视频\n")

    res = {"videos": {}, "summary": {}}
    attack_names = list(AA.ATTACKS)
    if n_attacks:
        attack_names = attack_names[:n_attacks]

    per_attack = {a: {"escaped": 0, "total": 0, "cers": [], "snrs": []} for a in attack_names}
    defense_names = list(AA.DEFENSES)
    per_defense = {d: {"recovered": 0, "total": 0} for d in defense_names}

    for v in vids:
        print(f"== {v.name}")
        wav = load_audio_from_video(v)
        base_text = transcribe_array(wav)
        base_audit = audit(base_text)
        print(f"   原始转写: {base_text[:70]!r}")
        print(f"   原始审核: score={base_audit['score']:.2f} {base_audit['verdict']} {base_audit['labels'][:4]}")
        if base_audit["verdict"] == "pass":
            print("   （原始就未被判风险，跳过）\n")
            continue
        entry = {"clean_text": base_text, "clean_audit": {k: base_audit[k] for k in ("score", "verdict", "labels")},
                 "attacks": {}}

        for name in attack_names:
            try:
                t0 = time.time()
                adv = AA.ATTACKS[name](wav)
                txt = transcribe_array(adv)
                a = audit(txt)
                c = AA.cer(base_text, txt)
                s = AA.snr_db(wav, adv)
                escaped = a["verdict"] == "pass"
                entry["attacks"][name] = {"text": txt[:110], "score": round(a["score"], 3),
                                          "verdict": a["verdict"], "cer": round(c, 3),
                                          "snr_db": round(s, 1) if np.isfinite(s) else None,
                                          "escaped": escaped, "time_s": round(time.time() - t0, 1)}
                per_attack[name]["total"] += 1
                per_attack[name]["escaped"] += escaped
                per_attack[name]["cers"].append(c)
                if np.isfinite(s):
                    per_attack[name]["snrs"].append(s)
                flag = "逃逸成功" if escaped else "仍被拦截"
                print(f"   {name:14s} CER={c:.2f} SNR={s:6.1f}dB 审核={a['verdict']:6s} {flag}")

                # 只对成功逃逸的攻击测试防御是否能救回来
                if escaped:
                    for dn in defense_names:
                        if dn == "none":
                            continue
                        try:
                            rec = transcribe_array(AA.DEFENSES[dn](adv))
                            ra = audit(rec)
                            per_defense[dn]["total"] += 1
                            per_defense[dn]["recovered"] += ra["verdict"] != "pass"
                            entry["attacks"][name].setdefault("defenses", {})[dn] = {
                                "verdict": ra["verdict"], "score": round(ra["score"], 3)}
                        except Exception as e:
                            entry["attacks"][name].setdefault("defenses", {})[dn] = {"error": str(e)[:60]}
            except Exception as e:
                entry["attacks"][name] = {"error": f"{type(e).__name__}: {e}"}
                print(f"   {name:14s} 失败: {str(e)[:60]}")
        # ---- 自适应攻击：攻击者可以查询审核结果，定点消除风险词 ----
        for name, fn in AA.ADAPTIVE_ATTACKS.items():
            if n_attacks:
                break
            try:
                t0 = time.time()
                adv = fn(wav, transcribe_array, lambda t: audit(t)["score"])
                txt = transcribe_array(adv)
                a = audit(txt)
                c = AA.cer(base_text, txt); sn = AA.snr_db(wav, adv)
                escaped = a["verdict"] == "pass"
                entry["attacks"][name] = {"text": txt[:110], "score": round(a["score"], 3),
                                          "verdict": a["verdict"], "cer": round(c, 3),
                                          "snr_db": round(sn, 1) if np.isfinite(sn) else None,
                                          "escaped": escaped, "adaptive": True,
                                          "time_s": round(time.time() - t0, 1)}
                d = per_attack.setdefault(name, {"escaped": 0, "total": 0, "cers": [], "snrs": []})
                d["total"] += 1; d["escaped"] += escaped; d["cers"].append(c)
                if np.isfinite(sn):
                    d["snrs"].append(sn)
                print(f"   {name:14s} CER={c:.2f} SNR={sn:6.1f}dB 审核={a['verdict']:6s} "
                      f"{'逃逸成功' if escaped else '仍被拦截'}  [自适应, {time.time()-t0:.0f}s]")
                if escaped:
                    for dn in defense_names:
                        if dn == "none":
                            continue
                        try:
                            ra = audit(transcribe_array(AA.DEFENSES[dn](adv)))
                            per_defense[dn]["total"] += 1
                            per_defense[dn]["recovered"] += ra["verdict"] != "pass"
                            entry["attacks"][name].setdefault("defenses", {})[dn] = {
                                "verdict": ra["verdict"], "score": round(ra["score"], 3)}
                        except Exception:
                            pass
            except Exception as e:
                print(f"   {name:14s} 失败: {str(e)[:70]}")

        res["videos"][v.name] = entry
        print()

    # ---- 汇总 ----
    summary = {}
    for name, d in per_attack.items():
        if d["total"]:
            summary[name] = {"escape_rate": round(d["escaped"] / d["total"], 3),
                             "n": d["total"], "mean_cer": round(float(np.mean(d["cers"])), 3),
                             "mean_snr_db": round(float(np.mean(d["snrs"])), 1) if d["snrs"] else None}
    res["summary"]["attacks"] = dict(sorted(summary.items(), key=lambda kv: -kv[1]["escape_rate"]))
    res["summary"]["defenses"] = {k: {"recovery_rate": round(v["recovered"] / v["total"], 3), "n": v["total"]}
                                  for k, v in per_defense.items() if v["total"]}

    print("== 攻击成功率汇总（按端到端审核逃逸计）==")
    for k, v in res["summary"]["attacks"].items():
        snr = f"{v['mean_snr_db']:6.1f}dB" if v["mean_snr_db"] is not None else "   inf "
        print(f"  {k:14s} 逃逸率={v['escape_rate']:.2f} (n={v['n']})  平均CER={v['mean_cer']:.2f}  平均SNR={snr}")
    print("\n== 防御恢复率（在成功逃逸的样本上重新审核）==")
    for k, v in res["summary"]["defenses"].items():
        print(f"  {k:14s} 恢复率={v['recovery_rate']:.2f} (n={v['n']})")

    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "adv_audio.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {RESULT_DIR/'adv_audio.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", type=int, default=None, help="只跑前 N 种攻击")
    main(ap.parse_args().attacks)
