"""(长/短)视频内容解析与不良内容监测流水线。

设计参考 DFscan 的"预处理 → 多专家 → 融合决策 → 可视化报告"范式，并针对视频的时序冗余
采用"快慢双通道 + 场景关键帧"策略（对应 Looking Fast and Slow / 关键帧选择思想）：

  1. 解复用与探针     ffprobe 取时长/帧率/分辨率/有无音轨，据此区分长短视频与采样预算
  2. 快通道 (Fast)    低频均匀采样，计算 HSV 直方图差分做镜头切分，得到候选关键帧
  3. 关键帧选择       按场景切换强度 + 时间覆盖度做预算内的 top-k 选择（长视频关键）
  4. 慢通道 (Slow)    对快通道判为高风险的时间邻域做加密采样，降低漏检
  5. 帧级多专家       NSFW / CLIP 零样本风险类目 / 人脸检测 + 深伪专家
  6. 音频通道         ffmpeg 抽音轨 → Whisper ASR → 文本审核；并统计响度异常
  7. OCR 通道         关键帧文字识别 → 文本审核（弹幕、字幕、水印引流）
  8. 时序聚合         帧分数 → 连续高风险片段；视频级分数用 top-k 均值（抗单帧噪声）
  9. 报告             JSON + HTML（缩略图、时间轴、片段结论、各专家置信度）
"""
from __future__ import annotations
import json, math, shutil, subprocess, tempfile, time
from datetime import datetime
from pathlib import Path
import numpy as np

from ..config import (KEYFRAME_MAX, KEYFRAME_MIN_GAP, SCENE_DIFF_THRESH, FAST_SAMPLE_FPS,
                      SHORT_VIDEO_SEC, RISK_BLOCK, RISK_REVIEW, RESULT_DIR, HF_WHISPER, device,
                      thresholds)


# ---------------------------------------------------------------- 探针
def probe(path: str | Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format=duration,size,bit_rate:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames",
           "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    info = json.loads(out.stdout or "{}")
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fr = v.get("avg_frame_rate", "0/1")
    try:
        num, den = fr.split("/"); fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    dur = float(info.get("format", {}).get("duration", 0) or 0)
    return {"duration": dur, "fps": fps, "width": int(v.get("width", 0) or 0), "height": int(v.get("height", 0) or 0),
            "codec": v.get("codec_name", ""), "has_audio": a is not None,
            "size": int(info.get("format", {}).get("size", 0) or 0),
            "kind": "short" if dur <= SHORT_VIDEO_SEC else "long"}


# ---------------------------------------------------------------- 采样与关键帧
def _hist(frame_bgr) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return (h / (h.sum() + 1e-9)).flatten()


def fast_scan(path: str | Path, meta: dict, budget: int = KEYFRAME_MAX) -> list[dict]:
    """快通道：均匀低频采样 + 镜头切分，返回带场景切换强度的候选帧。

    长视频不可能逐帧处理（10 分钟 60fps = 36000 帧），这里用采样率上限与预算共同约束，
    保证处理时间与视频长度近似线性且有上界。
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    dur = meta["duration"] or 0
    # 采样间隔：短视频按 FAST_SAMPLE_FPS；长视频放宽到覆盖全长的 3~4 倍预算
    step = 1.0 / FAST_SAMPLE_FPS
    if dur > SHORT_VIDEO_SEC:
        step = max(step, dur / (budget * 3.5))
    cands, prev = [], None
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h = _hist(frame)
        diff = float(np.abs(h - prev).sum() / 2) if prev is not None else 1.0   # 首帧视为场景开始
        cands.append({"t": round(t, 3), "scene_diff": diff, "is_cut": diff >= SCENE_DIFF_THRESH})
        prev = h
        t += step
    cap.release()
    return cands


def select_keyframes(cands: list[dict], budget: int = KEYFRAME_MAX, min_gap: float = KEYFRAME_MIN_GAP) -> list[float]:
    """关键帧选择：优先镜头切换点，再用时间均匀补齐，保证覆盖度。"""
    if not cands:
        return []
    cuts = sorted([c for c in cands if c["is_cut"]], key=lambda c: -c["scene_diff"])
    chosen: list[float] = []

    def ok(t):
        return all(abs(t - x) >= min_gap for x in chosen)

    for c in cuts:
        if len(chosen) >= budget:
            break
        if ok(c["t"]):
            chosen.append(c["t"])
    if len(chosen) < budget:                      # 用均匀时间点补齐，避免长视频只盯着切换点
        span = cands[-1]["t"] - cands[0]["t"] or 1.0
        need = budget - len(chosen)
        for i in range(need):
            t = cands[0]["t"] + span * (i + 0.5) / need
            if ok(t):
                chosen.append(round(t, 3))
    return sorted(chosen)[:budget]


def grab_frames(path: str | Path, times: list[float]) -> list[tuple[float, np.ndarray]]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if ok:
            out.append((t, f))
    cap.release()
    return out


# ---------------------------------------------------------------- 音频通道
def extract_audio(path: str | Path, out_wav: Path) -> bool:
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav)]
    r = subprocess.run(cmd, capture_output=True, timeout=600)
    return r.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 1024


_WHISPER = {}


def _whisper():
    if "m" not in _WHISPER:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        proc = WhisperProcessor.from_pretrained(HF_WHISPER)
        mdl = WhisperForConditionalGeneration.from_pretrained(HF_WHISPER).eval()
        dev = device()
        try:
            mdl = mdl.to(dev)
        except Exception:
            dev = "cpu"
        _WHISPER.update({"p": proc, "m": mdl, "d": dev})
    return _WHISPER["p"], _WHISPER["m"], _WHISPER["d"]


def _is_hallucination(text: str) -> bool:
    """Whisper 在纯音乐/环境音片段上会产生重复的幻觉文本（实测动画片背景音乐被转写成
    "我看你了 我看你了 …"）。用重复率与词汇多样性做启发式过滤。"""
    t = text.strip()
    if len(t) < 2:
        return True
    # (a) 某个 n-gram 在整段中的占比过高
    for n in (2, 3, 4):
        if len(t) >= n * 3:
            grams = [t[i:i + n] for i in range(0, len(t) - n + 1)]
            if grams:
                top = max(set(grams), key=grams.count)
                if grams.count(top) / len(grams) > 0.25:
                    return True
    # (b) 某个 n-gram 连续重复（"小的小的小的小的…"），这是解码坍塌的典型形态
    import re as _re
    for n in (1, 2, 3, 4):
        if _re.search(r"(.{%d})\1{3,}" % n, t):
            return True
    # (c) 短文本里同一个 3-gram 出现 3 次以上
    if len(t) < 60:
        for n in (2, 3):
            grams = [t[i:i + n] for i in range(0, len(t) - n + 1)]
            if grams and max((grams.count(g) for g in set(grams)), default=0) >= 3:
                return True
    uniq = len(set(t)) / max(1, len(t))
    if len(t) > 12 and uniq < 0.28:
        return True
    COMMON_HALLUC = ["请不吝点赞", "订阅", "转发", "打赏", "明镜与点点栏目", "字幕由", "Amara.org",
                     "谢谢观看", "谢谢大家", "本视频由"]
    if any(h in t for h in COMMON_HALLUC) and len(t) < 30:
        return True
    return False


def transcribe(wav_path: Path, language: str = "chinese", chunk_sec: float = 28.0, max_chunks: int = 40,
               vad_rms: float = 0.006) -> list[dict]:
    """分块 ASR，返回带时间戳的文本片段。

    两道防幻觉措施：
      1. 能量门限（轻量 VAD）：块内 RMS 过低直接跳过，不喂给模型；
      2. 输出后处理：重复率过高或命中常见幻觉短语的转写被丢弃。
    """
    import torch, librosa
    proc, mdl, dev = _whisper()
    audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    total = len(audio) / sr
    segs = []
    n = min(max_chunks, max(1, math.ceil(total / chunk_sec)))
    for i in range(n):
        s, e = i * chunk_sec, min(total, (i + 1) * chunk_sec)
        chunk = audio[int(s * sr):int(e * sr)]
        if len(chunk) < sr * 0.4:
            continue
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < vad_rms:
            segs.append({"start": round(s, 2), "end": round(e, 2), "text": "", "skipped": "low_energy", "rms": round(rms, 5)})
            continue
        inp = proc(chunk, sampling_rate=16000, return_tensors="pt")
        feats = inp.input_features.to(dev)
        with torch.no_grad():
            ids = mdl.generate(feats, language=language, task="transcribe", max_new_tokens=180,
                               no_repeat_ngram_size=4, repetition_penalty=1.15)
        txt = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if not txt:
            continue
        if _is_hallucination(txt):
            segs.append({"start": round(s, 2), "end": round(e, 2), "text": "", "skipped": "hallucination",
                         "raw": txt[:80], "rms": round(rms, 5)})
            continue
        segs.append({"start": round(s, 2), "end": round(e, 2), "text": txt, "rms": round(rms, 5)})
    return segs


def audio_stats(wav_path: Path) -> dict:
    """响度与突变统计：尖叫/爆炸等异常声学事件的轻量代理指标。"""
    try:
        import librosa
        y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        if len(y) < sr:
            return {}
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
        cen = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        loud = rms / (np.median(rms) + 1e-9)
        return {"rms_mean": float(rms.mean()), "loud_peak_ratio": float(loud.max()),
                "loud_burst_frac": float((loud > 4).mean()), "zcr_mean": float(zcr.mean()),
                "centroid_mean": float(cen.mean())}
    except Exception:
        return {}


# ---------------------------------------------------------------- 时序聚合
def aggregate(frame_scores: list[tuple[float, float]], topk_frac: float = 0.10,
              seg_thresh: float = 0.4, min_seg: int = 1) -> dict:
    """帧级分数 → 视频级分数 + 高风险片段。

    视频级分数用 top-k 均值而非单帧最大值：单帧最大值对误报极其敏感，
    而"若干帧持续高分"才是真正的违规证据（与弱监督视频异常检测中的 top-k MIL 思路一致）。
    topk_frac 由网格搜索确定为 0.10（见 experiments/exp_prompt_tuning.py）。
    """
    if not frame_scores:
        return {"score": 0.0, "segments": [], "topk": []}
    fs = sorted(frame_scores, key=lambda x: x[0])
    scores = np.array([s for _, s in fs])
    k = max(1, int(len(scores) * topk_frac))
    topk_idx = np.argsort(-scores)[:k]
    video_score = float(scores[topk_idx].mean())
    segs, cur = [], None
    for t, s in fs:
        if s >= seg_thresh:
            if cur is None:
                cur = {"start": t, "end": t, "peak": s, "n": 1, "sum": s}
            else:
                cur["end"] = t; cur["peak"] = max(cur["peak"], s); cur["n"] += 1; cur["sum"] += s
        elif cur is not None:
            if cur["n"] >= min_seg:
                cur["mean"] = cur["sum"] / cur["n"]; segs.append(cur)
            cur = None
    if cur is not None and cur["n"] >= min_seg:
        cur["mean"] = cur["sum"] / cur["n"]; segs.append(cur)
    for s in segs:
        s.pop("sum", None)
        s["start"], s["end"] = round(s["start"], 2), round(s["end"], 2)
        s["peak"], s["mean"] = round(s["peak"], 3), round(s.get("mean", s["peak"]), 3)
    return {"score": video_score, "segments": segs,
            "topk": [(round(fs[i][0], 2), round(float(scores[i]), 3)) for i in sorted(topk_idx)]}


# ---------------------------------------------------------------- 主流程
def analyze_video(path: str | Path, out_dir: Path | None = None, budget: int = KEYFRAME_MAX,
                  do_audio: bool = True, do_ocr: bool = True, do_deepfake: bool = True,
                  slow_pass: bool = True, save_report: bool = True) -> dict:
    from . import image_experts as ie
    from .text_filter import analyze_text
    import cv2

    path = Path(path)
    t0 = time.time()
    out_dir = Path(out_dir) if out_dir else (RESULT_DIR / "reports" / f"{path.stem}_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = probe(path)
    timing = {}

    # --- 1. 快通道：镜头切分 + 关键帧选择 ---
    t = time.time()
    cands = fast_scan(path, meta, budget)
    times = select_keyframes(cands, budget)
    timing["fast_scan"] = time.time() - t
    n_cuts = sum(1 for c in cands if c["is_cut"])

    # --- 2. 帧级专家 ---
    t = time.time()
    frames = grab_frames(path, times)
    pil = [ie._to_pil(f) for _, f in frames]
    nsfw = ie.nsfw_batch(pil) if pil else []
    clip = ie.clip_risk_batch(pil) if pil else []
    timing["frame_experts"] = time.time() - t

    # --- 3. 人脸 + 深伪专家 ---
    t = time.time()
    df_by_idx: dict[int, float] = {}
    n_faces_total = 0
    if do_deepfake and frames:
        face_crops, owners = [], []
        for i, (_, f) in enumerate(frames):
            fs = ie.detect_faces(f)
            n_faces_total += len(fs)
            for (x, y, w, h) in fs[:2]:
                pad = int(0.15 * max(w, h))
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(f.shape[1], x + w + pad), min(f.shape[0], y + h + pad)
                face_crops.append(ie._to_pil(f[y0:y1, x0:x1])); owners.append(i)
        if face_crops:
            probs = ie.deepfake_batch(face_crops)
            for i, p in zip(owners, probs):
                df_by_idx[i] = max(df_by_idx.get(i, 0.0), float(p))
    timing["deepfake"] = time.time() - t

    # --- 4. 帧级融合 ---
    frame_records = []
    for i, ((ts, f), n, c) in enumerate(zip(frames, nsfw, clip)):
        fused = ie.fuse_frame(n, c, df_by_idx.get(i))
        frame_records.append({"t": ts, "idx": i, **fused})

    # --- 5. 慢通道：对高风险时刻加密采样，降低漏检 ---
    slow_records = []
    if slow_pass and frame_records:
        hot = [r["t"] for r in frame_records if r["score"] >= RISK_REVIEW]
        extra = []
        for ht in hot[:8]:
            for d in (-0.6, -0.3, 0.3, 0.6):
                tt = round(ht + d, 3)
                if 0 <= tt <= meta["duration"] and all(abs(tt - r["t"]) > 0.2 for r in frame_records):
                    extra.append(tt)
        extra = sorted(set(extra))[:24]
        if extra:
            t = time.time()
            ef = grab_frames(path, extra)
            epil = [ie._to_pil(f) for _, f in ef]
            en, ec = ie.nsfw_batch(epil), ie.clip_risk_batch(epil)
            for (ts, _), n, c in zip(ef, en, ec):
                fu = ie.fuse_frame(n, c, None)
                slow_records.append({"t": ts, "slow": True, **fu})
            timing["slow_pass"] = time.time() - t
    all_frames = sorted(frame_records + slow_records, key=lambda r: r["t"])

    # --- 6. OCR 通道 ---
    ocr_hits, ocr_score = [], 0.0
    if do_ocr and frames:
        t = time.time()
        # 只对分布均匀的一部分帧做 OCR（OCR 相对昂贵）
        idxs = list(range(0, len(frames), max(1, len(frames) // 12)))[:12]
        for i in idxs:
            ts, f = frames[i]
            txt = ie._ocr_text(f)
            if txt and len(txt) >= 3:
                r = analyze_text(txt, noisy_source=True)
                if r["score"] > 0.15:
                    ocr_hits.append({"t": ts, "text": txt[:160], "score": r["score"], "labels": r["labels"]})
                    ocr_score = max(ocr_score, r["score"])
        timing["ocr"] = time.time() - t

    # --- 7. 音频通道 ---
    audio_info = {"available": False}
    asr_score, asr_hits = 0.0, []
    if do_audio and meta["has_audio"]:
        t = time.time()
        wav = out_dir / "audio.wav"
        if extract_audio(path, wav):
            audio_info["available"] = True
            audio_info["stats"] = audio_stats(wav)
            try:
                from .asr import transcribe as asr_transcribe, backend_name
                segs = asr_transcribe(wav)
                audio_info["backend"] = backend_name()
                audio_info["transcript"] = segs
                cue_max = 0.0
                for sg in segs:
                    # 音频事件/情绪线索：爆炸、尖叫、哭喊这类"声先于形"的证据没有语音内容，
                    # 纯 ASR 完全捕捉不到，必须单独计入。
                    cue = float(sg.get("cue_risk", 0.0))
                    if cue > 0:
                        cue_max = max(cue_max, cue)
                    if not sg.get("text"):
                        continue
                    r = analyze_text(sg["text"], noisy_source=True)
                    sg["risk"] = r["score"]; sg["labels"] = r["labels"]
                    if r["score"] > 0.15:
                        asr_hits.append(sg); asr_score = max(asr_score, r["score"])
                audio_info["full_text"] = " ".join(s["text"] for s in segs if s.get("text"))[:2000]
                audio_info["n_skipped"] = sum(1 for s in segs if s.get("skipped"))
                audio_info["cue_risk"] = round(cue_max, 3)
                audio_info["cues"] = sorted({l for s in segs for l in s.get("cue_labels", [])})
            except Exception as e:
                audio_info["asr_error"] = f"{type(e).__name__}: {e}"
            try:
                wav.unlink()
            except Exception:
                pass
        timing["audio"] = time.time() - t

    # --- 8. 时序聚合与最终融合 ---
    th_review, th_block = thresholds()
    agg = aggregate([(r["t"], r["score"]) for r in all_frames], seg_thresh=th_review)
    visual_score = agg["score"]
    cue_risk = float(audio_info.get("cue_risk", 0.0))
    final = max(visual_score, asr_score, ocr_score)
    # 音频线索只做加权提升，不单独判定——愤怒语气本身不违规
    if cue_risk > 0 and visual_score > 0:
        final = min(1.0, final + 0.5 * cue_risk * visual_score)
    # 多模态一致性：视觉与音频/OCR 同时指向风险 → 提升置信度
    if visual_score >= th_review and max(asr_score, ocr_score) >= 0.35:
        final = min(1.0, final + 0.1)
    labels = sorted({l for r in all_frames for l in r["labels"]} |
                    {f"asr_{l}" for h in asr_hits for l in h["labels"]} |
                    {f"ocr_{l}" for h in ocr_hits for l in h["labels"]} |
                    set(audio_info.get("cues", [])))
    verdict = "block" if final >= th_block else ("review" if final >= th_review else "pass")

    # 按违规类目分别给分，而不是压成一个数。
    # 理由：不同类目的处置策略不同（暴力→限流，诈骗引流→封禁，辱骂→折叠），
    # 而且"一个 max 分数"会让不同性质的风险互相污染——实测在 XD-Violence 上，
    # 电影对白里的脏话把视觉暴力通道的 AUC 从 0.91 拉低到 0.72。
    def _cat(hits, prefix):
        out = {}
        for h in hits:
            for l in h.get("labels", []):
                out[l] = max(out.get(l, 0.0), float(h.get("risk", h.get("score", 0.0))))
        return out
    text_cats = {}
    for d_ in (_cat(asr_hits, "asr"), _cat(ocr_hits, "ocr")):
        for k_, v_ in d_.items():
            text_cats[k_] = max(text_cats.get(k_, 0.0), v_)
    category_scores = {
        "violence_visual": round(visual_score, 4),
        "audio_cue": round(cue_risk, 4),
        "offensive_speech": round(max([v for k, v in text_cats.items()
                                       if k in ("insult", "offensive")] or [0.0]), 4),
        "spam_fraud": round(max([v for k, v in text_cats.items()
                                 if k in ("spam", "fraud", "contact_wechat", "contact_qq",
                                          "contact_phone", "external_link", "short_link")] or [0.0]), 4),
        "gambling_porn_drugs": round(max([v for k, v in text_cats.items()
                                          if k in ("gambling", "porn", "drugs")] or [0.0]), 4),
    }

    result = {
        "path": str(path), "meta": meta, "score": float(final), "verdict": verdict, "labels": labels,
        "visual_score": float(visual_score), "asr_score": float(asr_score), "ocr_score": float(ocr_score),
        "segments": agg["segments"], "n_frames_analyzed": len(all_frames), "n_keyframes": len(frames),
        "n_scene_cuts": n_cuts, "n_faces": n_faces_total, "deepfake_max": max(df_by_idx.values()) if df_by_idx else 0.0,
        "audio": audio_info, "ocr_hits": ocr_hits, "asr_hits": asr_hits,
        "cue_risk": cue_risk, "category_scores": category_scores,
        "frames": all_frames, "timing": {k: round(v, 2) for k, v in timing.items()},
        "thresholds": {"review": round(th_review, 4), "block": round(th_block, 4)},
        "elapsed": round(time.time() - t0, 2), "analyzed_at": datetime.now().isoformat(timespec="seconds"),
    }
    if save_report:
        result["report_path"] = str(write_report(result, frames, out_dir))
    return result


# ---------------------------------------------------------------- 报告
def write_report(result: dict, frames: list[tuple[float, np.ndarray]], out_dir: Path) -> Path:
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = out_dir / "thumbs"; thumb_dir.mkdir(exist_ok=True)
    # 保存风险最高的若干帧缩略图作为证据
    by_t = {round(t, 3): f for t, f in frames}
    ranked = sorted(result["frames"], key=lambda r: -r["score"])[:8]
    thumbs = []
    for r in ranked:
        f = by_t.get(round(r["t"], 3))
        if f is None:
            continue
        h, w = f.shape[:2]
        scale = 240 / max(1, w)
        small = cv2.resize(f, (240, max(1, int(h * scale))))
        name = f"t{r['t']:.2f}.jpg".replace(".", "_", 1)
        cv2.imwrite(str(thumb_dir / name), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumbs.append({"t": r["t"], "file": f"thumbs/{name}", "score": r["score"], "labels": r["labels"]})
    result_slim = {k: v for k, v in result.items() if k != "frames"}
    result_slim["frames"] = result["frames"]
    result_slim["thumbs"] = thumbs
    (out_dir / "report.json").write_text(json.dumps(result_slim, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    html = _render_html(result, thumbs)
    p = out_dir / "report.html"
    p.write_text(html, encoding="utf-8")
    return p


LABEL_NAMES = {
    "contact_wechat": "微信引流", "contact_qq": "QQ 引流", "contact_phone": "手机号引流", "external_link": "外部链接",
    "short_link": "短链接", "spam": "垃圾广告", "fraud": "诈骗", "gambling": "赌博", "drugs": "毒品", "porn": "色情",
    "violence": "暴力", "insult": "辱骂", "offensive": "冒犯言论", "obfuscation": "刻意混淆", "gore": "血腥", "weapon": "武器",
    "smoking_drug": "烟草/毒品", "protest_crowd": "人群聚集", "fire_disaster": "火灾/灾害", "horror": "惊悚",
    "nsfw": "色情", "nsfw_cartoon": "色情（动画）", "deepfake_suspect": "疑似深度伪造",
}
TIMING_NAMES = {"fast_scan": "快速扫描 · 选关键帧", "frame_experts": "帧级视觉专家", "deepfake": "深伪检测",
                "ocr": "画面文字 OCR", "audio": "语音理解 ASR"}


def _label_name(l: str) -> str:
    for prefix, name in (("asr_", "语音·"), ("ocr_", "画面文字·")):
        if l.startswith(prefix):
            return name + _label_name(l[len(prefix):])
    if l in LABEL_NAMES:
        return LABEL_NAMES[l]
    base, _, dom = l.rpartition("_")
    if base in LABEL_NAMES and dom in ("cartoon", "game"):
        return f"{LABEL_NAMES[base]}（{'动画' if dom == 'cartoon' else '游戏'}）"
    return l


def _render_html(r: dict, thumbs: list[dict]) -> str:
    """独立报告页：配色与网页端一致；通过 /reports/ 访问时用 ECharts 画时间线，直接从磁盘打开时退回内联 SVG。"""
    from html import escape as e
    color = {"block": "#f53f3f", "review": "#ff7d00", "pass": "#00b42a"}[r["verdict"]]
    vname = {"block": "拦截", "review": "人工复核", "pass": "通过"}[r["verdict"]]
    th = r.get("thresholds", {})
    t_block, t_review = th.get("block", RISK_BLOCK), th.get("review", RISK_REVIEW)
    pts = [(f["t"], f["score"]) for f in r["frames"]]
    dur = max(1e-6, r["meta"]["duration"])
    poly = " ".join(f"{100 * t / dur:.2f},{100 - 100 * s:.2f}" for t, s in pts)
    tags = "".join(f'<span class="tag">{e(_label_name(l))}</span>' for l in r["labels"]) or '<span class="muted">无风险标签</span>'
    seg_html = "".join(
        f"<div class='seg'><b>{s['start']}s – {s['end']}s</b>　峰值 {s['peak']} · 均值 {s['mean']}（{s['n']} 帧）</div>"
        for s in r["segments"]) or "<div class='muted'>未发现连续高风险片段</div>"
    thumb_html = "".join(
        f"<figure><div class='blur'><img src='{t['file']}' alt='证据帧'></div><figcaption>{t['t']}s · {t['score']:.2f}<br>"
        f"<span class='muted'>{e('、'.join(_label_name(l) for l in t['labels']) or '无标签')}</span></figcaption></figure>"
        for t in thumbs) or "<div class='muted'>无高风险帧</div>"
    asr = "".join(f"<tr><td>{h['start']}–{h['end']}s</td><td>{e(h['text'])}</td><td>{h['risk']:.2f}</td>"
                  f"<td>{e('、'.join(_label_name(l) for l in h['labels']))}</td></tr>"
                  for h in r["asr_hits"]) or "<tr><td colspan=4 class='muted'>无风险语音片段</td></tr>"
    ocr = "".join(f"<tr><td>{h['t']}s</td><td>{e(h['text'])}</td><td>{h['score']:.2f}</td>"
                  f"<td>{e('、'.join(_label_name(l) for l in h['labels']))}</td></tr>"
                  for h in r["ocr_hits"]) or "<tr><td colspan=4 class='muted'>无风险画面文字</td></tr>"
    transcript = e((r["audio"].get("full_text") or "")[:1200]) if r["audio"].get("available") else "（无音轨或未启用）"
    total_t = max(1e-6, sum(r["timing"].values()))
    timing = "".join(f"<div class='tm'><span>{TIMING_NAMES.get(k, k)}</span><i style='width:{100 * v / total_t:.1f}%'></i><b>{v}s</b></div>"
                     for k, v in r["timing"].items())
    # 给 ECharts 用的精简数据；"</" 转义，防止文本里出现 </script> 提前结束脚本
    data = json.dumps({"dur": dur, "review": t_review, "block": t_block,
                       "frames": [[f["t"], round(f["score"], 4)] for f in r["frames"]],
                       "segments": [[s["start"], s["end"]] for s in r["segments"]],
                       "asr": [[h["start"], round(h["risk"], 3), h["text"]] for h in r["asr_hits"]],
                       "ocr": [[h["t"], round(h["score"], 3), h["text"]] for h in r["ocr_hits"]]},
                      ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>视频审核报告 · {e(Path(r['path']).name)}</title>
<style>
:root{{--text:#1d2129;--sub:#86909c;--line:#e5e6eb;--fill:#f7f8fa}}
*{{box-sizing:border-box}}
body{{font-family:"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",-apple-system,sans-serif;margin:0;background:#f2f3f5;color:var(--text);font-size:14px}}
.wrap{{max-width:1040px;margin:0 auto;padding:28px 24px}}
.head{{background:#fff;border-radius:14px;padding:22px 24px;margin-bottom:16px;display:flex;gap:20px;align-items:center;
  box-shadow:0 1px 2px rgba(0,0,0,.03),0 4px 12px rgba(0,0,0,.035);border-left:6px solid {color}}}
.head h1{{font-size:20px;margin:0 0 6px}}
.verdict{{display:inline-flex;align-items:center;padding:4px 14px;border-radius:20px;color:#fff;background:{color};font-weight:600}}
.score{{text-align:center;min-width:120px}} .score b{{display:block;font-size:34px;color:{color};font-variant-numeric:tabular-nums}}
h2{{font-size:15px;margin:0 0 12px}}
.card{{background:#fff;border-radius:12px;padding:18px 20px;margin-bottom:16px;box-shadow:0 1px 2px rgba(0,0,0,.03),0 4px 12px rgba(0,0,0,.035)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px}}
.kv{{background:var(--fill);border-radius:8px;padding:10px 12px}} .kv b{{display:block;font-size:18px;font-variant-numeric:tabular-nums}}
.muted{{color:var(--sub);font-size:12px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} td,th{{border-bottom:1px solid #f0f1f3;padding:8px;text-align:left;vertical-align:top}}
th{{background:var(--fill);color:#4e5969;font-weight:500;font-size:12px}}
figure{{margin:0;display:inline-block;width:156px;vertical-align:top;margin:0 10px 10px 0}}
.blur{{border-radius:8px;overflow:hidden;background:#000;position:relative}}
.blur img{{display:block;width:100%;filter:blur(14px) saturate(.6);transform:scale(1.08);transition:filter .25s,transform .25s}}
.blur:hover img{{filter:none;transform:none}}
.blur::after{{content:"悬停查看";position:absolute;right:6px;bottom:6px;font-size:11px;color:#fff;background:rgba(0,0,0,.45);padding:1px 6px;border-radius:4px}}
.blur:hover::after{{opacity:0}}
figcaption{{font-size:11px;text-align:center;margin-top:4px;line-height:1.5}}
.seg{{padding:7px 0;border-bottom:1px dashed #eee;font-size:13px}}
.tag{{display:inline-block;background:#f2f3f5;border-radius:10px;padding:2px 9px;margin:2px;font-size:12px;color:#4e5969}}
svg{{width:100%;height:140px;background:var(--fill);border-radius:8px}}
#tl{{height:260px;display:none}}
.tm{{display:grid;grid-template-columns:150px 1fr 60px;gap:10px;align-items:center;font-size:12.5px;margin:6px 0}}
.tm i{{display:block;height:8px;border-radius:4px;background:#165dff}} .tm b{{font-weight:500;text-align:right}}
@media print{{body{{background:#fff}} .card,.head{{box-shadow:none;border:1px solid var(--line);break-inside:avoid}}
  .blur img{{filter:none;transform:none}} .blur::after{{display:none}}}}
</style>
<div class="wrap">
<div class="head">
  <div style="flex:1;min-width:0"><h1>视频审核报告</h1>
    <div class="muted">{e(Path(r['path']).name)} · 分析于 {r['analyzed_at']} · 耗时 {r['elapsed']}s</div>
    <div style="margin-top:10px"><span class="verdict">{vname}</span>　{tags}</div></div>
  <div class="score"><b>{r['score']:.3f}</b><span class="muted">综合风险分</span></div>
</div>

<div class="card"><h2>视频信息与处理统计</h2><div class="grid">
<div class="kv"><b>{r['meta']['duration']:.1f}s</b><span class="muted">时长（{'短视频' if r['meta']['kind']=='short' else '长视频'}）</span></div>
<div class="kv"><b>{r['meta']['width']}×{r['meta']['height']}</b><span class="muted">分辨率 {r['meta']['fps']:.0f}fps</span></div>
<div class="kv"><b>{r['n_frames_analyzed']}</b><span class="muted">分析帧数</span></div>
<div class="kv"><b>{r['n_scene_cuts']}</b><span class="muted">镜头切换</span></div>
<div class="kv"><b>{r['n_faces']}</b><span class="muted">人脸数</span></div>
<div class="kv"><b>{r['visual_score']:.2f} / {r['asr_score']:.2f} / {r['ocr_score']:.2f}</b><span class="muted">视觉 / 语音 / 画面文字</span></div>
</div></div>

<div class="card"><h2>风险时间线</h2>
<div id="tl"></div>
<svg id="tl-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
<line x1="0" y1="{100-100*t_block}" x2="100" y2="{100-100*t_block}" stroke="#f53f3f" stroke-width=".4" stroke-dasharray="1.5"/>
<line x1="0" y1="{100-100*t_review}" x2="100" y2="{100-100*t_review}" stroke="#ff7d00" stroke-width=".4" stroke-dasharray="1.5"/>
<polyline points="{poly}" fill="none" stroke="#165dff" stroke-width=".8"/></svg>
<div class="muted" style="margin-top:6px">横轴时间 0–{dur:.0f}s，纵轴帧级风险分 0–1；红线 = 拦截阈值 {t_block:.3f}，橙线 = 复核阈值 {t_review:.3f}；红底 = 高风险片段</div></div>

<div class="card"><h2>高风险片段</h2>{seg_html}</div>
<div class="card"><h2>证据帧 <span class="muted">默认模糊，鼠标悬停才显示</span></h2>{thumb_html}</div>
<div class="card"><h2>语音识别风险片段</h2><table><tr><th>时间</th><th>识别文本</th><th>风险</th><th>标签</th></tr>{asr}</table>
<div class="muted" style="margin-top:8px;line-height:1.7">完整转写：{transcript}</div></div>
<div class="card"><h2>画面文字（OCR）风险</h2><table><tr><th>时间</th><th>文本</th><th>风险</th><th>标签</th></tr>{ocr}</table></div>
<div class="card"><h2>各阶段耗时</h2>{timing}</div>
<div class="muted" style="text-align:center">SocialNetSim · 本地沙箱生成的审核报告</div>
</div>
<script src="/static/vendor/echarts/6.1.0/echarts.min.js"></script>
<script>
// 通过网页端（/reports/）打开时能加载到本地 ECharts，画可缩放的时间线；从磁盘直接打开时保留上面的 SVG
(function () {{
  if (!window.echarts) return;
  var R = {data}, el = document.getElementById('tl');
  el.style.display = 'block'; document.getElementById('tl-svg').style.display = 'none';
  var hit = function (name, rows, y) {{ return rows.map(function (h) {{ return {{ value: [h[0], y], risk: h[1], text: h[2], kind: name }}; }}); }};
  echarts.init(el).setOption({{
    textStyle: {{ fontFamily: '"PingFang SC","Microsoft YaHei",sans-serif' }},
    tooltip: {{ trigger: 'item', confine: true, formatter: function (p) {{
      return p.seriesIndex === 0 ? p.value[0] + 's · 帧级风险 <b>' + p.value[1] + '</b>' : p.data.kind + ' · ' + p.value[0] + 's<br>' + p.data.text + '<br>风险 ' + p.data.risk; }} }},
    grid: [{{ left: 72, right: 20, top: 14, height: 150 }}, {{ left: 72, right: 20, top: 186, height: 40 }}],
    xAxis: [{{ type: 'value', min: 0, max: +R.dur.toFixed(1), axisLabel: {{ show: false }}, splitLine: {{ show: false }} }},
            {{ type: 'value', gridIndex: 1, min: 0, max: +R.dur.toFixed(1), axisLabel: {{ formatter: '{{value}}s', color: '#86909c' }}, splitLine: {{ show: false }} }}],
    yAxis: [{{ type: 'value', min: 0, max: 1, splitNumber: 2, axisLabel: {{ color: '#86909c' }}, splitLine: {{ lineStyle: {{ color: '#f0f1f3' }} }} }},
            {{ type: 'category', gridIndex: 1, data: ['画面文字', '语音'], axisLine: {{ show: false }}, axisTick: {{ show: false }}, axisLabel: {{ color: '#4e5969', fontSize: 11 }} }}],
    visualMap: {{ show: false, seriesIndex: 0, pieces: [{{ lt: R.review, color: '#00b42a' }}, {{ gte: R.review, lt: R.block, color: '#ff7d00' }}, {{ gte: R.block, color: '#f53f3f' }}] }},
    dataZoom: [{{ type: 'inside', xAxisIndex: [0, 1], zoomOnMouseWheel: 'ctrl', moveOnMouseWheel: false }}],
    series: [
      {{ type: 'line', data: R.frames, showSymbol: R.frames.length < 40, symbolSize: 5, areaStyle: {{ opacity: 0.08 }},
        markLine: {{ symbol: 'none', silent: true, data: [{{ yAxis: R.review, lineStyle: {{ color: '#ff7d00', type: 'dashed' }} }}, {{ yAxis: R.block, lineStyle: {{ color: '#f53f3f', type: 'dashed' }} }}], label: {{ show: false }} }},
        markArea: {{ silent: true, itemStyle: {{ color: 'rgba(245,63,63,.08)' }}, data: R.segments.map(function (s) {{ return [{{ xAxis: s[0] }}, {{ xAxis: s[1] }}]; }}) }} }},
      {{ type: 'scatter', xAxisIndex: 1, yAxisIndex: 1, symbol: 'roundRect', symbolSize: 13, data: hit('语音', R.asr, 1).concat(hit('画面文字', R.ocr, 0)),
        itemStyle: {{ color: function (p) {{ return p.data.risk >= R.block ? '#f53f3f' : p.data.risk >= R.review ? '#ff7d00' : '#165dff'; }} }} }}
    ]
  }});
}})();
</script>
</html>"""
