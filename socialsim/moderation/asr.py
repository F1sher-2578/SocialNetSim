"""语音理解统一接口：ASR 文本 + 情绪 + 音频事件。

两个后端，按可用性自动选择：

  SenseVoiceSmall（默认，推荐）
      中文识别显著优于同体量的 Whisper（实测"棋牌平台首充返水"被 Whisper 转写成
      "其排平台手冲繁水"，SenseVoice 完全正确），CPU 上实时倍率约 10×，
      且**同时输出情绪与音频事件标签**——愤怒、哭喊、掌声、BGM 等。
      这解决了纯 ASR 的一个根本盲区：爆炸、枪声、尖叫这类"声先于形"的事件没有语音内容，
      转写为空，但它们恰恰是暴力场景最强的音频证据。

  Whisper（回退）
      SenseVoice 不可用时使用，附带能量门限与防幻觉后处理。

两者都返回统一结构：[{start, end, text, emotion, events, ...}]
"""
from __future__ import annotations
import math
import os
import re
from pathlib import Path

import numpy as np

SR = 16000

# SenseVoice 的标签体系
EMOTIONS = ["HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"]
EVENTS = ["Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough"]

# 与暴力/不良内容相关的音频线索及其风险权重
AUDIO_RISK = {"ANGRY": 0.35, "FEARFUL": 0.4, "SAD": 0.1, "DISGUSTED": 0.2, "Cry": 0.35}

_BACKEND: dict = {}


def _sensevoice():
    if "sv" in _BACKEND:
        return _BACKEND["sv"]
    try:
        os.environ.pop("HF_HUB_OFFLINE", None)   # funasr 需要读取模型目录
        from funasr import AutoModel
        m = AutoModel(model="FunAudioLLM/SenseVoiceSmall", device="cpu",
                      disable_update=True, hub="hf", disable_pbar=True, disable_log=True)
        _BACKEND["sv"] = m
    except Exception as e:
        _BACKEND["sv"] = None
        _BACKEND["sv_error"] = f"{type(e).__name__}: {e}"
    return _BACKEND["sv"]


TAG_RE = re.compile(r"<\|([^|]+)\|>")


def parse_sensevoice(raw: str) -> dict:
    """解析 SenseVoice 输出中的标签，返回纯文本 + 情绪 + 事件。"""
    tags = TAG_RE.findall(raw or "")
    text = TAG_RE.sub("", raw or "").strip()
    emotion = next((t for t in tags if t in EMOTIONS), "")
    events = [t for t in tags if t in EVENTS]
    lang = next((t for t in tags if len(t) == 2 and t.islower()), "")
    return {"text": text, "emotion": emotion, "events": events, "language": lang, "tags": tags}


def audio_cue_risk(emotion: str, events: list[str]) -> tuple[float, list[str]]:
    """把情绪与音频事件转成风险分与标签。

    注意刻意保持保守：愤怒语气本身不违规，只作为**辅助证据**参与融合，
    不能单独触发拦截，否则激烈的正常辩论会被大量误伤。
    """
    score, labels = 0.0, []
    if emotion in AUDIO_RISK:
        score = max(score, AUDIO_RISK[emotion])
        labels.append(f"emotion_{emotion.lower()}")
    for e in events:
        if e in AUDIO_RISK:
            score = max(score, AUDIO_RISK[e])
            labels.append(f"event_{e.lower()}")
    return score, labels


def transcribe(wav_path: str | Path, language: str = "zh", chunk_sec: float = 30.0,
               max_chunks: int = 40, vad_rms: float = 0.006, backend: str = "auto") -> list[dict]:
    """分块转写，返回带时间戳、情绪与音频事件的片段。"""
    import librosa
    wav_path = Path(wav_path)
    audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    total = len(audio) / SR
    n = min(max_chunks, max(1, math.ceil(total / chunk_sec)))

    m = _sensevoice() if backend in ("auto", "sensevoice") else None
    if m is None:
        return _whisper_transcribe(audio, total, n, chunk_sec, vad_rms, language)

    import soundfile as sf
    import tempfile
    segs = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(n):
            s, e = i * chunk_sec, min(total, (i + 1) * chunk_sec)
            chunk = audio[int(s * SR):int(e * SR)]
            if len(chunk) < SR * 0.4:
                continue
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < vad_rms:
                segs.append({"start": round(s, 2), "end": round(e, 2), "text": "",
                             "emotion": "", "events": [], "skipped": "low_energy", "rms": round(rms, 5)})
                continue
            p = Path(td) / f"c{i}.wav"
            sf.write(str(p), chunk, SR)
            try:
                r = m.generate(input=str(p), language=language, use_itn=True, batch_size_s=60)
                parsed = parse_sensevoice(r[0].get("text", "") if r else "")
            except Exception as ex:
                segs.append({"start": round(s, 2), "end": round(e, 2), "text": "",
                             "emotion": "", "events": [], "skipped": f"error:{type(ex).__name__}"})
                continue
            cue, cue_labels = audio_cue_risk(parsed["emotion"], parsed["events"])
            segs.append({"start": round(s, 2), "end": round(e, 2), "text": parsed["text"],
                         "emotion": parsed["emotion"], "events": parsed["events"],
                         "cue_risk": round(cue, 3), "cue_labels": cue_labels,
                         "rms": round(rms, 5), "backend": "sensevoice"})
    return segs


# ---------------------------------------------------------------- Whisper 回退
def _whisper_transcribe(audio, total, n, chunk_sec, vad_rms, language) -> list[dict]:
    import torch
    from .video_pipeline import _whisper, _is_hallucination
    proc, mdl, dev = _whisper()
    lang = {"zh": "chinese", "en": "english"}.get(language, "chinese")
    segs = []
    for i in range(n):
        s, e = i * chunk_sec, min(total, (i + 1) * chunk_sec)
        chunk = audio[int(s * SR):int(e * SR)]
        if len(chunk) < SR * 0.4:
            continue
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < vad_rms:
            segs.append({"start": round(s, 2), "end": round(e, 2), "text": "", "emotion": "",
                         "events": [], "skipped": "low_energy", "rms": round(rms, 5)})
            continue
        inp = proc(chunk, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            ids = mdl.generate(inp.input_features.to(dev), language=lang, task="transcribe",
                               max_new_tokens=180, no_repeat_ngram_size=4, repetition_penalty=1.15)
        txt = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if not txt or _is_hallucination(txt):
            segs.append({"start": round(s, 2), "end": round(e, 2), "text": "", "emotion": "",
                         "events": [], "skipped": "hallucination" if txt else "empty",
                         "raw": txt[:80], "rms": round(rms, 5)})
            continue
        segs.append({"start": round(s, 2), "end": round(e, 2), "text": txt, "emotion": "",
                     "events": [], "rms": round(rms, 5), "backend": "whisper"})
    return segs


def backend_name() -> str:
    return "sensevoice" if _sensevoice() is not None else "whisper"
