"""针对语音审核链路的对抗攻击与防御。

攻击目标是本项目视频审核流水线里的**音频通道**：
违规视频（广告引流、辱骂、赌博话术）常常把风险信息放在语音里，
平台靠 Whisper 做语音转写后再送文本审核。攻击者只要让 ASR 转写失真，
就能让下游的文本审核完全失效——这是一条真实且低成本的规避路径。

攻击（按攻击者所需能力从低到高）：
  L0 信号处理类（零模型访问）：变速、变调、加噪、带通滤波、回声、音量压缩、
                                时间掩蔽（在关键词处插入短促噪声）
  L1 心理声学类（零模型访问）：在感知掩蔽阈下叠加噪声，人耳几乎听不出差别
  L2 白盒梯度类：对 Whisper 编码器做 PGD，使转写偏离原文（需模型梯度）

防御：
  语音增强（谱减法降噪）、带通限制、重采样归一化、
  多假设转写一致性检验（对同一段音频施加若干轻扰动，转写不一致即可疑），
  以及"ASR 失败时不放行"的安全默认策略（fail-closed）。

关键洞察：审核系统不应把"转写为空/乱码"当作安全。攻击者恰恰追求这个结果。
"""
from __future__ import annotations
import numpy as np

SR = 16000


# ---------------------------------------------------------------- L0 信号处理攻击
def speed_change(wav: np.ndarray, rate: float = 1.15) -> np.ndarray:
    import librosa
    return librosa.effects.time_stretch(wav.astype(np.float32), rate=rate)


def pitch_shift(wav: np.ndarray, steps: float = 2.0, sr: int = SR) -> np.ndarray:
    import librosa
    return librosa.effects.pitch_shift(wav.astype(np.float32), sr=sr, n_steps=steps)


def add_noise(wav: np.ndarray, snr_db: float = 10.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, len(wav))
    p_s = np.mean(wav ** 2) + 1e-12
    p_n = np.mean(n ** 2) + 1e-12
    n = n * np.sqrt(p_s / (p_n * 10 ** (snr_db / 10)))
    return (wav + n).astype(np.float32)


def bandpass(wav: np.ndarray, low: float = 300, high: float = 3400, sr: int = SR) -> np.ndarray:
    """限制到电话频带：保留可懂度但压缩 ASR 依赖的频谱细节。"""
    import scipy.signal as sps
    sos = sps.butter(6, [low / (sr / 2), min(0.99, high / (sr / 2))], btype="band", output="sos")
    return sps.sosfilt(sos, wav).astype(np.float32)


def echo(wav: np.ndarray, delay_s: float = 0.08, decay: float = 0.5, sr: int = SR) -> np.ndarray:
    d = int(delay_s * sr)
    out = np.copy(wav).astype(np.float32)
    if d < len(wav):
        out[d:] += decay * wav[:-d]
    return (out / (np.abs(out).max() + 1e-9)).astype(np.float32)


def time_mask(wav: np.ndarray, n_masks: int = 8, mask_ms: float = 60, seed: int = 0,
              sr: int = SR) -> np.ndarray:
    """时间掩蔽：在若干随机位置用短促噪声覆盖，破坏词边界。"""
    rng = np.random.default_rng(seed)
    out = np.copy(wav).astype(np.float32)
    w = int(mask_ms * sr / 1000)
    for _ in range(n_masks):
        if len(out) <= w:
            break
        s = int(rng.integers(0, len(out) - w))
        out[s:s + w] = rng.normal(0, np.abs(wav).mean() * 1.5, w)
    return out


def volume_compress(wav: np.ndarray, factor: float = 0.05) -> np.ndarray:
    """极度压低音量：很多 ASR 会直接输出空转写，而人仍可通过放大听清。"""
    return (wav * factor).astype(np.float32)


# ---------------------------------------------------------------- L1 心理声学掩蔽
def psychoacoustic_noise(wav: np.ndarray, eps_db: float = -25, seed: int = 0,
                         n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """在每个时频单元的能量之下按固定 dB 余量叠加噪声。

    简化的听觉掩蔽模型：强分量会掩蔽其邻近的弱分量，
    因此在信号自身能量以下 eps_db 的噪声人耳几乎察觉不到，却可能干扰 ASR 的声学特征。
    """
    import librosa
    rng = np.random.default_rng(seed)
    S = librosa.stft(wav.astype(np.float32), n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    scale = 10 ** (eps_db / 20)
    noise_mag = mag * scale * rng.uniform(0.5, 1.5, mag.shape)
    noise_phase = rng.uniform(-np.pi, np.pi, mag.shape)
    S2 = S + noise_mag * np.exp(1j * noise_phase)
    y = librosa.istft(S2, hop_length=hop, length=len(wav))
    return (y / (np.abs(y).max() + 1e-9) * (np.abs(wav).max() + 1e-9)).astype(np.float32)


def keyword_mask(wav: np.ndarray, transcribe_fn, risk_fn, win_s: float = 0.6,
                 hop_s: float = 0.3, max_masks: int = 6, sr: int = SR) -> np.ndarray:
    """定点消除：只抹掉承载风险词的那几百毫秒，其余音频原样保留。

    攻击者的先验很强——他自己写的稿子，当然知道哪几个词会触发审核。
    做法：滑窗静音，观察审核分数下降最多的窗口，贪心地依次消除。
    相比"整段加噪/变速"这类无差别破坏，本攻击对音质的损害极小，
    而且**恰好绕开了响度归一化、谱减法这类全局防御**——因为音频整体并没有被破坏。
    """
    base = risk_fn(transcribe_fn(wav))
    if base <= 0:
        return wav.astype(np.float32)
    out = np.copy(wav).astype(np.float32)
    n_win = int(win_s * sr)
    starts = list(range(0, max(1, len(wav) - n_win), int(hop_s * sr)))
    for _ in range(max_masks):
        best, best_score = None, risk_fn(transcribe_fn(out))
        if best_score <= 0.15:
            break
        for st in starts:
            trial = np.copy(out)
            trial[st:st + n_win] = 0.0
            sc = risk_fn(transcribe_fn(trial))
            if sc < best_score - 1e-6:
                best_score, best = sc, st
        if best is None:
            break
        out[best:best + n_win] = 0.0
        starts = [x for x in starts if abs(x - best) >= n_win]
    return out


def selective_overlay(wav: np.ndarray, transcribe_fn, risk_fn, win_s: float = 0.8,
                      snr_db: float = -3.0, max_overlays: int = 4, sr: int = SR) -> np.ndarray:
    """在风险词位置叠加响亮噪声（而非静音），听感上更像"环境干扰"，更难被察觉是人为处理。"""
    rng = np.random.default_rng(0)
    out = np.copy(wav).astype(np.float32)
    n_win = int(win_s * sr)
    starts = list(range(0, max(1, len(wav) - n_win), int(win_s * sr / 2)))
    for _ in range(max_overlays):
        best, best_score = None, risk_fn(transcribe_fn(out))
        if best_score <= 0.15:
            break
        for st in starts:
            trial = np.copy(out)
            seg = trial[st:st + n_win]
            amp = np.sqrt(np.mean(seg ** 2) + 1e-12) * (10 ** (-snr_db / 20))
            trial[st:st + n_win] = seg + rng.normal(0, amp, len(seg))
            sc = risk_fn(transcribe_fn(np.clip(trial, -1, 1)))
            if sc < best_score - 1e-6:
                best_score, best = sc, st
        if best is None:
            break
        seg = out[best:best + n_win]
        amp = np.sqrt(np.mean(seg ** 2) + 1e-12) * (10 ** (-snr_db / 20))
        out[best:best + n_win] = seg + rng.normal(0, amp, len(seg))
        starts = [x for x in starts if abs(x - best) >= n_win]
    return np.clip(out, -1, 1).astype(np.float32)


# 需要查询审核结果的"自适应"攻击，单独登记（签名与其他攻击不同）
ADAPTIVE_ATTACKS = {"keyword_mask": keyword_mask, "selective_overlay": selective_overlay}

ATTACKS = {
    "speed_1.15x": lambda w: speed_change(w, 1.15),
    "speed_0.85x": lambda w: speed_change(w, 0.85),
    "pitch_+3": lambda w: pitch_shift(w, 3.0),
    "pitch_-3": lambda w: pitch_shift(w, -3.0),
    "noise_snr10": lambda w: add_noise(w, 10),
    "noise_snr5": lambda w: add_noise(w, 5),
    "noise_snr0": lambda w: add_noise(w, 0),
    "bandpass_tel": lambda w: bandpass(w),
    "echo": lambda w: echo(w),
    "time_mask": lambda w: time_mask(w),
    "volume_0.05": lambda w: volume_compress(w, 0.05),
    "psycho_-25dB": lambda w: psychoacoustic_noise(w, -25),
    "psycho_-15dB": lambda w: psychoacoustic_noise(w, -15),
}


# ---------------------------------------------------------------- 评估指标
def cer(ref: str, hyp: str) -> float:
    """字符错误率（中文场景下比 WER 更合适）。"""
    ref = "".join((ref or "").split())
    hyp = "".join((hyp or "").split())
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[n] / m


def snr_db(clean: np.ndarray, adv: np.ndarray) -> float:
    n = min(len(clean), len(adv))
    c, a = clean[:n], adv[:n]
    noise = a - c
    ps, pn = float(np.mean(c ** 2)), float(np.mean(noise ** 2))
    return float("inf") if pn <= 1e-15 else float(10 * np.log10(ps / (pn + 1e-15)))


# ---------------------------------------------------------------- 防御
def spectral_subtraction(wav: np.ndarray, n_fft: int = 1024, hop: int = 256,
                         noise_frames: int = 6, alpha: float = 1.5) -> np.ndarray:
    """谱减法降噪：用开头若干帧估计噪声谱并减去。"""
    import librosa
    S = librosa.stft(wav.astype(np.float32), n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    noise = np.median(mag[:, :max(1, noise_frames)], axis=1, keepdims=True)
    clean = np.maximum(mag - alpha * noise, 0.05 * mag)
    y = librosa.istft(clean * np.exp(1j * phase), hop_length=hop, length=len(wav))
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def normalize_loudness(wav: np.ndarray, target_rms: float = 0.06) -> np.ndarray:
    """响度归一化：直接消解"压低音量"这类攻击。"""
    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-12
    return np.clip(wav * (target_rms / rms), -1.0, 1.0).astype(np.float32)


def resample_roundtrip(wav: np.ndarray, mid_sr: int = 8000, sr: int = SR) -> np.ndarray:
    """降采样再升采样：抹掉高频对抗成分。"""
    import librosa
    y = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=mid_sr)
    return librosa.resample(y, orig_sr=mid_sr, target_sr=sr).astype(np.float32)


DEFENSES = {
    "none": lambda w: w,
    "loudness_norm": normalize_loudness,
    "spectral_sub": spectral_subtraction,
    "resample_8k": resample_roundtrip,
    "norm+specsub": lambda w: spectral_subtraction(normalize_loudness(w)),
}


def consistency_check(transcribe_fn, wav: np.ndarray, n: int = 3) -> dict:
    """多假设一致性检验：对同一段音频施加若干轻微扰动分别转写。

    正常语音在轻扰动下转写基本稳定；对抗音频往往在扰动下转写剧烈变化，
    这个不一致性本身就是"该音频被动过手脚"的信号。
    """
    base = transcribe_fn(wav)
    variants = [add_noise(wav, 25, seed=i) for i in range(n)]
    outs = [transcribe_fn(v) for v in variants]
    cers = [cer(base, o) for o in outs]
    return {"base": base, "variants": outs, "mean_cer": float(np.mean(cers)),
            "max_cer": float(np.max(cers)), "suspicious": float(np.mean(cers)) > 0.4}
