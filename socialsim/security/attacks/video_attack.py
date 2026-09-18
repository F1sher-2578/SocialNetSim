"""针对视频审核流水线的攻击。

视频审核与图像审核有一个本质区别：**审核系统不会看每一帧**。
长视频必须抽样，因此攻击面从"像素"扩展到了"采样策略"本身。
这是视频场景独有、也是最廉价的攻击面。

攻击分三类：

  A. 时序稀疏攻击（利用采样策略）
       只需扰动少数被采样到的帧，或反过来——把违规内容放在不易被采样的位置。
       - sparse_frame_attack   只对 top-K 高风险帧加对抗扰动，其余帧不动
       - keyframe_evasion      把违规片段缩短到关键帧间隔以下，使其落在采样缝隙里
       - scene_flood           插入大量无关镜头切换，稀释关键帧预算

  B. 编码层攻击（利用解码与压缩）
       - recompress            低码率重压缩，抹掉细节
       - frame_rate / 分辨率下采样
       - 画中画：把违规内容缩到小窗口，周围填正常画面

  C. 跨模态错配攻击
       违规信息只放在一个模态里，并攻击该模态。
       例如风险全在语音中，就只攻击音频通道（见 audio_attack.py）。

防御思路：提高采样密度的下界、随机化采样时刻（使攻击者无法预测哪帧被看）、
多尺度检测、以及对"异常高的镜头切换率"本身做检测。
"""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- A. 时序稀疏
def sparse_frame_attack(video_path: str | Path, out_path: str | Path, frame_times: list[float],
                        clf, eps: float = 8 / 255, steps: int = 10, fps_out: int = 25) -> dict:
    """只对指定时刻的帧施加对抗扰动，重新编码成视频。

    攻击者的先验：审核系统倾向于在镜头切换点采样。
    只要攻破这些帧，视频级的 top-k 聚合分数就会被拉下来。
    """
    import cv2
    import torch
    from . import image_attack as IA

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_out
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_idx = {int(round(t * fps)) for t in frame_times}
    # 连同前后各一帧一起扰动，抵抗采样时刻的微小偏移
    expanded = set()
    for i in target_idx:
        expanded |= {i - 1, i, i + 1}

    tmp = Path(tempfile.mkdtemp()) / "adv.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n_perturbed = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in expanded:
            x = IA.to_tensor01(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), clf.dev)
            adv = IA.pgd(clf, x, eps, steps=steps, minimize=True)
            arr = (adv[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            n_perturbed += 1
        vw.write(frame)
        idx += 1
    cap.release(); vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(out_path)], capture_output=True)
    # 保留原音轨
    _mux_audio(video_path, out_path)
    return {"n_frames_total": n_total, "n_perturbed": n_perturbed,
            "perturb_ratio": n_perturbed / max(1, n_total), "eps": eps}


def keyframe_evasion(video_path: str | Path, out_path: str | Path, risky_range: tuple[float, float],
                     target_len: float = 0.5) -> dict:
    """把违规片段压缩到关键帧最小间隔以下，使其有较大概率落在采样缝隙里。

    这是一个纯剪辑操作，不需要任何模型访问权，成本极低。
    """
    s, e = risky_range
    keep = max(0.1, min(target_len, e - s))
    mid = (s + e) / 2
    ns, ne = mid - keep / 2, mid + keep / 2
    with tempfile.TemporaryDirectory() as td:
        parts = []
        for i, (a, b) in enumerate([(0, s), (ns, ne), (e, None)]):
            p = Path(td) / f"p{i}.mp4"
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-ss", str(max(0, a))]
            if b is not None:
                cmd += ["-to", str(b)]
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(p)]
            subprocess.run(cmd, capture_output=True)
            if p.exists() and p.stat().st_size > 1000:
                parts.append(p)
        lst = Path(td) / "list.txt"
        lst.write_text("\n".join(f"file '{p}'" for p in parts), encoding="utf-8")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(out_path)], capture_output=True)
    return {"original_risky_len": e - s, "new_risky_len": keep}


def scene_flood(video_path: str | Path, out_path: str | Path, n_flashes: int = 60,
                flash_ms: int = 40) -> dict:
    """插入大量单帧闪烁，制造虚假的镜头切换。

    关键帧选择算法优先取镜头切换点，攻击者用廉价的闪帧灌满这个预算，
    真正的违规片段就更难被选中。防御：对异常切换率本身报警。
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rng = np.random.default_rng(0)
    flash_at = set(rng.choice(max(1, n_total), size=min(n_flashes, max(1, n_total)), replace=False).tolist())
    flash_len = max(1, int(flash_ms * fps / 1000))
    tmp = Path(tempfile.mkdtemp()) / "flood.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    n_ins = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        vw.write(f)
        if idx in flash_at:
            color = rng.integers(0, 255, 3)
            flash = np.full((h, w, 3), color, np.uint8)
            for _ in range(flash_len):
                vw.write(flash)
            n_ins += flash_len
        idx += 1
    cap.release(); vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out_path)], capture_output=True)
    _mux_audio(video_path, out_path)
    return {"n_flashes": len(flash_at), "frames_inserted": n_ins}


# ---------------------------------------------------------------- B. 编码层
def recompress(video_path: str | Path, out_path: str | Path, crf: int = 40,
               scale: float = 1.0, fps: int | None = None) -> dict:
    vf = []
    if scale != 1.0:
        vf.append(f"scale=iw*{scale}:ih*{scale}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-c:v", "libx264", "-crf", str(crf)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-pix_fmt", "yuv420p", "-c:a", "copy", str(out_path)]
    subprocess.run(cmd, capture_output=True)
    return {"crf": crf, "scale": scale, "fps": fps,
            "size_ratio": Path(out_path).stat().st_size / max(1, Path(video_path).stat().st_size)}


def picture_in_picture(video_path: str | Path, out_path: str | Path, scale: float = 0.35,
                       bg_color: str = "white") -> dict:
    """画中画：把原视频缩到角落，周围填充正常背景。

    针对整帧分类器：违规内容只占画面的一小部分，全局特征被背景稀释。
    防御：多尺度/分块检测，而不只看整帧。
    """
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
                    "-vf", f"scale=iw*{scale}:ih*{scale},pad=iw/{scale}:ih/{scale}:(ow-iw)/2:(oh-ih)/2:{bg_color}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(out_path)],
                   capture_output=True)
    return {"scale": scale, "area_ratio": scale ** 2}


def _mux_audio(src: str | Path, dst: str | Path):
    """把原视频的音轨合并回处理后的视频（多数攻击只动画面）。"""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                            "stream=codec_type", "-of", "csv=p=0", str(src)],
                           capture_output=True, text=True)
    if "audio" not in probe.stdout:
        return
    tmp = Path(str(dst) + ".mux.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(dst), "-i", str(src),
                        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac",
                        "-shortest", str(tmp)], capture_output=True)
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1000:
        tmp.replace(dst)
    else:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------- 防御
def randomized_sampling_times(duration: float, n: int, seed: int | None = None) -> list[float]:
    """随机化采样时刻：攻击者无法预知哪些帧会被检查。

    分层随机：把时间轴均分成 n 段，每段内随机取一点。
    既保证覆盖均匀，又让采样时刻不可预测。
    """
    rng = np.random.default_rng(seed)
    edges = np.linspace(0, duration, n + 1)
    return [float(rng.uniform(edges[i], edges[i + 1])) for i in range(n)]


def detect_scene_flood(scene_cuts: int, duration: float, threshold: float = 2.0) -> dict:
    """镜头切换率异常检测：正常影视内容一般每秒远低于 2 次切换。"""
    rate = scene_cuts / max(1e-6, duration)
    return {"cut_rate_per_sec": round(rate, 3), "suspicious": rate > threshold, "threshold": threshold}
