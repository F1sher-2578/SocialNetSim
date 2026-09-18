"""构造带真值标签的测试视频集（不使用任何真实违规素材）。

三类来源：
  A. 合成帧 + 合成语音：画面烧录引流文字 / 侮辱字幕，语音由 macOS TTS 朗读脚本文本
  B. Wikimedia 知识共享图片：武器、火灾、人群等可公开获取的风险类目图片
  C. 正常长视频（Big Buck Bunny）与正常合成视频作为负样本
真值标签写入 data/videos/bench/labels.json，用于计算精确率/召回率与误报率。
"""
from __future__ import annotations
import json, subprocess, tempfile, shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "data" / "videos" / "bench"
IMGB = ROOT / "data" / "images" / "bench"
BENCH.mkdir(parents=True, exist_ok=True); IMGB.mkdir(parents=True, exist_ok=True)

# 中文字体候选，覆盖 macOS / Windows / Linux
FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc", "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


def get_font(size=42):
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            try:
                return ImageFont.truetype(f, size)
            except Exception:
                continue
    # 兜底：让 matplotlib 在系统已注册的字体里找一个支持中文的
    try:
        from matplotlib import font_manager
        for name in ("PingFang", "Microsoft YaHei", "SimHei", "Noto Sans CJK", "WenQuanYi"):
            for fp in font_manager.fontManager.ttflist:
                if name.lower() in fp.name.lower():
                    return ImageFont.truetype(fp.fname, size)
    except Exception:
        pass
    print("  [提示] 未找到中文字体，画面文字会显示为方块，OCR 相关样本将失效。"
          "Ubuntu 可 sudo apt install fonts-wqy-zenhei。")
    return ImageFont.load_default()


def make_frame(text: str, bg=(40, 60, 90), size=(854, 480), sub: str = "") -> np.ndarray:
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    # 背景纹理，避免纯色导致模型行为异常
    rng = np.random.default_rng(abs(hash(text)) % 2**31)
    arr = np.array(img).astype(np.int16)
    arr += rng.integers(-12, 12, arr.shape, dtype=np.int16)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    if text:
        d.text((40, size[1] // 2 - 60), text, font=get_font(44), fill=(255, 255, 255))
    if sub:
        d.text((40, size[1] - 90), sub, font=get_font(30), fill=(255, 230, 120))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def tts(text: str, out: Path, voice="Tingting", rate=180) -> bool:
    """合成中文语音。

    优先用 macOS 自带的 `say`；非 macOS 环境依次尝试 pyttsx3 / edge-tts，
    都不可用时返回 False（此时生成的测试视频没有音轨，语音相关的评测会自动跳过）。
    """
    import shutil as _sh
    if _sh.which("say"):
        aiff = out.with_suffix(".aiff")
        r = subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text], capture_output=True)
        if r.returncode != 0 or not aiff.exists():
            r = subprocess.run(["say", "-r", str(rate), "-o", str(aiff), text], capture_output=True)
        if aiff.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(out)],
                           capture_output=True)
            aiff.unlink(missing_ok=True)
            if out.exists():
                return True
    # 非 macOS 回退
    try:
        import pyttsx3
        eng = pyttsx3.init()
        eng.setProperty("rate", rate)
        eng.save_to_file(text, str(out))
        eng.runAndWait()
        if out.exists() and out.stat().st_size > 1024:
            return True
    except Exception:
        pass
    try:
        tmp_mp3 = out.with_suffix(".mp3")
        r = subprocess.run(["edge-tts", "--voice", "zh-CN-XiaoxiaoNeural", "--text", text,
                            "--write-media", str(tmp_mp3)], capture_output=True, timeout=120)
        if r.returncode == 0 and tmp_mp3.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp_mp3), "-ar", "16000", "-ac", "1", str(out)],
                           capture_output=True)
            tmp_mp3.unlink(missing_ok=True)
            return out.exists()
    except Exception:
        pass
    print("  [提示] 未找到可用的语音合成工具，测试视频将不含音轨。"
          "macOS 自带 say；其他系统可 pip install pyttsx3 或 edge-tts。")
    return False


def build_video(name: str, frames: list[np.ndarray], secs_per_frame: float, audio_text: str | None,
                out_dir: Path = BENCH, fps: int = 12) -> Path:
    tmp = Path(tempfile.mkdtemp())
    vid = tmp / "v.mp4"
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        for _ in range(int(fps * secs_per_frame)):
            vw.write(cv2.resize(f, (w, h)))
    vw.release()
    out = out_dir / f"{name}.mp4"
    if audio_text:
        wav = tmp / "a.wav"
        if tts(audio_text, wav):
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(vid), "-i", str(wav),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
                           capture_output=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(vid), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", str(out)], capture_output=True)
    else:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(vid), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(out)], capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


# ---------------- Wikimedia Commons 检索（知识共享/公有领域素材） ----------------
# 说明：不使用任何真实违规素材。风险类目仅取"武器展品/火灾/人群集会/吸烟"等可公开获取的
# 知识共享图片，用于验证 CLIP 零样本风险类目是否被正确触发。
WIKI_QUERIES = {
    "weapon": ["handgun revolver museum", "assault rifle museum exhibit"],
    "fire_disaster": ["house fire firefighter", "building fire flames"],
    "protest_crowd": ["protest demonstration crowd banner", "street march rally people"],
    "smoking_drug": ["person smoking cigarette", "cigarette smoking ashtray"],
    "normal_scenery": ["mountain landscape lake", "forest path nature"],
    "normal_food": ["plate of food meal", "breakfast table food"],
    "normal_face": ["portrait photograph man face", "portrait photograph woman face"],
}
UA = {"User-Agent": "SocialNetSim-course-project/1.0 (educational, local sandbox)"}


def _commons_search(term: str, n: int = 4, retries: int = 5) -> list[tuple[str, str, str]]:
    import urllib.request, urllib.parse, json as _json, time as _time
    q = urllib.parse.urlencode({"action": "query", "generator": "search",
                                "gsrsearch": f"filetype:bitmap {term}", "gsrnamespace": 6, "gsrlimit": n,
                                "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 800,
                                "format": "json"})
    for attempt in range(retries):
        try:
            req = urllib.request.Request("https://commons.wikimedia.org/w/api.php?" + q, headers=UA)
            d = _json.load(urllib.request.urlopen(req, timeout=40))
            out = []
            for p in d.get("query", {}).get("pages", {}).values():
                ii = p.get("imageinfo")
                if ii and ii[0].get("thumburl"):
                    lic = ii[0].get("extmetadata", {}).get("LicenseShortName", {}).get("value", "?")
                    out.append((p["title"], ii[0]["thumburl"], lic))
            return out
        except Exception as e:
            _time.sleep(3 * (attempt + 1))
    return []


def fetch_wiki_images(per_cat: int = 3) -> dict:
    import urllib.request, json as _json, time as _time
    cache = IMGB / "sources.json"
    known = _json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    out = {}
    for cat, terms in WIKI_QUERIES.items():
        out[cat] = []
        for ti, term in enumerate(terms):
            if len(out[cat]) >= per_cat:
                break
            for i, (title, url, lic) in enumerate(_commons_search(term, 4)):
                if len(out[cat]) >= per_cat:
                    break
                p = IMGB / f"{cat}_{ti}{i}.jpg"
                if not p.exists():
                    try:
                        req = urllib.request.Request(url, headers=UA)
                        with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
                            f.write(r.read())
                        _time.sleep(1.2)
                    except Exception as e:
                        print(f"  fetch fail {cat}: {str(e)[:70]}")
                        continue
                if p.exists() and p.stat().st_size > 8000:
                    out[cat].append(str(p))
                    known[p.name] = {"title": title, "license": lic, "url": url}
            _time.sleep(1.5)
    cache.write_text(_json.dumps(known, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    print("下载 Wikimedia 风险类目图片 …")
    wiki = fetch_wiki_images(per_cat=3)
    for k, v in wiki.items():
        print(f"  {k}: {len(v)} 张")

    labels = {}

    # 1) 广告引流（画面烧录 + 语音）
    f = [make_frame("限时福利", (30, 90, 60), sub="加微信 xk8899 领取内部优惠"),
         make_frame("兼职日结", (60, 40, 90), sub="兼职刷单日入三百，私信我"),
         make_frame("点击链接", (80, 60, 30), sub="点击链接领红包 t.cn/abc123")]
    p = build_video("ad_scam_short", f, 2.5, "兼职刷单日入三百，加微信 x k 八八九九，点击链接领红包，无抵押秒批")
    labels[p.name] = {"risky": True, "categories": ["spam", "fraud"], "modality": ["ocr", "asr"], "kind": "short"}

    # 2) 辱骂/冒犯（语音为主，画面正常）
    f = [make_frame("直播回放", (50, 70, 100)), make_frame("第 2 段", (70, 50, 100))]
    p = build_video("insult_speech_short", f, 3.0, "你这个傻逼给我滚出去，脑残废物，去死吧")
    labels[p.name] = {"risky": True, "categories": ["insult", "offensive"], "modality": ["asr"], "kind": "short"}

    # 3) 赌博引流（画面文字混淆 + 语音）
    f = [make_frame("娱乐城", (90, 30, 40), sub="百 家 乐 返 水 五 十"),
         make_frame("首充优惠", (40, 30, 90), sub="棋牌平台首充返水50%，速来")]
    p = build_video("gambling_short", f, 2.5, "棋牌平台首充返水百分之五十，百家乐，六合彩，速来下注")
    labels[p.name] = {"risky": True, "categories": ["gambling"], "modality": ["ocr", "asr"], "kind": "short"}

    # 4) 视觉风险：武器/火灾/人群（无语音风险）
    vis = []
    for cat in ("weapon", "fire_disaster", "protest_crowd"):
        for pth in wiki.get(cat, []):
            im = cv2.imread(pth)
            if im is not None:
                vis.append(cv2.resize(im, (854, 480)))
    if vis:
        p = build_video("visual_risk_short", vis, 3.0, "这是一段没有语音风险的视频，请注意画面内容")
        labels[p.name] = {"risky": True, "categories": ["weapon", "fire_disaster", "protest_crowd"],
                          "modality": ["visual"], "kind": "short"}

    # 5) 正常短视频（合成）
    f = [make_frame("今天的风景很好", (60, 110, 140)), make_frame("周末去公园散步", (70, 120, 90)),
         make_frame("记录生活", (110, 100, 70))]
    p = build_video("benign_synthetic_short", f, 3.0, "今天天气真好，出去公园散步，拍了很多好看的照片，周末过得很充实")
    labels[p.name] = {"risky": False, "categories": [], "modality": [], "kind": "short"}

    # 6) 正常图片短视频（Wikimedia 正常类目）
    ben = []
    for cat in ("normal_scenery", "normal_food", "normal_face"):
        for pth in wiki.get(cat, []):
            im = cv2.imread(pth)
            if im is not None:
                ben.append(cv2.resize(im, (854, 480)))
    if ben:
        p = build_video("benign_photos_short", ben, 3.0, "分享一些旅行和美食的照片，希望大家喜欢")
        labels[p.name] = {"risky": False, "categories": [], "modality": [], "kind": "short"}

    # 7) 长视频：正常内容中间插入风险片段（考察时序定位能力）
    long_frames = []
    for i in range(14):
        long_frames.append(make_frame(f"正常内容 第 {i+1} 段", (50 + i * 8, 90, 120 - i * 4)))
    long_frames[8] = make_frame("福利", (90, 30, 40), sub="加微信 xk8899 领内部优惠，兼职日结")
    long_frames[9] = make_frame("福利", (95, 35, 45), sub="点击链接 t.cn/abc123 秒批")
    p = build_video("long_mixed_with_ad", long_frames, 6.0,
                    "这是一段较长的视频，前面都是正常内容，中间会出现一段推广内容，加微信领取内部优惠，兼职日结，后面继续正常内容")
    labels[p.name] = {"risky": True, "categories": ["spam", "fraud"], "modality": ["ocr", "asr"],
                      "kind": "long", "risky_range": [48, 60]}

    # 8) 正常长视频（真实素材）
    bbb = ROOT / "data" / "videos" / "bbb_360p.webm"
    if bbb.exists():
        labels[bbb.name] = {"risky": False, "categories": [], "modality": [], "kind": "long", "external": str(bbb)}

    (BENCH / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n测试集：")
    for k, v in labels.items():
        print(f"  {k:32s} risky={v['risky']} {v['kind']:5s} {v['categories']}")
    return labels


if __name__ == "__main__":
    main()
