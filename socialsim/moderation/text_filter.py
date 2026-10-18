"""文本不良内容检测（多专家融合 + 变体归一化防御）。

专家：
  E1 规则/词表     —— 涉黄、涉赌、涉毒、暴恐、辱骂、诈骗引流、违规联系方式
  E2 COLD RoBERTa —— 中文冒犯性语言检测（thu-coai/roberta-base-cold）
  E3 引流/广告结构 —— 联系方式、外链、诱导话术的正则与结构特征

防御要点：攻击者常用同音字、拆字、插入零宽字符/符号、繁体、拼音、火星文来绕过关键词匹配。
normalize() 做多路归一化，并对"归一化后才命中"的样本额外标记 obfuscation，因为刻意混淆本身就是强风险信号。
"""
from __future__ import annotations
import re, unicodedata, functools
import numpy as np

# ---------------- 词表（教学用途，精简版；生产环境应接入权威词库） ----------------
LEXICON = {
    "porn": ["约炮", "裸聊", "色情", "黄片", "成人影片", "一夜情", "情色", "av资源", "福利视频", "开车视频", "小电影"],
    "gambling": ["赌博", "博彩", "棋牌", "百家乐", "六合彩", "下注", "返水", "真人娱乐城", "彩票内幕", "北京赛车"],
    "drugs": ["冰毒", "大麻", "摇头丸", "麻古", "海洛因", "迷药", "听话水"],
    "violence": ["杀了你", "砍死", "血腥", "枪支", "制爆", "炸药", "军火", "自制枪"],
    "insult": ["傻逼", "煞笔", "废物", "垃圾人", "滚蛋", "去死", "脑残", "智障", "贱人", "狗东西", "白痴"],
    "fraud": ["刷单", "兼职日结", "无抵押贷款", "内部优惠", "躺赚", "日入过千", "免费领取", "秒批", "代办证件", "征信修复"],
    "spam": ["加微信", "加v信", "私聊我", "扫码进群", "点击链接", "详情私信", "招代理"],
}
LABEL_WEIGHT = {"porn": 0.95, "gambling": 0.85, "drugs": 0.95, "violence": 0.9,
                "insult": 0.7, "fraud": 0.8, "spam": 0.6}

CONTACT_PATTERNS = [
    # 中文关键词可直接匹配；wx/vx 这类两字母缩写必须前后有分隔符，
    # 否则任意英文长串都会命中（实测 OCR 识别的片尾演职员表被判为引流）。
    (r"(?:微信|威信|v信|weixin)\s*[:：=]?\s*[a-zA-Z0-9_-]{5,20}", "contact_wechat", 0.7),
    (r"(?<![a-z0-9])(?:wx|vx)\s*[:：=]\s*[a-zA-Z0-9_-]{5,20}", "contact_wechat", 0.7),
    (r"(?:qq|扣扣|企鹅)\s*[:：=]?\s*\d{5,12}", "contact_qq", 0.7),
    (r"(?<!\d)1[3-9]\d{9}(?!\d)", "contact_phone", 0.6),
    (r"(?:https?://|www\.)[^\s，。！？]+", "external_link", 0.45),
    (r"[a-z0-9]+\.(?:cn|com|net|top|xyz|vip)/[a-zA-Z0-9]{3,}", "short_link", 0.55),
]
# 招揽语：噪声来源（OCR/ASR）下，联系方式与外链必须与招揽语同现才算违规，
# 否则片尾字幕、水印、官网地址会被大量误判。
SOLICIT_RE = re.compile(
    r"(加|领取|领|优惠|私信|私聊|点击|联系|咨询|扫码|免费|福利|红包|返水|首充|下单|代理|兼职|日结|贷款|"
    r"add\s+me|contact\s+me|dm\s+me|free\s+gift|click\s+here)", re.I)

# 同音/形近替换表（攻击者常用的字符级混淆）
HOMOPHONE = {"沙比": "傻逼", "傻b": "傻逼", "sb": "傻逼", "煞笔": "傻逼", "沙雕": "傻屌",
             "wei信": "微信", "v信": "微信", "威信": "微信", "溦信": "微信",
             "du博": "赌博", "堵博": "赌博", "涩情": "色情", "瑟情": "色情"}
ZERO_WIDTH = "".join(chr(c) for c in (0x200b, 0x200c, 0x200d, 0xfeff, 0x2060))
SEPARATORS = r"[\s\.\*\-_~•·|/\\+=,，。、:：;；!！?？\^#\$%&\(\)\[\]【】<>「」\"']"


@functools.lru_cache(maxsize=1)
def _opencc():
    try:
        import opencc
        return opencc.OpenCC("t2s")
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def _cold_model():
    """延迟加载 COLD 中文冒犯语言检测模型。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from ..config import HF_COLD, device
    tok = AutoTokenizer.from_pretrained(HF_COLD)
    mdl = AutoModelForSequenceClassification.from_pretrained(HF_COLD).eval()
    dev = device()
    try:
        mdl = mdl.to(dev)
    except Exception:
        dev = "cpu"
    return tok, mdl, dev


def normalize(text: str) -> dict:
    """多路归一化，返回若干变体，任一变体命中词表即视为命中。"""
    raw = text or ""
    t = unicodedata.normalize("NFKC", raw)                    # 全角→半角、兼容字符
    t = "".join(ch for ch in t if ch not in ZERO_WIDTH)        # 去零宽字符
    cc = _opencc()
    t = cc.convert(t) if cc else t                             # 繁体→简体
    lower = t.lower()
    compact = re.sub(SEPARATORS, "", lower)                    # 去分隔符（对抗"傻 逼"/"傻*逼"）
    dedup = re.sub(r"(.)\1{2,}", r"\1", compact)               # 压缩重复字符（"傻傻傻逼"）
    homo = dedup
    for k, v in HOMOPHONE.items():
        if k.isascii() and len(k) <= 3:
            # "sb" 这类短拉丁缩写必须按词边界替换，否则在任意英文长串里都能捞出来
            # （实测把 "workstationsblender" 中的 sb 判成辱骂）
            homo = re.sub(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", v, homo)
        else:
            homo = homo.replace(k, v)
    try:
        from pypinyin import lazy_pinyin
        pinyin = "".join(lazy_pinyin(re.sub(r"[^一-鿿]", "", t)))
    except Exception:
        pinyin = ""
    return {"raw": raw, "nfkc": t, "compact": compact, "dedup": dedup, "homo": homo, "pinyin": pinyin}


@functools.lru_cache(maxsize=1)
def _pinyin_lexicon() -> dict:
    """词表的拼音索引，用于捕捉纯拼音绕过（如 shabi / dubo）。"""
    try:
        from pypinyin import lazy_pinyin
    except Exception:
        return {}
    out = {}
    for label, words in LEXICON.items():
        for w in words:
            py = "".join(lazy_pinyin(w))
            if len(py) >= 4:
                out.setdefault(py, []).append((label, w))
    return out


def rule_expert(text: str, allow_pinyin: bool = True, require_solicit: bool = False) -> dict:
    """E1+E3：词表命中 + 联系方式/外链结构，并区分是否经过混淆。

    allow_pinyin=False 用于 OCR 识别结果等噪声文本：
    OCR 的拉丁串误识别率高，拼音子串匹配会产生大量误报（实测把片尾字幕判成违规）。
    """
    v = normalize(text)
    hits, labels, obf = [], set(), False
    for label, words in LEXICON.items():
        for w in words:
            wl = w.lower()
            w_compact = re.sub(SEPARATORS, "", wl)
            in_raw = wl in v["nfkc"].lower()
            in_norm = (w_compact in v["compact"]) or (w_compact in v["dedup"]) or (w_compact in v["homo"])
            if in_raw or in_norm:
                hits.append({"label": label, "word": w, "obfuscated": (not in_raw) and in_norm})
                labels.add(label)
                obf = obf or ((not in_raw) and in_norm)
    # 纯拼音绕过（"shabi"）：检查汉字转写出的拼音串；拉丁串必须按**词边界**匹配，
    # 否则任意长串里都能捞出子串（实测 OCR 噪声因此大量误报）。
    if allow_pinyin:
        latin_tokens = set(re.findall(r"[a-z]{3,}", v["dedup"]))
        for py, items in _pinyin_lexicon().items():
            if not py or len(py) < 5:
                continue
            hit = (py in v["pinyin"]) or (py in latin_tokens)
            if hit and not any(h["word"] == items[0][1] for h in hits):
                hits.append({"label": items[0][0], "word": items[0][1], "obfuscated": True, "via": "pinyin"})
                labels.add(items[0][0]); obf = True
    struct = []
    has_solicit = bool(SOLICIT_RE.search(v["nfkc"]))
    for pat, name, w in CONTACT_PATTERNS:
        m = re.search(pat, v["nfkc"], re.I)
        if not m:
            continue
        if require_solicit and not has_solicit:
            continue        # 噪声来源下，孤立的网址/号码不算违规
        struct.append({"label": name, "weight": w, "match": m.group(0)[:40]}); labels.add(name)
    score = 0.0
    for h in hits:
        score = max(score, LABEL_WEIGHT.get(h["label"], 0.5))
    for st in struct:
        score = max(score, st["weight"])
    if obf and score > 0:
        score = min(1.0, score + 0.15)      # 刻意混淆 → 加权
        labels.add("obfuscation")
    if len(hits) >= 3:
        score = min(1.0, score + 0.05)
    return {"score": float(score), "labels": sorted(labels), "hits": hits, "structures": struct,
            "obfuscated": obf, "normalized": v["dedup"][:200],
            "stages": {k: v[k][:200] for k in ("raw", "nfkc", "compact", "dedup", "homo", "pinyin")}}


def cold_expert(texts: list[str]) -> list[float]:
    """E2：COLD 中文冒犯性语言检测，返回 offensive 概率。"""
    if not texts:
        return []
    import torch
    tok, mdl, dev = _cold_model()
    out = []
    for i in range(0, len(texts), 16):
        batch = [t[:480] for t in texts[i:i + 16]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            p = torch.softmax(mdl(**enc).logits.float(), -1)[:, 1]
        out += p.cpu().tolist()
    return out


def analyze_text(text: str, use_model: bool = True, noisy_source: bool = False) -> dict:
    """融合规则专家与模型专家。规则给出可解释证据，模型补充泛化能力。

    noisy_source=True 表示文本来自 OCR/ASR 等含识别噪声的通道，
    此时关闭拼音匹配并提高判定保守度。
    """
    if not (text or "").strip():
        return {"score": 0.0, "labels": [], "experts": {}}
    rule = rule_expert(text, allow_pinyin=not noisy_source, require_solicit=noisy_source)
    experts = {"rule": rule}
    score = rule["score"]
    labels = set(rule["labels"])
    if use_model:
        try:
            # 对归一化后的文本也过一遍模型，抵消字符混淆对模型的干扰
            v = normalize(text)
            probs = cold_expert([text, v["dedup"]])
            p = float(max(probs))
            experts["cold"] = {"offensive_prob": p, "raw": probs[0], "normalized": probs[1]}
            if p >= 0.5:
                labels.add("offensive")
            # 融合：取最大值（任一专家高置信即判风险），并对双专家一致的情况小幅提升
            fused = max(score, p)
            if score >= 0.4 and p >= 0.5:
                fused = min(1.0, fused + 0.1)
            score = fused
        except Exception as e:
            experts["cold"] = {"error": f"{type(e).__name__}: {e}"}
    return {"score": float(min(1.0, score)), "labels": sorted(labels), "experts": experts}
