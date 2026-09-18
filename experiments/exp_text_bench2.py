"""合并文本基准：COLD（冒犯言论）+ 垃圾广告/赌博引流 + 正常内容。

动机：单用 COLD 评估会低估规则专家——COLD 只标注"冒犯性言论"，完全不覆盖广告引流、
赌博、诈骗这类平台高发违规；而 COLD 模型对这类文本输出接近 0。反之只用广告样本又会
低估模型。合并基准才能说明"规则 + 模型"融合的必要性。

三个切片：
  S1 COLD 冒犯 vs COLD 正常     —— 考察语义类违规
  S2 广告/赌博/诈骗 vs 正常内容 —— 考察模式类违规
  S3 S1 ∪ S2                    —— 平台真实混合场景
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from socialsim.moderation.text_filter import rule_expert, cold_expert
from socialsim.simulation import content_gen as cg
from socialsim.config import RESULT_DIR, ROOT
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


def metrics(y, score, th=0.5):
    pred = (np.array(score) >= th).astype(int)
    d = {"acc": accuracy_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
         "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0)}
    try:
        d["auc"] = roc_auc_score(y, score)
    except Exception:
        d["auc"] = float("nan")
    return {k: round(float(v), 4) for k, v in d.items()}


def fuse(rule, model):
    f = np.maximum(rule, model)
    return np.where((rule >= 0.4) & (model >= 0.5), np.minimum(1.0, f + 0.1), f)


def build_spam_slice(n: int, seed: int) -> tuple[list[str], np.ndarray]:
    """正样本：水军模板生成的广告/赌博/诈骗文本；负样本：正常用户帖子与评论。"""
    rng = random.Random(seed)
    pos = []
    while len(pos) < n // 2:
        s = rng.choice(["spam", "boost", "smear"])
        if s == "spam":
            pos.append(cg.bot_comment("spam", rng, rng.uniform(0.3, 0.9)))
        else:
            pos.append(cg.bot_post(s, rng, "#热搜", rng.uniform(0.3, 0.9))[0])
    # 只保留确实属于违规类的（boost/smear 中的普通夸赞不算违规）
    from socialsim.moderation.text_filter import LEXICON
    bad_words = set(LEXICON["fraud"] + LEXICON["spam"] + LEXICON["gambling"] + LEXICON["insult"])
    pos = [t for t in pos if any(w in t for w in bad_words)][:n // 2]
    neg = []
    while len(neg) < n // 2:
        if rng.random() < 0.5:
            neg.append(cg.normal_post(rng, rng.sample(list(cg.NORMAL_TEMPLATES), 2))[0])
        else:
            neg.append(cg.normal_comment(rng))
    texts = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    return texts, y


def run_slice(name: str, texts: list[str], y: np.ndarray) -> dict:
    rule = np.array([rule_expert(t)["score"] for t in texts])
    model = np.array(cold_expert(texts))
    f = fuse(rule, model)
    out = {"n": len(texts), "positive_rate": round(float(y.mean()), 3),
           "rule_only": metrics(y, rule), "model_only": metrics(y, model), "fused": metrics(y, f)}
    print(f"\n== {name} (n={len(texts)}, 正例率 {y.mean():.2f}) ==")
    for k in ("rule_only", "model_only", "fused"):
        v = out[k]
        print(f"  {k:12s} AUC={v['auc']:.4f} F1={v['f1']:.4f} P={v['precision']:.4f} R={v['recall']:.4f}")
    return out


def main(n_cold: int = 1200, n_spam: int = 800, seed: int = 0):
    df = pd.read_csv(ROOT / "data" / "raw" / "COLD_test.csv").dropna(subset=["TEXT", "label"])
    df = df.sample(n=min(n_cold, len(df)), random_state=seed)
    cold_texts = df["TEXT"].astype(str).tolist()
    cold_y = df["label"].astype(int).values

    spam_texts, spam_y = build_spam_slice(n_spam, seed)

    res = {}
    res["S1_COLD_offensive"] = run_slice("S1 COLD 冒犯性言论", cold_texts, cold_y)
    res["S2_spam_ads"] = run_slice("S2 广告/赌博/诈骗引流", spam_texts, spam_y)
    res["S3_combined"] = run_slice("S3 合并（平台真实混合场景）",
                                   cold_texts + spam_texts, np.concatenate([cold_y, spam_y]))

    # 按 COLD 的 topic 细分，看模型在不同议题上的差异
    by_topic = {}
    for topic in sorted(df["topic"].unique()):
        sub = df[df["topic"] == topic]
        if len(sub) < 50:
            continue
        t = sub["TEXT"].astype(str).tolist(); yy = sub["label"].astype(int).values
        m = np.array(cold_expert(t))
        by_topic[topic] = metrics(yy, m)
    res["cold_by_topic_model_only"] = by_topic
    print("\n== COLD 分议题（模型专家）==")
    for k, v in by_topic.items():
        print(f"  {k:8s} AUC={v['auc']:.4f} F1={v['f1']:.4f}")

    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "text_bench_combined.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {RESULT_DIR/'text_bench_combined.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", type=int, default=1200); ap.add_argument("--spam", type=int, default=800)
    a = ap.parse_args(); main(a.cold, a.spam)
