"""文本审核层在公开中文基准 COLD 上的评估。

COLD (Chinese Offensive Language Dataset, Deng et al., EMNLP 2022) 是中文冒犯性语言检测的
公开基准，测试集 5,323 条，按 topic 分为 race / gender / region 三类。

本实验评估三种配置，用于论证"规则 + 模型"融合的必要性：
  R  仅规则/词表专家
  M  仅 COLD 模型专家
  F  融合（本项目实际使用的配置）
并额外测量对抗混淆下的鲁棒性：对测试集施加字符级混淆攻击后各配置的召回率变化。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from socialsim.moderation.text_filter import rule_expert, cold_expert, analyze_text
from socialsim.config import RESULT_DIR, ROOT
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


def metrics(y, score, th=0.5) -> dict:
    pred = (np.array(score) >= th).astype(int)
    out = {"acc": accuracy_score(y, pred), "f1": f1_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred)}
    try:
        out["auc"] = roc_auc_score(y, score)
    except Exception:
        out["auc"] = float("nan")
    return {k: round(float(v), 4) for k, v in out.items()}


def obfuscate(text: str, rng: np.random.Generator, level: float = 0.6) -> str:
    """字符级混淆攻击：模拟真实用户绕过关键词过滤的常见手法。"""
    import re
    SEP = ["*", " ", ".", "-", "·", "_"]
    chars = list(text)
    out = []
    for ch in chars:
        out.append(ch)
        if "一" <= ch <= "鿿" and rng.random() < level * 0.35:
            out.append(rng.choice(SEP))
    s = "".join(out)
    if rng.random() < level:
        s = s.replace("傻逼", "沙比").replace("垃圾", "辣鸡").replace("微信", "威信")
    return s


def main(n: int = 1500, seed: int = 0):
    df = pd.read_csv(ROOT / "data" / "raw" / "COLD_test.csv")
    print("COLD 测试集:", df.shape, df.columns.tolist())
    col_t = "TEXT" if "TEXT" in df.columns else df.columns[-1]
    col_y = "label" if "label" in df.columns else df.columns[0]
    df = df.dropna(subset=[col_t, col_y])
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
    texts = df[col_t].astype(str).tolist()
    y = df[col_y].astype(int).values
    print(f"样本 {len(texts)}，冒犯类占比 {y.mean():.3f}")

    res = {"n": len(texts), "positive_rate": float(y.mean())}
    t0 = time.time()
    rule = np.array([rule_expert(t)["score"] for t in texts])
    res["rule_time_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    model = np.array(cold_expert(texts))
    res["model_time_s"] = round(time.time() - t0, 1)
    fuse = np.maximum(rule, model)
    fuse = np.where((rule >= 0.4) & (model >= 0.5), np.minimum(1.0, fuse + 0.1), fuse)

    res["clean"] = {"rule_only": metrics(y, rule), "model_only": metrics(y, model), "fused": metrics(y, fuse)}
    print("\n== 原始文本 ==")
    for k, v in res["clean"].items():
        print(f"  {k:12s} AUC={v['auc']:.4f} F1={v['f1']:.4f} P={v['precision']:.4f} R={v['recall']:.4f}")

    # --- 对抗混淆 ---
    rng = np.random.default_rng(seed)
    obf = [obfuscate(t, rng) for t in texts]
    rule_o = np.array([rule_expert(t)["score"] for t in obf])
    model_o = np.array(cold_expert(obf))
    fuse_o = np.maximum(rule_o, model_o)
    fuse_o = np.where((rule_o >= 0.4) & (model_o >= 0.5), np.minimum(1.0, fuse_o + 0.1), fuse_o)
    res["obfuscated"] = {"rule_only": metrics(y, rule_o), "model_only": metrics(y, model_o), "fused": metrics(y, fuse_o)}
    print("\n== 字符级混淆攻击后 ==")
    for k, v in res["obfuscated"].items():
        d = v["recall"] - res["clean"][k]["recall"]
        print(f"  {k:12s} AUC={v['auc']:.4f} F1={v['f1']:.4f} R={v['recall']:.4f} (召回变化 {d:+.4f})")

    # --- 消融：关掉归一化防御 ---
    import socialsim.moderation.text_filter as tf
    orig = tf.normalize
    tf.normalize = lambda t: {"raw": t, "nfkc": t, "compact": t, "dedup": t, "homo": t, "pinyin": ""}
    rule_nodef = np.array([rule_expert(t)["score"] for t in obf])
    tf.normalize = orig
    res["obfuscated_no_normalization"] = {"rule_only": metrics(y, rule_nodef)}
    print(f"\n== 混淆攻击 + 关闭归一化防御 ==\n  rule_only    R={res['obfuscated_no_normalization']['rule_only']['recall']:.4f} "
          f"(开启防御时 {res['obfuscated']['rule_only']['recall']:.4f})")

    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "text_bench_cold.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {RESULT_DIR/'text_bench_cold.json'}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("-n", type=int, default=1500)
    main(ap.parse_args().n)
