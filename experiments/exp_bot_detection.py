"""水军检测实验：特征组消融、跨团伙泛化、规避强度曲线。

三个问题：
  Q1 哪类特征真正有用？画像特征最易伪造，去掉后检测是否仍成立？
  Q2 面对训练时没见过的新团伙，检测器还能不能检出？（对应 DFscan 强调的跨算法泛化）
  Q3 水军把规避等级提高，检测率如何下降？攻防的边界在哪？

Q3 是本实验最有价值的部分：只报告一个高 AUC 是没有意义的，
必须画出"拟人度提升 → 检测率下降"的曲线，才能说明检测器的真实能力边界。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from socialsim.config import RESULT_DIR, DB_PATH
from socialsim.db import init_db
from socialsim.simulation import seed as SEED, bot_engine as BE, bot_detector as BD


def rebuild(n_users: int = 80, fresh: bool = True) -> dict:
    """重建沙箱：正常用户基线 + 覆盖 5 策略 × 3 规避等级的水军团伙。"""
    if fresh and DB_PATH.exists():
        DB_PATH.unlink()
    init_db()
    st = SEED.seed_normal(n_users=n_users)
    print(f"基线：{st}")
    from datetime import timedelta
    from sqlalchemy import select
    from socialsim.db import session_scope
    from socialsim.models import Post, User
    with session_scope() as s:
        hot = s.execute(select(Post).where(Post.repost_of == None)
                        .order_by(Post.like_count.desc()).limit(10)).scalars().all()
        hot_ids = [p.id for p in hot]
        uid = s.execute(select(User.id).where(User.is_bot == False)).scalars().first()

    plans = [
        ("控评刷量-ev0", "boost", str(hot_ids[0]), 14, 0),
        ("抹黑攻击-ev0", "smear", str(hot_ids[1]), 10, 0),
        ("话题带节奏-ev0", "hashtag_push", "#支持星光洗面奶", 12, 0),
        ("互粉抱团-ev0", "follow_farm", str(uid), 8, 0),
        ("广告引流-ev0", "spam", "", 6, 0),
        ("控评刷量-ev1", "boost", str(hot_ids[2]), 12, 1),
        ("抹黑攻击-ev1", "smear", str(hot_ids[3]), 10, 1),
        ("话题带节奏-ev1", "hashtag_push", "#飞鹿跑鞋yyds", 10, 1),
        ("控评刷量-ev2", "boost", str(hot_ids[4]), 12, 2),
        ("抹黑攻击-ev2", "smear", str(hot_ids[5]), 10, 2),
        ("话题带节奏-ev2", "hashtag_push", "#云端奶茶必喝", 10, 2),
    ]
    for i, (name, strat, target, n, ev) in enumerate(plans):
        st = BE.run_campaign(name, strat, target, n_bots=n, intensity=1.0,
                             start=SEED.SIM_NOW - timedelta(days=1 + i % 6, hours=2 + i),
                             seed=100 + i, evasion=ev)
        print(f"  {st['group']:26s} 账号{st['bots']:3d} 赞{st['likes']:4d} 评{st['comments']:4d} "
              f"帖{st['posts']:4d} 关{st['follows']:4d}")
    return {"campaigns": len(plans)}


def main(rebuild_db: bool = True, n_users: int = 80):
    if rebuild_db:
        print("== 重建沙箱 ==")
        rebuild(n_users)

    print("\n== 提取特征 ==")
    df = BD.build_features()
    print(f"账号 {len(df)}，其中水军 {int(df.is_bot.sum())}，特征 {len(BD.FEATURES)} 维")

    res = BD.evaluate(df)
    out = {"n_accounts": len(df), "n_bots": int(df.is_bot.sum()), "n_features": len(BD.FEATURES)}

    print("\n== Q0 整体性能（5 折交叉验证）==")
    tbl = {}
    for k in ("RandomForest", "GradientBoosting", "LogisticRegression", "IsolationForest(unsup)", "RuleBaseline"):
        v = res.get(k, {})
        tbl[k] = {kk: round(float(vv), 4) for kk, vv in v.items() if isinstance(vv, (int, float))}
        print(f"  {k:24s} AUC={v.get('auc', float('nan')):.4f} F1={v.get('f1', float('nan')):.4f} "
              f"P={v.get('precision', float('nan')):.4f} R={v.get('recall', float('nan')):.4f}")
    out["models"] = tbl

    print("\n== Q0b 低误报率下的召回（真实风控更关心）==")
    for k in ("RandomForest", "LightGBM", "LogisticRegression"):
        v = res.get(k, {})
        for fk in ("recall@fpr1%", "recall@fpr5%"):
            if v.get(fk):
                r = v[fk]
                print(f"  {k:20s} {fk:14s} 召回={r['recall']:.4f} 实际误报率={r['actual_fpr']:.4f} 阈值={r['threshold']}")
    out["recall_at_low_fpr"] = {k: {fk: res[k][fk] for fk in ("recall@fpr1%", "recall@fpr5%") if fk in res.get(k, {})}
                                for k in ("RandomForest", "LightGBM", "LogisticRegression") if k in res}

    print("\n== Q0c 团伙级检出（团伙内 >50% 账号被标记即视为该团伙被检出）==")
    cl = res.get("campaign_level", {})
    for g, v in cl.items():
        if g.startswith("_"):
            continue
        print(f"  {g:26s} n={v['n']:2d} 标记比例={v['flagged_frac']:.2f} 平均分={v['mean_score']:.3f} "
              f"{'已检出' if v['campaign_detected'] else '漏过'}")
    if "_summary" in cl:
        print(f"  团伙级召回：{cl['_summary']['n_detected']}/{cl['_summary']['n_campaigns']} = "
              f"{cl['_summary']['campaign_recall']:.3f}")
    out["campaign_level"] = cl

    print("\n== Q0d 无监督协同团伙发现（完全不用标签）==")
    uc = res.get("unsupervised_coordination", {})
    if uc:
        print(f"  发现 {uc['n_communities']} 个协同社区，共圈出 {uc['n_accounts_flagged']} 个账号")
        print(f"  纯度={uc['precision']:.3f} 覆盖水军={uc['recall']:.3f} 平均社区纯度={uc['mean_purity']:.3f}")
        for c in uc.get("communities", [])[:6]:
            print(f"    规模{c['size']:3d} 水军{c['n_bots']:3d} 纯度{c['purity']:.2f} "
                  f"主导团伙={c['dominant_campaign'] or '-'}")
    out["unsupervised_coordination"] = uc

    print("\n== Q1 特征组消融 ==")
    out["ablation"] = res["ablation"]
    for k, v in res["ablation"].items():
        print(f"  {k:24s} 维数={v['n_feat']:2d} AUC={v['auc']:.4f} F1={v['f1']:.4f} R={v['recall']:.4f}")

    print("\n== Q2 跨团伙泛化（训练时剔除该团伙）==")
    g1, g2 = res["generalization"], res["generalization_no_profile"]
    out["generalization"] = {"all_features": g1, "no_profile_features": g2}
    for k in sorted(g1):
        if k.startswith("_"):
            continue
        print(f"  {k:26s} n={g1[k]['n']:2d} 全特征检出率={g1[k]['detect_rate@0.5']:.2f} "
              f"去画像={g2[k]['detect_rate@0.5']:.2f} 最低分={g1[k]['min_score']:.3f}")
    print(f"  正常用户误报率：全特征 {g1['_normal_fpr@0.5']:.3f} / 去画像 {g2['_normal_fpr@0.5']:.3f}")

    print("\n== Q3 规避强度 → 检测率 ==")
    prob = np.array(res["RandomForest"]["prob"])
    df = df.copy(); df["score"] = prob
    ev_of = {}
    from sqlalchemy import select
    from socialsim.db import session_scope
    from socialsim.models import BotCampaign
    with session_scope() as s:
        for c in s.execute(select(BotCampaign)).scalars():
            grp = f"C{c.id}-{c.strategy}" + (f"-ev{(c.stats or {}).get('evasion', 0)}"
                                             if (c.stats or {}).get("evasion") else "")
            ev_of[grp] = int((c.stats or {}).get("evasion", 0))
    df["evasion"] = df["bot_group"].map(ev_of).fillna(-1).astype(int)
    curve = {}
    for ev in sorted(x for x in df["evasion"].unique() if x >= 0):
        sub = df[df["evasion"] == ev]
        curve[f"evasion_{ev}"] = {
            "n": len(sub), "mean_score": round(float(sub["score"].mean()), 4),
            "detect_rate@0.5": round(float((sub["score"] >= 0.5).mean()), 4),
            "detect_rate@0.8": round(float((sub["score"] >= 0.8).mean()), 4),
            "min_score": round(float(sub["score"].min()), 4)}
        c = curve[f"evasion_{ev}"]
        print(f"  规避等级 {ev}: n={c['n']:3d} 平均分={c['mean_score']:.3f} "
              f"检出率@0.5={c['detect_rate@0.5']:.3f} @0.8={c['detect_rate@0.8']:.3f} 最低分={c['min_score']:.3f}")
    # 跨规避等级泛化：只用低规避等级训练，测高规避等级
    from sklearn.ensemble import RandomForestClassifier
    hard = {}
    for test_ev in sorted(x for x in df["evasion"].unique() if x > 0):
        tr = df[(df["evasion"] < test_ev) | (df["is_bot"] == 0)]
        te = df[df["evasion"] == test_ev]
        if tr["is_bot"].nunique() < 2 or len(te) == 0:
            continue
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)
        m.fit(tr[BD.FEATURES].values, tr["is_bot"].values)
        p = m.predict_proba(te[BD.FEATURES].values)[:, 1]
        hard[f"train<{test_ev}_test={test_ev}"] = {
            "n": len(te), "detect_rate@0.5": round(float((p >= 0.5).mean()), 4),
            "mean_score": round(float(p.mean()), 4)}
    out["evasion_curve"] = curve
    out["evasion_generalization"] = hard
    if hard:
        print("\n  只用低规避等级训练，检测更高规避等级：")
        for k, v in hard.items():
            print(f"    {k}: n={v['n']} 检出率={v['detect_rate@0.5']:.3f} 平均分={v['mean_score']:.3f}")

    out["feature_importance"] = res["feature_importance"]
    df.to_csv(RESULT_DIR / "bot_features.csv", index=False)
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "bot_detection_metrics.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # 回写风险分供前端展示
    from socialsim.db import session_scope as ss
    from socialsim.models import User
    with ss() as s:
        for uid, p in zip(df["user_id"], df["score"]):
            u = s.get(User, int(uid))
            u.bot_score = float(p); u.bot_flag = bool(p >= 0.5)
    print(f"\n已保存 {RESULT_DIR/'bot_detection_metrics.json'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rebuild", action="store_true")
    ap.add_argument("--users", type=int, default=80)
    a = ap.parse_args()
    main(not a.no_rebuild, a.users)
