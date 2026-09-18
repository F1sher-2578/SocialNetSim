"""水军账号检测器：行为特征 + 内容相似度 + 协同图特征 → 监督分类 / 无监督异常 / 协同团伙发现。"""
from __future__ import annotations
import json, math, pickle
from collections import defaultdict, Counter
from datetime import timedelta
import numpy as np
import pandas as pd
import networkx as nx
from sqlalchemy import select
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, average_precision_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from ..db import session_scope
from ..models import User, Post, Comment, Like, Follow, ActionLog
from ..config import MODEL_DIR, RESULT_DIR
from .seed import SIM_NOW

SYNC_WINDOW = timedelta(minutes=15)
# 五类特征组，按"越难伪造越靠后"排列。分组便于消融，验证"去掉易伪造的画像特征后是否仍可检测"。
#   A 画像   最易伪造（换个昵称头像即可）
#   B 行为   中等（需要牺牲效率）
#   C 内容   中等（LLM 改写即可去模板化）
#   D 图结构 较难（需要真实社交关系）
#   E 协同   最难——规避协同检测等于放弃控评效率本身
GROUPS = {
    "profile": ["account_age_days", "nick_digit_ratio", "bio_empty", "default_avatar", "reg_burst"],
    "behavior": ["n_actions", "actions_per_day", "active_days", "n_posts", "n_comments", "n_likes",
                 "hour_entropy", "night_ratio", "iet_cv", "iet_median_min", "burst_ratio",
                 "comment_like_ratio", "target_concentration", "unique_target_ratio",
                 "iet_ks_vs_human", "seq_compress_ratio"],
    "content": ["n_texts", "self_dup", "cross_dup", "len_cv", "emoji_rate", "punct_rate"],
    "graph": ["n_follow_out", "n_followers", "ff_ratio", "reciprocity", "kcore", "clustering"],
    "coordination": ["sync_partners", "sync_ratio", "community_size", "community_new_ratio",
                     "colike_jaccard_max", "cocomment_jaccard_max", "min_dt_median", "xcorr_peak"],
}
FEATURES = [f for g in GROUPS.values() for f in g]


def _entropy(counts):
    tot = sum(counts)
    if tot == 0:
        return 0.0
    ps = [c / tot for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in ps) / math.log2(24)


def build_features(now=SIM_NOW) -> pd.DataFrame:
    with session_scope() as s:
        users = s.execute(select(User)).scalars().all()
        logs = s.execute(select(ActionLog)).scalars().all()
        follows = s.execute(select(Follow)).scalars().all()
        posts = s.execute(select(Post.author_id, Post.content)).all()
        comments = s.execute(select(Comment.author_id, Comment.content)).all()
    uid2logs = defaultdict(list)
    for l in logs:
        uid2logs[l.user_id].append(l)
    out_f, in_f = defaultdict(set), defaultdict(set)
    for f in follows:
        out_f[f.follower_id].add(f.followee_id); in_f[f.followee_id].add(f.follower_id)
    texts = defaultdict(list)
    for uid, c in posts + comments:
        if c and not c.startswith("转发 //"):
            texts[uid].append(c)
    # ---- 内容重复度（TF-IDF 字符 n-gram 余弦）----
    all_texts, owners = [], []
    for uid, ts in texts.items():
        for t in ts:
            all_texts.append(t); owners.append(uid)
    self_dup, cross_dup = defaultdict(float), defaultdict(float)
    if len(all_texts) >= 2:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(all_texts)
        sim = (X @ X.T).toarray()
        np.fill_diagonal(sim, 0)
        owners_arr = np.array(owners)
        for uid in texts:
            idx = np.where(owners_arr == uid)[0]
            if len(idx) == 0:
                continue
            same = np.isin(owners_arr, [uid])
            sub_self = sim[np.ix_(idx, idx)]
            self_dup[uid] = float(sub_self.max(axis=1).mean()) if len(idx) > 1 else 0.0
            sub_cross = sim[idx][:, ~same]
            cross_dup[uid] = float((sub_cross.max(axis=1) > 0.6).mean()) if sub_cross.shape[1] else 0.0
    # ---- 协同同步：同一目标 15 分钟窗口内的共同行为 ----
    by_target = defaultdict(list)
    for l in logs:
        if l.action in ("like", "comment", "repost", "follow") and l.target_id is not None:
            by_target[(l.action, l.target_id)].append((l.created_at, l.user_id))
    sync_pairs = Counter()
    sync_actions = Counter()
    for key, lst in by_target.items():
        lst.sort()
        for i, (t1, u1) in enumerate(lst):
            j = i + 1
            while j < len(lst) and lst[j][0] - t1 <= SYNC_WINDOW:
                u2 = lst[j][1]
                if u1 != u2:
                    sync_pairs[(min(u1, u2), max(u1, u2))] += 1
                    sync_actions[u1] += 1; sync_actions[u2] += 1
                j += 1
    # 共行为集合：共点赞、共评论目标。协同水军必然在目标集合上高度重叠。
    colike, cocomment = defaultdict(set), defaultdict(set)
    for l in logs:
        if l.target_id is None:
            continue
        if l.action == "like":
            colike[l.user_id].add(l.target_id)
        elif l.action == "comment":
            cocomment[l.user_id].add(l.target_id)

    def _jaccard_max(sets: dict, uid: int, min_size: int = 2) -> float:
        a = sets.get(uid, set())
        if len(a) < min_size:
            return 0.0
        best = 0.0
        for v, b in sets.items():
            if v == uid or len(b) < min_size:
                continue
            inter = len(a & b)
            if inter:
                best = max(best, inter / len(a | b))
        return best

    # 同一目标上、与他人动作的最小时间差（协同水军的最小Δt会异常小且集中）
    min_dt = defaultdict(list)
    for key, lst in by_target.items():
        lst.sort()
        for i, (t1, u1) in enumerate(lst):
            best = None
            for j, (t2, u2) in enumerate(lst):
                if u2 == u1:
                    continue
                d = abs((t2 - t1).total_seconds())
                best = d if best is None else min(best, d)
            if best is not None:
                min_dt[u1].append(best)

    # 动作时间序列的互相关峰值：把每个账号的动作离散成分钟级直方图，
    # 与同社区其他账号做归一化互相关，峰值高说明行动节奏被同一个脚本驱动。
    def _hist(uid, bins=240, span=86400 * 7):
        ts = [l.created_at for l in uid2logs.get(uid, []) if l.action != "login"]
        if not ts:
            return None
        h = np.zeros(bins)
        for t in ts:
            k = int((now - t).total_seconds() / span * bins)
            if 0 <= k < bins:
                h[k] += 1
        return h - h.mean() if h.sum() else None

    G = nx.Graph()
    for (a, b), w in sync_pairs.items():
        if w >= 2:
            G.add_edge(a, b, weight=w)
    # 关注关系图（无向化）上的结构特征
    FG = nx.Graph()
    FG.add_nodes_from([u.id for u in users])
    for f in follows:
        FG.add_edge(f.follower_id, f.followee_id)
    FG.remove_edges_from(nx.selfloop_edges(FG))
    try:
        kcore_of = nx.core_number(FG)
    except Exception:
        kcore_of = {}
    try:
        clustering_of = nx.clustering(FG)
    except Exception:
        clustering_of = {}

    comm_of = {}
    if G.number_of_edges():
        for comm in nx.community.louvain_communities(G, weight="weight", seed=0):
            for u in comm:
                comm_of[u] = comm
    # 人类行为的 IET 参考分布：用真实用户（非水军标签无关，这里用全体的稳健分位）做基线
    all_iets = []
    for u in users:
        ls = sorted(uid2logs.get(u.id, []), key=lambda l: l.created_at)
        ts = [l.created_at for l in ls if l.action != "login"]
        all_iets += [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    human_ref = np.log10(np.array([x for x in all_iets if x > 0]) + 1) if all_iets else np.array([1.0])

    # 注册时间聚簇：同一小时内注册的账号数（批量注册的直接证据）
    reg_hour = Counter(u.created_at.replace(minute=0, second=0, microsecond=0) for u in users)

    import gzip as _gzip
    rows = []
    for u in users:
        ls = sorted(uid2logs.get(u.id, []), key=lambda l: l.created_at)
        acts = [l for l in ls if l.action != "login"]
        age = max((now - u.created_at).total_seconds() / 86400, 0.05)
        hours = Counter(l.created_at.hour for l in acts)
        times = [l.created_at for l in acts]
        iets = [(t2 - t1).total_seconds() for t1, t2 in zip(times, times[1:])]
        iet_cv = (np.std(iets) / (np.mean(iets) + 1e-9)) if len(iets) >= 2 else 1.5
        iet_med = (np.median(iets) / 60) if iets else 1e4
        burst = 0.0
        if times:
            for i, t in enumerate(times):
                j = i
                while j < len(times) and times[j] - t <= timedelta(minutes=10):
                    j += 1
                burst = max(burst, (j - i) / len(times))
        tc = Counter(l.target_id for l in acts if l.target_id is not None)
        n_t = sum(tc.values())
        comm = comm_of.get(u.id, set())
        comm_users = [x for x in users if x.id in comm]
        n_c, n_l, n_p = sum(1 for l in acts if l.action == "comment"), sum(1 for l in acts if l.action == "like"), sum(1 for l in acts if l.action in ("post", "repost"))
        # 行为序列的可压缩性：脚本驱动的账号动作序列高度规律，gzip 压缩比明显更高
        seq = "".join({"post": "P", "comment": "C", "like": "L", "follow": "F", "repost": "R"}.get(l.action, "?")
                      for l in acts)
        seq_ratio = (len(_gzip.compress(seq.encode())) / max(1, len(seq))) if len(seq) >= 8 else 1.0
        # IET 分布与人类基线的 KS 距离
        ks = 0.0
        if len(iets) >= 5:
            try:
                from scipy.stats import ks_2samp
                ks = float(ks_2samp(np.log10(np.array([x for x in iets if x > 0]) + 1), human_ref).statistic)
            except Exception:
                ks = 0.0
        txts = texts.get(u.id, [])
        lens = np.array([len(t) for t in txts]) if txts else np.array([0.0])
        emoji_rate = float(np.mean([sum(1 for ch in t if ord(ch) > 0x1F000) / max(1, len(t)) for t in txts])) if txts else 0.0
        punct_rate = float(np.mean([sum(1 for ch in t if ch in "！!？?~。.…") / max(1, len(t)) for t in txts])) if txts else 0.0
        dts = min_dt.get(u.id, [])
        h_u = _hist(u.id)
        xcorr = 0.0
        if h_u is not None and np.std(h_u) > 0:
            comm_ids = [x for x in comm if x != u.id][:12]
            for v in comm_ids:
                h_v = _hist(v)
                if h_v is None or np.std(h_v) == 0:
                    continue
                c_ = np.correlate(h_u, h_v, mode="valid")
                denom = np.linalg.norm(h_u) * np.linalg.norm(h_v)
                if denom > 0:
                    xcorr = max(xcorr, float(np.max(c_) / denom))
        rows.append({
            "user_id": u.id, "username": u.username, "nickname": u.nickname, "is_bot": int(u.is_bot), "bot_group": u.bot_group,
            "account_age_days": age, "n_actions": len(acts), "actions_per_day": len(acts) / min(age, 30),
            "active_days": len({l.created_at.date() for l in acts}),
            "n_posts": n_p, "n_comments": n_c, "n_likes": n_l,
            "n_follow_out": len(out_f[u.id]), "n_followers": len(in_f[u.id]),
            "ff_ratio": len(in_f[u.id]) / (len(out_f[u.id]) + 1),
            "reciprocity": (len(out_f[u.id] & in_f[u.id]) / len(out_f[u.id])) if out_f[u.id] else 0.0,
            "hour_entropy": _entropy([hours.get(h, 0) for h in range(24)]),
            "night_ratio": sum(hours.get(h, 0) for h in range(0, 6)) / max(1, len(acts)),
            "iet_cv": float(iet_cv), "iet_median_min": float(iet_med), "burst_ratio": burst,
            "n_texts": len(texts.get(u.id, [])), "self_dup": self_dup.get(u.id, 0.0), "cross_dup": cross_dup.get(u.id, 0.0),
            "target_concentration": (max(tc.values()) / n_t) if n_t else 0.0,
            "unique_target_ratio": (len(tc) / n_t) if n_t else 1.0,
            "nick_digit_ratio": sum(ch.isdigit() for ch in u.nickname) / max(1, len(u.nickname)),
            "bio_empty": int(not u.bio.strip()), "default_avatar": int(u.avatar in ("🙂", "👤")),
            "reg_burst": reg_hour[u.created_at.replace(minute=0, second=0, microsecond=0)],
            "iet_ks_vs_human": ks, "seq_compress_ratio": seq_ratio,
            "len_cv": float(lens.std() / (lens.mean() + 1e-9)) if len(lens) > 1 else 0.0,
            "emoji_rate": emoji_rate, "punct_rate": punct_rate,
            "kcore": kcore_of.get(u.id, 0), "clustering": clustering_of.get(u.id, 0.0),
            "colike_jaccard_max": _jaccard_max(colike, u.id),
            "cocomment_jaccard_max": _jaccard_max(cocomment, u.id),
            "min_dt_median": float(np.median(dts)) if dts else 1e5,
            "xcorr_peak": xcorr,
            "sync_partners": G.degree(u.id) if u.id in G else 0,
            "sync_ratio": sync_actions.get(u.id, 0) / max(1, len(acts)),
            "community_size": len(comm), "community_new_ratio": (np.mean([(now - x.created_at).days < 7 for x in comm_users]) if comm_users else 0.0),
            "comment_like_ratio": n_c / (n_l + 1),
        })
    df = pd.DataFrame(rows)
    df.attrs["graph"] = G
    return df


def _cv_prob(model, X, y, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]


def ablation(df: pd.DataFrame, seed: int = 0) -> dict:
    """特征组消融：单组、留一组、全部。验证画像特征被伪装后检测是否仍然成立。"""
    y = df["is_bot"].values
    out = {}
    combos = {f"only_{g}": cols for g, cols in GROUPS.items()}
    combos.update({f"drop_{g}": [c for gg, cc in GROUPS.items() if gg != g for c in cc] for g in GROUPS})
    combos["all"] = FEATURES
    combos["behavior+content+graph"] = GROUPS["behavior"] + GROUPS["content"] + GROUPS["graph"]
    for name, cols in combos.items():
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed)
        prob = _cv_prob(m, df[cols].values, y, seed)
        pred = (prob >= 0.5).astype(int)
        out[name] = {"n_feat": len(cols), "auc": roc_auc_score(y, prob), "f1": f1_score(y, pred),
                     "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred)}
    return out


def leave_one_campaign_out(df: pd.DataFrame, seed: int = 0, cols=None) -> dict:
    """跨团伙泛化：训练时剔除某个水军团伙，测试能否检出这个"没见过"的团伙。
    对应 DFscan 强调的跨伪造算法泛化性问题。"""
    cols = cols or FEATURES
    y = df["is_bot"].values
    groups = sorted({g for g in df["bot_group"] if g})
    out = {}
    for g in groups:
        test_mask = (df["bot_group"] == g).values
        train_mask = ~test_mask & ((df["is_bot"] == 0).values | (df["bot_group"] != g).values)
        Xtr, ytr = df.loc[train_mask, cols].values, y[train_mask]
        if len(set(ytr)) < 2:
            continue
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed).fit(Xtr, ytr)
        prob = m.predict_proba(df.loc[test_mask, cols].values)[:, 1]
        out[g] = {"n": int(test_mask.sum()), "detect_rate@0.5": float((prob >= 0.5).mean()),
                  "mean_score": float(prob.mean()), "min_score": float(prob.min())}
    # 正常用户的误报率（用全部团伙训练）
    m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed)
    prob_all = _cv_prob(m, df[cols].values, y, seed)
    out["_normal_fpr@0.5"] = float((prob_all[y == 0] >= 0.5).mean())
    return out


def recall_at_fpr(y, score, target_fpr: float = 0.01) -> dict:
    """低误报率下的召回。真实平台误封成本极高，整体 F1 会掩盖这一点。"""
    y = np.array(y); score = np.array(score)
    neg = np.sort(score[y == 0])
    if len(neg) == 0:
        return {}
    k = max(0, int(np.ceil((1 - target_fpr) * len(neg))) - 1)
    th = float(neg[min(k, len(neg) - 1)])
    pred = score > th
    return {"target_fpr": target_fpr, "threshold": round(th, 4),
            "recall": round(float(pred[y == 1].mean()), 4),
            "actual_fpr": round(float(pred[y == 0].mean()), 4)}


def campaign_level(df: pd.DataFrame, score: np.ndarray, th: float = 0.5,
                   min_frac: float = 0.5) -> dict:
    """团伙级指标：一个团伙中超过 min_frac 的账号被标记，即认为该团伙被检出。

    真实风控关心的是"这个团伙有没有被端掉"，而不是每个账号是否都被标中。
    """
    out = {}
    for g in sorted({x for x in df["bot_group"] if x}):
        m = (df["bot_group"] == g).values
        frac = float((score[m] >= th).mean())
        out[g] = {"n": int(m.sum()), "flagged_frac": round(frac, 4),
                  "campaign_detected": bool(frac >= min_frac),
                  "mean_score": round(float(score[m].mean()), 4)}
    n_det = sum(1 for v in out.values() if v["campaign_detected"])
    out["_summary"] = {"n_campaigns": len(out), "n_detected": n_det,
                       "campaign_recall": round(n_det / max(1, len(out)), 4)}
    return out


def unsupervised_coordination(df: pd.DataFrame, min_weight: int = 3,
                              min_size: int = 3) -> dict:
    """无监督协同团伙发现：完全不使用标签。

    真实平台没有 ground truth，只能靠"行为异常同步"这类无监督信号先圈出可疑群体，
    再交人工复核。这里用共行为相似度网络 + Louvain 社区划分，
    并用真值标签仅作**事后评估**（纯度、召回）。
    """
    G = df.attrs.get("graph")
    if G is None or G.number_of_edges() == 0:
        return {}
    H = nx.Graph((a, b, d) for a, b, d in G.edges(data=True) if d.get("weight", 0) >= min_weight)
    if H.number_of_nodes() == 0:
        return {}
    comms = [c for c in nx.community.louvain_communities(H, weight="weight", seed=0) if len(c) >= min_size]
    truth = dict(zip(df["user_id"], df["is_bot"]))
    groups = dict(zip(df["user_id"], df["bot_group"]))
    detail, tp, total = [], 0, 0
    for c in comms:
        ids = sorted(c)
        bots = sum(truth.get(i, 0) for i in ids)
        gs = Counter(groups.get(i, "") for i in ids if groups.get(i))
        detail.append({"size": len(ids), "n_bots": int(bots),
                       "purity": round(bots / len(ids), 4),
                       "dominant_campaign": gs.most_common(1)[0][0] if gs else "",
                       "campaign_coverage": round(gs.most_common(1)[0][1] / len(ids), 4) if gs else 0.0})
        tp += bots; total += len(ids)
    n_bots_all = int(df["is_bot"].sum())
    return {"n_communities": len(comms), "n_accounts_flagged": total,
            "precision": round(tp / max(1, total), 4),
            "recall": round(tp / max(1, n_bots_all), 4),
            "mean_purity": round(float(np.mean([d["purity"] for d in detail])), 4) if detail else 0.0,
            "communities": sorted(detail, key=lambda d: -d["size"])[:12]}


def evaluate(df: pd.DataFrame, seed: int = 0) -> dict:
    X, y = df[FEATURES].values, df["is_bot"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    }
    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=seed, verbose=-1)
    except Exception:
        pass
    res = {}
    for name, m in models.items():
        prob = cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)
        res[name] = {"auc": roc_auc_score(y, prob), "ap": average_precision_score(y, prob), "f1": f1_score(y, pred),
                     "precision": precision_score(y, pred), "recall": recall_score(y, pred),
                     "confusion": confusion_matrix(y, pred).tolist(),
                     "recall@fpr1%": recall_at_fpr(y, prob, 0.01),
                     "recall@fpr5%": recall_at_fpr(y, prob, 0.05)}
        if name == "RandomForest":
            res[name]["prob"] = prob.tolist()
            fpr, tpr, _ = roc_curve(y, prob); res[name]["roc"] = [fpr.tolist(), tpr.tolist()]
    # 无监督：孤立森林（不使用标签）
    iso = IsolationForest(n_estimators=300, contamination=0.3, random_state=seed).fit(X)
    score = -iso.score_samples(X)
    res["IsolationForest(unsup)"] = {"auc": roc_auc_score(y, score), "ap": average_precision_score(y, score)}
    # 规则基线：协同同步 + 新账号 + 内容重复
    rule = ((df["sync_ratio"] > 0.5) & (df["account_age_days"] < 10)) | (df["cross_dup"] > 0.5) | (df["community_new_ratio"] > 0.8)
    res["RuleBaseline"] = {"f1": f1_score(y, rule.astype(int)), "precision": precision_score(y, rule.astype(int)), "recall": recall_score(y, rule.astype(int))}
    rf = models["RandomForest"].fit(X, y)
    res["feature_importance"] = dict(sorted(zip(FEATURES, rf.feature_importances_.tolist()), key=lambda kv: -kv[1]))
    res["campaign_level"] = campaign_level(df, np.array(res["RandomForest"]["prob"]))
    res["unsupervised_coordination"] = unsupervised_coordination(df)
    res["ablation"] = ablation(df, seed)
    res["generalization"] = leave_one_campaign_out(df, seed)
    res["generalization_no_profile"] = leave_one_campaign_out(
        df, seed, cols=GROUPS["behavior"] + GROUPS["content"] + GROUPS["graph"])
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "bot_detector.pkl", "wb") as f:
        pickle.dump({"model": rf, "features": FEATURES}, f)
    return res


def score_all() -> dict:
    """为全部账号打分（交叉验证概率，避免训练集上的乐观偏差）并写回数据库。"""
    df = build_features()
    res = evaluate(df)
    prob = np.array(res["RandomForest"]["prob"])
    with session_scope() as s:
        for uid, p in zip(df["user_id"], prob):
            u = s.get(User, int(uid)); u.bot_score = float(p); u.bot_flag = bool(p >= 0.5)
    df["bot_score"] = prob
    df.to_csv(RESULT_DIR / "bot_features.csv", index=False)
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("prob", "roc")} if isinstance(v, dict) else v for k, v in res.items()}
    with open(RESULT_DIR / "bot_detection_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return res


def explain(user_id: int) -> dict:
    df = pd.read_csv(RESULT_DIR / "bot_features.csv") if (RESULT_DIR / "bot_features.csv").exists() else build_features()
    row = df[df["user_id"] == user_id]
    if row.empty:
        return {}
    row = row.iloc[0]
    try:
        imp = pickle.load(open(MODEL_DIR / "bot_detector.pkl", "rb"))["model"].feature_importances_
    except Exception:
        imp = np.ones(len(FEATURES))
    evid = []
    for f, w in zip(FEATURES, imp):
        pct = float((df[f] <= row[f]).mean())
        evid.append({"feature": f, "value": float(row[f]), "percentile": pct, "weight": float(w), "score": float(w * abs(pct - 0.5) * 2)})
    evid.sort(key=lambda e: -e["score"])
    return {"user_id": int(user_id), "bot_score": float(row.get("bot_score", 0.0)), "evidence": evid[:8]}
