"""关系网络视图：把关注图与协同行为图合成一张供前端绘制的图。

两张图共用一套坐标（在两者的并集上做力导向布局），前端切换边类型时节点不动，
便于对照"关注关系里看不出来、协同行为才暴露"的团伙：
默认沙箱里控评、抹黑、带节奏、广告四类团伙有 32 个账号没有任何关注边，
它们的协同只体现在同时去点赞、评论同一个目标。
"""
from __future__ import annotations
import math

import networkx as nx
from sqlalchemy import select, func

from ..db import session_scope
from ..models import User, Follow, ActionLog
from . import bot_detector as BD

_CACHE: dict = {}


def _signature() -> tuple:
    with session_scope() as s:
        return (s.execute(select(func.count(User.id))).scalar(),
                s.execute(select(func.count(Follow.id))).scalar(),
                s.execute(select(func.count(ActionLog.id))).scalar())


def _structure() -> dict:
    """布局、边与社区只依赖行为数据，按数据签名缓存；检测分每次请求时重新读取。"""
    df = BD.build_features()
    sync = df.attrs["graph"]            # 同一目标 15 分钟窗口内共同行为 ≥2 次的账号对
    with session_scope() as s:
        follows = s.execute(select(Follow.follower_id, Follow.followee_id)).all()
    follow_edges = sorted({(min(a, b), max(a, b)) for a, b in follows if a != b})

    union = nx.Graph()
    union.add_edges_from(follow_edges, weight=1.0)
    for a, b, d in sync.edges(data=True):
        # 协同边权重更大，让同一团伙在布局里聚得更紧
        w = 1.0 + d["weight"]
        union.add_edge(a, b, weight=union[a][b]["weight"] + w if union.has_edge(a, b) else w)
    # 逐个连通块布局：整图一起算时，斥力会把几个小连通块推得很远，主图被压成一团
    pos = {}
    comps = sorted(nx.connected_components(union), key=len, reverse=True)
    if comps:
        main = union.subgraph(comps[0])
        pos.update(nx.spring_layout(main, seed=7, weight="weight", iterations=300,
                                    k=1.8 / math.sqrt(max(1, main.number_of_nodes()))))
        for i, c in enumerate(comps[1:]):          # 其余小连通块排在主图下方
            sub = nx.spring_layout(union.subgraph(c), seed=7, scale=0.1)
            pos.update({n: (-0.9 + 0.32 * i + x, -1.25 + y) for n, (x, y) in sub.items()})

    # 与 bot_detector.unsupervised_coordination 相同的参数，社区编号按规模从大到小
    strong = nx.Graph((a, b, d) for a, b, d in sync.edges(data=True) if d.get("weight", 0) >= 3)
    comms = []
    if strong.number_of_edges():
        comms = [c for c in nx.community.louvain_communities(strong, weight="weight", seed=0) if len(c) >= 3]
    comms.sort(key=len, reverse=True)
    community_of = {n: i + 1 for i, c in enumerate(comms) for n in c}

    # 没有任何关注/协同边的账号放在主图右侧的网格里，水军在上、正常用户在下
    isolated = [int(u) for u in df["user_id"] if int(u) not in pos]
    is_bot = dict(zip(df["user_id"].astype(int), df["is_bot"].astype(int)))
    isolated.sort(key=lambda u: (-is_bot.get(u, 0), u))
    cols = max(1, math.ceil(math.sqrt(len(isolated) / 2))) if isolated else 1
    for i, u in enumerate(isolated):
        pos[u] = (1.35 + 0.16 * (i % cols), -0.95 + 0.16 * (i // cols))

    return {
        "pos": {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()},
        "isolated": set(isolated),
        "follow": [[int(a), int(b)] for a, b in follow_edges],
        "sync": [[int(a), int(b), int(d["weight"])] for a, b, d in sync.edges(data=True)],
        "community_of": community_of,
        "n_communities": len(comms),
    }


def graph_view() -> dict:
    sig = _signature()
    if _CACHE.get("sig") != sig:
        _CACHE.update(sig=sig, data=_structure())
    st = _CACHE["data"]
    with session_scope() as s:
        users = s.execute(select(User)).scalars().all()
        fo = dict(s.execute(select(Follow.follower_id, func.count(Follow.id)).group_by(Follow.follower_id)).all())
        fa = dict(s.execute(select(Follow.followee_id, func.count(Follow.id)).group_by(Follow.followee_id)).all())
        nodes = []
        for u in users:
            x, y = st["pos"].get(u.id, (0.0, 0.0))
            nodes.append({"id": u.id, "name": u.nickname, "avatar": u.avatar, "is_bot": bool(u.is_bot),
                          "group": u.bot_group or "", "community": st["community_of"].get(u.id, 0),
                          "bot_score": round(u.bot_score or 0, 4), "bot_flag": bool(u.bot_flag),
                          "following": int(fo.get(u.id, 0)), "followers": int(fa.get(u.id, 0)),
                          "isolated": u.id in st["isolated"], "x": round(x, 4), "y": round(y, 4)})
    communities = []
    for cid in range(1, st["n_communities"] + 1):
        members = [n for n in nodes if n["community"] == cid]
        groups = [n["group"] for n in members if n["group"]]
        top = max(set(groups), key=groups.count) if groups else ""
        communities.append({"id": cid, "size": len(members), "n_bots": sum(n["is_bot"] for n in members),
                            "dominant_group": top})
    return {"nodes": nodes, "edges": {"follow": st["follow"], "sync": st["sync"]}, "communities": communities,
            "stats": {"n_nodes": len(nodes), "n_follow": len(st["follow"]), "n_sync": len(st["sync"]),
                      "n_isolated": len(st["isolated"]),
                      "n_isolated_bots": sum(1 for n in nodes if n["isolated"] and n["is_bot"])}}
