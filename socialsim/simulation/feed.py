"""信息流排序：没有推荐系统，水军的"刷量控评"就无法转化为曝光，模拟也就失去意义。

提供两种排序：
  hot       —— 热度 + 时间衰减（类似微博热门流），水军刷量可直接提升排名
  following —— 关注流，按时间倒序
并输出"曝光度"指标，用于量化水军行动对内容分发的实际影响。
"""
from __future__ import annotations
import math
import re
from datetime import datetime
from sqlalchemy import select
from ..db import session_scope
from ..models import Post, User, Follow, Like
from .seed import SIM_NOW


def hot_score(post: Post, now: datetime, gravity: float = 1.5) -> float:
    """Hacker News 式热度：(互动量) / (小时数 + 2)^gravity。"""
    engage = post.like_count + 2 * post.comment_count + 3 * post.repost_count
    age_h = max(0.1, (now - post.created_at).total_seconds() / 3600)
    return (engage + 1) / math.pow(age_h + 2, gravity)


def get_feed(mode: str = "hot", user_id: int | None = None, limit: int = 30, offset: int = 0,
             now: datetime | None = None, hide_blocked: bool = True) -> list[dict]:
    now = now or SIM_NOW
    with session_scope() as s:
        q = select(Post).where(Post.repost_of == None)
        if hide_blocked:
            q = q.where(Post.mod_status != "block")
        posts = s.execute(q.order_by(Post.created_at.desc()).limit(1500)).scalars().all()
        if mode == "following" and user_id:
            ids = {f for (f,) in s.execute(select(Follow.followee_id).where(Follow.follower_id == user_id)).all()}
            ids.add(user_id)
            posts = [p for p in posts if p.author_id in ids]
            posts.sort(key=lambda p: p.created_at, reverse=True)
        else:
            posts.sort(key=lambda p: -hot_score(p, now))
        page = posts[offset:offset + limit]
        authors = {u.id: u for u in s.execute(select(User).where(User.id.in_([p.author_id for p in page]))).scalars()}
        out = []
        for rank, p in enumerate(page, offset + 1):
            a = authors.get(p.author_id)
            out.append({
                "id": p.id, "rank": rank, "content": p.content, "hashtags": p.hashtags,
                "created_at": p.created_at.strftime("%Y-%m-%d %H:%M"),
                "likes": p.like_count, "comments": p.comment_count, "reposts": p.repost_count,
                "hot": round(hot_score(p, now), 4),
                "media_type": p.media_type, "media_path": p.media_path,
                "mod_status": p.mod_status, "risk_score": round(p.risk_score or 0, 3),
                "risk_labels": p.risk_labels, "mod_report": p.mod_report,
                "author": {"id": a.id if a else 0, "nickname": a.nickname if a else "?",
                           "username": a.username if a else "", "group": a.bot_group if a else "",
                           "avatar": a.avatar if a else "👤", "is_bot": bool(a.is_bot) if a else False,
                           "bot_score": round(a.bot_score or 0, 3) if a else 0,
                           "bot_flag": bool(a.bot_flag) if a else False},
            })
        return out


TOPIC_RE = re.compile(r"#([^\s#，。！？,.!?]+)")


def trending_topics(top_n: int = 10, window_hours: float = 6.0) -> list[dict]:
    """热搜榜：按话题在任意 6 小时窗口内的讨论人数峰值排序。

    与真实平台的趋势榜一样看"突增"而不是累计总量，这也正是话题带节奏的水军
    在短时间内集中发帖就能冲上榜单的原因。bot_ratio 是真值水军发帖占比，只在审核透视下展示。
    """
    from datetime import timedelta
    win = timedelta(hours=window_hours)
    with session_scope() as s:
        rows = s.execute(select(Post.hashtags, Post.content, Post.author_id, Post.created_at, User.is_bot)
                         .join(User, User.id == Post.author_id)
                         .where(Post.mod_status != "block")).all()
    events: dict[str, list] = {}
    for tags, content, author, at, is_bot in rows:
        for t in set(TOPIC_RE.findall(f"{tags or ''} {content or ''}")):
            events.setdefault(t, []).append((at, author, bool(is_bot)))
    out = []
    for t, ev in events.items():
        ev.sort(key=lambda e: e[0])
        peak, j = 0, 0
        for i in range(len(ev)):
            while ev[i][0] - ev[j][0] > win:
                j += 1
            peak = max(peak, len({a for _, a, _ in ev[j:i + 1]}))
        out.append({"topic": t, "peak_authors": peak, "posts": len(ev),
                    "bot_ratio": round(sum(b for *_, b in ev) / len(ev), 3)})
    out.sort(key=lambda d: (-d["peak_authors"], -d["posts"]))
    return out[:top_n]


def exposure_report(now: datetime | None = None, top_n: int = 30) -> dict:
    """量化水军对信息流的影响：热门榜中被水军推高的内容占比。"""
    now = now or SIM_NOW
    feed = get_feed("hot", limit=top_n, now=now, hide_blocked=False)
    with session_scope() as s:
        bot_ids = {u for (u,) in s.execute(select(User.id).where(User.is_bot == True)).all()}
        stats = []
        for item in feed:
            likes = s.execute(select(Like.user_id).where(Like.post_id == item["id"])).scalars().all()
            bot_likes = sum(1 for u in likes if u in bot_ids)
            stats.append({"post_id": item["id"], "rank": item["rank"], "likes": item["likes"],
                          "bot_likes": bot_likes,
                          "bot_like_ratio": round(bot_likes / max(1, len(likes)), 3),
                          "author_is_bot": item["author"]["is_bot"]})
    inflated = [x for x in stats if x["bot_like_ratio"] >= 0.3]
    return {"top_n": top_n, "n_inflated": len(inflated),
            "inflated_ratio": round(len(inflated) / max(1, len(stats)), 3),
            "bot_authored_in_top": sum(1 for x in stats if x["author_is_bot"]),
            "detail": stats}
