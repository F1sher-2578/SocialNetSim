"""生成沙箱基础数据：正常用户及其 30 天的发帖/评论/点赞/关注行为（带昼夜节律与热度偏好）。"""
from __future__ import annotations
import math, random
from datetime import datetime, timedelta
from sqlalchemy import select, func
from ..db import session_scope, init_db
from ..models import User, Post, Comment, Like, Follow, ActionLog
from ..config import SIM_DAYS, RANDOM_SEED
from . import personas, content_gen as cg

SIM_NOW = datetime(2026, 9, 18, 12, 0, 0)

# 昼夜活跃度分布（小时 -> 权重）
DIURNAL = [0.15, 0.08, 0.04, 0.02, 0.02, 0.03, 0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.95, 0.9, 0.6, 0.55, 0.6, 0.7, 0.85, 0.95, 1.0, 1.0, 0.8, 0.4]
NIGHT_OWL = [0.6, 0.5, 0.35, 0.15, 0.05, 0.02, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0, 1.0, 0.9, 0.8]


def draw_time(rng: random.Random, day_offset: int, night_owl: bool = False) -> datetime:
    w = NIGHT_OWL if night_owl else DIURNAL
    hour = rng.choices(range(24), weights=w)[0]
    base = SIM_NOW - timedelta(days=day_offset)
    return base.replace(hour=hour, minute=rng.randint(0, 59), second=rng.randint(0, 59))


def log(s, uid, action, target, t, campaign=""):
    s.add(ActionLog(user_id=uid, action=action, target_id=target, created_at=t, campaign=campaign))


def seed_normal(n_users: int = 60, days: int = SIM_DAYS, seed: int = RANDOM_SEED) -> dict:
    init_db()
    rng = random.Random(seed)
    stats = {"users": 0, "posts": 0, "comments": 0, "likes": 0, "follows": 0}
    with session_scope() as s:
        if s.execute(select(func.count(User.id))).scalar():
            return {"skipped": True}
        users = []
        for i in range(n_users):
            p = personas.normal_persona(rng, i)
            age = rng.choice([rng.randint(1, 6), rng.randint(7, 60), rng.randint(60, 400), rng.randint(60, 400)])   # 含新注册正常用户
            u = User(username=p["username"], nickname=p["nickname"], bio=p["bio"], avatar=p["avatar"],
                     is_bot=False, created_at=SIM_NOW - timedelta(days=age, hours=rng.randint(0, 23)), password="123456")
            s.add(u); s.flush()
            users.append((u, p))
        stats["users"] = len(users)
        # 关注关系：按兴趣相似 + 少量随机，30% 互相关注
        pairs = set()
        for u, p in users:
            k = rng.randint(4, 18)
            cands = [(v, q) for v, q in users if v.id != u.id]
            weights = [1 + 3 * len(set(p["interests"]) & set(q["interests"])) for v, q in cands]
            chosen = set()
            while len(chosen) < min(k, len(cands)):
                v, q = rng.choices(cands, weights=weights)[0]
                chosen.add(v.id)
            for vid in chosen:
                t = u.created_at + timedelta(days=rng.randint(0, max(1, (SIM_NOW - u.created_at).days)))
                if (u.id, vid) not in pairs:
                    pairs.add((u.id, vid))
                    s.add(Follow(follower_id=u.id, followee_id=vid, created_at=min(t, SIM_NOW)))
                    stats["follows"] += 1
                if rng.random() < 0.3 and (vid, u.id) not in pairs:
                    pairs.add((vid, u.id))
                    s.add(Follow(follower_id=vid, followee_id=u.id, created_at=min(t + timedelta(hours=rng.randint(1, 48)), SIM_NOW)))
                    stats["follows"] += 1
        s.flush()
        # 逐日生成行为
        posts: list[Post] = []
        liked = set()
        for day in range(days, -1, -1):
            for u, p in users:
                if (SIM_NOW - timedelta(days=day)) < u.created_at:
                    continue
                n_actions = _poisson(rng, p["activity"] * 2.2)
                for _ in range(n_actions):
                    t = draw_time(rng, day, p["night_owl"])
                    r = rng.random()
                    recent = [x for x in posts if x.created_at < t][-300:]
                    if r < 0.22 or not recent:
                        text, tags = cg.normal_post(rng, p["interests"])
                        post = Post(author_id=u.id, content=text, hashtags=tags, created_at=t, mod_status="pass")
                        s.add(post); s.flush(); posts.append(post); stats["posts"] += 1
                        log(s, u.id, "post", post.id, t)
                    elif r < 0.55:
                        target = _pick_post(rng, recent, t)
                        c = Comment(post_id=target.id, author_id=u.id, content=cg.normal_comment(rng), created_at=t)
                        s.add(c); target.comment_count += 1; stats["comments"] += 1
                        log(s, u.id, "comment", target.id, t)
                    else:
                        target = _pick_post(rng, recent, t)
                        if (target.id, u.id) in liked:
                            continue
                        liked.add((target.id, u.id))
                        s.add(Like(post_id=target.id, user_id=u.id, created_at=t)); target.like_count += 1; stats["likes"] += 1
                        log(s, u.id, "like", target.id, t)
            s.flush()
    return stats


def _poisson(rng: random.Random, lam: float) -> int:
    L, k, p = math.exp(-lam), 0, 1.0
    while p > L:
        k += 1; p *= rng.random()
    return k - 1


def _pick_post(rng: random.Random, recent: list[Post], t: datetime) -> Post:
    """按新鲜度与热度（优先连接）选择互动目标。"""
    ws = []
    for x in recent:
        age_h = max(0.5, (t - x.created_at).total_seconds() / 3600)
        ws.append((1 + x.like_count + 2 * x.comment_count) / (age_h ** 0.8))
    return rng.choices(recent, weights=ws)[0]


if __name__ == "__main__":
    print(seed_normal())
