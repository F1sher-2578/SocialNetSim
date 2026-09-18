"""水军账号模拟引擎（仅作用于本地沙箱数据库）。

策略：
  boost        —— 刷量/控评：对目标帖集中点赞、转发、正面评论，并关注作者
  smear        —— 抹黑：对目标帖集中发布负面/辱骂评论（部分含违规内容，供审核模块检测）
  hashtag_push —— 带节奏：批量发布带指定话题的帖子并互相点赞，制造热搜假象
  follow_farm  —— 互粉抱团：批量注册账号互相关注并关注目标用户
  spam         —— 广告/引流：在热门帖下发布引流、诈骗类评论
"""
from __future__ import annotations
import random
from datetime import datetime, timedelta
from sqlalchemy import select, func
from ..db import session_scope
from ..models import User, Post, Comment, Like, Follow, ActionLog, BotCampaign
from . import personas, content_gen as cg
from .seed import SIM_NOW, log

STRATEGIES = ["boost", "smear", "hashtag_push", "follow_farm", "spam"]


# 唯一键去重缓存：SQLAlchemy 的待插入记录在 flush 前查不到，必须在内存里去重
_EDGE_CACHE: dict[tuple[int, str], set] = {}


def _cache(s, kind: str) -> set:
    key = (id(s), kind)
    if key not in _EDGE_CACHE:
        if kind == "follow":
            _EDGE_CACHE[key] = {(a, b) for a, b in s.execute(select(Follow.follower_id, Follow.followee_id)).all()}
        else:
            _EDGE_CACHE[key] = {(a, b) for a, b in s.execute(select(Like.post_id, Like.user_id)).all()}
    return _EDGE_CACHE[key]


def _clear_cache(s):
    for k in [k for k in _EDGE_CACHE if k[0] == id(s)]:
        _EDGE_CACHE.pop(k, None)


def _follow(s, a: int, b: int, t: datetime, group: str, stats: dict) -> bool:
    if a == b:
        return False
    c = _cache(s, "follow")
    if (a, b) in c:
        return False
    c.add((a, b))
    s.add(Follow(follower_id=a, followee_id=b, created_at=t)); stats["follows"] += 1
    log(s, a, "follow", b, t, group)
    return True


def _like(s, post, user_id: int, t: datetime, group: str, stats: dict) -> bool:
    c = _cache(s, "like")
    if (post.id, user_id) in c:
        return False
    c.add((post.id, user_id))
    s.add(Like(post_id=post.id, user_id=user_id, created_at=t)); post.like_count += 1; stats["likes"] += 1
    log(s, user_id, "like", post.id, t, group)
    return True


def _register_bots(s, rng, n, group, reg_time: datetime, mode: str, disguise: int = 0):
    """批量注册：突发模式在同一小时内集中注册；分散模式分布在数天内。"""
    bots = []
    base_idx = (s.execute(select(func.count(User.id)).where(User.is_bot == True)).scalar() or 0)
    for i in range(n):
        p = personas.bot_persona(rng, base_idx + i, group, disguise=disguise)
        if mode == "burst":
            t = reg_time + timedelta(seconds=rng.randint(0, 3600))
        else:
            t = reg_time - timedelta(days=rng.randint(0, 6), hours=rng.randint(0, 23))
        u = User(username=p["username"], nickname=p["nickname"], bio=p["bio"], avatar=p["avatar"],
                 is_bot=True, bot_group=group, created_at=t, password="123456")
        s.add(u); s.flush(); bots.append(u)
        log(s, u.id, "login", None, t, group)
    return bots


def _camouflage(s, rng, bots, reg_time: datetime, start: datetime, group: str, stats: dict, depth: int = 1):
    """养号伪装：在注册到行动之间制造稀疏但“像人”的活动，降低新号/零内容/目标集中等特征的区分度。"""
    from .seed import draw_time
    normal_posts = s.execute(select(Post).where(Post.repost_of == None).order_by(Post.created_at.desc()).limit(400)).scalars().all()
    real_users = s.execute(select(User.id).where(User.is_bot == False)).scalars().all()
    days = max(1, (start - reg_time).days)
    for b in bots:
        n_ev = rng.randint(6, 14) * depth * depth
        for _ in range(n_ev):
            day_off = rng.randint(0, days)
            t = draw_time(rng, day_off, night_owl=False)
            t = t - (SIM_NOW - start)                           # 以 start 为“现在”重新对齐
            if t < reg_time or t > start:
                continue
            r = rng.random()
            if r < 0.4 and normal_posts:
                _like(s, rng.choice(normal_posts), b.id, t, group, stats)
            elif r < 0.65:
                text, tags = cg.normal_post(rng, personas.TOPICS[:5])
                p = Post(author_id=b.id, content=text, hashtags=tags, created_at=t, mod_status="pass")
                s.add(p); s.flush(); stats["posts"] += 1
                log(s, b.id, "post", p.id, t, group)
            elif r < 0.85 and normal_posts:
                p = rng.choice(normal_posts)
                c = Comment(post_id=p.id, author_id=b.id, content=cg.normal_comment(rng), created_at=t)
                s.add(c); p.comment_count += 1; stats["comments"] += 1
                log(s, b.id, "comment", p.id, t, group)
            elif real_users:
                _follow(s, b.id, rng.choice(real_users), t, group, stats)
        # 养号成功的账号会获得少量真实粉丝，使 n_followers / ff_ratio 不再是区分特征
        for r in rng.sample(real_users, k=min(len(real_users), rng.randint(2, 6) * depth)):
            _follow(s, r, b.id, start - timedelta(days=rng.randint(0, max(1, days))), group, stats)
    s.flush()


def _action_times(rng, n, start: datetime, timing: str, intensity: float):
    """生成 n 个动作的时间：突发模式为近似等间隔的密集序列；分散模式为数小时内随机。"""
    if timing == "burst":
        gap = max(15, int(120 / intensity))
        return [start + timedelta(seconds=i * gap + rng.randint(-5, 5)) for i in range(n)]
    span = int(6 * 3600 / intensity)
    return sorted(start + timedelta(seconds=rng.randint(0, span)) for _ in range(n))


def run_campaign(name: str, strategy: str, target: str = "", n_bots: int = 10, intensity: float = 1.0,
                 timing: str = "burst", content_mode: str = "template", start: datetime | None = None,
                 seed: int | None = None, variation: float = 0.5, evasion: int = 0) -> dict:
    """evasion 规避等级（对检测器的对抗）：
       0 = 朴素水军：新注册、集中突发、模板化内容
       1 = 养号+分散：账号提前 30~180 天注册、行动分散在数小时、作息符合昼夜节律
       2 = 深度伪装：在 1 的基础上混入正常“伪装行为”（随机点赞/正常发帖/关注真人）、内容强改写、降低目标集中度
    """
    assert strategy in STRATEGIES, strategy
    if evasion >= 1:
        timing = "spread"
        variation = max(variation, 0.8)
    rng = random.Random(seed if seed is not None else random.randint(0, 10**6))
    start = start or SIM_NOW - timedelta(hours=rng.randint(2, 48))
    stats = {"bots": 0, "likes": 0, "comments": 0, "posts": 0, "follows": 0, "reposts": 0, "flagged_text": 0}
    with session_scope() as s:
        camp = BotCampaign(name=name, strategy=strategy, target=target, n_bots=n_bots, intensity=intensity,
                           timing=timing, content_mode=content_mode, status="running",
                           stats={"evasion": evasion})
        s.add(camp); s.flush()
        group = f"C{camp.id}-{strategy}" + (f"-ev{evasion}" if evasion else "")
        # 注册时间：突发模式在行动前 1~3 天集中注册；规避模式提前 30~180 天“养号”
        if evasion >= 1:
            reg_time = start - timedelta(days=rng.randint(30, 180), hours=rng.randint(0, 23))
        else:
            reg_time = start - timedelta(days=rng.randint(1, 3), hours=rng.randint(0, 5))
        bots = _register_bots(s, rng, n_bots, group, reg_time, "burst" if timing == "burst" else "spread", disguise=evasion)
        stats["bots"] = len(bots)
        if evasion >= 1:
            _camouflage(s, rng, bots, reg_time, start, group, stats, depth=evasion)

        target_post = None
        if strategy in ("boost", "smear"):
            target_post = s.get(Post, int(target)) if str(target).isdigit() else None
            if target_post is None:   # 默认选最热的帖子
                target_post = s.execute(select(Post).order_by(Post.like_count.desc())).scalars().first()
        if strategy in ("boost", "smear"):
            times = _action_times(rng, len(bots) * 3, start, timing, intensity)
            ti = 0
            for b in bots:
                # 点赞
                t = times[ti]; ti += 1
                _like(s, target_post, b.id, t, group, stats)
                # 评论
                if rng.random() < 0.85:
                    t = times[ti]; ti += 1
                    text = _gen_comment(strategy, rng, content_mode, variation)
                    c = Comment(post_id=target_post.id, author_id=b.id, content=text, created_at=t)
                    s.add(c); target_post.comment_count += 1; stats["comments"] += 1
                    log(s, b.id, "comment", target_post.id, t, group)
                else:
                    ti += 1
                # 关注作者 / 转发
                t = times[ti]; ti += 1
                if strategy == "boost":
                    if rng.random() < 0.7:
                        _follow(s, b.id, target_post.author_id, t, group, stats)
                    if rng.random() < 0.5:
                        rp = Post(author_id=b.id, content="转发 // " + target_post.content[:60], repost_of=target_post.id,
                                  created_at=t, mod_status="pass")
                        s.add(rp); target_post.repost_count += 1; stats["reposts"] += 1
                        log(s, b.id, "repost", target_post.id, t, group)
        elif strategy == "hashtag_push":
            tag = target if target.startswith("#") else f"#{target or '热门话题'}"
            times = _action_times(rng, len(bots) * 4, start, timing, intensity)
            ti = 0
            made = []
            for b in bots:
                t = times[ti]; ti += 1
                text, tags = cg.bot_post("hashtag_push", rng, tag, variation)
                p = Post(author_id=b.id, content=text, hashtags=tags, created_at=t, mod_status="pass")
                s.add(p); s.flush(); made.append(p); stats["posts"] += 1
                log(s, b.id, "post", p.id, t, group)
            for b in bots:   # 互相点赞
                for p in rng.sample(made, k=min(3, len(made))):
                    if p.author_id == b.id:
                        continue
                    t = times[ti % len(times)]; ti += 1
                    _like(s, p, b.id, t, group, stats)
        elif strategy == "follow_farm":
            tgt_user = None
            if str(target).isdigit():
                tgt_user = s.get(User, int(target))
            times = _action_times(rng, len(bots) * (len(bots) - 1) // 2 + len(bots), start, timing, intensity)
            ti = 0
            for i, b in enumerate(bots):
                for c in bots[i + 1:]:
                    t = times[ti % len(times)]; ti += 1
                    _follow(s, b.id, c.id, t, group, stats)
                    _follow(s, c.id, b.id, t + timedelta(seconds=rng.randint(1, 30)), group, stats)
                if tgt_user is not None:
                    t = times[ti % len(times)]; ti += 1
                    _follow(s, b.id, tgt_user.id, t, group, stats)
        elif strategy == "spam":
            hot = s.execute(select(Post).where(Post.repost_of == None).order_by(Post.like_count.desc()).limit(15)).scalars().all()
            times = _action_times(rng, len(bots) * 3, start, timing, intensity)
            ti = 0
            for b in bots:
                for p in rng.sample(hot, k=min(3, len(hot))):
                    t = times[ti % len(times)]; ti += 1
                    text = cg.bot_comment("spam", rng, variation)
                    c = Comment(post_id=p.id, author_id=b.id, content=text, created_at=t)
                    s.add(c); p.comment_count += 1; stats["comments"] += 1
                    log(s, b.id, "comment", p.id, t, group)
        camp.status = "done"; camp.stats = dict(stats, evasion=evasion)
        s.flush()
        _clear_cache(s)
        stats["campaign_id"] = camp.id
        stats["group"] = group
    return stats


def _gen_comment(strategy: str, rng, content_mode: str, variation: float) -> str:
    if content_mode == "llm" and cg.llm_available():
        goal = "正面夸赞并鼓励购买" if strategy == "boost" else "负面质疑"
        out = cg.llm_generate(f"请生成一条社交媒体评论，20字以内，风格口语化，目的：{goal}。只输出评论。")
        if out:
            return out
    return cg.bot_comment(strategy, rng, variation)


def default_campaigns(seed: int = 7, with_evasive: bool = True) -> list[dict]:
    """一键生成 4 个典型水军团伙（用于实验与演示）。"""
    rng = random.Random(seed)
    out = []
    with session_scope() as s:
        hot = s.execute(select(Post).where(Post.repost_of == None).order_by(Post.like_count.desc()).limit(6)).scalars().all()
        hot_ids = [p.id for p in hot]
        some_user = s.execute(select(User).where(User.is_bot == False)).scalars().first().id
    out.append(run_campaign("控评刷量-A", "boost", str(hot_ids[0]), n_bots=14, intensity=1.5, timing="burst",
                            start=SIM_NOW - timedelta(days=2, hours=3), seed=seed, variation=0.4))
    out.append(run_campaign("抹黑攻击-B", "smear", str(hot_ids[1]), n_bots=10, intensity=1.0, timing="burst",
                            start=SIM_NOW - timedelta(days=1, hours=1), seed=seed + 1, variation=0.5))
    out.append(run_campaign("话题带节奏-C", "hashtag_push", "#支持星光洗面奶", n_bots=12, intensity=1.2, timing="spread",
                            start=SIM_NOW - timedelta(days=4, hours=6), seed=seed + 2, variation=0.6))
    out.append(run_campaign("互粉抱团-D", "follow_farm", str(some_user), n_bots=8, intensity=1.0, timing="burst",
                            start=SIM_NOW - timedelta(days=6, hours=2), seed=seed + 3))
    out.append(run_campaign("广告引流-E", "spam", "", n_bots=6, intensity=0.8, timing="spread",
                            start=SIM_NOW - timedelta(days=3, hours=5), seed=seed + 4))
    if with_evasive:
        out.append(run_campaign("养号控评-F(规避1)", "boost", str(hot_ids[2]), n_bots=12, intensity=1.0,
                                start=SIM_NOW - timedelta(days=1, hours=8), seed=seed + 5, evasion=1))
        out.append(run_campaign("深度伪装抹黑-G(规避2)", "smear", str(hot_ids[3]), n_bots=12, intensity=1.0,
                                start=SIM_NOW - timedelta(days=2, hours=4), seed=seed + 6, evasion=2))
        out.append(run_campaign("伪装带节奏-H(规避2)", "hashtag_push", "#飞鹿跑鞋yyds", n_bots=10, intensity=1.0,
                                start=SIM_NOW - timedelta(days=3, hours=2), seed=seed + 7, evasion=2))
    return out
