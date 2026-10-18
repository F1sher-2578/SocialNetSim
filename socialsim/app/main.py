"""SocialNetSim Web 应用：仿真社交平台 + 水军控制台 + 审核后台 + 安全实验台。

仅在本地沙箱运行（默认绑定 127.0.0.1），不与任何真实平台交互。
前端是 static/ 下的 Vue 3 + Element Plus + ECharts，第三方文件全部本地托管，断网可用。
"""
from __future__ import annotations
import json, mimetypes, shutil, time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, func, desc

from ..config import UPLOAD_DIR, RESULT_DIR, DATA_DIR, VIDEO_DIR, RISK_BLOCK, RISK_REVIEW
from ..db import session_scope, init_db
from ..models import User, Post, Comment, Like, Follow, BotCampaign, ModerationRecord, ChatMessage
from ..simulation import bot_engine, bot_detector
from ..simulation.feed import get_feed, exposure_report, trending_topics
from ..simulation.seed import SIM_NOW

# Windows 上 mimetypes 会读注册表，.js 可能被识别成 text/plain，浏览器会拒绝执行 ES 模块
mimetypes.add_type("text/javascript", ".js")

app = FastAPI(title="SocialNetSim", description="社交网络拟真沙箱", version="2.0")
app.add_middleware(GZipMiddleware, minimum_size=1000)
STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/reports", StaticFiles(directory=str(RESULT_DIR)), name="reports")
# 示例素材预览：只暴露图片与视频两个子目录，不暴露数据库
IMAGE_DIR = DATA_DIR / "images"
for _d in (IMAGE_DIR, VIDEO_DIR):
    _d.mkdir(parents=True, exist_ok=True)
app.mount("/media/images", StaticFiles(directory=str(IMAGE_DIR)), name="media_images")
app.mount("/media/videos", StaticFiles(directory=str(VIDEO_DIR)), name="media_videos")

TASKS: dict[str, dict] = {}


@app.middleware("http")
async def _revalidate_own_assets(request: Request, call_next):
    # 业务代码改动后浏览器要立刻拿到新版本（带 ETag，未改动时只回 304）；vendor 目录带版本号，可以长期缓存
    resp = await call_next(request)
    path = request.url.path
    if path == "/" or (path.startswith("/static/") and not path.startswith("/static/vendor/")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.on_event("startup")
def _startup():
    init_db()


def _read_result(name: str):
    p = RESULT_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _default_user_id(s) -> int | None:
    """沙箱里没有登录，"我"固定为第一个正常用户（发帖、关注流都以他的身份）。"""
    return s.execute(select(User.id).where(User.is_bot == False).order_by(User.id)).scalars().first()


# ---------------------------------------------------------------- 页面
@app.get("/", response_class=HTMLResponse)
def index():
    p = Path(__file__).parent / "templates" / "index.html"
    return p.read_text(encoding="utf-8")


@app.get("/api/meta")
def api_meta():
    with session_scope() as s:
        me = s.get(User, _default_user_id(s))
        me_out = {"id": me.id, "nickname": me.nickname, "username": me.username, "avatar": me.avatar,
                  "bio": me.bio} if me else None
    return {"sim_now": SIM_NOW.strftime("%Y-%m-%d %H:%M"), "me": me_out,
            "thresholds": {"review": RISK_REVIEW, "block": RISK_BLOCK}}


# ---------------------------------------------------------------- 社交基本功能
@app.get("/api/feed")
def api_feed(mode: str = "hot", user_id: int | None = None, limit: int = 25, offset: int = 0,
             hide_blocked: bool = True):
    return {"items": get_feed(mode, user_id, limit, offset, hide_blocked=hide_blocked)}


@app.get("/api/trending")
def api_trending(top_n: int = 10):
    return {"topics": trending_topics(top_n)}


def _user_counts(s) -> tuple[dict, dict, dict]:
    posts = dict(s.execute(select(Post.author_id, func.count(Post.id)).group_by(Post.author_id)).all())
    fo = dict(s.execute(select(Follow.follower_id, func.count(Follow.id)).group_by(Follow.follower_id)).all())
    fa = dict(s.execute(select(Follow.followee_id, func.count(Follow.id)).group_by(Follow.followee_id)).all())
    return posts, fo, fa


def _user_dict(u: User, posts: dict, fo: dict, fa: dict) -> dict:
    return {"id": u.id, "username": u.username, "nickname": u.nickname, "avatar": u.avatar,
            "bio": u.bio, "is_bot": bool(u.is_bot), "bot_group": u.bot_group,
            "bot_score": round(u.bot_score or 0, 4), "bot_flag": bool(u.bot_flag),
            "created_at": u.created_at.strftime("%Y-%m-%d"), "posts": int(posts.get(u.id, 0)),
            "following": int(fo.get(u.id, 0)), "followers": int(fa.get(u.id, 0))}


@app.get("/api/users")
def api_users(limit: int = 50, only_bots: bool = False, sort: str = "bot_score"):
    with session_scope() as s:
        q = select(User)
        if only_bots:
            q = q.where(User.is_bot == True)
        col = {"bot_score": User.bot_score, "created_at": User.created_at, "id": User.id}.get(sort, User.bot_score)
        users = s.execute(q.order_by(desc(col)).limit(limit)).scalars().all()
        counts = _user_counts(s)
        return {"users": [_user_dict(u, *counts) for u in users]}


@app.get("/api/user/{uid}")
def api_user(uid: int):
    """账号画像抽屉：基本信息 + 最近发帖 + 判定依据。"""
    with session_scope() as s:
        u = s.get(User, uid)
        if not u:
            raise HTTPException(404)
        user = _user_dict(u, *_user_counts(s))
        ps = s.execute(select(Post).where(Post.author_id == uid).order_by(Post.created_at.desc()).limit(5)).scalars().all()
        posts = [{"id": p.id, "content": p.content, "created_at": p.created_at.strftime("%Y-%m-%d %H:%M"),
                  "likes": p.like_count, "comments": p.comment_count, "mod_status": p.mod_status} for p in ps]
    try:
        explain = bot_detector.explain(uid)
    except Exception:
        explain = {}
    return {"user": user, "posts": posts, "explain": explain}


@app.get("/api/post/{pid}")
def api_post(pid: int):
    with session_scope() as s:
        p = s.get(Post, pid)
        if not p:
            raise HTTPException(404)
        cs = s.execute(select(Comment).where(Comment.post_id == pid).order_by(Comment.created_at)).scalars().all()
        authors = {u.id: u for u in s.execute(select(User).where(
            User.id.in_([c.author_id for c in cs] + [p.author_id]))).scalars()}

        def author(uid):
            a = authors.get(uid)
            if not a:
                return {"id": uid, "nickname": "?", "avatar": "👤", "is_bot": False, "bot_score": 0, "group": ""}
            return {"id": a.id, "nickname": a.nickname, "username": a.username, "avatar": a.avatar,
                    "is_bot": bool(a.is_bot), "bot_score": round(a.bot_score or 0, 3),
                    "bot_flag": bool(a.bot_flag), "group": a.bot_group or ""}

        return {
            "post": {"id": p.id, "content": p.content, "created_at": p.created_at.strftime("%Y-%m-%d %H:%M"),
                     "likes": p.like_count, "comments": p.comment_count, "reposts": p.repost_count,
                     "media_type": p.media_type, "media_path": p.media_path,
                     "mod_status": p.mod_status, "risk_score": round(p.risk_score or 0, 3),
                     "risk_labels": p.risk_labels, "mod_report": p.mod_report, "author": author(p.author_id)},
            "comments": [{"id": c.id, "content": c.content, "created_at": c.created_at.strftime("%m-%d %H:%M"),
                          "mod_status": c.mod_status, "risk_score": round(c.risk_score or 0, 3),
                          "risk_labels": c.risk_labels, "author": author(c.author_id)} for c in cs],
        }


class PostIn(BaseModel):
    content: str
    user_id: int | None = None
    moderate: bool = True


@app.post("/api/post")
def api_create_post(body: PostIn):
    from ..moderation.pipeline import moderate_post
    with session_scope() as s:
        uid = body.user_id or _default_user_id(s)
        p = Post(author_id=uid, content=body.content, created_at=datetime.now(), mod_status="pending")
        s.add(p); s.flush()
        pid = p.id
    result = moderate_post(pid) if body.moderate else {}
    text = (result.get("parts") or {}).get("text") or {}
    return {"post_id": pid, "moderation": {k: v for k, v in result.items() if k != "parts"},
            "experts": text.get("experts", {})}


class CommentIn(BaseModel):
    post_id: int
    content: str
    user_id: int | None = None


@app.post("/api/comment")
def api_create_comment(body: CommentIn):
    from ..moderation.pipeline import moderate_comment
    with session_scope() as s:
        uid = body.user_id or _default_user_id(s)
        c = Comment(post_id=body.post_id, author_id=uid, content=body.content, created_at=datetime.now())
        s.add(c); s.flush()
        cid = c.id
        p = s.get(Post, body.post_id)
        if p:
            p.comment_count += 1
    return {"comment_id": cid, "moderation": moderate_comment(cid)}


# ---------------------------------------------------------------- 水军控制台
class CampaignIn(BaseModel):
    name: str = "自定义任务"
    strategy: str = "boost"
    target: str = ""
    n_bots: int = 10
    intensity: float = 1.0
    timing: str = "burst"
    content_mode: str = "template"
    variation: float = 0.5
    evasion: int = 0


@app.post("/api/campaign")
def api_campaign(body: CampaignIn):
    stats = bot_engine.run_campaign(body.name, body.strategy, body.target, body.n_bots, body.intensity,
                                    body.timing, body.content_mode, variation=body.variation,
                                    evasion=body.evasion)
    return {"stats": stats}


@app.get("/api/campaigns")
def api_campaigns():
    with session_scope() as s:
        cs = s.execute(select(BotCampaign).order_by(BotCampaign.id.desc())).scalars().all()
        return {"campaigns": [{"id": c.id, "name": c.name, "strategy": c.strategy, "target": c.target,
                               "n_bots": c.n_bots, "timing": c.timing, "intensity": c.intensity,
                               "status": c.status, "stats": c.stats or {},
                               "created_at": c.created_at.strftime("%Y-%m-%d %H:%M")} for c in cs]}


@app.post("/api/detect_bots")
def api_detect_bots():
    res = bot_detector.score_all()
    return {"metrics": {k: {kk: vv for kk, vv in v.items() if kk != "prob"}
                        if isinstance(v, dict) else v for k, v in res.items()}}


MODEL_KEYS = ("RandomForest", "GradientBoosting", "LogisticRegression", "LightGBM",
              "IsolationForest(unsup)", "RuleBaseline")


def _normalize_bot_metrics(m: dict | None) -> dict:
    """两种写法统一成一种：实验脚本把模型放在 models 下、泛化拆成两份；网页评估是平铺的。"""
    if not m:
        return {}
    out = {k: v for k, v in m.items()
           if k not in MODEL_KEYS + ("models", "generalization", "generalization_no_profile", "recall_at_low_fpr")}
    models = {k: dict(v) for k, v in (m.get("models") or {}).items()}
    for k in MODEL_KEYS:
        if isinstance(m.get(k), dict):
            models[k] = dict(m[k])
    for k, v in (m.get("recall_at_low_fpr") or {}).items():
        models.setdefault(k, {}).update(v)
    roc = dict(m.get("roc") or {})
    for k, v in models.items():
        if "roc" in v:
            roc[k] = v.pop("roc")
    g = m.get("generalization") or {}
    out["models"], out["roc"] = models, roc
    out["generalization"] = g if "all_features" in g else {
        "all_features": g, "no_profile_features": m.get("generalization_no_profile") or {}}
    return out


@app.get("/api/bot_metrics")
def api_bot_metrics():
    """离线实验结果为底，网页上「重新评估」更新的部分覆盖在上面；规避等级相关指标只来自离线实验。"""
    exp_p, live_p = RESULT_DIR / "bot_detection_metrics.json", RESULT_DIR / "bot_detection_live.json"
    exp = _normalize_bot_metrics(_read_result(exp_p.name))
    live = _normalize_bot_metrics(_read_result(live_p.name))
    use_live = bool(live) and (not exp or live_p.stat().st_mtime >= exp_p.stat().st_mtime)
    out = dict(exp)
    if use_live:
        out.update({k: v for k, v in live.items() if v})
    stamp = lambda p: datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if p.exists() else None
    out["source"] = "live" if use_live else ("experiment" if exp else "none")
    out["experiment_at"], out["live_at"] = stamp(exp_p), stamp(live_p)
    return out


@app.get("/api/bot_explain/{uid}")
def api_bot_explain(uid: int):
    return bot_detector.explain(uid)


@app.get("/api/graph")
def api_graph():
    from ..simulation.graph_view import graph_view
    return graph_view()


@app.get("/api/exposure")
def api_exposure(top_n: int = 30):
    return exposure_report(top_n=top_n)


# ---------------------------------------------------------------- 审核后台
@app.post("/api/moderate/text")
def api_mod_text(body: dict):
    from ..moderation.text_filter import analyze_text
    return analyze_text(body.get("text", ""))


@app.post("/api/moderate/upload")
async def api_mod_upload(background: BackgroundTasks, file: UploadFile = File(...), kind: str = Form("auto")):
    """上传图片/视频并异步审核，返回 task_id 供轮询。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch for ch in Path(file.filename or "upload").name if ch.isalnum() or ch in "._-")
    dest = UPLOAD_DIR / f"{int(time.time())}_{safe}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    task_id = f"t{int(time.time()*1000)}"
    TASKS[task_id] = {"status": "running", "file": str(dest), "started": time.time()}
    is_video = dest.suffix.lower() in VIDEO_EXT
    background.add_task(_run_moderation, task_id, str(dest), is_video if kind == "auto" else kind == "video")
    return {"task_id": task_id, "file": dest.name, "is_video": is_video}


def _run_moderation(task_id: str, path: str, is_video: bool):
    try:
        if is_video:
            from ..moderation.video_pipeline import analyze_video
            r = analyze_video(path, out_dir=RESULT_DIR / "reports" / Path(path).stem)
            r["report_url"] = f"/reports/reports/{Path(path).stem}/report.html"
            r.pop("frames", None)
        else:
            from ..moderation.image_experts import analyze_image
            r = analyze_image(path)
        with session_scope() as s:
            s.add(ModerationRecord(target_type="upload", target_id=0,
                                   media_type="video" if is_video else "image",
                                   risk_score=float(r.get("score", 0)),
                                   verdict=r.get("verdict") or ("block" if r.get("score", 0) >= RISK_BLOCK
                                                                else "review" if r.get("score", 0) >= RISK_REVIEW else "pass"),
                                   labels=",".join(r.get("labels", [])),
                                   report_path=r.get("report_path", ""),
                                   elapsed=r.get("elapsed", 0)))
        TASKS[task_id].update({"status": "done", "result": r, "elapsed": time.time() - TASKS[task_id]["started"]})
    except Exception as e:
        import traceback
        TASKS[task_id].update({"status": "error", "error": f"{type(e).__name__}: {e}",
                               "trace": traceback.format_exc()[-1500:]})


VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@app.get("/api/samples/media")
def api_sample_media():
    """本地已有的示例素材（自建中文切片、XD-Violence、风险类目图片），演示时不用再找文件。"""
    def ls(root: Path, sub: str, exts: tuple, limit: int):
        d = root / sub
        if not d.exists():
            return []
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in exts)[:limit]
        return [{"name": f.name, "path": f"{root.name}/{sub}/{f.name}", "size": f.stat().st_size} for f in files]
    labels = {}
    lp = VIDEO_DIR / "bench" / "labels.json"
    if lp.exists():
        try:
            labels = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            pass
    own = ls(VIDEO_DIR, "bench", VIDEO_EXT, 50)
    for v in own:
        v["truth"] = labels.get(v["name"], {})
    return {"own_videos": own, "xd_videos": ls(VIDEO_DIR, "xdviolence", VIDEO_EXT, 200),
            "images": ls(IMAGE_DIR, "bench", IMAGE_EXT, 200)}


class SampleIn(BaseModel):
    path: str      # 形如 videos/bench/gambling_short.mp4，必须落在 data/images 或 data/videos 之下


@app.post("/api/moderate/sample")
def api_mod_sample(body: SampleIn, background: BackgroundTasks):
    p = (DATA_DIR / body.path).resolve()
    if not any(p.is_relative_to(d.resolve()) for d in (IMAGE_DIR, VIDEO_DIR)) or not p.is_file():
        raise HTTPException(400, "只能选择 data/images 或 data/videos 下的示例素材")
    is_video = p.suffix.lower() in VIDEO_EXT
    # 复制一份到上传目录，报告目录按"时间戳_文件名"区分，与网页上传走同一条流水线
    dest = UPLOAD_DIR / f"{int(time.time())}_{p.name}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dest)
    task_id = f"t{int(time.time()*1000)}"
    TASKS[task_id] = {"status": "running", "file": str(dest), "started": time.time()}
    background.add_task(_run_moderation, task_id, str(dest), is_video)
    return {"task_id": task_id, "file": dest.name, "is_video": is_video}


@app.get("/api/task/{task_id}")
def api_task(task_id: str):
    t = TASKS.get(task_id)
    if not t:
        raise HTTPException(404)
    return t


@app.get("/api/moderation_records")
def api_mod_records(limit: int = 50):
    with session_scope() as s:
        rs = s.execute(select(ModerationRecord).order_by(ModerationRecord.id.desc()).limit(limit)).scalars().all()
        return {"records": [{"id": r.id, "type": r.target_type, "target_id": r.target_id,
                             "media": r.media_type, "score": round(r.risk_score, 3), "verdict": r.verdict,
                             "labels": r.labels, "elapsed": round(r.elapsed, 2),
                             "report_path": r.report_path,
                             "created_at": r.created_at.strftime("%m-%d %H:%M")} for r in rs]}


@app.post("/api/moderate/scan_all")
def api_scan_all(background: BackgroundTasks, limit: int = 300):
    """批量审核平台上的存量文本内容。"""
    task_id = f"scan{int(time.time()*1000)}"
    TASKS[task_id] = {"status": "running", "started": time.time()}
    background.add_task(_scan_all, task_id, limit)
    return {"task_id": task_id}


def _scan_all(task_id: str, limit: int):
    from ..moderation.pipeline import verdict_of
    from ..moderation.text_filter import analyze_text
    try:
        with session_scope() as s:
            posts = s.execute(select(Post.id, Post.content).where(Post.mod_status.in_(["pending", "pass"])).limit(limit)).all()
            comments = s.execute(select(Comment.id, Comment.content).limit(limit * 2)).all()
        n_block = n_review = 0
        labels: dict[str, int] = {}
        with session_scope() as s:
            for pid, content in posts:
                r = analyze_text(content or "")
                v = verdict_of(r["score"])
                p = s.get(Post, pid)
                p.mod_status = v; p.risk_score = r["score"]; p.risk_labels = ",".join(r["labels"])
                n_block += v == "block"; n_review += v == "review"
                for l in r["labels"]:
                    labels[l] = labels.get(l, 0) + 1
            for cid, content in comments:
                r = analyze_text(content or "")
                v = verdict_of(r["score"])
                c = s.get(Comment, cid)
                c.mod_status = v; c.risk_score = r["score"]; c.risk_labels = ",".join(r["labels"])
                n_block += v == "block"; n_review += v == "review"
                for l in r["labels"]:
                    labels[l] = labels.get(l, 0) + 1
        TASKS[task_id].update({"status": "done", "result": {"posts": len(posts), "comments": len(comments),
                                                            "blocked": n_block, "review": n_review,
                                                            "labels": labels},
                               "elapsed": time.time() - TASKS[task_id]["started"]})
    except Exception as e:
        TASKS[task_id].update({"status": "error", "error": str(e)})


# ---------------------------------------------------------------- 统计
@app.get("/api/stats")
def api_stats():
    with session_scope() as s:
        n_users = s.execute(select(func.count(User.id))).scalar()
        n_bots = s.execute(select(func.count(User.id)).where(User.is_bot == True)).scalar()
        n_flag = s.execute(select(func.count(User.id)).where(User.bot_flag == True)).scalar()
        n_posts = s.execute(select(func.count(Post.id))).scalar()
        n_comments = s.execute(select(func.count(Comment.id))).scalar()
        n_likes = s.execute(select(func.count(Like.id))).scalar()
        n_follows = s.execute(select(func.count(Follow.id))).scalar()
        n_camp = s.execute(select(func.count(BotCampaign.id))).scalar()
        blocked = s.execute(select(func.count(Post.id)).where(Post.mod_status == "block")).scalar()
        review = s.execute(select(func.count(Post.id)).where(Post.mod_status == "review")).scalar()
        cblocked = s.execute(select(func.count(Comment.id)).where(Comment.mod_status == "block")).scalar()
    m = api_bot_metrics()
    rf = (m.get("models") or {}).get("RandomForest") or {}
    unseen = [v.get("detect_rate@0.5") for v in (m.get("evasion_generalization") or {}).values()]
    return {"users": n_users, "bots": n_bots, "flagged": n_flag, "posts": n_posts, "comments": n_comments,
            "likes": n_likes, "follows": n_follows, "campaigns": n_camp,
            "blocked_posts": blocked, "review_posts": review, "blocked_comments": cblocked,
            "sim_now": SIM_NOW.strftime("%Y-%m-%d %H:%M"),
            "detect": {"auc": rf.get("auc"), "f1": rf.get("f1"),
                       "unseen_evasion_rate": min(unseen) if unseen else None}}


# ---------------------------------------------------------------- 安全：聊天机器人
class ChatIn(BaseModel):
    message: str
    session_id: str = "default"
    guard: bool = True
    context: str = ""      # 模拟助手读取到的平台内容（不可信），用于演示间接注入


@app.post("/api/chat")
def api_chat(body: ChatIn):
    from ..security.chatbot import chat
    return chat(body.message, body.session_id, guard=body.guard, untrusted_context=body.context or None)


@app.get("/api/chat/samples")
def api_chat_samples():
    from ..security.guard import JAILBREAK_SAMPLES, INDIRECT_INJECTION_SAMPLES, BENIGN_SAMPLES
    return {"direct": JAILBREAK_SAMPLES, "indirect": INDIRECT_INJECTION_SAMPLES, "benign": BENIGN_SAMPLES}


@app.get("/api/chat/history")
def api_chat_history(session_id: str = "default", limit: int = 40):
    with session_scope() as s:
        ms = s.execute(select(ChatMessage).where(ChatMessage.session_id == session_id)
                       .order_by(ChatMessage.id.desc()).limit(limit)).scalars().all()
        return {"messages": [{"role": m.role, "content": m.content, "guard": m.guard or {},
                              "at": m.created_at.strftime("%H:%M:%S")} for m in reversed(ms)]}


# ---------------------------------------------------------------- 安全：对抗实验
@app.get("/api/security/results")
def api_sec_results():
    out = {}
    for name in ("adv_image.json", "adv_text.json", "adv_video.json", "adv_audio.json",
                 "deepfake_eval.json", "text_bench_combined.json", "text_bench_cold.json", "uatr.json",
                 "chatbot_guard.json", "video_bench_xdv.json", "prompt_tuning.json"):
        d = _read_result(name)
        if d is not None:
            out[name.replace(".json", "")] = d
    return out


@app.get("/api/health")
def api_health():
    return {"ok": True, "time": datetime.now().isoformat(timespec="seconds")}
