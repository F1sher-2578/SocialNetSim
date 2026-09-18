"""SocialNetSim Web 应用：仿真社交平台 + 水军控制台 + 审核后台 + 安全实验台。

仅在本地沙箱运行（默认绑定 127.0.0.1），不与任何真实平台交互。
"""
from __future__ import annotations
import json, shutil, time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, func, desc

from ..config import UPLOAD_DIR, RESULT_DIR, ROOT, RISK_BLOCK, RISK_REVIEW
from ..db import session_scope, init_db
from ..models import User, Post, Comment, Like, Follow, BotCampaign, ModerationRecord, ActionLog, ChatMessage
from ..simulation import bot_engine, bot_detector
from ..simulation.feed import get_feed, exposure_report
from ..simulation.seed import SIM_NOW

app = FastAPI(title="SocialNetSim", description="社交网络拟真沙箱", version="1.0")
STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/reports", StaticFiles(directory=str(RESULT_DIR)), name="reports")

TASKS: dict[str, dict] = {}


@app.on_event("startup")
def _startup():
    init_db()


# ---------------------------------------------------------------- 页面
@app.get("/", response_class=HTMLResponse)
def index():
    p = Path(__file__).parent / "templates" / "index.html"
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------- 社交基本功能
@app.get("/api/feed")
def api_feed(mode: str = "hot", user_id: int | None = None, limit: int = 25, offset: int = 0,
             hide_blocked: bool = True):
    return {"items": get_feed(mode, user_id, limit, offset, hide_blocked=hide_blocked)}


@app.get("/api/users")
def api_users(limit: int = 50, only_bots: bool = False, sort: str = "bot_score"):
    with session_scope() as s:
        q = select(User)
        if only_bots:
            q = q.where(User.is_bot == True)
        col = {"bot_score": User.bot_score, "created_at": User.created_at, "id": User.id}.get(sort, User.bot_score)
        users = s.execute(q.order_by(desc(col)).limit(limit)).scalars().all()
        out = []
        for u in users:
            n_posts = s.execute(select(func.count(Post.id)).where(Post.author_id == u.id)).scalar()
            n_fo = s.execute(select(func.count(Follow.id)).where(Follow.follower_id == u.id)).scalar()
            n_fa = s.execute(select(func.count(Follow.id)).where(Follow.followee_id == u.id)).scalar()
            out.append({"id": u.id, "username": u.username, "nickname": u.nickname, "avatar": u.avatar,
                        "bio": u.bio, "is_bot": bool(u.is_bot), "bot_group": u.bot_group,
                        "bot_score": round(u.bot_score or 0, 4), "bot_flag": bool(u.bot_flag),
                        "created_at": u.created_at.strftime("%Y-%m-%d"), "posts": n_posts,
                        "following": n_fo, "followers": n_fa})
        return {"users": out}


@app.get("/api/post/{pid}")
def api_post(pid: int):
    with session_scope() as s:
        p = s.get(Post, pid)
        if not p:
            raise HTTPException(404)
        cs = s.execute(select(Comment).where(Comment.post_id == pid).order_by(Comment.created_at)).scalars().all()
        authors = {u.id: u for u in s.execute(select(User).where(
            User.id.in_([c.author_id for c in cs] + [p.author_id]))).scalars()}
        a = authors.get(p.author_id)
        return {
            "post": {"id": p.id, "content": p.content, "created_at": p.created_at.strftime("%Y-%m-%d %H:%M"),
                     "likes": p.like_count, "comments": p.comment_count, "reposts": p.repost_count,
                     "media_type": p.media_type, "media_path": p.media_path,
                     "mod_status": p.mod_status, "risk_score": round(p.risk_score or 0, 3),
                     "risk_labels": p.risk_labels, "mod_report": p.mod_report,
                     "author": {"id": a.id, "nickname": a.nickname, "avatar": a.avatar,
                                "is_bot": bool(a.is_bot), "bot_score": round(a.bot_score or 0, 3)} if a else None},
            "comments": [{"id": c.id, "content": c.content, "created_at": c.created_at.strftime("%m-%d %H:%M"),
                          "mod_status": c.mod_status, "risk_score": round(c.risk_score or 0, 3),
                          "risk_labels": c.risk_labels,
                          "author": {"id": c.author_id,
                                     "nickname": authors[c.author_id].nickname if c.author_id in authors else "?",
                                     "avatar": authors[c.author_id].avatar if c.author_id in authors else "👤",
                                     "is_bot": bool(authors[c.author_id].is_bot) if c.author_id in authors else False,
                                     "bot_score": round(authors[c.author_id].bot_score or 0, 3) if c.author_id in authors else 0}}
                         for c in cs],
        }


class PostIn(BaseModel):
    content: str
    user_id: int | None = None
    moderate: bool = True


@app.post("/api/post")
def api_create_post(body: PostIn):
    from ..moderation.pipeline import moderate_post
    with session_scope() as s:
        uid = body.user_id
        if uid is None:
            uid = s.execute(select(User.id).where(User.is_bot == False)).scalars().first()
        p = Post(author_id=uid, content=body.content, created_at=datetime.now(), mod_status="pending")
        s.add(p); s.flush()
        pid = p.id
    result = moderate_post(pid) if body.moderate else {}
    return {"post_id": pid, "moderation": {k: v for k, v in result.items() if k != "parts"}}


class CommentIn(BaseModel):
    post_id: int
    content: str
    user_id: int | None = None


@app.post("/api/comment")
def api_create_comment(body: CommentIn):
    from ..moderation.pipeline import moderate_comment
    with session_scope() as s:
        uid = body.user_id or s.execute(select(User.id).where(User.is_bot == False)).scalars().first()
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
    return {"metrics": {k: {kk: vv for kk, vv in v.items() if kk not in ("prob", "roc")}
                        if isinstance(v, dict) else v for k, v in res.items()}}


@app.get("/api/bot_explain/{uid}")
def api_bot_explain(uid: int):
    return bot_detector.explain(uid)


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
    is_video = dest.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v")
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
    from ..moderation.pipeline import moderate_comment, verdict_of
    from ..moderation.text_filter import analyze_text
    try:
        with session_scope() as s:
            posts = s.execute(select(Post.id, Post.content).where(Post.mod_status.in_(["pending", "pass"])).limit(limit)).all()
            comments = s.execute(select(Comment.id, Comment.content).limit(limit * 2)).all()
        n_block = n_review = 0
        with session_scope() as s:
            for pid, content in posts:
                r = analyze_text(content or "")
                v = verdict_of(r["score"])
                p = s.get(Post, pid)
                p.mod_status = v; p.risk_score = r["score"]; p.risk_labels = ",".join(r["labels"])
                n_block += v == "block"; n_review += v == "review"
            for cid, content in comments:
                r = analyze_text(content or "")
                v = verdict_of(r["score"])
                c = s.get(Comment, cid)
                c.mod_status = v; c.risk_score = r["score"]; c.risk_labels = ",".join(r["labels"])
                n_block += v == "block"; n_review += v == "review"
        TASKS[task_id].update({"status": "done", "result": {"posts": len(posts), "comments": len(comments),
                                                            "blocked": n_block, "review": n_review},
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
    metrics = {}
    p = RESULT_DIR / "bot_detection_metrics.json"
    if p.exists():
        try:
            metrics = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"users": n_users, "bots": n_bots, "flagged": n_flag, "posts": n_posts, "comments": n_comments,
            "likes": n_likes, "follows": n_follows, "campaigns": n_camp,
            "blocked_posts": blocked, "review_posts": review, "blocked_comments": cblocked,
            "sim_now": SIM_NOW.strftime("%Y-%m-%d %H:%M"), "bot_metrics": metrics}


# ---------------------------------------------------------------- 安全：聊天机器人
class ChatIn(BaseModel):
    message: str
    session_id: str = "default"
    guard: bool = True


@app.post("/api/chat")
def api_chat(body: ChatIn):
    from ..security.chatbot import chat
    return chat(body.message, body.session_id, guard=body.guard)


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
                 "deepfake_eval.json", "text_bench_combined.json", "uatr.json", "chatbot_guard.json",
                 "video_bench_xdv.json"):
        p = RESULT_DIR / name
        if p.exists():
            try:
                out[name.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return out


@app.get("/api/health")
def api_health():
    return {"ok": True, "time": datetime.now().isoformat(timespec="seconds")}
