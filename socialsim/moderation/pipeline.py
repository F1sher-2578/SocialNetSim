"""统一审核入口：文本 / 图片 / 视频 → 风险分数、标签、报告。

具体专家模型在 text_filter / image_experts / video_pipeline 中实现；本模块负责调度与落库。
"""
from __future__ import annotations
import json, time, traceback
from datetime import datetime
from pathlib import Path
from sqlalchemy import select
from ..db import session_scope
from ..models import Post, Comment, ModerationRecord
from ..config import RISK_BLOCK, RISK_REVIEW, RESULT_DIR


def verdict_of(score: float) -> str:
    if score >= RISK_BLOCK:
        return "block"
    if score >= RISK_REVIEW:
        return "review"
    return "pass"


def moderate_text(text: str) -> dict:
    from .text_filter import analyze_text
    return analyze_text(text)


def moderate_image(path: str) -> dict:
    from .image_experts import analyze_image
    return analyze_image(path)


def moderate_video(path: str, out_dir: Path | None = None) -> dict:
    from .video_pipeline import analyze_video
    return analyze_video(path, out_dir=out_dir)


def moderate_post(post_id: int) -> dict:
    """对帖子做多模态审核并写回状态。"""
    t0 = time.time()
    with session_scope() as s:
        post = s.get(Post, post_id)
        if post is None:
            return {}
        content, media_type, media_path = post.content, post.media_type, post.media_path
    result = {"score": 0.0, "labels": [], "parts": {}}
    try:
        tr = moderate_text(content or "")
        result["parts"]["text"] = tr
        result["score"] = max(result["score"], tr["score"])
        result["labels"] += tr["labels"]
        if media_type == "image" and media_path:
            ir = moderate_image(media_path)
            result["parts"]["image"] = ir
            result["score"] = max(result["score"], ir["score"]); result["labels"] += ir["labels"]
        elif media_type == "video" and media_path:
            vr = moderate_video(media_path, out_dir=RESULT_DIR / "reports" / f"post_{post_id}")
            result["parts"]["video"] = {k: v for k, v in vr.items() if k not in ("frames",)}
            result["score"] = max(result["score"], vr["score"]); result["labels"] += vr["labels"]
            result["report_path"] = vr.get("report_path", "")
        result["labels"] = sorted(set(result["labels"]))
    except Exception as e:  # 模型缺失/失败时转人工复核而不是放行
        result["error"] = f"{type(e).__name__}: {e}"
        result["trace"] = traceback.format_exc()[-800:]
        result["score"] = max(result["score"], RISK_REVIEW)
        result["labels"].append("error")
    v = verdict_of(result["score"])
    with session_scope() as s:
        post = s.get(Post, post_id)
        post.mod_status = v; post.risk_score = float(result["score"]); post.risk_labels = ",".join(result["labels"])
        post.mod_report = result.get("report_path", "")
        rec = ModerationRecord(target_type="post", target_id=post_id, media_type=media_type, risk_score=float(result["score"]),
                               verdict=v, labels=",".join(result["labels"]), details=_slim(result), report_path=result.get("report_path", ""),
                               elapsed=time.time() - t0)
        s.add(rec)
    result["verdict"] = v; result["elapsed"] = time.time() - t0
    return result


def moderate_comment(comment_id: int) -> dict:
    with session_scope() as s:
        c = s.get(Comment, comment_id)
        if c is None:
            return {}
        r = moderate_text(c.content or "")
        v = verdict_of(r["score"])
        c.mod_status = v; c.risk_score = float(r["score"]); c.risk_labels = ",".join(r["labels"])
        s.add(ModerationRecord(target_type="comment", target_id=comment_id, media_type="text", risk_score=float(r["score"]),
                               verdict=v, labels=",".join(r["labels"]), details=_slim(r)))
        r["verdict"] = v
        return r


def _slim(d, limit=6000):
    try:
        js = json.dumps(d, ensure_ascii=False, default=str)
        return json.loads(js) if len(js) < limit else {"summary": js[:limit]}
    except Exception:
        return {"summary": str(d)[:limit]}
