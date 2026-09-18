"""ORM 数据模型：用户/帖子/评论/点赞/关注/审核记录/水军任务/行为日志。"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, Index, JSON)
from sqlalchemy.orm import relationship
from .db import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    nickname = Column(String(64), nullable=False)
    bio = Column(Text, default="")
    avatar = Column(String(16), default="🙂")
    password = Column(String(128), default="")      # 沙箱内明文占位，仅用于演示登录
    is_bot = Column(Boolean, default=False)          # 真实标签（用于评估检测器）
    bot_group = Column(String(64), default="")       # 所属水军团伙/任务
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    bot_score = Column(Float, default=0.0)           # 检测器给出的水军概率
    bot_flag = Column(Boolean, default=False)        # 检测器判定
    posts = relationship("Post", back_populates="author")


class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    author_id = Column(Integer, ForeignKey("users.id"), index=True)
    content = Column(Text, default="")
    media_type = Column(String(16), default="text")   # text / image / video
    media_path = Column(String(256), default="")
    hashtags = Column(String(256), default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    like_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    repost_count = Column(Integer, default=0)
    repost_of = Column(Integer, nullable=True)
    # 审核
    mod_status = Column(String(16), default="pending")   # pending / pass / review / block
    risk_score = Column(Float, default=0.0)
    risk_labels = Column(String(256), default="")
    mod_report = Column(String(256), default="")         # 报告文件路径
    author = relationship("User", back_populates="posts")


class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True)
    author_id = Column(Integer, ForeignKey("users.id"), index=True)
    content = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    mod_status = Column(String(16), default="pass")
    risk_score = Column(Float, default=0.0)
    risk_labels = Column(String(256), default="")


class Like(Base):
    __tablename__ = "likes"
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey("posts.id"), index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    __table_args__ = (Index("ix_like_unique", "post_id", "user_id", unique=True),)


class Follow(Base):
    __tablename__ = "follows"
    id = Column(Integer, primary_key=True)
    follower_id = Column(Integer, ForeignKey("users.id"), index=True)
    followee_id = Column(Integer, ForeignKey("users.id"), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (Index("ix_follow_unique", "follower_id", "followee_id", unique=True),)


class ActionLog(Base):
    """统一行为日志（用于水军检测的时序/协同特征）。"""
    __tablename__ = "action_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)
    action = Column(String(16), index=True)        # post / comment / like / follow / repost / login
    target_id = Column(Integer, nullable=True)     # post id 或 user id
    created_at = Column(DateTime, index=True)
    campaign = Column(String(64), default="")


class BotCampaign(Base):
    __tablename__ = "bot_campaigns"
    id = Column(Integer, primary_key=True)
    name = Column(String(64))
    strategy = Column(String(32))          # boost / smear / hashtag_push / follow_farm
    target = Column(String(128), default="")
    n_bots = Column(Integer, default=10)
    intensity = Column(Float, default=1.0)
    timing = Column(String(16), default="burst")   # burst / spread
    content_mode = Column(String(16), default="template")
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(16), default="created")
    stats = Column(JSON, default=dict)


class ModerationRecord(Base):
    __tablename__ = "moderation_records"
    id = Column(Integer, primary_key=True)
    target_type = Column(String(16))       # post / comment / upload
    target_id = Column(Integer, index=True)
    media_type = Column(String(16))
    risk_score = Column(Float, default=0.0)
    verdict = Column(String(16))           # pass / review / block
    labels = Column(String(256), default="")
    details = Column(JSON, default=dict)
    report_path = Column(String(256), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    elapsed = Column(Float, default=0.0)


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), index=True)
    role = Column(String(16))
    content = Column(Text)
    guard = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
