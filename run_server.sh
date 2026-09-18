#!/bin/bash
# 启动本地沙箱 Web 服务（仅绑定 127.0.0.1）
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn socialsim.app.main:app --host 127.0.0.1 --port 8000 "$@"
