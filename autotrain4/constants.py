"""
全局默认常量（单源）。

显式配置（JSON / settings / CLI）永远优先；这里只统一「没配时」的默认值，
避免 auto_train2 / common / webview2 / playwright / PyQt 各写一套 8842/8843。
"""

from __future__ import annotations

# DLCV 训练前端（dev server / 壳端内嵌页常见端口以 8843 为准；历史 8842 仅作兼容探测）
DEFAULT_FRONTEND_URL = "http://localhost:8843"

# DLCV 训练后端 API
DEFAULT_BACKEND_URL = "http://127.0.0.1:9887"

# 推理服务（Test / 报告），与训练 Backend 不同端口
DEFAULT_INFERENCE_URL = "http://127.0.0.1:9890"
