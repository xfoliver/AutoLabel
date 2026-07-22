"""
AutoTrain 4.0 Web 前端后端

提供 REST + SSE API，并为 Vue SPA 提供静态文件服务。

启动：
    cd web_frontend
    pip install -r requirements.txt
    npm install
    npm run build
    python app.py

开发模式：
    python app.py          # 后端 5000
    npm run dev            # 前端 5173（带 proxy）
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="dist/assets", static_url_path="/assets")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
AUTO_TRAIN = PROJECT_ROOT / "auto_train4.0.py"
DATASET_ROOT = PROJECT_ROOT / "AI训练平台测试"
BASELINE_DIR = PROJECT_ROOT / "Baseline Models"
RECORD_DIR = PROJECT_ROOT / "record"
FAILED_DIR = PROJECT_ROOT / "failed_runs"


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _scan_dataset_dirs() -> list:
    if not DATASET_ROOT.exists():
        return []
    result = []
    for child in sorted(DATASET_ROOT.iterdir()):
        if not child.is_dir():
            continue
        train_dir = child / "训练集"
        images = []
        if train_dir.exists():
            images = [
                p for p in train_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
        result.append({
            "name": child.name,
            "path": str(child),
            "type": child.name.split("-")[-1] if "-" in child.name else "未知",
            "count": len(images),
            "updatedAt": datetime.fromtimestamp(child.stat().st_mtime).strftime("%Y-%m-%d"),
        })
    return result


def _scan_models() -> list:
    models = []
    if BASELINE_DIR.exists():
        for p in sorted(BASELINE_DIR.glob("*.dvp")):
            models.append({
                "name": p.stem,
                "path": str(p),
                "kind": "baseline",
                "updatedAt": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d"),
            })
    config_dir = PROJECT_ROOT / "config"
    if config_dir.exists():
        for p in sorted(config_dir.glob("auto_latest_*.json")):
            models.append({
                "name": p.stem,
                "path": str(p),
                "kind": "latest_config",
                "updatedAt": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d"),
            })
    return models


def _scan_tasks() -> list:
    tasks = []
    for src_dir in (RECORD_DIR, FAILED_DIR):
        if not src_dir.exists():
            continue
        for child in sorted(src_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not child.is_dir():
                continue
            status = "done" if src_dir == RECORD_DIR else "error"
            detail_parts = []
            webm_files = list(child.glob("*.webm"))
            if webm_files:
                detail_parts.append(f"{len(webm_files)} 个录屏")
            exc_file = child / "exception.json"
            if exc_file.exists():
                status = "error"
                detail_parts.append("含异常日志")
            tasks.append({
                "id": child.name,
                "name": child.name,
                "detail": " · ".join(detail_parts) or "训练任务",
                "status": status,
                "path": str(child),
                "timestamp": datetime.fromtimestamp(child.stat().st_mtime).isoformat(),
            })
    return tasks


@app.route("/")
def index():
    return send_from_directory("dist", "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "backend": True,
        "frontend": True,
        "project_root": str(PROJECT_ROOT),
    })


@app.route("/api/run")
def api_run():
    prompt = request.args.get("prompt", "").strip()
    if not prompt:
        return Response(
            _sse_event({"type": "error", "message": "请输入训练需求"}),
            mimetype="text/event-stream",
        )

    cmd = [sys.executable, str(AUTO_TRAIN), "--prompt", prompt]
    logger.info("执行命令: %s", " ".join(cmd))

    def generate():
        start_time = time.time()
        yield _sse_event({"type": "start", "prompt": prompt})

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        try:
            for line in process.stdout:
                line = line.rstrip("\n")
                if line:
                    yield _sse_event({"type": "log", "line": line})
        except GeneratorExit:
            process.terminate()
            raise
        finally:
            process.wait()
            elapsed = time.time() - start_time
            yield _sse_event(
                {
                    "type": "done",
                    "code": process.returncode,
                    "elapsed": round(elapsed, 2),
                }
            )

    return Response(generate(), mimetype="text/event-stream")


@app.route("/run")
def legacy_run():
    """兼容旧版前端入口。"""
    return api_run()


@app.route("/api/tasks")
def tasks():
    return jsonify(_scan_tasks())


@app.route("/api/tasks/<task_id>")
def task_detail(task_id):
    for src_dir in (RECORD_DIR, FAILED_DIR):
        target = src_dir / task_id
        if target.exists():
            return jsonify({
                "id": task_id,
                "path": str(target),
                "files": [p.name for p in target.iterdir()],
            })
    return jsonify({"error": "task not found"}), 404


@app.route("/api/datasets")
def datasets():
    return jsonify(_scan_dataset_dirs())


@app.route("/api/models")
def models():
    return jsonify(_scan_models())


@app.route("/api/templates")
def templates():
    return jsonify([
        {"title": "OCR 基础训练", "desc": "标准参数", "prompt": "训练一个 OCR 模型"},
        {"title": "DET 快速实验", "desc": "低 Epoch", "prompt": "训练一个目标检测模型，epoch 设为 5"},
        {"title": "Baseline 对比", "desc": "自动评估", "prompt": "训练一个分类模型并对比 baseline"},
        {"title": "批量任务", "desc": "多模型队列", "prompt": "依次训练 OCR、CLS、DET 三个模型"},
    ])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
