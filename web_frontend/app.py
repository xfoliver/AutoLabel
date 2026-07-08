"""
AutoTrain 4.0 简易 Web 前端

启动：
    python web_frontend/app.py

然后浏览器打开 http://127.0.0.1:5000

功能：
- 提供一个输入框，用户直接输入自然语言训练需求。
- 点击「开始训练」后，后台调用 auto_train4.0.py --prompt "..." 并实时流式输出日志。
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask, Response, render_template, request, stream_with_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 项目根目录（app.py 位于 web_frontend/ 下）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
AUTO_TRAIN = PROJECT_ROOT / "auto_train4.0.py"


def _send_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run")
def run():
    prompt = request.args.get("prompt", "").strip()
    if not prompt:
        return Response(
            _send_event({"type": "error", "message": "请输入训练需求"}),
            mimetype="text/event-stream",
        )

    cmd = [sys.executable, str(AUTO_TRAIN), "--prompt", prompt]
    logger.info("执行命令: %s", " ".join(cmd))

    @stream_with_context
    def generate():
        start_time = time.time()
        yield _send_event({"type": "start", "prompt": prompt})

        env = os.environ.copy()
        # 确保子进程 stdout/stderr 使用 UTF-8，避免中文乱码
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
                    yield _send_event({"type": "log", "line": line})
        except GeneratorExit:
            process.terminate()
            raise
        finally:
            process.wait()
            elapsed = time.time() - start_time
            yield _send_event(
                {
                    "type": "done",
                    "code": process.returncode,
                    "elapsed": round(elapsed, 2),
                }
            )

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
