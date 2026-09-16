# AutoLabel

自然语言 / 短类别名 → 实例分割自动标注。输出 **LabelMe 兼容 JSON**（与图片同名旁路文件），可直接进 LabelMe / LabelMeAI 复核。

标注引擎：**SAM3.1**（本机内置）以及设置里用 Base URL / API Key 自行添加的视觉模型。主进程不加载 SAM3 / torch。SAM3 权重不随软件分发（电脑上的模型文件保持不动）。

```
用户（GUI / CLI）
  → Intent（LLM 可选）  label=显示名, model_prompt=视觉描述
  → Capability Check
  → Plan（扫图、跳过已有 JSON）
  → Preview 3～5 张
  → 确认
  → Batch 全量
  → work JSON + overlay
  → Commit 旁路 JSON 到数据集目录
```

---

## 快速开始

```bash
# 1. 依赖（AutoLabel 主进程环境，不要用 sam3 Conda）
pip install -r AutoLabel/requirements.txt

# 2. 本地配置（不要提交密钥）
copy AutoLabel\config\settings.example.json AutoLabel\config\settings.json
copy AutoLabel\config\backends.example.json AutoLabel\config\backends.json

# 3. 单测
python -m pytest AutoLabel/tests -q

# 4. GUI
python AutoLabel/main.py

# 5. CLI（Mock，无需 GPU）
python AutoLabel/cli.py --dataset <图片目录> --label target --description "visual description" --preview --backend mock
```

SAM3.1 实标需要：

1. 仓库旁有官方 `sam3/`（**禁止修改**；SAM 3.1 与 SAM3 共用这套推理代码）
2. Conda 环境 `C:\dlcv\envs\sam3\python.exe` 已装好 SAM3
3. `backends.json` 的 `annotation_engine` 为 `sam3.1`
4. 本机有 `models/sam3.1_multiplex.pt`（仓库根 `models/` 或便携包 `models/`）

启动 GUI 后会探测 `http://127.0.0.1:9892/health`；未运行则自动拉起 Worker，等 `model_loaded=true` 再推理。关闭 GUI 时会结束**本程序拉起**的 Worker。

Windows 便携包（无 Python / Conda）见仓库 `packaging/README_PORTABLE.md`。构建：

```bat
packaging\build_portable.bat
```

---

## 目录结构

```
AutoLabel/
├── main.py                 # PyQt GUI
├── cli.py                  # 无头 CLI
├── requirements.txt
├── config/
│   ├── settings.example.json
│   ├── backends.example.json
│   ├── settings.json       # 本地，gitignore
│   └── backends.json       # 本地，gitignore
├── autolabel/              # 编排、几何、导出（禁止 import sam3/torch）
├── sam3_worker/            # 仅用 sam3 Conda 启动的 HTTP 服务
├── ui/                     # PyQt 界面
├── tests/
├── fixtures/golden/        # LabelMe JSON 金样
└── runs/                   # Worker 日志 / 一次性推理输出（本地）
```

一次 run 写在数据集旁，不覆盖原图：

```
<dataset>/work/autolabel/autolabel-YYYYMMDD_HHMMSS/
├── meta/intent.json, plan.json
├── preview/                # 试标 JSON
├── json/                   # 全量 work JSON
├── stage1/                 # transform、raw mask、debug
├── overlays/
└── report/                 # 复核队列、commit 清单
```

Commit 后才会把 JSON 拷到图片同级：`foo.jpg` → `foo.json`。默认 `skip_existing`，已有旁路文件不覆盖。

---

## 环境

| 进程 | Python | 职责 |
|------|--------|------|
| AutoLabel GUI / CLI | 普通环境（PyQt5、opencv、requests、openai） | 任务、Intent、后处理、LabelMe、UI |
| SAM3 Worker | `C:\dlcv\envs\sam3\python.exe` | 加载 SAM3 一次并常驻，HTTP 推理 |

约束：

- **不要改** 仓库根目录旁的 `sam3/` 官方代码
- AutoLabel 主进程 **禁止** `import sam3` / `import torch`
- 不要把 AutoLabel 业务文件塞进 `sam3/`
- Worker 只绑定 `127.0.0.1`，模型常驻；禁止每张图重新 load

SAM3 环境里需要能 `import sam3`（官方仓库在 `PYTHONPATH` 上）。Worker 启动时会把 `AutoLabel/` 与 `../sam3` 注入子进程 `PYTHONPATH`，并清掉父进程的 `PYTHONPATH`，避免 PyQt / 错误的 torch 泄漏进去。

---

## 配置

两套配置分开，**不要混用 Key**。

| 通道 | 文件 | 用途 |
|------|------|------|
| LLM | `config/settings.json` | 自然语言 → Intent（短 `label` + 英文 `model_prompt`） |
| Vision | `config/backends.json` | 选哪个分割后端、阈值、Worker 路径 |

`settings.json` / `backends.json` 已在仓库根 `.gitignore`。提交只用 `*.example.json`。也可复用 `../pyqt_frontend/settings.json`。

### LLM（`settings.json`）

```json
{
  "llm_base_url": "http://127.0.0.1:3000/v1",
  "llm_api_key": "",
  "llm_model": "kimi-k3",
  "llm_protocol": "openai",
  "intent_mode": "auto"
}
```

`intent_mode`：

| 值 | 行为 |
|----|------|
| `auto` | 短名词走 simple；句子才调 LLM |
| `llm` | 始终走 LLM |
| `simple` | 不调 LLM，把输入当类别名 |

「AI 识别名称」从数据集里随机抽最多 5 张图，一次发给模型，由它自己写出给 SAM3.1 的英文 prompt。设置里填 `gpt-image-2` 即可，内部自动改走 `gpt-4o` 看图，不生成 mask。

Intent 规则：`label` 是 LabelMe 显示名；发给 SAM3 的是 `model_prompt`（用户或「AI 识别名称」给出的视觉描述，代码不改写成固定词表）。整句任务指令不要塞进 prompt。GUI 短名称只改 LabelMe 类别，不会覆盖已经很好的 `model_prompt`。Worker 按 0.05 召回，再用最小面积过滤过小连通域；置信度滑条不再把整张图打空。

### 标注引擎（`backends.json`）

设置页的 **标注引擎**（`annotation_engine`）只有两项：

| 引擎 | 内部 id | 行为 |
|------|---------|------|
| SAM3 | `sam3_local` | 本机 Worker 文本 prompt 实例分割 |
| chat-image | `gpt_image_mask` | `/images/edits` 彩色 mask → 拆实例 |

CLI `--backend mock` 仍可用于无 GPU 测试。

SAM3 常用选项（`sam3_local.options`）：

| 字段 | 默认 | 含义 |
|------|------|------|
| `python_path` | `C:\dlcv\envs\sam3\python.exe` | Worker 解释器 |
| `host` / `port` | `127.0.0.1` / `9892` | 只监听本机 |
| `score_threshold` | `0.55` | 复核用。SAM3 实际按 0.05 召回；低分目标不再被滑条直接打空 |
| `min_area` | `64` | 丢掉过小连通域（px）。GUI 滑条上限 16384 |
| `nms_iou_threshold` | `0.7` | bbox NMS |
| `epsilon_ratio` | `0.001` | 轮廓多边形简化 |
| `startup_timeout` | `600` | 等模型加载（秒） |
| `timeout` | `180` | 单张推理超时 |

GUI 设置页可改 LLM 与当前 backend；置信度 / 最小面积也可在「新建任务」高级选项里临时覆盖。

---

## GUI

```bash
python AutoLabel/main.py
```

四步：

1. **新建任务**：选图片文件夹 → 自然语言或短名（可用「AI 识别名称」）→ 可选阈值
2. **预览标注**：试标若干张，看 overlay / JSON，确认质量
3. **批量处理**：试标确认后才允许全量（安全闸）
4. **结果导出**：Commit 旁路 JSON 到数据集；未确认不写原目录

侧栏「检查连接」探测 SAM3 `/health`。首次加载模型可能要几分钟，状态会从 `loading` 变为 `ok`。

---

## CLI

在仓库根目录执行（`cli.py` 会把 `AutoLabel/` 与仓库根加入 `sys.path`）。

```bash
# 只出 Plan
python AutoLabel/cli.py --dataset D:\data\images --label target --plan-only --backend sam3_local

# 试标
python AutoLabel/cli.py --dataset D:\data\images --label target --preview --backend sam3_local --intent-mode simple

# 试标 + 全量（必须 --yes）
python AutoLabel/cli.py --dataset D:\data\images --label target --preview --batch --yes --backend sam3_local

# 全量后 Commit 旁路 JSON
python AutoLabel/cli.py --dataset D:\data\images --label target --preview --batch --yes --commit --backend sam3_local

# Mock 不碰 GPU
python AutoLabel/cli.py --dataset D:\data\images --label target --preview --backend mock
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--dataset` | 图片目录（递归扫图，跳过 `work/`） |
| `--label` / `--description` / `--exclude` | 手工 Intent |
| `--prompt` | 自然语言，覆盖手工字段 |
| `--backend` | 覆盖引擎：`sam3` / `chat-image`，或内部 id（`sam3_local` / `gpt_image_mask` / `mock`） |
| `--intent-mode` | `auto` / `llm` / `simple` |
| `--score-threshold` / `--min-area` | 覆盖 SAM3 过滤 |
| `--preview-count` | 试标张数，默认 5 |
| `--yes` | 确认试标，允许 `--batch` |
| `--commit` | 把 work JSON 写成图片旁路文件 |

全量没有 `--yes` 会退出码 `3`。已有同名 JSON 默认跳过。

---

## SAM3 Worker

主进程只走 HTTP。Worker 由 `autolabel/sam3_worker_ctl.py` 按需拉起。

手动验证：

```bash
# Stage A：单张（会当场 load 模型）
C:\dlcv\envs\sam3\python.exe AutoLabel\sam3_worker\run_once.py --image D:\data\001.jpg --prompt "<text prompt>"

# Stage B：常驻
C:\dlcv\envs\sam3\python.exe AutoLabel\sam3_worker\server.py --host 127.0.0.1 --port 9892
```

接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | `{status, model_loaded, device}`；加载中 `status=loading` |
| POST | `/segment` | `{image_path, prompt, score_threshold, run_dir}` → instances（mask / bbox / score） |

环境变量：`SAM3_WORKER_HOST`、`SAM3_WORKER_PORT`、`SAM3_WORKER_RUNS`。日志默认 `AutoLabel/runs/sam3/worker.log`。

---

## 标注流水线

1. **Intent**：短 `label` + `model_prompt` + 可选 exclude / description
2. **Plan**：列图、统计已有 JSON、检查 backend 能力（无实例 mask 则硬失败）
3. **Stage 1**：backend 出 per-instance binary mask + score + bbox
4. **Stage 2**：形态学 / 可选 GrabCut（GPT Image 路径）、`findContours` → `approxPolyDP` → 面积过滤
5. **Export**：LabelMe polygon JSON（`imageData=null`，坐标为原图像素）
6. **Review**：按置信度 / 质量风险分桶，写入 `report/`
7. **Commit**：work JSON → `<stem>.json` 旁路

图片读入会做 EXIF 方向校正，多边形写回校正后的宽高。`work/` 与 `__pycache__` 不会当数据集扫进去。

支持的图片后缀：`.jpg` `.jpeg` `.png` `.bmp` `.tif` `.tiff` `.webp` `.gif`。

---

## 输出 JSON

与 LabelMe 5.x 对齐，shape 为 polygon：

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "target",
      "points": [[10.0, 20.0], [80.0, 20.0], [80.0, 90.0]],
      "group_id": 0,
      "shape_type": "polygon",
      "flags": {
        "autolabel": true,
        "model_confidence": 0.91,
        "quality_risk": 0.12
      },
      "mask": null
    }
  ],
  "imagePath": "001.jpg",
  "imageData": null,
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

审核被拦的图会记空 JSON + `moderation_blocked`，**不会 Commit**。

---

## 测试

```bash
python -m pytest AutoLabel/tests -q
```

覆盖导出金样、capability、mock 流水线、SAM3 client/NMS、Worker 拉起、几何 / EXIF、GrabCut、UI 缩放等。GUI 在 `QT_QPA_PLATFORM=offscreen` 下不会去探 9892。

---

## 依赖

主进程见 `requirements.txt`：PyQt5、numpy、opencv-python、Pillow、requests、openai、python-dotenv、tqdm、pytest。

SAM3 Worker 额外需要官方 `sam3` 及其 CUDA / torch 栈，只装在 `C:\dlcv\envs\sam3`。

可选：仓库里的 `autotrain4` 若存在，Intent JSON 解析会复用它的 `safe_parse_json`；没有则走本地正则 JSON 抽取。

---

## 常见问题

**Worker 起不来 / health 一直 down**  
检查 `python_path` 是否真是 sam3 环境、9892 是否被占、`AutoLabel/runs/sam3/worker.log`。首次 load 可能接近 `startup_timeout`（默认 600s）。

**主进程 import 了 torch**  
架构错误。推理必须留在 Worker。主进程只允许 `requests` 调 `127.0.0.1:9892`。

**多边形和画面错位**  
先看该次 run 的 `stage1/debug/<stem>/` 与日志里的 `[IMAGE DEBUG]`（EXIF / PIL / OpenCV / SAM3 尺寸是否一致）。

**不想覆盖已有标注**  
保持默认 `overwrite_policy=skip_existing`。Commit 前 JSON 只在 `work/`，删 work 不影响原图。

**只想推代码、不要测试集**  
不要提交 `数据集测试/`、`work/`、`runs/`、`config/settings.json`、`config/backends.json`。SAM3 官方仓是旁边的 `sam3/`，不要当 AutoLabel 源码改。
