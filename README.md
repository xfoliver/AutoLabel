# AutoTrain 4.0 — DLCV 视觉训练自动化工作台

AutoTrain 4.0 使用自然语言驱动 DLCV AI Training Platform，负责解析训练需求、匹配数据集、生成配置、操作训练界面、监控任务，以及在训练后执行推理评估和 Baseline 对比。

项目支持命令行、PyQt 桌面端和 Vue Web 控制台。三种入口最终都由 `auto_train4.0.py` 编排，并调用 `auto_train2.0.py` 完成 DLCV 平台自动化。

## 当前能力

- 一句话创建 OCR、CLS、DET、DETR、ISEG、SSEG、旋转框检测任务。
- 从一条提示词中提取多个任务并按顺序执行；单个任务失败时继续后续任务，结束后统一返回失败汇总。
- 按任务独立解析数据增强，支持随机翻转、随机固定角度旋转、随机裁剪、亮度/对比度等选项，避免多任务参数串扰。
- 自动扫描并匹配数据集，置信度不足时停止执行，防止选错数据。
- 使用 LLM 分析训练需求并调整项目说明和超参数。
- 通过 Playwright 网页模式或 WebView2 CDP 壳端模式操作 DLCV 平台。
- WebBridge 失败时尝试通过 DLCV 后端 API 启动训练。
- 自动生成 `.dva` 项目文件并收集最终 `.dvp` 模型。
- 训练完成后可生成推理报告，或与 `Baseline Models/` 中的模型对比。
- 非训练输入自动进入通用对话；多行中文回复使用 JSON 行协议传给前端，避免回复只显示一半。
- 每次训练使用独立的 `work/<run_id>/`，成功任务只保留核心产物，失败任务保留错误复现证据。

## 运行流程

```text
自然语言提示词
    ↓
LLM 解析一个或多个任务
    ↓
逐任务匹配数据集并创建独立 run 目录
    ↓
需求判断与配置生成
    ↓
AutoTrain 2.0 启动 DLCV 平台
    ↓
WebBridge 配置界面并启动训练（失败时 API 兜底）
    ↓
监控状态并收集 .dva / .dvp
    ↓
清理运行期冗余文件
    ↓
可选：推理报告 / Baseline 对比
```

## 项目结构

```text
kimi_gui_automation/
├── auto_train4.0.py              # 自然语言、多任务、归档和训练后操作入口
├── auto_train2.0.py              # DLCV 平台启动、UI 自动化与训练监控
├── autotrain4/                   # AutoTrain 4.0 核心模块
│   ├── common.py                 # 任务映射、默认值和路径工具
│   ├── nlu.py                    # 训练/测试/通用对话意图解析
│   ├── resolver.py               # 数据集扫描与匹配
│   ├── judge.py                  # LLM 需求判断
│   ├── config_builder.py         # 任务配置生成
│   ├── llm_client.py             # OpenAI 兼容 LLM 客户端
│   ├── inference_client.py       # DLCV 推理接口客户端
│   ├── evaluator.py              # 推理报告和 Baseline 对比
│   ├── run_context.py            # run 目录、异常记录和产物清理
│   └── output_protocol.py        # 子进程与前端的结构化回复协议
├── kimi_webbridge/               # Playwright / WebView2 CDP 适配器
├── config/                       # 各任务基础模板和运行时最新配置
├── pyqt_frontend/                # PyQt5 桌面工作台
├── web_frontend/                 # Vue 3 + Vite + Flask Web 工作台
├── tests/                        # 多任务、回复协议和产物策略回归测试
├── AI训练平台测试/               # 默认数据集扫描根目录（本地数据，不建议提交）
├── Baseline Models/              # 可选 Baseline .dvp（不建议直接提交）
├── record/                       # 通用对话历史等运行数据
├── failed_runs/                  # 未创建正常 run 前的早期失败记录
└── README.md                     # PyQt“查看文档”读取的文件
```

## 环境要求

- Windows 10/11
- Python 3.11 或兼容版本
- Node.js 18+（仅 Web 前端需要）
- 已安装 DLCV AI Training Platform
- 可访问的 OpenAI 兼容 LLM；自然语言解析没有规则兜底，因此 API Key 是必需的
- Playwright Chromium，或支持远程调试的 WebView2 壳端
- DLCV Test 推理服务（只在生成推理报告时需要，默认 `http://127.0.0.1:9890`）

### 安装核心依赖

当前仓库没有统一的根目录 `requirements.txt`，可先安装核心依赖：

```powershell
python -m pip install openai python-dotenv requests tqdm PyYAML psutil playwright
python -m playwright install chromium
```

使用 PyQt 桌面端时：

```powershell
python -m pip install -r pyqt_frontend/requirements.txt
```

使用 Web 控制台时：

```powershell
python -m pip install -r web_frontend/requirements.txt
```

## LLM 配置

程序启动时会读取项目根目录的 `.env`。推荐使用 AutoTrain 通用变量：

```dotenv
AUTOTRAIN_LLM_BASE_URL=https://api.deepseek.com
AUTOTRAIN_LLM_API_KEY=your_api_key
AUTOTRAIN_LLM_MODEL=deepseek-chat
```

也兼容 DeepSeek 专用变量：

```dotenv
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_MODEL=deepseek-chat
```

优先级为显式传入值、`AUTOTRAIN_LLM_*`、`DEEPSEEK_*`。不要把 `.env` 或带有密钥的 `pyqt_frontend/settings.json` 提交到 Git。

## DLCV 平台配置

平台 EXE 按以下顺序确定：

1. 命令行 `--platform-exe`。
2. 环境变量 `DLCV_TRAIN_EXE`。
3. 当前任务 JSON 模板中的 `platform.executable_path`。
4. 自动扫描常见安装目录。

默认服务地址：

| 服务 | 默认地址 | 用途 |
|---|---|---|
| DLCV 训练后端 | `http://127.0.0.1:9887` | 项目、超参数和训练状态 API |
| DLCV 训练前端 | `http://localhost:8842` 或模板中的地址 | Playwright 操作的训练页面 |
| WebView2 CDP | `127.0.0.1:9222` | 连接桌面壳端页面 |
| DLCV Test 推理后端 | `http://127.0.0.1:9890` | 新模型推理和 Baseline 对比 |
| AutoTrain Flask | `http://127.0.0.1:5000` | Web 控制台 API 与生产页面 |
| Vite 开发服务器 | `http://127.0.0.1:5173` | Web 前端热更新 |

## 数据集约定

默认扫描目录是 `AI训练平台测试/`。一级目录名称应包含任务关键词，训练数据放在其中一个候选子目录中：

```text
AI训练平台测试/
├── 1-OCR/训练集/
├── 2-CLS/训练集/
├── 3-DET/训练集/
├── 4-DETR/训练/
├── 5-ISEG/1-训练集/
├── 6-SSEG/训练集/
└── 7-ROTATE/训练集/
```

支持的训练集子目录名称依次包括：`训练集`、`1-训练集`、`训练`、`train`、`dataset`。也可以使用 `--scan-dir` 更换根目录（支持多根，见下），或用 `--folder` 为第一个任务指定目录。

### 多根目录浅扫

数据分散在多个磁盘/目录时，可一次指定多个扫描根。每个根下仍只扫**直接子目录**（与原先单根行为一致），再合并候选交给 LLM 选择：

```powershell
# 方式 1：重复 --scan-dir
python auto_train4.0.py --prompt "训练 OCR" --scan-dir Z:\AI训练平台测试 --scan-dir Y:\datasets

# 方式 2：单参数内用 ; 或 | 分隔
python auto_train4.0.py --prompt "训练 OCR" --scan-dir "Z:\AI训练平台测试;Y:\datasets;C:\data"
```

不存在的根会跳过并打警告；全部不存在则报错。多根下若出现同名项目文件夹，匹配结果以完整路径区分（LLM 会收到 `path` 字段）。

### 自然语言模糊定位（锚点有界深搜）

除完整绝对路径与 `--folder` 外，提示词里可以用**口语盘符 + 目录名/项目名**描述数据位置，无需写准路径：

```powershell
python auto_train4.0.py --prompt "帮我训练一个 iseg 模型，训练集是 z 盘 zzh 文件夹中的易格技术，默认参数" --generate-config-only
```

解析顺序：

1. CLI `--folder`
2. 提示词中的完整绝对路径（如 `Z:\项目\2-训练集实例分割`）
3. **NL 锚点深搜**：从 `z盘`/`Z:` 等推出盘符，结合 `zzh`、`易格` 等片段在锚点下有界递归（默认深度 4），按项目名/任务关键字/训练集结构打分，必要时 LLM 消歧
4. 回退到 `--scan-dir` 多根浅扫 + 任务关键字

限制与安全：

- 必须能推出可搜索线索（盘符片段、目录名或项目名）；**不会**在无关键词时整盘扫描 `Z:\`
- 跳过 `work`、`checkpoints`、`.git` 等目录
- `--scan-depth N` 调整深度；`--no-nl-locate` 可关闭该层（仅显式路径 / 浅扫）
- LLM 若把口语描述脑补成不存在的完整路径（如 `Z:\zzh\易格技术`），会被降级为 `dataset_hint` 并继续走锚点深搜，而不是直接失败
- 提示词里的 `iseg`/`sseg` 等简写会复核纠正任务类型，避免与语义分割等混淆

本机验收示例（口语定位，无需准确路径）：

```text
prompt: 帮我训练一个iseg模型 训练集是z盘zzh文件夹中的易格技术 默认参数
→ dataset_dir: \\...\zzh\A-易格技术\A260804-温州易格-奥图\2-训练集实例分割\1-训练集
```

## 命令行使用

### 单任务训练

```powershell
python auto_train4.0.py --prompt "训练一个 OCR 模型"
```

### 多任务顺序训练

```powershell
python auto_train4.0.py --prompt "训练 CLS 模型，启用随机固定角度旋转；再训练 OCR 模型；再训练 SSEG 模型并启用随机翻转"
```

每个任务都会创建独立配置和 run 目录。批量任务的中间任务不会等待训练后交互，也不会保持窗口打开；最后一个任务才应用 `keep-alive` 和 `keep-browser-open` 设置。

### 只生成配置

```powershell
python auto_train4.0.py --prompt "训练一个 DET 模型" --generate-config-only
```

### 只生成 DVA，不启动平台

```powershell
python auto_train4.0.py --prompt "训练一个 CLS 模型" --dry-run
```

### 使用 WebView2 壳端

```powershell
python auto_train4.0.py `
  --prompt "训练一个 OCR 模型" `
  --adapter webview2_cdp `
  --platform-exe "C:\dlcv\Lib\site-packages\dlcv_train\DLCV AI Training Platform-win_x64.exe" `
  --cdp-port 9222
```

### 测试最近生成的模型

```powershell
python auto_train4.0.py --prompt "测试最近训练的 OCR 模型" --test-latest
```

也可以用 `--latest-config` 指定 `auto_latest_*.json`，使用 `--baseline-model` 指定 `.dvp` 基准模型。

### 通用对话

```powershell
python auto_train4.0.py --prompt "最近一次 OCR 任务为什么失败？"
```

当 LLM 判断输入不需要训练或测试时，程序会结合最近三个 run 目录回答，并将历史默认保存到 `record/chat_history.json`。使用 `--clear-chat-history` 可在本次启动前清空历史。

### 常用参数

| 参数 | 说明 |
|---|---|
| `--prompt` | 必填，自然语言需求 |
| `--scan-dir` | 数据集扫描根，可重复或用 `;`/`|` 分隔多根；默认 `AI训练平台测试` |
| `--folder` | 直接指定第一个任务的数据集项目目录（绝对路径或任一扫描根下的相对名） |
| `--adapter` | `playwright` 或 `webview2_cdp`；解析器还保留测试适配器选项 |
| `--platform-exe` | DLCV 壳端 EXE 路径 |
| `--cdp-port` | WebView2 CDP 端口，默认 `9222` |
| `--auto-threshold` | 数据集自动匹配阈值，默认 `0.55` |
| `--config-out` | 指定生成配置的保存路径 |
| `--generate-config-only` | 生成 JSON 配置后停止 |
| `--dry-run` | 生成 `.dva` 后停止，不启动训练 |
| `--force-new` | 强制创建新项目 |
| `--reuse` / `--no-reuse` | 是否复用已运行的平台，默认复用 |
| `--keep-alive` / `--no-keep-alive` | 训练后是否保持 DLCV 平台运行，默认保持 |
| `--keep-browser-open` / `--no-keep-browser-open` | 训练后是否等待用户关闭窗口，默认等待 |
| `--record` / `--no-record` | 是否启用录屏，默认启用；成功后会按归档策略清理 |
| `--post-train` / `--no-post-train` | 是否询问生成报告或对比 Baseline，默认询问 |
| `--full-train` | 训练集比例设为 100%，用于可复现性验证 |
| `--test-latest` | 跳过训练，测试最近一次模型 |
| `--inference-api-url` | 推理服务地址，默认 `http://127.0.0.1:9890` |
| `--chat-history` | 通用对话记录路径 |

运行 `python auto_train4.0.py --help` 可查看完整参数。

## work 目录与产物策略

每个训练任务创建一个独立目录：

```text
<匹配到的项目目录>/work/<项目目录名>-<YYYYMMDD_HHMMSS>/
```

例如：

```text
AI训练平台测试/2-CLS/work/2-CLS-20260722_130954/
```

### 训练成功

只有子流程返回成功并且 run 根目录中找到最终 `.dvp`，才按成功任务收敛。核心保留内容为：

```text
config.json                 # 本次最终配置
runtime_info.json           # Python、系统、进程和任务序号等环境信息
train.log                   # 训练日志（平台实际生成时）
*.dva                       # DLCV 项目文件
*.dvp                       # 最终模型，主要产物
inference_report.json       # 可选，执行训练后推理时生成
baseline_compare_report.json# 可选，执行 Baseline 对比时生成
```

成功后会删除不再需要的浏览器 Profile、重复 stdout、检查点、截图、录屏以及平台生成的重复时间戳临时目录。

### 训练失败

失败任务保留用于复现问题的证据：

```text
config.json
runtime_info.json
exception.json
stderr.log                  # 实际产生时保留
train.log                   # 实际产生时保留
screenshots/                # 错误现场截图
screen_recording.mp4        # 错误现场录屏
checkpoints/                # 只保留最后一个 .dvp 检查点
*.dva
*.dvp                       # 若失败前已经生成
```

浏览器用户数据、重复 stdout、空的辅助进程日志和多余检查点仍会删除。如果流程返回成功但没有找到最终 `.dvp`，会按失败任务保留诊断证据。

如果在正常 run 创建前发生异常，错误会保存到：

```text
failed_runs/<时间戳>/work/early-failure-<时间戳>/exception.json
```

## 训练后推理与 Baseline

训练完成后，最后一个任务会询问是否：

1. 为新模型生成 `inference_report.json`。
2. 将新模型与 Baseline 比较并生成 `baseline_compare_report.json`。
3. 直接结束。

Baseline 查找顺序：命令行 `--baseline-model`、提示词中的 `.dvp` 路径、`Baseline Models/` 中以任务类型开头的最新模型。推理服务默认使用 `http://127.0.0.1:9890`。

## PyQt 桌面端

启动：

```powershell
python pyqt_frontend/main.py
```

主要功能：

- 自然语言训练需求输入和智能建议。
- 快捷模板的使用、保存和删除。
- 分阶段实时日志与当前任务状态。
- 训练完成后的报告/对比选择。
- 系统设置：DLCV 地址、EXE、Adapter、CDP 和 LLM。
- 顶部“查看文档”直接读取项目根目录的 `README.md`。

本地设置保存在 `pyqt_frontend/settings.json`。该文件可能包含 API Key，不应提交到 GitHub。

## Web 控制台

技术栈：Vue 3、Vite、Pinia、Vue Router 和 Flask。

### 生产模式

```powershell
cd web_frontend
npm install
npm run build
python app.py
```

访问 `http://127.0.0.1:5000`。

### 开发模式

终端一：

```powershell
cd web_frontend
python app.py
```

终端二：

```powershell
cd web_frontend
npm install
npm run dev
```

访问 `http://127.0.0.1:5173`，Vite 会把 `/api` 代理到 Flask。

### API

| 路由 | 说明 |
|---|---|
| `GET /api/health` | AutoTrain Web 后端健康状态 |
| `GET /api/run?prompt=...` | 通过 SSE 启动任务并流式返回日志 |
| `GET /api/tasks` | 最近任务列表 |
| `GET /api/tasks/<task_id>` | 单个任务文件列表 |
| `GET /api/datasets` | 数据集列表 |
| `GET /api/models` | Baseline 模型和最新配置 |
| `GET /api/templates` | 内置快捷模板 |

## 配置模板

任务与模板的对应关系：

| 任务 | 模板 |
|---|---|
| OCR | `config/ocr_train_automation_config.json` |
| CLS / 分类 | `config/cls_train_automation_config.json` |
| DET / 目标检测 | `config/det_train_automation_config.json` |
| DETR | `config/detr_train_automation_config.json` |
| ISEG / 实例分割 | `config/iseg_train_automation_config.json` |
| SSEG / 语义分割 | `config/sseg_train_automation_config.json` |
| 旋转框检测 | `config/rotate_train_automation_config.json` |

`ConfigBuilder` 会基于模板覆盖当前数据集、run 目录、`.dva` 路径、Adapter、录屏目录、增强选项和 LLM 调整。每次还会生成 `config/auto_latest_<任务>.json`，供最近模型测试和界面展示使用。

模板目前可能含开发机器的绝对路径。提交或部署前应将数据集、work 和平台 EXE 改为当前环境的路径或示例值。

## 测试

运行回归测试：

```powershell
python -m unittest tests.test_multitask_regressions -v
```

测试覆盖：

- 多任务中间任务和最后任务的进程参数。
- 多行 Unicode 助手回复的完整传输。
- 不同任务的数据增强隔离。
- LLM 判断只接收当前任务上下文。
- 成功任务的最小产物集合。
- 失败任务的诊断证据和最后检查点。
- “返回成功但没有最终模型”按失败保留证据。

## GitHub 提交建议

至少排除：

```gitignore
.env
**/settings.json
**/__pycache__/
*.py[cod]
*.log
.idea/
.vscode/
.ruff_cache/

record/
failed_runs/
webbridge_screenshots/
web_frontend/node_modules/
web_frontend/dist/

config/auto_latest_*.json
config/auto_*-20*.json
Baseline Models/
AI训练平台测试/
*.dva
*.dvp
*.mp4
```

基础任务模板应保留，但应清除本机绝对路径。大型模型如确实需要发布，请使用 Git LFS，并确认模型和数据集具有分发权限。

## 常见问题

### 提示 LLM 未配置

确认根目录 `.env` 中已设置 `AUTOTRAIN_LLM_API_KEY` 或 `DEEPSEEK_API_KEY`，并已安装 `openai` 与 `python-dotenv`。

### 数据集匹配置信度不足

检查一级目录名称是否包含任务关键词，并确认训练集子目录符合约定。确定目录无误时可使用 `--folder`，不建议盲目降低 `--auto-threshold`。

### WebBridge 无法连接

- Playwright：确认 DLCV 前端地址可访问，并已执行 `python -m playwright install chromium`。
- WebView2：确认 EXE 路径正确、`9222` 未被占用，并选择 `webview2_cdp`。
- 自动化操作失败后会尝试后端 API 兜底；两条路径都失败时查看 run 中的 `exception.json`、截图和录屏。

### Web 页面为空白

生产模式必须先在 `web_frontend/` 执行 `npm run build`。开发模式直接访问 Vite 的 `http://127.0.0.1:5173`。

### 助手回复只显示前半段

当前版本使用 `[CHAT_RESPONSE_JSON]` 结构化协议传输多行回复，PyQt 和 Web 状态层会解析完整内容。若仍出现问题，请确认前后端均为当前版本，不要混用旧版入口文件。

### 训练返回成功但 work 中没有 `.dvp`

当前版本不会把这种情况当作有效成功，而会保留失败诊断证据。检查 `exception.json`、`train.log`、`stderr.log` 以及最后检查点。

## 安全说明

- 不要提交 `.env`、API Key、本机用户目录、数据集或训练录屏。
- `pyqt_frontend/settings.json` 是本机设置，不是可公开分发的默认配置。
- `config/auto_latest_*.json` 是运行产物，可能暴露本机数据集和 work 路径。
- 在公开仓库发布 DLCV 平台相关文件、模型和数据前，请先确认许可证与分发权限。

## 许可证

当前项目为内部工具。未经授权，请勿商用或二次分发。
