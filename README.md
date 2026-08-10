# AutoTrain 4.0 — DLCV 视觉训练自动化工作台

AutoTrain 4.0 用自然语言驱动 DLCV AI Training Platform：解析训练/测试/定位需求、匹配数据集、生成配置、操作训练界面、监控任务，并在训练后做推理评估与 Baseline 对比。

入口支持命令行、PyQt 桌面端和 Vue Web 控制台。三种入口最终都由 `auto_train4.0.py` 编排，训练执行仍交给 `auto_train2.0.py`（WebBridge UI 自动化）。

自然语言链路**依赖 OpenAI 兼容 LLM**（默认 DeepSeek），本地仅做规范、纠偏与文件系统搜索，没有“纯规则也能完整跑通”的兜底。

---

## 当前能力

- **多意图模式**（由 LLM 判定，本地再纠偏）：
  - `train`：完整训练链路
  - `test`：测试最近模型 / 评估
  - `locate`：只定位数据集路径，**不**生成训练配置、不启动平台
  - `chat`：通用对话
  - `dry_run`：生成配置 / `.dva` 后停止（也可由 CLI `--dry-run` 强制）
- 一句话创建 OCR、CLS、DET、DETR、ISEG、SSEG、旋转框检测任务。
- 一条提示词可拆出多个任务并**顺序执行**；单任务失败不中断后续，结束后汇总失败项。
- 按任务独立解析数据增强（翻转、固定角度旋转、裁剪、亮度/对比度等），避免多任务串扰。
- **数据集定位四级优先级**：
  1. CLI `--folder`
  2. 提示词 / 意图中的**真实存在**的完整绝对路径
  3. **NL 锚点有界深搜**（口语盘符 + 目录片段 + 项目名）
  4. `--scan-dir` 多根浅扫（depth=1）+ 任务关键字 + LLM 选择
- LLM 把口语脑补成不存在路径（如 `Z:\zzh\易格技术`）时会**降级为 hint** 并继续深搜，而不是直接失败。
- 提示词中的 `iseg` / `sseg` 等简写会复核纠正任务类型。
- 多根扫描根、自定义模型输出目录（`--output-dir`）。
- Playwright 网页 或 WebView2 CDP 壳端；WebBridge 失败时 API 兜底启动训练。
- 自动生成 `.dva`，收集最终 `.dvp`；成功/失败分别收敛 `work/<run_id>/` 产物。
- 训练后可选推理报告或 Baseline 对比。
- 多行中文助手回复经 `[CHAT_RESPONSE_JSON]` 协议传给前端，避免截断。

---

## 运行流程

```text
自然语言提示词
        ↓
LLM 解析 intent_action + 一个或多个 task
        ↓
┌───────┼────────┬─────────┐
│       │        │         │
train  test    locate     chat
│       │        │         │
│       │        │         └→ 结合最近 run 上下文回复
│       │        └→ 四级定位 → 打印路径 / CHAT_RESPONSE（不训练）
│       └→ 读 auto_latest_*.json + 最新 .dvp → 推理/对比
│
逐任务：匹配数据集 → 独立 run 目录
        ↓
需求判断（Judge）+ 配置生成（ConfigBuilder）
        ↓
auto_train2.0.py + WebBridge（失败则 API）
        ↓
监控并收集 .dva / .dvp → 清理冗余
        ↓
可选：推理报告 / Baseline 对比
```

---

## 项目结构

```text
kimi_gui_automation/
├── auto_train4.0.py              # 自然语言编排入口（多任务 / locate / test / chat）
├── auto_train2.0.py              # DLCV 平台启动、UI 自动化与训练监控
├── autotrain4/                   # AutoTrain 4.0 核心
│   ├── common.py                 # 任务映射、路径/盘符/NL 线索与锚点工具
│   ├── nlu.py                    # 意图解析（train/test/locate/chat）与纠偏
│   ├── resolver.py               # 显式路径 / NL 深搜 / 多根浅扫
│   ├── judge.py                  # LLM 需求可行性判断
│   ├── config_builder.py         # 基于模板生成任务 JSON
│   ├── llm_client.py             # OpenAI 兼容客户端 + intent/select_dataset
│   ├── inference_client.py       # DLCV 推理 HTTP 客户端
│   ├── evaluator.py              # 推理报告与 Baseline 对比
│   ├── run_context.py            # run 目录、异常与产物清理
│   └── output_protocol.py        # 子进程 ↔ 前端结构化回复
├── kimi_webbridge/               # Playwright / WebView2 CDP 适配器
├── config/                       # 任务模板 + 运行时 auto_latest_*.json
├── pyqt_frontend/                # PyQt5 桌面工作台
├── web_frontend/                 # Vue 3 + Vite + Flask Web 控制台
├── tests/                        # 回归测试（多任务 / NL 定位 / locate / 产物策略）
├── AI训练平台测试/               # 默认数据集扫描根（本地数据，勿提交）
├── Baseline Models/              # 可选 Baseline .dvp（勿直接提交）
├── record/                       # 对话历史、录屏等运行数据
├── failed_runs/                  # run 创建前的早期失败记录
├── FLOWCHART.md                  # 调用链示意（可能略旧，以本 README 为准）
└── README.md                     # 本文档（PyQt「查看文档」也会读它）
```

权威实现在仓库根目录；`dlcv-auto-train/` 若存在仅为历史快照，不要以其为准改代码。

---

## 环境要求

- Windows 10/11
- Python 3.11 或兼容版本
- Node.js 18+（仅 Web 前端）
- 已安装 DLCV AI Training Platform
- 可访问的 OpenAI 兼容 LLM（**必需**）
- Playwright Chromium，或支持远程调试的 WebView2 壳端
- DLCV Test 推理服务（仅推理报告需要，默认 `http://127.0.0.1:9890`）

### 安装核心依赖

仓库根目录暂无统一 `requirements.txt`，可先：

```powershell
python -m pip install openai python-dotenv requests tqdm PyYAML psutil playwright
python -m playwright install chromium
```

```powershell
# PyQt 桌面端
python -m pip install -r pyqt_frontend/requirements.txt

# Web 控制台
python -m pip install -r web_frontend/requirements.txt
```

---

## LLM 配置

启动时读取项目根目录 `.env`。推荐：

```dotenv
AUTOTRAIN_LLM_BASE_URL=https://api.deepseek.com
AUTOTRAIN_LLM_API_KEY=your_api_key
AUTOTRAIN_LLM_MODEL=deepseek-chat
```

兼容 DeepSeek 专用变量：

```dotenv
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_MODEL=deepseek-chat
```

优先级：显式传入 → `AUTOTRAIN_LLM_*` → `DEEPSEEK_*`。  
**不要**提交 `.env` 或含密钥的 `pyqt_frontend/settings.json`。

---

## DLCV 平台配置

平台 EXE 解析顺序：

1. `--platform-exe`
2. 环境变量 `DLCV_TRAIN_EXE`
3. 任务 JSON 模板中的 `platform.executable_path`
4. 常见安装目录自动扫描

| 服务 | 默认地址 | 用途 |
|---|---|---|
| DLCV 训练后端 | `http://127.0.0.1:9887` | 项目、超参、训练状态 API |
| DLCV 训练前端 | `http://localhost:8842` 或模板地址 | Playwright 操作页 |
| WebView2 CDP | `127.0.0.1:9222` | 桌面壳端页面 |
| DLCV Test 推理 | `http://127.0.0.1:9890` | 新模型推理 / Baseline 对比 |
| AutoTrain Flask | `http://127.0.0.1:5000` | Web 控制台 API 与生产页 |
| Vite 开发服 | `http://127.0.0.1:5173` | 前端热更新 |

---

## 数据集约定

默认扫描根：`AI训练平台测试/`。一级目录名宜含任务关键词，训练数据放在候选子目录中：

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

识别的训练集子目录名（优先级顺序）：`训练集`、`1-训练集`、`训练`、`train`、`dataset`。

可用 `--scan-dir` 换根（支持多根），或用 `--folder` 为**第一个**任务指定目录。

### 多根目录浅扫

每个根下只扫**直接子目录**，再合并候选给 LLM：

```powershell
# 重复 --scan-dir
python auto_train4.0.py --prompt "训练 OCR" --scan-dir Z:\AI训练平台测试 --scan-dir Y:\datasets

# 单参数内用 ; 或 | 分隔
python auto_train4.0.py --prompt "训练 OCR" --scan-dir "Z:\AI训练平台测试;Y:\datasets;C:\data"
```

不存在的根会跳过并警告；全部不存在则报错。同名文件夹靠完整 `path` 消歧。

### 自然语言模糊定位（锚点有界深搜）

除完整绝对路径与 `--folder` 外，可用**口语盘符 + 目录名/项目名**描述位置：

```powershell
python auto_train4.0.py --prompt "帮我训练一个 iseg 模型，训练集是 z 盘 zzh 文件夹中的易格技术，默认参数" --generate-config-only
```

**解析顺序：**

1. CLI `--folder`
2. 提示词中的完整绝对路径（且优先要求真实存在；LLM 幻觉路径会降级）
3. **NL 锚点深搜**：`z盘`/`Z:` → 盘符，结合 `zzh`、`易格` 等在锚点下有界递归（默认深度 4），按项目名 / 任务关键字 / 训练集结构打分，必要时 LLM 消歧
4. 回退 `--scan-dir` 多根浅扫 + 任务关键字

**限制与安全：**

- 必须有可搜索线索（路径片段或项目名）；**禁止**无关键词时整盘扫 `Z:\`
- 跳过 `work`、`checkpoints`、`.git`、`node_modules` 等
- `--scan-depth N` 调深度；`--no-nl-locate` 关闭深搜层
- LLM 脑补不存在路径 → 降为 `dataset_hint`，继续锚点深搜
- `iseg`/`sseg` 等简写会纠正错误的 task_type

**本机验收示例：**

```text
prompt: 帮我训练一个iseg模型 训练集是z盘zzh文件夹中的易格技术 默认参数
→ task_type: 实例分割
→ dataset_dir: ...\zzh\A-易格技术\...\2-训练集实例分割\1-训练集
```

### 仅定位数据集（locate）

不需要开训、只想确认数据在哪时：

```powershell
python auto_train4.0.py --prompt "帮我定位 z 盘 zzh 里易格技术的 iseg 训练集"
```

行为：

- `mode=locate`，`needs_training=false`
- 走与训练相同的四级匹配
- 打印 `[DATASET_DIR]`、`[PROJECT_DIR]` 与结构化中文说明
- **不**写训练配置、**不**调 `auto_train2.0.py`

含「定位 / 查找 / 在哪 / locate」等语气且无明确开训动词时，应判为 locate，而不是误开训。

---

## 命令行使用

### 单任务训练

```powershell
python auto_train4.0.py --prompt "训练一个 OCR 模型"
```

### 多任务顺序训练

```powershell
python auto_train4.0.py --prompt "训练 CLS 模型，启用随机固定角度旋转；再训练 OCR 模型；再训练 SSEG 模型并启用随机翻转"
```

每个任务独立配置与 run 目录。批量时中间任务不 `keep-alive` / 不长时间占窗口；**仅最后一个任务**应用用户的 keep 设置。批量任务默认 `--force-new` 且不 `--reuse`。

### 只生成配置

```powershell
python auto_train4.0.py --prompt "训练一个 DET 模型" --generate-config-only
```

### 只生成 DVA，不启动平台

```powershell
python auto_train4.0.py --prompt "训练一个 CLS 模型" --dry-run
```

### 指定模型输出目录

```powershell
python auto_train4.0.py --prompt "训练 iseg，数据在 z盘zzh 易格" --output-dir "D:\models\yige"
# run 落在 D:\models\yige\work\<run_id>\
```

也支持在提示词中写「模型输出到 `D:\...`」（与训练集路径区分）。

### WebView2 壳端

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

可用 `--latest-config` 指定 `auto_latest_*.json`，`--baseline-model` 指定 `.dvp`。

### 通用对话

```powershell
python auto_train4.0.py --prompt "最近一次 OCR 任务为什么失败？"
```

非训练/测试/定位时进入对话，结合最近约 3 个 run 目录回答；历史默认 `record/chat_history.json`。`--clear-chat-history` 可在启动前清空。

### 常用参数

| 参数 | 说明 |
|---|---|
| `--prompt` | 必填，自然语言需求 |
| `--scan-dir` | 扫描根，可重复或 `;`/`\|` 分隔多根；默认 `AI训练平台测试` |
| `--folder` | 第一个任务的数据集目录（绝对路径或任一扫描根下相对名） |
| `--scan-depth` | NL 锚点相对最大深度，默认 `4` |
| `--no-nl-locate` | 关闭 NL 深搜，仅显式路径 / folder / 浅扫 |
| `--output-dir` | 模型与 work 归档根；run 在 `<output-dir>/work/<run_id>/` |
| `--adapter` | `playwright` / `webview2_cdp`（另保留测试用选项） |
| `--platform-exe` | DLCV 壳端 EXE |
| `--cdp-port` | WebView2 CDP 端口，默认 `9222` |
| `--auto-threshold` | 数据集自动匹配阈值，默认 `0.55` |
| `--config-out` | 生成配置保存路径 |
| `--generate-config-only` | 生成 JSON 后停止 |
| `--dry-run` | 生成 `.dva` 后停止 |
| `--force-new` | 强制新建项目 |
| `--reuse` / `--no-reuse` | 是否复用已运行平台，默认复用（批量任务不复用） |
| `--keep-alive` / `--no-keep-alive` | 训练后是否保持平台，默认保持 |
| `--keep-browser-open` / `--no-keep-browser-open` | 训练后是否等待关窗，默认等待 |
| `--record` / `--no-record` | 是否录屏，默认开；成功后按策略清理 |
| `--record-dir` | 录屏目录，默认 `./record` |
| `--post-train` / `--no-post-train` | 是否询问报告/Baseline，默认询问 |
| `--full-train` | 训练集比例 100%（可复现验证） |
| `--test-latest` | 跳过训练，测最近模型 |
| `--latest-config` | 配合 test-latest 的配置路径 |
| `--baseline-model` | 基准 `.dvp` |
| `--inference-api-url` | 推理服务，默认 `http://127.0.0.1:9890` |
| `--inference-report-path` | 推理报告路径 |
| `--no-progress` | 关闭推理进度条 |
| `--chat-history` | 对话历史路径 |
| `--clear-chat-history` | 启动前清空对话历史 |

完整列表：`python auto_train4.0.py --help`。

---

## work 目录与产物策略

默认每个训练任务：

```text
<匹配到的项目目录>/work/<项目目录名>-<YYYYMMDD_HHMMSS>/
```

若指定了 `--output-dir`（或提示词中的输出目录）：

```text
<output-dir>/work/<run_id>/
```

示例：

```text
AI训练平台测试/2-CLS/work/2-CLS-20260722_130954/
```

### 训练成功

子流程成功 **且** run 根目录找到最终 `.dvp` 才按成功收敛。保留：

```text
config.json
runtime_info.json
train.log                   # 若平台生成
*.dva
*.dvp                       # 主产物
inference_report.json       # 可选
baseline_compare_report.json# 可选
```

会删除浏览器 Profile、重复 stdout、多余检查点、截图、录屏及平台重复时间戳临时目录。

### 训练失败

保留复现证据：

```text
config.json
runtime_info.json
exception.json
stderr.log / train.log      # 若有
screenshots/
screen_recording.mp4
checkpoints/                # 仅最后一个 .dvp 检查点
*.dva / *.dvp               # 若已生成
```

若返回成功但无最终 `.dvp`，按失败保留诊断证据。

run 创建前异常：

```text
failed_runs/<时间戳>/work/early-failure-<时间戳>/exception.json
```

---

## 训练后推理与 Baseline

最后一个任务（或单任务）在 `--post-train` 开启时会询问：

1. 为新模型生成 `inference_report.json`
2. 与 Baseline 比较 → `baseline_compare_report.json`
3. 直接结束

Baseline 顺序：`--baseline-model` → 提示词中的 `.dvp` → `Baseline Models/` 下以任务类型开头的最新模型。

---

## PyQt 桌面端

```powershell
python pyqt_frontend/main.py
```

- 自然语言输入与建议、快捷模板
- 分阶段日志与任务状态、训练后报告/对比
- 设置：DLCV 地址、EXE、Adapter、CDP、LLM
- 「查看文档」读取根目录 `README.md`

本地设置：`pyqt_frontend/settings.json`（可能含 Key，勿提交）。

---

## Web 控制台

技术栈：Vue 3、Vite、Pinia、Vue Router、Flask。

### 生产

```powershell
cd web_frontend
npm install
npm run build
python app.py
```

访问 `http://127.0.0.1:5000`。

### 开发

```powershell
# 终端 1
cd web_frontend
python app.py

# 终端 2
cd web_frontend
npm install
npm run dev
```

访问 `http://127.0.0.1:5173`（`/api` 代理到 Flask）。

| 路由 | 说明 |
|---|---|
| `GET /api/health` | 健康检查 |
| `GET /api/run?prompt=...` | SSE 启动任务并流式日志 |
| `GET /api/tasks` | 最近任务 |
| `GET /api/tasks/<task_id>` | 任务文件列表 |
| `GET /api/datasets` | 数据集列表 |
| `GET /api/models` | Baseline 与最新配置 |
| `GET /api/templates` | 快捷模板 |

---

## 配置模板

| 任务 | 模板 |
|---|---|
| OCR | `config/ocr_train_automation_config.json` |
| CLS / 分类 | `config/cls_train_automation_config.json` |
| DET / 目标检测 | `config/det_train_automation_config.json` |
| DETR | `config/detr_train_automation_config.json` |
| ISEG / 实例分割 | `config/iseg_train_automation_config.json` |
| SSEG / 语义分割 | `config/sseg_train_automation_config.json` |
| 旋转框检测 | `config/rotate_train_automation_config.json` |

`ConfigBuilder` 基于模板写入数据集、run、`.dva`、Adapter、录屏、增强与 LLM 调整，并更新 `config/auto_latest_<任务>.json`。

模板可能含本机绝对路径；提交前请改为示例值。

---

## 核心模块职责（简表）

| 模块 | 职责 |
|---|---|
| `auto_train4.0.py` | CLI、编排、locate/test/chat/train、调 2.0、后处理 |
| `autotrain4/nlu.py` | LLM 意图 → `TrainIntent`；路径可信性；task/mode 纠偏 |
| `autotrain4/resolver.py` | 数据集四级匹配与打分 |
| `autotrain4/common.py` | 别名、NL 线索、锚点、路径工具 |
| `autotrain4/judge.py` | 是否 proceed 与调整建议 |
| `autotrain4/config_builder.py` | 模板 → 最终 JSON |
| `autotrain4/llm_client.py` | chat / extract_intent / select_dataset / match_categories |
| `auto_train2.0.py` | 启平台、WebBridge 点选、监控训练 |

---

## 测试

```powershell
python -m unittest tests.test_multitask_regressions -v
```

当前覆盖包括：

- 多任务中间/末任务传给 2.0 的参数策略
- 多行 Unicode 助手回复协议
- 数据增强按任务隔离；Judge 只收当前任务上下文
- 显式路径、多根浅扫、folder 跨根
- NL 线索提取、锚点、嵌套发现、噪声过滤、work/深度限制
- 幻觉路径降级、iseg 任务类型纠正
- **locate 不触发训练**；task_context 措辞
- `--output-dir` 归档位置
- 成功/失败产物清理策略；「成功但无 dvp」按失败保留
- 训练 API 防重复启动

---

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

保留基础任务模板并清掉本机绝对路径。大模型请用 Git LFS，并确认分发权限。

---

## 常见问题

### 提示 LLM 未配置

根目录 `.env` 设置 `AUTOTRAIN_LLM_API_KEY` 或 `DEEPSEEK_API_KEY`，并安装 `openai`、`python-dotenv`。

### 数据集匹配置信度不足

检查一级目录是否含任务关键词、训练集子目录是否符合约定。确定目录时用 `--folder`，勿盲目降低 `--auto-threshold`。口语场景确认是否带盘符/目录/项目名线索。

### 口语路径被拼成错误绝对路径

当前版本会忽略不存在的幻觉路径并走 NL 深搜。若仍失败，可加 `--folder` 或写真实绝对路径。

### 只想找数据却启动了训练

使用「定位 / 查找 / 在哪」等表述，且不要带「训练/开训」。日志中 `mode` 应为 `locate`。

### WebBridge 无法连接

- Playwright：前端地址可达，`python -m playwright install chromium`
- WebView2：EXE 正确、CDP 端口空闲、`--adapter webview2_cdp`
- 双路径失败时看 run 内 `exception.json`、截图、录屏

### Web 页面空白

生产模式需先 `web_frontend` 下 `npm run build`；开发模式访问 Vite `5173`。

### 助手回复只显示一半

需前后端都支持 `[CHAT_RESPONSE_JSON]` 协议，勿混用旧入口。

### 训练成功但 work 中无 `.dvp`

不会当作有效成功，会保留失败诊断。检查 `exception.json`、`train.log`、`stderr.log` 与最后检查点。

---

## 安全说明

- 勿提交 `.env`、API Key、本机用户目录、数据集或训练录屏
- `pyqt_frontend/settings.json` 为本地设置
- `config/auto_latest_*.json` 为运行产物，可能含本机路径
- 公开分发平台相关文件、模型、数据前确认许可证

---

## 许可证

内部工具。未经授权请勿商用或二次分发。
