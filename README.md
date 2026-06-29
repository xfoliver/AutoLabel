# DLCV AI 训练平台自动化工具

本项目提供从自然语言指令到模型训练的端到端自动化能力，支持 OCR、分类（CLS）、目标检测（DET）、DETR、实例分割（ISEG）、语义分割（SSEG）、旋转框检测（ROTATE）等多种任务类型。

---

## 项目架构

```text
用户指令
  │
  ├──► auto_train4.0.py ──► autotrain4/ ──► config/ ──► auto_train2.0.py ──► kimi_webbridge/ ──► 浏览器/平台
  │
  └──► auto_train3.0.py （旧入口，仍可独立使用）
```

### 核心组件

| 组件 | 说明 |
|---|---|
| `auto_train2.0.py` | 执行引擎：加载配置 → 生成 `.dva` → 启动平台 → WebBridge 操作前端 → 监控训练 |
| `auto_train3.0.py` | 旧版入口：通过命令行参数（`--task OCR --latest`）选择数据集并调用 2.0 |
| `auto_train4.0.py` | 新版入口：通过自然语言指令（`--prompt`）自动解析意图、匹配数据集并调用 2.0 |
| `autotrain4/` | 4.0 的模块化实现包，包含 NLU、数据集匹配、配置生成、LLM 客户端 |
| `config/` | 各任务类型的模板配置文件，以及运行时生成的最终配置 |
| `kimi_webbridge/` | WebBridge 抽象层，支持 `playwright`/`skill`/`http`/`stub` 多种适配器 |
| `AI训练平台测试/` | 默认数据集根目录，按任务类型存放数据集文件夹 |

---

## 环境要求

- Python >= 3.8
- Windows 系统（当前平台 exe 为 Windows 版本）
- 已安装 `sentence-transformers`、`rapidfuzz`、`openai`、`requests`、`psutil`、`pyyaml`、`playwright`

### 依赖检查

```bash
python -c "import sentence_transformers, rapidfuzz, openai, requests, psutil, yaml, playwright; print('OK')"
```

---

## 快速开始

### 1. 自然语言一句话训练（推荐）

```bash
# 分类
python auto_train4.0.py --prompt "训练一个cls模型来辨别王不留行和白芷"

# OCR
python auto_train4.0.py --prompt "训练一个ocr模型识别图片中的文字"

# 目标检测
python auto_train4.0.py --prompt "训练一个det模型来识别螺丝"

# DETR
python auto_train4.0.py --prompt "训练一个detr模型来检测作弊行为"

# 实例分割
python auto_train4.0.py --prompt "训练一个实例分割模型"

# 语义分割
python auto_train4.0.py --prompt "训练一个语义分割模型"
```

### 2. 仅生成配置，不启动训练

```bash
python auto_train4.0.py --prompt "训练一个cls模型来辨别王不留行和白芷" --generate-config-only
```

### 3. Dry-run（只生成 .dva，不启动平台）

```bash
python auto_train4.0.py --prompt "训练一个cls模型来辨别王不留行和白芷" --dry-run
```

### 4. 手动指定数据集文件夹

```bash
python auto_train4.0.py --prompt "训练一个分类模型" --folder 2-CLS
```

### 5. 旧版命令行入口（3.0）

```bash
python auto_train3.0.py --task OCR --latest
python auto_train3.0.py --task CLS --folder 2-CLS
```

---

## 4.0 自然语言入口详解

### 核心流程

```text
用户输入 --prompt
    │
    ▼
auto_train4.0.py
    ├── 1. NLU 解析 → task_type + categories + dataset_hint
    ├── 2. 语义数据集匹配 → 找到最匹配的文件夹
    ├── 3. 生成 config/auto_latest_{任务类型}.json
    └── 4. 调用 auto_train2.0.py --config xxx.json
              │
              ▼
        生成 .dva → 启平台 → WebBridge 点击训练 → 监控
```

### 支持的指令形式

```bash
# 中文类别名
python auto_train4.0.py --prompt "训练一个分类模型识别王不留行和白芷"

# 英文类别名（数据集为中文时也能语义匹配）
python auto_train4.0.py --prompt "train a cls model to classify tulip and chrysanthemum"

# 只指定任务
python auto_train4.0.py --prompt "训练一个目标检测模型" --folder 3-DET

# 切换 WebBridge adapter
python auto_train4.0.py --prompt "训练一个cls模型" --adapter stub --dry-run
```

### 4.0 命令行参数

| 参数 | 说明 |
|---|---|
| `--prompt` | **必填**。自然语言训练指令。 |
| `--scan-dir` | 数据集根目录，默认 `AI训练平台测试`。 |
| `--folder` | 直接指定文件夹名，跳过语义匹配。 |
| `--config-out` | 指定生成的配置文件路径。 |
| `--adapter` | 覆盖 WebBridge adapter：`playwright` / `skill` / `http` / `stub`。 |
| `--auto-threshold` | 自动执行的匹配置信度阈值，默认 `0.55`。 |
| `--generate-config-only` | 仅生成配置，不调用 2.0。 |
| `--dry-run` | 传给 2.0：只生成 `.dva`，不启动平台。 |
| `--force-new` | 传给 2.0：强制创建新项目。 |
| `--keep-alive` | 传给 2.0：训练结束后保持平台运行（默认开启）。 |
| `--reuse` | 传给 2.0：复用已运行平台实例（默认开启）。 |

---

## 2.0 执行引擎详解

`auto_train2.0.py` 是实际干活的引擎，通常由 3.0/4.0 自动调用，也可以单独使用：

```bash
python auto_train2.0.py --config config/auto_latest_分类.json
```

### 2.0 执行流程

1. **加载配置**：读取 JSON 配置，展开 `%VAR%` 环境变量，校验必填项。
2. **生成 `.dva`**：根据配置生成/更新项目文件，自动检测数据集类别。
3. **启动平台**：启动 DLCV AI Training Platform 桌面应用，或复用已有实例。
4. **服务就绪探测**：等待后端（`:9887`）和前端（`:8843`）就绪。
5. **后端 API**：确保项目存在，将超参数写入后端。
6. **WebBridge 自动化**：在浏览器中设置参数、保存、点击开始训练。
7. **监控训练**：轮询训练状态直到完成/失败/中止。

### 2.0 命令行参数

| 参数 | 说明 |
|---|---|
| `--config` | 配置文件路径。 |
| `--force-new` | 强制创建新项目。 |
| `--dry-run` | 仅生成 `.dva`，不启动平台。 |
| `--keep-alive` | 训练结束后保持平台运行。 |
| `--reuse` | 复用已运行的平台/后端实例。 |

---

## 数据集组织规范

默认扫描目录为 `AI训练平台测试/`，建议按以下结构组织：

```text
AI训练平台测试/
├── 1-OCR/
│   └── 训练集/
├── 2-CLS/
│   └── 训练集/
│       ├── 类别A/
│       ├── 类别B/
│       └── 类别C/
├── 3-DET/
│   └── 训练集/
├── 4-DETR/
│   └── 训练/
├── 5-ISEG/
│   └── 1-训练集/
├── 6-SSEG/
│   └── 训练集/
└── 7-ROTATE/
    └── 训练集/
```

数据集子目录探测优先级：

```python
["训练集", "1-训练集", "训练", "train", "dataset"]
```

---

## WebBridge 适配器

WebBridge 是连接 2.0 与真实浏览器的抽象层，支持多种适配器：

| 适配器 | 说明 | 适用场景 |
|---|---|---|
| `playwright` | 直接用 Python 启动 Chromium/Edge，DOM 操作 | 默认推荐，稳定可控 |
| `skill` | 连接本机 Kimi WebBridge daemon（`:10086`） | 已有 daemon 环境 |
| `http` | 通过 REST API 连接外部 WebBridge 服务 | 远程/分布式 |
| `stub` | 测试桩，不操作真实浏览器 | 调试流程 |

默认配置使用 `playwright`，可在配置中修改：

```json
{
  "webbridge": {
    "enabled": true,
    "adapter": "playwright",
    "frontend_url": "http://localhost:8843",
    "headless": false,
    "browser": "chromium"
  }
}
```

---

## 配置模板

`config/` 目录包含各任务类型的模板：

| 任务类型 | 模板文件 |
|---|---|
| OCR | `config/ocr_train_automation_config.json` |
| 分类 | `config/cls_train_automation_config.json` |
| 目标检测 | `config/det_train_automation_config.json` |
| DETR | `config/detr_train_automation_config.json` |
| 实例分割 | `config/iseg_train_automation_config.json` |
| 语义分割 | `config/sseg_train_automation_config.json` |
| 旋转框检测 | `config/rotate_train_automation_config.json` |

运行时生成的最终配置会覆盖模板中的 `project` 路径信息，保存为 `config/auto_latest_{任务类型}.json`。

---

## LLM 配置（可选）

4.0 默认使用规则 + `sentence-transformers` 语义向量匹配，无需 LLM。如需启用 LLM 兜底处理复杂语义，配置环境变量：

```bash
export AUTOTRAIN_LLM_BASE_URL="https://api.openai.com/v1"
export AUTOTRAIN_LLM_API_KEY="sk-xxx"
export AUTOTRAIN_LLM_MODEL="gpt-3.5-turbo"
```

支持任意 OpenAI 兼容接口。

---

## 常见问题

### 1. 中文乱码

Windows 终端建议设置为 UTF-8：

```bash
chcp 65001
set PYTHONIOENCODING=utf-8
```

### 2. 首次运行慢

首次使用 `sentence-transformers` 会下载多语言模型，需要联网，后续运行会缓存。

### 3. 数据集类别不匹配

如果提示「数据集类别不匹配」，说明用户指定的类别和数据集子文件夹不一致。可以：

- 检查数据集类别名是否正确
- 用 `--folder` 强制指定文件夹
- 对于描述性任务（如「识别图片中的文字」），系统会自动跳过类别匹配

### 4. WebBridge 参数跳过

某些任务页面的字段名和模板不一致时，Playwright 会跳过不存在的参数。这通常不影响训练启动，但可能需要根据实际页面调整 `kimi_webbridge/playwright_adapter.py` 中的 `HP_CONTROL_MAP`。

---

## 文件结构

```text
.
├── auto_train2.0.py              # 执行引擎
├── auto_train3.0.py              # 旧版命令行入口
├── auto_train4.0.py              # 新版自然语言入口
├── autotrain4/                   # 4.0 模块包
│   ├── __init__.py
│   ├── common.py                 # 常量、任务映射
│   ├── nlu.py                    # 自然语言意图解析
│   ├── resolver.py               # 语义数据集匹配
│   ├── config_builder.py         # 配置生成
│   └── llm_client.py             # 可选 LLM 客户端
├── config/                       # 配置模板和生成配置
├── kimi_webbridge/               # WebBridge 抽象层
├── AI训练平台测试/                # 默认数据集根目录
└── README.md                     # 本文档
```

---

## 更新日志

- **4.0**：新增自然语言入口、语义数据集匹配、任务描述过滤、LLM 可选兜底。
- **3.0**：基于命令行参数的数据集自动选择和配置生成。
- **2.0**：基于 Kimi WebBridge 的可视化训练自动化执行引擎。
