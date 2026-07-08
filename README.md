# AutoTrain 4.0 使用说明

AutoTrain 4.0 是 DLCV AI 训练平台的**自然语言入口**。你只需用一句话描述训练需求，它就能自动解析意图、匹配数据集、生成训练配置，并调用 `auto_train2.0.py` 完成前端可视化训练。

4.0 采用 **LLM（默认 DeepSeek）驱动**的决策链路：意图解析、数据集选择、类别对齐、超参与数据增强推荐、需求可行性判断均由 LLM 完成。同时还新增了**数据增强自然语言配置**、**训练完成保持页面**、**一键测试最新模型**和**自动推理对比**等功能。

---

## 核心特性

- **自然语言触发**：支持中文/英文别名识别任务类型（`cls`/`分类`、`det`/`目标检测`、`OCR`、`DETR`、`iseg`/`实例分割`、`sseg`/`语义分割`、`rotate`/`旋转框检测`）。
- **LLM（DeepSeek）驱动全流程**：意图解析、数据集选择、类别对齐、超参与数据增强推荐、需求可行性判断均直接调用 DeepSeek/OpenAI 兼容 API。
- **保留 WebBridge 可视化**：不改动 2.0 执行引擎，前端点击流程（数据分析 → 数据增强 → 训练参数 → 开始训练）完全保留。
- **数据增强自然语言配置**：在 `--prompt` 中直接说明需要哪些增强，Agent 自动在前端勾选、保存并处理确认弹窗。
- **训练完成保持页面**：默认训练结束后浏览器窗口停留在训练完成页，按 Enter 才关闭。
- **一键测试最新模型**：无需手动找模型路径和测试集，自动复用最近一次训练的配置。
- **自动推理对比**：可与基准模型（baseline）对比同一张测试集，统计结果一致率。
- **可读的概率格式**：report 中的 score/probability 使用普通小数（如 `0.99470669`），不使用科学计数法。

---

## 环境要求

AutoTrain 4.0 的意图解析、数据集匹配、需求判断均依赖 LLM，**必须配置可用的 LLM API**。

默认直接接入 DeepSeek（已内置默认 key，也可通过环境变量覆盖），或任意 OpenAI 兼容接口。

---

## 快速开始

### 1. 一句话训练

```bash
python auto_train4.0.py --prompt "训练一个 cls 模型来辨别王不留行和白芷"
```

### 2. 仅生成配置（不启动训练）

```bash
python auto_train4.0.py --prompt "训练一个 cls 模型来辨别王不留行和白芷" --generate-config-only
```

### 3. 生成 .dva 做 dry-run

```bash
python auto_train4.0.py --prompt "训练一个 cls 模型来辨别王不留行和白芷" --dry-run
```

### 4. 手动指定文件夹（跳过语义匹配）

```bash
python auto_train4.0.py --prompt "训练一个分类模型" --folder 2-CLS
```

### 5. 指定数据增强

在提示中直接说明需要的数据增强，Agent 会自动勾选并保存：

```bash
python auto_train4.0.py --prompt "训练一个分类模型，使用随机翻转和随机裁剪做数据增强"
```

支持的数据增强选项：

- 调整图像大小
- 随机比例缩放
- 随机翻转
- 随机固定角度旋转
- 随机裁剪
- 随机调整亮度
- 随机调整对比度
- 随机 JPG 编码

### 6. 训练完成后自动关闭浏览器窗口

默认训练完成后浏览器窗口保持打开。如需自动关闭：

```bash
python auto_train4.0.py --prompt "训练一个分类模型" --no-keep-browser-open
```

### 7. 测试刚刚训练好的模型（训练集 = 测试集）

```bash
# 方式一：用自然语言触发测试意图
python auto_train4.0.py --prompt "现在帮我测试一下刚刚训练的模型"

# 方式二：显式指定 --test-latest
python auto_train4.0.py --prompt "测试分类模型" --test-latest

# 方式三：指定具体配置文件（测某一次训练）
python auto_train4.0.py --prompt "测试分类模型" --test-latest --latest-config config/auto_latest_分类.json
```

测试模式会自动：

1. 找到 `config/auto_latest_{任务类型}.json` 中最近一次的训练配置。
2. 从配置里读出 `project.dataset_dir`（训练集）和 `project.work_dir`。
3. 在 `work_dir` 里找最新的 `.dvp` 模型。
4. 用训练集作为测试集，调用推理接口生成 `test_report_{模型名}.json`。

### 8. 与基准模型对比

```bash
python auto_train4.0.py --prompt "测试刚才训练的分类模型" --baseline-model "Baseline Models/CLS-baseline_20260630_154101.dvp"
```

跑完后 report 的 `summary` 里会有：

- `match_count`：两张模型结果一致的图片数
- `mismatch_count`：两张模型结果不一致的图片数
- `mismatched_images`：不一致的明细

---

## 命令行参数

| 参数 | 说明 |
|---|---|
| `--prompt` | **必填**。自然语言训练/测试指令。 |
| `--scan-dir` | 数据集根目录，默认 `AI训练平台测试`。 |
| `--folder` | 直接指定文件夹名，跳过语义匹配。 |
| `--config-out` | 指定生成的配置文件路径。 |
| `--adapter` | 覆盖 WebBridge adapter，当前仅支持 `playwright`。 |
| `--auto-threshold` | 自动执行的匹配置信度阈值，默认 `0.55`。 |
| `--generate-config-only` | 仅生成配置，不调用 `auto_train2.0.py`。 |
| `--dry-run` | 传给 2.0：只生成 `.dva`，不启动平台。 |
| `--force-new` | 传给 2.0：强制创建新项目。 |
| `--keep-alive` | 传给 2.0：训练结束后保持平台运行（默认开启）。 |
| `--keep-browser-open` | 传给 2.0：训练结束后保持浏览器窗口打开，按 Enter 关闭（默认开启）。 |
| `--reuse` | 传给 2.0：复用已运行的平台/后端实例（默认开启）。 |
| `--full-train` | 训练集拉到 100%（`train_set_val=100, val_set_val=0`）。 |
| `--test-latest` | 跳过训练，直接测试最近一次训练生成的模型。 |
| `--latest-config` | 配合 `--test-latest`，指定要回看的配置文件路径。 |
| `--test-image-dir` | 训练完成后用于推理测试的图片目录（训练模式专用）。 |
| `--baseline-model` | 基准模型 `.dvp` 路径，用于与新模型推理结果对比。 |
| `--inference-api-url` | DLCV 推理接口地址，默认 `http://127.0.0.1:9890`。 |
| `--inference-report-path` | 推理对比报告保存路径。 |
| `--skip-inference` | 跳过训练后的推理对比。 |
| `--no-progress` | 关闭推理进度条显示。 |

---

## 前端自动化流程

当启用 WebBridge（默认 `playwright`）时，AutoTrain 4.0 会按以下顺序在浏览器中完成操作：

1. **数据分析**（可关闭）：点击侧边栏「数据分析」→ 加载数据集 → 加载尺寸分布。
2. **数据增强**（可选）：点击侧边栏「数据增强」→ 打开增强策略弹窗 → 勾选用户指定的增强选项 → 关闭弹窗 → 点击页面右下角「保存」→ 处理二次确认弹窗。
3. **训练参数**：点击侧边栏「训练参数」→ 展开所有参数面板 → 按配置设置超参 → 保存 → 开始训练 → 处理确认弹窗。
4. **监控训练**：轮询后端训练状态，直到完成/失败/中止。
5. **保持页面**：训练完成后浏览器窗口停留在当前页面，按 Enter 关闭。

---

## 数据增强

数据增强可以通过自然语言在 `--prompt` 中直接指定，例如：

```bash
python auto_train4.0.py --prompt "训练目标检测模型，做随机翻转、随机裁剪和随机调整亮度"
```

LLM 会解析出增强选项并写入 `ui_flow.data_augmentation`，WebBridge 在前端自动勾选对应复选框。

如需跳过数据增强，不要在提示中提到即可；也可以在生成的配置中将 `ui_flow.data_augmentation` 设为空列表。

---

## 测试模式详解

### 触发方式

AutoTrain 4.0 会通过自然语言判断你是否想测试模型。以下关键词都会进入测试模式：

- 测试、试一下、试下、评估、评测、验证、跑一遍
- test / testing / evaluate / eval / infer / inference

例如：

```bash
python auto_train4.0.py --prompt "帮我测试一下刚刚训练的分类模型"
python auto_train4.0.py --prompt "evaluate the latest cls model"
```

### 模型定位逻辑

1. 如果指定了 `--latest-config`，直接使用该配置文件。
2. 否则在 `config/auto_latest_*.json` 中找：
   - 如果意图里识别出任务类型（如“分类”），优先用 `auto_latest_分类.json`。
   - 否则用修改时间最新的 `auto_latest_*.json`。
3. 从配置中读取 `project.work_dir`，取里面最新的 `.dvp` 模型。
4. 从配置中读取 `project.dataset_dir`，作为测试图片目录。

### 推理服务要求

测试模式需要本地 DLCV 推理服务已经启动，默认地址为：

```text
http://127.0.0.1:9890
```

如果服务在其他地址，用 `--inference-api-url` 指定。

---

## 推理报告格式

报告是一个 JSON 文件，主要结构如下：

```json
{
  "new_model_path": ".../2-CLS-xxx.dvp",
  "baseline_model_path": ".../CLS-baseline_xxx.dvp",
  "image_dir": ".../2-CLS/训练集",
  "summary": {
    "total_images": 264,
    "new_success_count": 264,
    "baseline_success_count": 264,
    "match_count": 264,
    "mismatch_count": 0,
    "new_failed_images": [],
    "baseline_failed_images": []
  },
  "mismatched_images": [],
  "per_image_results": [
    {
      "image": "IMG_xxx.jpg",
      "new_model": {
        "success": true,
        "result": { "results": [ { "score": 0.99470669, "category_name": "王不留行" } ] },
        "error": null
      },
      "baseline_model": {
        "success": true,
        "result": { ... },
        "error": null,
        "match": true
      }
    }
  ]
}
```

### 结果对比规则

默认对比两张模型结果中置信度最高的 top-1 项：

1. `category_name` 不同 → 不一致。
2. `category_name` 相同但 `score` 差距 > 0.05 → 不一致。
3. 其他情况 → 一致。

如果结果结构不是列表或 top-1 项不可比较，则回退到 JSON 字符串归一化对比。

如需更严格的对比逻辑，可以修改 `autotrain4/evaluator.py` 中的 `_compare_results` 方法。

---

## 匹配策略

当用户输入包含具体类别时，4.0 会按以下策略匹配数据集：

1. **任务类型过滤**：只找对应任务文件夹（`cls` 只匹配分类，不会误匹配 `detr`）。
2. **LLM 选择数据集**：由 LLM 在候选文件夹中选出最匹配的一项。
3. **LLM 类别对齐**：将用户类别与数据集类别进行对应，覆盖中英文、同义词、缩写等。
4. **置信度检查**：LLM 返回的置信度低于 `--auto-threshold`（默认 0.55）时会拒绝执行并提示。

如果用户指定的类别和数据集类别不匹配，系统会**拒绝执行**并提示。

---

## LLM 配置

AutoTrain 4.0 **依赖 LLM** 完成意图解析、数据集匹配、需求判断和训练顾问，**必须配置可用的 LLM API**。

默认直接接入 DeepSeek（已内置默认 key，也可通过环境变量覆盖），或任意 OpenAI 兼容接口：

```bash
# 通用 OpenAI 兼容接口
export AUTOTRAIN_LLM_BASE_URL="https://api.openai.com/v1"
export AUTOTRAIN_LLM_API_KEY="sk-xxx"
export AUTOTRAIN_LLM_MODEL="gpt-3.5-turbo"

# 或仅覆盖 DeepSeek key
export DEEPSEEK_API_KEY="sk-xxx"
```

支持任意 OpenAI 兼容接口（如 Kimi、通义千问、本地 vLLM 等）。

---

## 第三层需求判断

在意图解析和数据集匹配之后，AutoTrain 4.0 会调用 DeepSeek 进行最终需求评审：

1. 判断需求是否明确、可行。
2. 判断选中的数据集是否真正满足用户意图。
3. 输出是否继续执行训练的最终决策和必要修正建议。

若 DeepSeek 判断不可继续，流程会终止并返回理由。

---

## 项目结构

```text
auto_train4.0.py              # 4.0 主入口
autotrain4/
  ├── common.py               # 任务类型映射、常量、默认 WebBridge 配置
  ├── nlu.py                  # 自然语言意图解析（含 test 意图、数据增强解析）
  ├── resolver.py             # LLM 驱动的数据集匹配与类别对齐
  ├── config_builder.py       # 配置生成、超参 tuning、LLM 训练顾问
  ├── judge.py                # DeepSeek 第三层需求判断
  ├── llm_client.py           # LLM 客户端（OpenAI 兼容 / DeepSeek）
  ├── inference_client.py     # DLCV 推理接口客户端
  └── evaluator.py            # 推理结果评估与对比报告生成
```

---

## 平台 exe 路径

AutoTrain 4.0 会按以下优先级自动探测 DLCV 训练平台可执行文件：

1. 环境变量 `DLCV_TRAIN_EXE`
2. 模板配置中的 `platform.executable_path`
3. 常见安装路径：
   - `C:/dlcv/Lib/site-packages/dlcv_train`
   - `C:/Program Files/dlcv_train`
   - `C:/Program Files (x86)/dlcv_train`

支持的文件名：

- `DLCV AI Training Platform-win_x64.exe`
- `DLCV AI Training Platform-win_64.exe`
- `DLCV AI Training Platform.exe`

---

## 示例

```bash
# 中文类别名
python auto_train4.0.py --prompt "训练一个分类模型识别王不留行和白芷"

# 英文类别名（数据集为中文时也能语义匹配）
python auto_train4.0.py --prompt "train a cls model to classify wangbuliu and baizhi"

# DETR
python auto_train4.0.py --prompt "训练一个 DETR 模型检测作弊行为"

# 不指定类别，只指定任务
python auto_train4.0.py --prompt "训练一个目标检测模型" --folder 3-DET

# 指定数据增强
python auto_train4.0.py --prompt "训练一个分类模型，使用随机翻转和随机裁剪"

# 训练完立刻测试
python auto_train4.0.py --prompt "训练一个 cls 模型来辨别王不留行和白芷" --test-image-dir "AI训练平台测试/2-CLS/训练集"

# 只测试最新模型
python auto_train4.0.py --prompt "测试刚刚训练的分类模型"

# 测试并对比 baseline
python auto_train4.0.py --prompt "测试刚刚训练的分类模型" --baseline-model "Baseline Models/CLS-baseline_20260630_154101.dvp"
```

---

## 注意事项

- **LLM 是必需的**：未配置 LLM 时，4.0 无法解析意图和匹配数据集，会直接报错。
- 首次调用 LLM 需要联网。
- Windows 终端建议使用 UTF-8 编码：执行 `chcp 65001` 或设置 `PYTHONIOENCODING=utf-8`。
- 4.0 会覆盖 `config/auto_latest_{任务类型}.json`，如需保留历史配置请用 `--config-out`。
- 测试模式复用训练集作为测试集，主要用于快速验证模型可用性和与 baseline 的一致性；如需客观评估泛化能力，建议用独立的测试集并通过 `--test-image-dir` 指定。
- 训练完成后浏览器窗口默认保持打开，按 Enter 关闭；自动化场景可使用 `--no-keep-browser-open`。
