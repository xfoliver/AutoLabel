# AutoLabel（自动标注 MVP）

与 `autotrain4/` 平级的 **实例分割（ISEG）自动标注** 工具：自然语言 / 表单 → 能力校验 → 计划确认 → 试标 3～5 张 → 全量 AI mask → 确定性几何 → LabelMeAI 旁路 JSON → 复核清单 → Commit。

设计依据：仓库根目录 `自动化标注.md`。

默认视觉后端是 **SAM3 local worker**（独立 Conda 环境 + HTTP）。旧的 `gpt_image_mask` 仍保留，可随时切回。

## 快速开始

```bash
# 依赖（可与 pyqt_frontend 共用环境）
pip install -r AutoLabel/requirements.txt

# 单测
python -m pytest AutoLabel/tests -q

# CLI（MockBackend，无需 GPU）
python AutoLabel/cli.py --dataset <图片目录> --label 王不留行 --description "圆粒药材" --preview --backend mock
python AutoLabel/cli.py --dataset <图片目录> --label 王不留行 --preview --batch --yes --backend mock

# CLI（SAM3：需 sam3 Conda + Worker）
python AutoLabel/cli.py --dataset <图片目录> --label 手机 --intent-mode simple --preview --backend sam3_local

# PyQt 前端
python AutoLabel/main.py
```

## SAM3 架构（默认）

SAM3 官方仓库 `sam3/` **禁止修改**。AutoLabel 主进程也 **禁止 import sam3 / torch**。

```
用户
↓
AutoLabel UI / CLI
↓
Intent（LLM 可选）→ label=手机, model_prompt=phone
↓
sam3_local backend  (HTTP only)
↓
127.0.0.1:9892  SAM3 Worker   ← C:\dlcv\envs\sam3\python.exe
↓
SAM3  binary masks + boxes + scores
↓
score / area / NMS
↓
findContours → approxPolyDP → LabelMe JSON → Preview
```

Worker 模型只加载一次并常驻。启动 AutoLabel 后：

1. 探测 `http://127.0.0.1:9892/health`
2. 若未运行，用 `C:\dlcv\envs\sam3\python.exe AutoLabel\sam3_worker\server.py` 拉起
3. 等 `model_loaded=true` 再推理

单独验证（Stage A / B）：

```bash
# Stage A 单张
C:\dlcv\envs\sam3\python.exe AutoLabel\sam3_worker\run_once.py --image D:\data\001.jpg --prompt phone

# Stage B 常驻
C:\dlcv\envs\sam3\python.exe AutoLabel\sam3_worker\server.py --host 127.0.0.1 --port 9892
```

切 backend：编辑 `AutoLabel/config/backends.json` 的 `active_backend`：

| 值 | 说明 |
|----|------|
| `sam3_local` | 默认。HTTP → SAM3 Worker |
| `gpt_image_mask` | 旧 GPT Image 彩色 mask 路径 |
| `mock` | 本地合成，开发 / 单测 |

## 双通道配置（LLM + Vision）

AutoLabel 使用 **两套独立配置**，不要混用 Key 字段：

| 通道 | 配置位置 | 用途 |
|------|----------|------|
| **LLM** | `config/settings.json` 的 `llm_*` | 自然语言意图 / LabelSpec（可选） |
| **Vision** | `config/backends.json` 的 active backend | 出 instance mask |

`intent_mode`（settings.json）：

- `auto`：短词（phone / 手机）跳过 LLM；长句走 LLM，失败则回退 simple
- `simple`：永远不调用 LLM
- `llm`：必须走 LLM

`config/settings.json` 与 `config/backends.json` **已 gitignore**（只提交 `*.example.json`）。

## 换模型 / API

编辑 `AutoLabel/config/backends.json`（可从 `backends.example.json` 复制）：

| id | type | 说明 |
|----|------|------|
| `sam3_local` | `sam3_local` | 本地 SAM3 Worker（默认） |
| `mock` | `mock` | 本地合成 instance mask |
| `http_seg` | `http_api` | 任意返回 instance mask 的 HTTP API |
| `gpt_image_mask` | `openai_image_edits` | OpenAI 兼容 `/images/edits`：彩色 Mask → 本地拆实例 |
| `dlcv_9890` | `dlcv` | 本地 `http://127.0.0.1:9890` InferenceClient |
| `bad_vlm` | `no_instance` | 负例：Capability Check 会拒绝 |

### gpt-image-2 Mask 约定（旧路径，仍可用）

- 请求：multipart `model` + `prompt` + `image` → `POST {base_url}/images/edits`
- 响应：`data[0].b64_json` 黑底彩色 PNG；**每个实例一种纯色**（当前按单类处理）
- 本地：量化颜色 → `InstancePred` → 现有 Stage2 polygon / LabelMeAI
- 无 per-instance confidence → 结果进抽检/复核，不做高置信自动接受
- API 返回尺寸可能与请求 `size` 不一致；用 stretch `TransformMeta` 映回原图
- 贴边可调（`backends.json` → `gpt_image_mask.options`）：`grabcut`（默认 true，原图 GrabCut 贴边）、`grabcut_iters` / `grabcut_pad_px` / `grabcut_sure_fg_erode` / `grabcut_sure_bg_dilate`、`quantize`（默认 8）、`epsilon_ratio`（默认 0.0005）、`mask_close_ksize` / `mask_open_ksize`（默认 0，避免肥边）；更大 `size` 通常更贴边但更贵
- **目标语义不要写进 backend config**。类别 / 描述 / 排除项走 UI 或 CLI 的 `label` / `description` / `exclude`（进入 LabelSpec）。`prompt_extra` 仅作可选空字段，默认不注入任何领域文本

### HTTP API 约定（可换厂商）

`POST {base_url}/v1/segment`

```json
{
  "model": "...",
  "image_base64": "...",
  "label_spec": { "class_names": ["..."], "targets": [...] }
}
```

响应需含 `instances[].mask_png_base64` 与建议的 `transform`（TransformMeta）。**谁做 resize/letterbox，谁就必须返回 TransformMeta**。

## 流程闸门

1. **Capability Check**：ISEG 要求 `instance_mask=true`
2. **Plan Preview**：烧算力前可改意图
3. **试标确认**：未确认不能全量
4. **Commit**：先写 `work/autolabel/<run_id>/`，确认后才旁路 `图片.json`；默认 **skip 已有 JSON**
5. 无 `model_confidence` 时不做「高置信自动接受」，全部进抽检/复核桶

## 目录

```text
AutoLabel/
  main.py cli.py
  autolabel/          # 领域库（含 inference/sam3_local.py HTTP 客户端）
  sam3_worker/        # 独立 SAM3 进程（必须用 sam3 Conda 启动）
  ui/                 # PyQt（顶栏 / 侧栏 / 新建任务 demo 布局）
  config/backends.json
  runs/sam3/          # Worker debug mask / overlay
  tests/ fixtures/
```

## 非目标（本版）

CLS/DET/OCR/SSEG 真实实现、像素级复核编辑器、Stage1 直接吐 polygon、多类别色表图例、用图像模型做样本生成。
