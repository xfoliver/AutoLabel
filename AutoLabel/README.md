# AutoLabel（自动标注 MVP）

与 `autotrain4/` 平级的 **实例分割（ISEG）自动标注** 工具：自然语言 / 表单 → 能力校验 → 计划确认 → 试标 3～5 张 → 全量 AI mask → 确定性几何 → LabelMeAI 旁路 JSON → 复核清单 → Commit。

设计依据：仓库根目录 `自动化标注.md`。

## 快速开始

```bash
# 依赖（可与 pyqt_frontend 共用环境）
pip install -r AutoLabel/requirements.txt

# 单测
python -m pytest AutoLabel/tests -q

# CLI（MockBackend，无需 GPU）
python AutoLabel/cli.py --dataset <图片目录> --label 王不留行 --description "圆粒药材" --preview
python AutoLabel/cli.py --dataset <图片目录> --label 王不留行 --preview --batch --yes
python AutoLabel/cli.py --dataset <图片目录> --label 王不留行 --preview --batch --yes --commit

# PyQt 前端
python AutoLabel/main.py
```

## 双通道配置（LLM + Vision）

AutoLabel 使用 **两套独立配置**，不要混用 Key 字段：

| 通道 | 配置位置 | 用途 |
|------|----------|------|
| **LLM** | `config/settings.json` 的 `llm_*` | 自然语言意图 / LabelSpec |
| **Vision** | `config/backends.json` 的 active backend | 出 instance mask / 彩色 Mask |

`config/settings.json` 与 `config/backends.json` **已 gitignore**（只提交 `*.example.json`）。把 API Key 写在本地文件即可；若 Key 曾出现在聊天记录中，外传前请轮换。

示例（NewAPI 网关）：

- LLM：`llm_base_url=http://192.168.1.3:3000/v1`，`llm_model=kimi-k3`
- Vision：`gpt_image_mask` → `gpt-image-2`，调用 `/v1/images/edits`，本地解析彩色 Mask

运行时不依赖 CC Switch。

## 换模型 / API

编辑 `AutoLabel/config/backends.json`（可从 `backends.example.json` 复制）：

| id | type | 说明 |
|----|------|------|
| `mock` | `mock` | 本地合成 instance mask，开发默认 |
| `http_seg` | `http_api` | 任意返回 instance mask 的 HTTP API（`base_url` + Bearer Key） |
| `gpt_image_mask` | `openai_image_edits` | OpenAI 兼容 `/images/edits`（如 gpt-image-2）：彩色 Mask → 本地拆实例 |
| `dlcv_9890` | `dlcv` | 本地 `http://127.0.0.1:9890` InferenceClient |
| `bad_vlm` | `no_instance` | 负例：Capability Check 会拒绝 |

设置 `active_backend` 即可切换。

### gpt-image-2 Mask 约定（MVP）

- 请求：multipart `model` + `prompt` + `image` → `POST {base_url}/images/edits`
- 响应：`data[0].b64_json` 黑底彩色 PNG；**每个实例一种纯色**（当前按单类处理）
- 本地：量化颜色 → `InstancePred` → 现有 Stage2 polygon / LabelMeAI
- 无 per-instance confidence → 结果进抽检/复核，不做高置信自动接受
- API 返回尺寸可能与请求 `size` 不一致；用 stretch `TransformMeta` 映回原图
- 贴边可调（`backends.json` → `gpt_image_mask.options`）：`grabcut`（默认 true，原图 GrabCut 贴边）、`grabcut_iters` / `grabcut_pad_px` / `grabcut_sure_fg_erode` / `grabcut_sure_bg_dilate`、`quantize`（默认 8）、`epsilon_ratio`（默认 0.0005）、`mask_close_ksize` / `mask_open_ksize`（默认 0，避免肥边）、`prompt_extra`；更大 `size` 通常更贴边但更贵

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
  autolabel/          # 领域库
  ui/                 # PyQt
  config/backends.json
  tests/ fixtures/
```

## 非目标（本版）

CLS/DET/OCR/SSEG 真实实现、像素级复核编辑器、Stage1 直接吐 polygon、多类别色表图例、用图像模型做样本生成。
