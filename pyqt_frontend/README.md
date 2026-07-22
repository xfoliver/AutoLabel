# AutoTrain 4.0 — DLCV AI 训练自动化平台

AutoTrain 4.0 是一个面向 DLCV（深度学习计算机视觉）的 AI 训练自动化平台。用户只需用自然语言描述训练需求，后台即可自动完成数据集匹配、超参决策、UI 自动化配置、模型训练监控与后续评估报告生成。

---

## 目录

- [项目简介](#项目简介)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
  - [环境准备](#环境准备)
  - [启动 Web 前端](#启动-web-前端)
  - [启动 PyQt 前端](#启动-pyqt-前端)
- [核心功能](#核心功能)
- [前端说明](#前端说明)
  - [Web 前端](#web-前端)
  - [PyQt 前端](#pyqt-前端)
- [后端说明](#后端说明)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 项目简介

AutoTrain 4.0 提供两种人机交互方式：

1. **Web 前端**：基于 Vue 3 + Vite 的现代化浏览器控制台，适合远程访问和团队协作。
2. **PyQt 前端**：基于 PyQt5 的本地桌面应用，适合单机直接使用，对中文输入法更友好。

无论使用哪种前端，底层都调用 `auto_train4.0.py` 完成实际的训练流程，并通过 SSE 实时回传日志与状态。

---

## 目录结构

```
kimi_gui_automation/
├── auto_train4.0.py              # 核心训练自动化入口
├── auto_train2.0.py              # WebBridge 自动化入口（被 4.0 调用）
├── autotrain4/                   # 核心库：LLM 客户端、决策、配置生成、评估等
│   ├── llm_client.py
│   ├── config_builder.py
│   ├── evaluator.py
│   ├── judge.py
│   ├── nlu.py
│   └── ...
├── web_frontend/                 # Web 前端（Vue 3 + Flask）
│   ├── app.py                    # Flask 后端入口
│   ├── package.json
│   ├── vite.config.js
│   └── src/                      # Vue 组件、视图、状态管理、API 封装
├── pyqt_frontend/                # PyQt 桌面前端
│   ├── main.py                   # PyQt 主程序
│   ├── backend.py                # 训练工作线程
│   ├── suggestions.py            # 智能建议引擎
│   └── quick_templates.json      # 用户保存的快捷模板（运行时生成）
├── kimi_webbridge/               # WebBridge 适配器（Playwright / CDP）
│   ├── base.py
│   ├── playwright_adapter.py
│   └── ...
├── config/                       # 训练配置归档
├── AI训练平台测试/               # 测试数据集
├── Baseline Models/              # Baseline 模型文件
├── record/                       # 训练录屏与成功记录
├── failed_runs/                  # 失败运行记录
├── probe_playwright_attach.py    # WebView2 / Playwright attach 探针
├── probe_webview2_cdp.py         # WebView2 CDP 端口探测探针
└── README.md                     # 本文件
```

---

## 快速开始

### 环境准备

1. 安装 Python 3.11+，并确保已安装项目依赖：

```bash
pip install -r requirements.txt
```

2. 如果需要使用 PyQt 前端，额外安装 PyQt5：

```bash
pip install PyQt5
```

3. 配置环境变量（参考 `.env.example`）：

```bash
# Kimi / Moonshot API Key（用于 LLM 决策与建议）
MOONSHOT_API_KEY=your_key_here
```

### 启动 Web 前端

生产模式：

```bash
cd web_frontend
npm install
npm run build
python app.py
```

浏览器访问 `http://127.0.0.1:5000`。

开发模式（前后端分离，热更新）：

```bash
# 终端 1：启动 Flask 后端
cd web_frontend
python app.py

# 终端 2：启动 Vue 开发服务器
cd web_frontend
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`，API 请求会自动代理到 `http://127.0.0.1:5000`。

### 启动 PyQt 前端

```bash
python pyqt_frontend/main.py
```

---

## 核心功能

- **自然语言驱动训练**：输入如「训练一个 OCR 模型，不使用数据增强」，系统自动解析意图。
- **数据集智能匹配**：根据任务类型与数据特征，匹配 `AI训练平台测试/` 下的最佳数据集。
- **超参自动决策**：由 LLM 顾问推荐 Epoch、Batch Size、Learning Rate 等关键参数。
- **UI 自动化配置**：通过 Kimi WebBridge + Playwright 自动操控训练平台前端页面。
- **训练实时监控**：SSE 流式回传日志，实时显示训练进度、阶段状态与预计剩余时间。
- **快捷模板**：保存常用 prompt 为模板，一键填充。
- **录屏与归档**：每次运行自动录屏，成功/失败归档到 `record/` 或 `failed_runs/`。
- **训练后动作**：支持生成推理报告、对比 Baseline 模型等。

---

## 前端说明

### Web 前端

技术栈：Vue 3 + Vite + Pinia + Vue Router + Flask。

页面：

| 页面 | 路径 | 说明 |
|------|------|------|
| 工作台 | `/` | prompt 输入、运行、日志、进度、指标 |
| 训练任务 | `/tasks` | 历史任务列表与状态筛选 |
| 数据集 | `/datasets` | 扫描展示 `AI训练平台测试/` 数据集 |
| 模型中心 | `/models` | Baseline 与最新训练产物 |
| 系统设置 | `/settings` | 后端地址、CDP 端口、默认 Adapter |

API 列表：

| 路由 | 说明 |
|------|------|
| `GET /api/health` | 健康检查 |
| `GET /api/run?prompt=...` | SSE 启动训练 |
| `GET /api/tasks` | 最近任务 |
| `GET /api/datasets` | 数据集列表 |
| `GET /api/models` | 模型列表 |
| `GET /api/templates` | 快捷模板 |

### PyQt 前端

技术栈：PyQt5 + Python 标准库。

主要界面：

- 左侧边栏：品牌、导航、平台状态
- 顶部栏：标题、查看文档
- 训练需求卡片：prompt 输入、建议气泡、开始训练、指标概览
- 运行日志卡片：SSE 实时日志、阶段折叠、状态 badge
- 快捷模板卡片：可点击填充、保存为模板、管理（查看/删除）
- 本次任务卡片：状态同步、关闭浏览器

用户自定义模板保存在 `pyqt_frontend/quick_templates.json`，最多保存 8 个。

---

## 后端说明

核心入口 `auto_train4.0.py`：

```bash
python auto_train4.0.py --prompt "训练一个 OCR 模型"
```

它会依次执行：

1. 自然语言意图解析（NLU）
2. 任务规划与数据集匹配
3. LLM 超参决策
4. 生成训练配置
5. 启动后端服务
6. 调用 `auto_train2.0.py` 进行 WebBridge UI 自动化
7. 监控训练进度
8. 生成报告或后续动作

### WebBridge 适配器

`kimi_webbridge/` 提供多种 adapter：

- `playwright`：默认，直接操控浏览器访问 `http://localhost:8842`
- `webview2_cdp`：连接 WebView2 壳端 CDP 调试口（实验性）
- `stub` / `skill` / `http`：测试或 fallback 用

---

## 配置说明

环境变量（`.env`）：

| 变量 | 说明 |
|------|------|
| `MOONSHOT_API_KEY` | Kimi / Moonshot API Key |
| `DLCV_PLATFORM_EXE` | WebView2 壳端 EXE 路径（可选） |
| `DLCV_CDP_PORT` | CDP 调试端口，默认 9222（可选） |

Web 前端配置：修改 `web_frontend/src/api/index.js` 中的 `BASE` 或 Vite proxy。

PyQt 前端配置：通过界面「系统设置」页面临时填写，当前仅本地生效。

---

## 常见问题

### Q：PyQt 前端保存模板时提示 `name 'json' is not defined`
A：确保使用的是最新代码，已修复 `json` 导入问题。清除缓存后重启：

```bash
rd /s /q pyqt_frontend\__pycache__
python pyqt_frontend/main.py
```

### Q：PyQt 前端中文输入法不显示拼音
A：已改用 `QPlainTextEdit` 并调整调色板，大部分 Windows 输入法可正常显示。如仍有问题，可尝试切换输入法或反馈具体输入法版本。

### Q：Web 前端运行后页面空白
A：确保已执行 `npm run build` 生成 `dist/` 目录，且 Flask 的 `static_folder` 指向 `dist/assets`。

### Q：如何切换到壳端（WebView2）自动化？
A：目前 Web 前端和 PyQt 前端默认仍走网页端 `playwright`。要支持壳端需要：

1. 给 `kimi_webbridge` 新增 `webview2_cdp` adapter
2. 在启动流程中注入 `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS` 并启动 EXE
3. 用 `connect_over_cdp` attach 到目标 page

可参考 `probe_webview2_cdp.py` 和 `probe_playwright_attach.py` 两个探针脚本。

---

## 许可证

本项目为内部工具，未经授权不得商用或二次分发。
