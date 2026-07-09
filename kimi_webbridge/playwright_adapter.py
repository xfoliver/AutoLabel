"""
基于 Playwright 的 WebBridge 适配器。

不依赖 Kimi WebBridge daemon，直接用 Python 启动 Chromium/Edge 浏览器，
通过 DOM 选择器完成 DLCV AI 训练平台训练参数页的填写、保存并开始训练。

适用场景：
- Kimi WebBridge daemon 不稳定或无法连接
- 希望更直接地控制浏览器、截图、重试
- 前端 dev server 端口不固定，需要稳定兜底

配置示例：
    {
      "webbridge": {
        "enabled": true,
        "adapter": "playwright",
        "frontend_url": "http://localhost:8843",
        "headless": false,
        "browser": "chromium",
        "slow_mo": 300,
        "step_delay": 1.0,
        "screenshot_dir": "./webbridge_screenshots",
        "timeout": 120
      }
    }
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright

from .base import BaseWebBridge, WebBridgeError, WebBridgeResult


class PlaywrightWebBridgeAdapter(BaseWebBridge):
    """基于 Playwright 的真实浏览器适配器。"""

    HP_CONTROL_MAP: Dict[str, Dict[str, Any]] = {
        "train_set_val": {"label": "训练集：", "type": "text", "suffix": "%"},
        "val_set_val": {"label": "验证集：", "type": "text", "suffix": "%"},
        "copy_times": {"label": "数据倍增", "type": "number"},
        "max_epochs": {"label": "训练代数", "type": "number"},
        "batch_size": {"label": "批量大小", "type": "number"},
        "lr": {"label": "学习率", "type": "number"},
        "weight_decay": {"label": "权重衰减", "type": "number"},
        "distill_ratio": {"label": "蒸馏比例（OCR）", "type": "number"},
        "val_interval": {"label": "评估间隔", "type": "number"},
        "train_num_workers": {"label": "训练Dataloader进程数", "type": "number"},
        "val_num_workers": {"label": "验证Dataloader进程数", "type": "number"},
        "random_seed": {"label": "随机种子", "type": "number"},
        "mask_factor": {"label": "Mask放大系数", "type": "number"},
        "enable_fp16": {"label": "启用AMP混合精度", "type": "checkbox"},
        "save_error_data": {"label": "保存过漏检数据", "type": "checkbox"},
        "auto_deploy": {"label": "自动加速模型(.dvt)", "type": "checkbox"},
        "keep_consistency": {"label": "保持一致性", "type": "checkbox"},
        "use_ok_images": {"label": "使用OK图训练", "type": "checkbox"},
        "use_thread_dataloader": {"label": "使用线程读取数据", "type": "checkbox"},
    }

    # 数据增强面板里可能出现的选项（用于归一化和日志）
    DATA_AUGMENTATION_OPTIONS = [
        "调整图像大小",
        "随机比例缩放",
        "随机翻转",
        "随机固定角度旋转",
        "随机裁剪",
        "随机调整亮度",
        "随机调整对比度",
        "随机JPG编码",
    ]

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        """统一解析配置中的布尔值，兼容字符串 'True'/'False'。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.browser_type = config.get("browser", "chromium").lower()
        self.headless = config.get("headless", False)
        self.slow_mo = int(config.get("slow_mo", 300))
        self.step_delay = float(config.get("step_delay", 1.0))
        self.wait_loading_done = bool(config.get("wait_loading_done", False))
        self.post_click_delay = float(config.get("post_click_delay", 2.0))
        self.launch_args = config.get("launch_args", ["--disable-blink-features=AutomationControlled"])
        self.video_dir = config.get("video_dir")
        self.video_size = config.get("video_size", {"width": 1280, "height": 720})
        self.record_frontend = self._parse_bool(config.get("record_frontend", True))
        self.record_dir = config.get("record_dir", "./record")
        self.record_name = config.get("record_name")
        self.record_size = config.get("record_size", {"width": 1280, "height": 720})
        Path(self.screenshot_dir).mkdir(parents=True, exist_ok=True)
        self._playwright = None
        self._browser = None
        self._context = None
        self._page: Optional[Page] = None
        self._record_saved_path: Optional[str] = None

    # ------------------------------------------------------------------
    # BaseWebBridge 接口
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """启动 Playwright 浏览器。"""
        try:
            self._playwright = sync_playwright().start()
            browser_cls = getattr(self._playwright, self.browser_type, None)
            if browser_cls is None:
                raise WebBridgeError(f"不支持的浏览器类型: {self.browser_type}")
            self._browser = browser_cls.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
                args=self.launch_args,
            )

            context_options: Dict[str, Any] = {}
            if self.video_dir:
                video_dir = Path(self.video_dir).resolve()
                video_dir.mkdir(parents=True, exist_ok=True)
                context_options["record_video_dir"] = str(video_dir)
                context_options["record_video_size"] = self.video_size

            if self.record_frontend:
                record_dir = Path(self.record_dir).resolve()
                record_dir.mkdir(parents=True, exist_ok=True)
                context_options["record_video_dir"] = str(record_dir)
                context_options["record_video_size"] = self.record_size
                print(f"[PlaywrightWebBridge] 前端录屏已启用，保存目录: {record_dir}")

            self._context = self._browser.new_context(**context_options)
            self.session_id = f"playwright-{int(time.time())}"
        except Exception as e:
            raise WebBridgeError(f"Playwright 启动浏览器失败: {e}")

    def navigate(self, url: str) -> WebBridgeResult:
        if not self._context:
            self.connect()
        try:
            if self._page and not self._page.is_closed():
                self._page.close()
            self._page = self._context.new_page()
            self._page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
            return WebBridgeResult(success=True, message=f"navigated to {url}")
        except PlaywrightTimeoutError as e:
            raise WebBridgeError(f"页面加载超时: {e}")
        except Exception as e:
            raise WebBridgeError(f"导航失败: {e}")

    def get_current_url(self) -> str:
        """返回当前页面 URL；若页面未打开则返回空字符串。"""
        try:
            if self._page and not self._page.is_closed():
                return self._page.url
        except Exception:
            pass
        return ""

    def execute_task(self, task_description: str) -> WebBridgeResult:
        """兼容接口：将自然语言描述解析为超参后应用。"""
        hp = self._parse_task_prompt(task_description)
        return self.apply_hyperparameters(hp)

    def click(self, element_description: str) -> WebBridgeResult:
        self._ensure_page()
        try:
            self._page.get_by_role("button", name=element_description).click(timeout=10000)
            self._sleep_step()
            return WebBridgeResult(success=True)
        except Exception as e:
            raise WebBridgeError(f"点击失败: {e}")

    def fill(self, element_description: str, value: str) -> WebBridgeResult:
        self._ensure_page()
        try:
            self._set_input_by_label(element_description, value)
            self._sleep_step()
            return WebBridgeResult(success=True)
        except Exception as e:
            raise WebBridgeError(f"填充失败: {e}")

    def screenshot(self, name: str = "screenshot") -> WebBridgeResult:
        self._ensure_page()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        save_path = str(Path(self.screenshot_dir).resolve() / filename)
        try:
            self._page.screenshot(path=save_path, full_page=True)
            return WebBridgeResult(success=True, screenshot_path=save_path)
        except Exception as e:
            raise WebBridgeError(f"截图失败: {e}")

    def get_page_state(self) -> WebBridgeResult:
        self._ensure_page()
        try:
            return WebBridgeResult(
                success=True,
                data={
                    "url": self._page.url,
                    "title": self._page.title(),
                },
            )
        except Exception as e:
            raise WebBridgeError(f"获取页面状态失败: {e}")

    def close(self) -> None:
        video = None
        try:
            if self._page:
                try:
                    video = self._page.video
                except Exception:
                    pass
                self._page.close()
                self._page = None
            if self._context:
                self._context.close()
                self._context = None
            if self._browser:
                self._browser.close()
                self._browser = None
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
        except Exception as e:
            print(f"关闭 Playwright 时出错: {e}")

        # Playwright 的视频路径要在页面/上下文关闭后才可用。
        # 重命名则必须在 context / browser / playwright 全部关闭后进行，
        # 否则 Windows 上 Playwright 仍持有视频文件句柄，会导致 WinError 32。
        video_path = None
        if video:
            try:
                video_path = video.path()
            except Exception:
                pass

        if video_path and self.record_frontend:
            self._save_recording(video_path)
        elif video_path:
            print(f"[PlaywrightWebBridge] 页面录制视频: {video_path}")

        self.session_id = None

    def _save_recording(self, video_path: str) -> Optional[str]:
        """将 Playwright 自动命名的录屏文件重命名为 模型+训练结束时间 并保存到 record 目录。"""
        if not self.record_dir:
            return None
        try:
            import shutil

            src = Path(video_path)
            # Playwright 视频在 context 关闭后通常立即可用，但在 Windows 上偶尔需要多等一会儿
            for _ in range(50):
                if src.exists() and src.stat().st_size > 0:
                    break
                time.sleep(0.1)
            if not src.exists():
                return None

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            name = self.record_name or "autotrain"
            safe_name = re.sub(r'[\\/:*?"<>|]+', "_", str(name)).strip() or "autotrain"
            filename = f"{safe_name}_{timestamp}.webm"
            target = Path(self.record_dir).resolve() / filename
            target.parent.mkdir(parents=True, exist_ok=True)

            try:
                src.rename(target)
            except OSError:
                # Windows 文件仍被占用时使用复制+删除兜底
                shutil.copy2(str(src), str(target))
                src.unlink()

            self._record_saved_path = str(target)
            print(f"[PlaywrightWebBridge] 前端录屏已保存: {target}")
            return self._record_saved_path
        except Exception as e:
            print(f"[PlaywrightWebBridge] 保存录屏失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 完整训练流程：数据分析 → 数据增强 → 训练参数 → 开始训练
    # ------------------------------------------------------------------
    def run_training_workflow(
        self,
        ui_flow: Dict[str, Any],
        hyperparameters: Dict[str, Any],
    ) -> WebBridgeResult:
        """
        从项目信息页开始，按菜单顺序完成：
        数据分析 → 数据增强（如需要）→ 训练参数 → 开始训练。
        """
        self._ensure_page()

        # 1. 数据分析
        if ui_flow.get("data_analysis", True):
            self._log_step("点击侧边菜单：数据分析")
            self._click_sidebar("数据分析")
            self._sleep_step(2.0)

            self._log_step("点击：加载数据集")
            self._click_text("加载数据集")
            self._wait_loading_done(timeout=60)
            self._sleep_step(1.0)

            self._log_step("等待并点击：图像分布旁边的加载尺寸分布")
            self._wait_for_text("图像分布", timeout=30)
            self._click_near_anchor("加载尺寸分布", "图像分布")
            self._wait_loading_done(timeout=60)
            self._sleep_step(1.0)

        # 2. 数据增强
        aug_options = ui_flow.get("data_augmentation", [])
        if aug_options:
            self._log_step("点击侧边菜单：数据增强")
            self._click_sidebar("数据增强")
            self._sleep_step(2.0)

            self._log_step("点击页面上的数据增强按钮，打开增强策略弹窗")
            self._click_page_button("数据增强")
            self._sleep_step(2.0)

            # 等待增强策略弹窗渲染出来（以第一个选项为标志）
            first_opt = self._normalize_augmentation_name(aug_options[0])
            if first_opt:
                self._wait_for_text(first_opt, timeout=10)

            for opt in aug_options:
                normalized = self._normalize_augmentation_name(opt)
                if not normalized:
                    self._log_step(f"跳过未识别的增强选项: {opt}")
                    continue
                self._log_step(f"勾选数据增强：{normalized}")
                self._check_augmentation_option(normalized)
                self._sleep_step(0.8)

            self._log_step("点击数据增强弹窗确认/保存")
            # 可能先弹确认窗，再弹"数据增强操作已修改,是否保存"，需要连续处理
            for _ in range(3):
                try:
                    self._click_dialog_confirm(["保存", "确定", "确认", "是", "Yes", "OK"])
                except WebBridgeError:
                    break
                self._sleep_step(1.0)
                # 检查是否还有弹窗
                try:
                    has_dialog = self._page.evaluate(
                        """() => !!document.querySelector('.el-message-box, .ant-modal, [role=dialog], .modal, .dialog')"""
                    )
                    if not has_dialog:
                        break
                except Exception:
                    break
            self._sleep_step(1.5)

            # 选择完数据增强后，点击页面右下角的保存按钮（会触发二次确认弹窗）
            self._log_step("点击数据增强页面右下角保存按钮")
            self._click_page_button("保存")
            self._sleep_step(2.0)

            self._log_step("处理保存后的确认弹窗")
            self._handle_confirm_dialog()
            self._sleep_step(1.0)

        # 3. 训练参数
        self._log_step("点击侧边菜单：训练参数")
        self._click_sidebar("训练参数")
        self._ensure_parameter_page()

        # 4. 设置超参并保存（不点开始训练，由下一步统一点）
        self.apply_hyperparameters(hyperparameters, start_training=False)

        # 5. 开始训练
        self._log_step("点击：开始训练")
        self._click_button("开始训练")
        self._sleep_step(2.0)
        self._handle_confirm_dialog()
        self._sleep_step(1.0)

        return WebBridgeResult(success=True, message="完整训练流程已执行完毕")

    # ------------------------------------------------------------------
    # 训练参数自动化
    # ------------------------------------------------------------------
    def apply_hyperparameters(
        self,
        hyperparameters: Dict[str, Any],
        start_training: bool = True,
    ) -> WebBridgeResult:
        """在训练参数页批量设置超参并触发训练。"""
        if not hyperparameters:
            return WebBridgeResult(success=True, message="没有超参需要设置")

        self._ensure_page()
        self._wait_for_parameter_page()
        self._expand_sections()
        self._sleep_step(1.0)

        skipped: List[str] = []
        for key, value in hyperparameters.items():
            if key == "img_scale":
                skipped.append(f"{key}={value}")
                continue
            if key not in self.HP_CONTROL_MAP:
                skipped.append(key)
                continue

            meta = self.HP_CONTROL_MAP[key]
            try:
                display_value = self._format_value(value, meta)
                self._log_step(f"设置 {meta['label']} = {display_value}")
                self._set_field(meta["label"], display_value, meta["type"])
                self._sleep_step()
            except WebBridgeError as e:
                err_msg = str(e)
                # 页面中不存在该控件时仅跳过，避免不同任务类型字段不一致导致整体失败
                if any(err_msg.startswith(prefix) for prefix in ("label not found", "input not found", "checkbox not found")):
                    skipped.append(f"{key}({meta['label']})")
                    continue
                return WebBridgeResult(success=False, message=f"设置 {key} 失败: {e}")
            except Exception as e:
                return WebBridgeResult(success=False, message=f"设置 {key} 失败: {e}")

        if skipped:
            print(f"[PlaywrightWebBridge] 以下参数在当前页面未找到对应控件，已跳过: {skipped}")

        # 保存
        self._log_step("点击：保存")
        self._click_button("保存")
        self._sleep_step(2.0)

        if start_training:
            # 开始训练
            self._log_step("点击：开始训练")
            self._click_button("开始训练")
            self._sleep_step(2.0)

            # 处理可能出现的确认弹窗
            self._handle_confirm_dialog()
            self._sleep_step(1.0)

        return WebBridgeResult(success=True, message="训练参数已设置并保存")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _ensure_page(self) -> None:
        if self._page is None or self._page.is_closed():
            raise WebBridgeError("Playwright 页面未初始化，请先调用 navigate()")

    def _wait_for_parameter_page(self, timeout: int = 60) -> None:
        """等待训练参数页真正加载：URL 包含 parameter_edit 且页面出现“批量大小"。"""
        start = time.time()
        last_log = start
        while time.time() - start < timeout:
            try:
                url_ok = "parameter_edit" in self._page.url
                has_batch = self._page.locator("text=批量大小").count() > 0
                if url_ok and has_batch:
                    return
            except Exception:
                pass
            now = time.time()
            if now - last_log >= 10:
                print(f"[PlaywrightWebBridge] 等待训练参数页... 已等待 {int(now - start)}s, url={self._page.url if self._page else 'none'}")
                last_log = now
            time.sleep(0.5)
        raise WebBridgeError("等待训练参数页超时")

    def _expand_sections(self) -> None:
        """点击各参数面板标题，确保展开。"""
        sections = ["数据参数", "训练参数", "评估参数", "运行参数", "部署参数"]
        for section in sections:
            try:
                locator = self._page.locator(f"text={section}")
                if locator.count() > 0 and locator.first.is_visible():
                    locator.first.click(timeout=3000)
                    time.sleep(0.2)
            except Exception:
                pass

    def _format_value(self, value: Any, meta: Dict[str, Any]) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return repr(value)
        return str(value)

    def _set_field(self, label: str, value: str, field_type: str = "text") -> None:
        """根据中文标签定位并设置控件。"""
        if field_type == "checkbox":
            self._set_checkbox_by_label(label, value.lower() == "true")
        else:
            self._set_input_by_label(label, value)

    def _set_input_by_label(self, label: str, value: str) -> None:
        """通过 label 文本找到关联 input 并填入值。"""
        # 先尝试 Playwright 内置的 get_by_label（支持 for 属性和 label 包裹 input）
        try:
            input_locator = self._page.get_by_label(label, exact=False)
            if input_locator.count() > 0:
                input_locator.first.scroll_into_view_if_needed(timeout=5000)
                input_locator.first.fill(value, timeout=5000)
                input_locator.first.blur()
                return
        except Exception:
            pass

        # 兜底：用 JS 在 DOM 中根据 label 文本查找 input
        js_code = """
        (args) => {
            const labelText = args[0];
            const value = args[1];
            const labels = Array.from(document.querySelectorAll('*'));
            const labelEl = labels.find(el => Array.from(el.childNodes).some(n => n.nodeType === 3 && n.textContent.trim() === labelText));
            if (!labelEl) return {error: 'label not found: ' + labelText};
            let input = null;
            if (labelEl.htmlFor) {
                input = document.getElementById(labelEl.htmlFor);
            }
            if (!input || input.tagName !== 'INPUT') {
                let container = labelEl;
                for (let i = 0; i < 6; i++) {
                    input = container.querySelector('input');
                    if (input) break;
                    container = container.parentElement;
                    if (!container) break;
                }
            }
            if (!input) return {error: 'input not found for: ' + labelText};
            input.scrollIntoView({block: 'center'});
            input.focus();
            input.select();
            const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
            if (nativeSetter) nativeSetter.call(input, value);
            else input.value = value;
            ['input', 'change', 'blur'].forEach(e => input.dispatchEvent(new Event(e, {bubbles: true})));
            return {ok: true, value: input.value};
        }
        """
        result = self._page.evaluate(js_code, [label, value])
        if isinstance(result, dict) and result.get("ok"):
            return
        raise WebBridgeError(result.get("error") or f"设置 {label} 失败")

    def _set_checkbox_by_label(self, label: str, checked: bool) -> None:
        """通过 label 文本找到关联 checkbox 并勾选/取消。"""
        try:
            locator = self._page.get_by_label(label, exact=False).locator('input[type="checkbox"]')
            if locator.count() > 0:
                locator.first.scroll_into_view_if_needed(timeout=5000)
                if checked:
                    locator.first.check(timeout=5000)
                else:
                    locator.first.uncheck(timeout=5000)
                return
        except Exception:
            pass

        js_code = """
        (args) => {
            const labelText = args[0];
            const checked = args[1];
            const labels = Array.from(document.querySelectorAll('*'));
            const labelEl = labels.find(el => Array.from(el.childNodes).some(n => n.nodeType === 3 && n.textContent.trim() === labelText));
            if (!labelEl) return {error: 'label not found: ' + labelText};
            let cb = null;
            if (labelEl.htmlFor) cb = document.getElementById(labelEl.htmlFor);
            if (!cb || cb.type !== 'checkbox') {
                let container = labelEl.parentElement;
                for (let i = 0; i < 6; i++) {
                    cb = container.querySelector('input[type="checkbox"]');
                    if (cb) break;
                    container = container.parentElement;
                    if (!container) break;
                }
            }
            if (!cb) return {error: 'checkbox not found for: ' + labelText};
            cb.scrollIntoView({block: 'center'});
            cb.checked = checked;
            ['change', 'click'].forEach(e => cb.dispatchEvent(new Event(e, {bubbles: true})));
            return {ok: true};
        }
        """
        result = self._page.evaluate(js_code, [label, checked])
        if isinstance(result, dict) and result.get("ok"):
            return
        raise WebBridgeError(result.get("error") or f"设置 {label} 失败")

    def _click_button(self, text: str) -> None:
        """点击指定文本的按钮。"""
        try:
            self._page.get_by_role("button", name=text).click(timeout=10000)
            return
        except Exception:
            pass

        # 兜底：DOM 按文本点击
        js_code = f"""
        () => {{
            const text = {repr(text)};
            let el = Array.from(document.querySelectorAll('button, [role="button"]')).find(e => e.textContent.trim() === text || e.getAttribute('title') === text);
            if (!el) el = Array.from(document.querySelectorAll('*')).find(e => e.textContent.trim() === text || e.getAttribute('title') === text);
            if (!el) return {{error: 'button not found: ' + text}};
            let target = el;
            while (target && target.tagName !== 'BUTTON' && target.getAttribute('role') !== 'button' && target.parentElement) target = target.parentElement;
            target.click();
            return {{ok: true}};
        }}
        """
        result = self._page.evaluate(js_code)
        if isinstance(result, dict) and result.get("ok"):
            return
        raise WebBridgeError(result.get("error") or f"点击 {text} 失败")

    def _handle_confirm_dialog(self) -> None:
        """仅当检测到 dialog 时才点击确认。"""
        self._sleep_step(0.5)
        try:
            has_dialog = self._page.evaluate(
                """() => !!document.querySelector('.el-message-box, .ant-modal, [role=dialog], .modal, .dialog')"""
            )
            if not has_dialog:
                return
        except Exception:
            return

        try:
            self._click_dialog_confirm(["保存", "确定", "确认", "是", "Yes", "OK"])
            self._sleep_step(0.3)
        except WebBridgeError:
            pass

    def _click_dialog_confirm(self, candidates: Optional[List[str]] = None) -> None:
        """
        在弹出的 dialog 里点击确认按钮。
        优先匹配候选文字，匹配不到则点 dialog 里最后一个按钮（通常是确认）。
        """
        keywords = candidates or ["保存", "确定", "确认", "是", "Yes", "OK", "应用"]
        js_code = f"""
        () => {{
            const keywords = {json.dumps(keywords, ensure_ascii=False)};
            const dialogs = document.querySelectorAll('.el-message-box, .ant-modal, [role=dialog], .modal, .dialog');
            if (!dialogs.length) return {{error: 'no dialog found'}};
            const dialog = dialogs[dialogs.length - 1];
            let buttons = Array.from(dialog.querySelectorAll('button, [role="button"], .el-button, .ant-btn'));
            if (!buttons.length) return {{error: 'no buttons in dialog'}};

            for (const kw of keywords) {{
                const btn = buttons.find(b => b.textContent.trim().includes(kw));
                if (btn) {{
                    btn.scrollIntoView({{block: 'center'}});
                    btn.click();
                    return {{ok: true, text: btn.textContent.trim()}};
                }}
            }}

            // 兜底：点最后一个按钮（通常右边是确认）
            const last = buttons[buttons.length - 1];
            last.scrollIntoView({{block: 'center'}});
            last.click();
            return {{ok: true, text: last.textContent.trim(), fallback: true}};
        }}
        """
        result = self._page.evaluate(js_code)
        if isinstance(result, dict) and result.get("ok"):
            self._log_step(f"点击弹窗按钮: {result.get('text')}")
            return
        raise WebBridgeError(result.get("error") or "点击弹窗确认按钮失败")

    def _sleep_step(self, seconds: Optional[float] = None) -> None:
        """每步之间停顿，便于肉眼观察。"""
        time.sleep(seconds if seconds is not None else self.step_delay)

    def _log_step(self, message: str) -> None:
        """打印当前步骤，配合停顿让流程可追溯。"""
        print(f"[PlaywrightWebBridge] {message}")

    def _wait_loading_done(self, timeout: int = 30) -> None:
        """
        控制加载后的等待行为。
        默认不等待 loading 结束（wait_loading_done=false），只固定停顿 post_click_delay 秒，
        然后直接继续下一步。如需严格等待 spinner 消失，可在配置里设 wait_loading_done=true。
        """
        if not self.wait_loading_done:
            self._sleep_step(self.post_click_delay)
            return

        start = time.time()
        while time.time() - start < timeout:
            try:
                has_loading = self._page.evaluate(
                    """() => {
                        const selectors = [
                            '.el-loading-spinner',
                            '.el-loading-mask',
                            '.ant-spin',
                            '.loading',
                            '.spinner',
                            '[class*="loading"]',
                            '[class*="spin"]'
                        ];
                        return selectors.some(s => !!document.querySelector(s));
                    }"""
                )
                if not has_loading:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        print("[PlaywrightWebBridge] 等待 loading 消失超时，继续执行")

    def _click_text(self, text: str) -> None:
        """点击页面上文本匹配的元素（侧边栏菜单、普通按钮均可）。"""
        try:
            locator = self._page.get_by_text(text, exact=True)
            if locator.count() > 0:
                locator.first.click(timeout=10000)
                return
        except Exception:
            pass

        try:
            locator = self._page.get_by_text(text, exact=False)
            if locator.count() > 0:
                locator.first.click(timeout=10000)
                return
        except Exception:
            pass

        # JS 兜底
        js_code = f"""
        () => {{
            const text = {repr(text)};
            let el = Array.from(document.querySelectorAll('*')).find(e => e.textContent.trim() === text || e.getAttribute('title') === text);
            if (!el) return {{error: 'text not found: ' + text}};
            el.scrollIntoView({{block: 'center'}});
            el.click();
            return {{ok: true}};
        }}
        """
        result = self._page.evaluate(js_code)
        if isinstance(result, dict) and result.get("ok"):
            return
        raise WebBridgeError(result.get("error") or f"点击 {text} 失败")

    def _click_near_anchor(self, target: str, anchor: str) -> None:
        """在 anchor 文字所在的区域里点击 target 文字对应的元素。"""
        js_code = f"""
        () => {{
            const anchorText = {repr(anchor)};
            const targetText = {repr(target)};
            const all = Array.from(document.querySelectorAll('*'));
            const anchorEl = all.find(el => Array.from(el.childNodes).some(n => n.nodeType === 3 && n.textContent.trim() === anchorText));
            if (!anchorEl) return {{error: 'anchor not found: ' + anchorText}};

            let container = anchorEl.parentElement;
            let targetEl = null;
            for (let i = 0; i < 10; i++) {{
                targetEl = Array.from(container.querySelectorAll('*')).find(
                    el => Array.from(el.childNodes).some(n => n.nodeType === 3 && n.textContent.trim() === targetText)
                );
                if (targetEl) break;
                container = container.parentElement;
                if (!container) break;
            }}
            if (!targetEl) return {{error: 'target not found near anchor: ' + targetText}};
            targetEl.scrollIntoView({{block: 'center'}});
            targetEl.click();
            return {{ok: true}};
        }}
        """
        result = self._page.evaluate(js_code)
        if isinstance(result, dict) and result.get("ok"):
            return
        raise WebBridgeError(result.get("error") or f"点击 {target}（锚点 {anchor}）失败")

    def _click_page_button(self, text: str) -> None:
        """
        点击主内容区里的按钮（排除侧边栏 aside / nav）。
        用于区分侧边栏菜单和页面内同名的按钮，例如「数据增强」。
        """
        js_code = f"""
        () => {{
            const text = {repr(text)};
            // 1. 优先找 button 元素
            let candidates = Array.from(document.querySelectorAll('button, [role="button"]'))
                .filter(el => el.textContent.trim() === text || el.getAttribute('title') === text);
            // 2. 排除侧边栏里的
            candidates = candidates.filter(el => !el.closest('aside') && !el.closest('nav') && !el.closest('[class*="menu"]'));
            // 3. 没有 button 再找主内容区任意元素
            if (candidates.length === 0) {{
                candidates = Array.from(document.querySelectorAll('main, [class*="content"], [class*="panel"] *'))
                    .filter(el => el.textContent.trim() === text);
            }}
            if (candidates.length === 0) return {{error: 'page button not found: ' + text}};
            // 点最后一个（通常页面主按钮在菜单项之后渲染）
            const target = candidates[candidates.length - 1];
            target.scrollIntoView({{block: 'center'}});
            target.click();
            return {{ok: true, count: candidates.length}};
        }}
        """
        result = self._page.evaluate(js_code)
        if isinstance(result, dict) and result.get("ok"):
            return
        raise WebBridgeError(result.get("error") or f"点击页面按钮 {text} 失败")

    def _set_checkbox_by_text(self, text: str, checked: bool) -> None:
        """通过包含文本定位 checkbox（支持数据增强面板中的选项）。"""
        try:
            locator = self._page.locator(
                f'input[type="checkbox"]'
            ).filter(has=self._page.locator(f"text={text}"))
            if locator.count() == 0:
                locator = self._page.get_by_text(text, exact=False).locator('xpath=..').locator('input[type="checkbox"]')
            if locator.count() > 0:
                locator.first.scroll_into_view_if_needed(timeout=5000)
                if checked:
                    locator.first.check(timeout=5000)
                else:
                    locator.first.uncheck(timeout=5000)
                return
        except Exception:
            pass

        # JS 兜底：找文本包含 text 的最近 checkbox
        js_code = f"""
        () => {{
            const text = {repr(text)};
            const all = Array.from(document.querySelectorAll('*'));
            const el = all.find(e => Array.from(e.childNodes).some(n => n.nodeType === 3 && n.textContent.trim().includes(text)));
            if (!el) return {{error: 'text not found: ' + text}};
            let cb = el.querySelector('input[type="checkbox"]');
            if (!cb) {{
                let container = el.parentElement;
                for (let i = 0; i < 8; i++) {{
                    cb = container.querySelector('input[type="checkbox"]');
                    if (cb) break;
                    container = container.parentElement;
                    if (!container) break;
                }}
            }}
            if (!cb) return {{error: 'checkbox not found for: ' + text}};
            cb.scrollIntoView({{block: 'center'}});
            cb.checked = {str(checked).lower()};
            ['change', 'click'].forEach(e => cb.dispatchEvent(new Event(e, {{bubbles: true}})));
            return {{ok: true}};
        }}
        """
        result = self._page.evaluate(js_code)
        if isinstance(result, dict) and result.get("ok"):
            return
        raise WebBridgeError(result.get("error") or f"设置 {text} 失败")

    def _is_text_visible(self, text: str) -> bool:
        """判断页面上是否存在指定文字。"""
        try:
            return self._page.evaluate(
                f"() => document.body.innerText.includes({repr(text)})"
            )
        except Exception:
            return False

    def _wait_for_text(self, text: str, timeout: int = 10) -> None:
        """等待页面上出现指定文字。"""
        start = time.time()
        while time.time() - start < timeout:
            if self._is_text_visible(text):
                return
            time.sleep(0.3)
        raise WebBridgeError(f"等待文字出现超时: {text}")

    def _click_sidebar(self, text: str) -> None:
        """点击侧边栏菜单项，优先滚动到可见区域再点。"""
        try:
            # 侧边栏通常是 aside / nav / menu 里的文字
            locator = self._page.locator("aside, nav, [class*='menu'], [class*='sidebar']").get_by_text(text, exact=False)
            if locator.count() > 0:
                locator.first.scroll_into_view_if_needed(timeout=5000)
                locator.first.click(timeout=10000)
                return
        except Exception:
            pass
        # 兜底用通用文本点击
        self._click_text(text)

    def _ensure_parameter_page(self, timeout: int = 30) -> None:
        """确保当前已经在训练参数页，否则点击侧边栏训练参数并等待。"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                url_ok = "parameter_edit" in self._page.url
                has_batch = self._page.locator("text=批量大小").count() > 0
                if url_ok and has_batch:
                    return
            except Exception:
                pass
            # 还没到参数页，尝试点击训练参数
            try:
                self._click_sidebar("训练参数")
            except WebBridgeError:
                pass
            time.sleep(1.0)
        raise WebBridgeError("未能进入训练参数页")

    def _check_augmentation_option(self, option: str, checked: bool = True) -> None:
        """
        在数据增强弹窗中勾选/取消某个增强策略。
        兼容 Element UI 的自定义 checkbox（小正方形）以及原生 checkbox。
        若当前弹窗没有该选项，则跳过并记录警告，避免任务中断。
        """
        if not self._is_text_visible(option):
            print(f"[PlaywrightWebBridge] 数据增强弹窗中未找到选项 '{option}'，跳过")
            return

        js_code = f"""
        (args) => {{
            const text = args[0];
            const wantChecked = args[1];
            const all = Array.from(document.querySelectorAll('*'));
            const labelEl = all.find(el => Array.from(el.childNodes).some(n => n.nodeType === 3 && n.textContent.trim().includes(text)));
            if (!labelEl) return {{error: 'text not found: ' + text}};

            // 1. 优先找 Element UI 风格的 el-checkbox
            let cb = labelEl.closest('.el-checkbox');
            if (!cb) {{
                let container = labelEl.parentElement;
                for (let i = 0; i < 8; i++) {{
                    const found = container.querySelector('.el-checkbox');
                    if (found) {{ cb = found; break; }}
                    container = container.parentElement;
                    if (!container) break;
                }}
            }}
            if (cb) {{
                cb.scrollIntoView({{block: 'center'}});
                const isChecked = cb.classList.contains('is-checked') ||
                                  (cb.querySelector('input[type="checkbox"]') && cb.querySelector('input[type="checkbox"]').checked);
                if (Boolean(isChecked) !== Boolean(wantChecked)) {{
                    const target = cb.querySelector('.el-checkbox__inner') || cb.querySelector('input[type="checkbox"]') || cb;
                    target.click();
                }}
                return {{ok: true, type: 'el-checkbox'}};
            }}

            // 2. 兜底：原生 checkbox
            let input = labelEl.querySelector('input[type="checkbox"]');
            if (!input) {{
                let container = labelEl.parentElement;
                for (let i = 0; i < 8; i++) {{
                    input = container.querySelector('input[type="checkbox"]');
                    if (input) break;
                    container = container.parentElement;
                    if (!container) break;
                }}
            }}
            if (!input) return {{error: 'checkbox not found for: ' + text}};
            input.scrollIntoView({{block: 'center'}});
            if (input.checked !== wantChecked) {{
                input.click();
                ['change', 'click'].forEach(e => input.dispatchEvent(new Event(e, {{bubbles: true}})));
            }}
            return {{ok: true, type: 'native'}};
        }}
        """
        result = self._page.evaluate(js_code, [option, checked])
        if isinstance(result, dict) and result.get("ok"):
            return
        raise WebBridgeError(result.get("error") or f"勾选 {option} 失败")

    def _normalize_augmentation_name(self, raw: str) -> Optional[str]:
        """把用户/LLM 返回的增强名归一化为标准选项。"""
        text = str(raw).strip().replace("随即", "随机")
        for option in self.DATA_AUGMENTATION_OPTIONS:
            if option.lower() in text.lower() or text.lower() in option.lower():
                return option
        return None

    # ------------------------------------------------------------------
    # 自然语言提示解析（execute_task 兼容用）
    # ------------------------------------------------------------------
    _PARSE_PATTERNS = {
        re.compile(r"设置「(.+?)」为\s*(.+?)。"): "set",
        re.compile(r"勾选「(.+?)」"): "check_true",
        re.compile(r"取消勾选「(.+?)」"): "check_false",
        re.compile(r"设置图片缩放尺寸为宽\s*(\d+).*高\s*(\d+)"): "img_scale",
    }

    _LABEL_TO_KEY: Dict[str, str] = {
        "训练集比例": "train_set_val",
        "验证集比例": "val_set_val",
        "数据倍增": "copy_times",
        "训练代数": "max_epochs",
        "批量大小": "batch_size",
        "学习率": "lr",
        "权重衰减": "weight_decay",
        "评估间隔": "val_interval",
        "蒸馏比例": "distill_ratio",
        "Mask放大系数": "mask_factor",
        "随机种子": "random_seed",
        "训练数据读取线程数": "train_num_workers",
        "验证数据读取线程数": "val_num_workers",
        "启用AMP混合精度": "enable_fp16",
        "保存过漏检数据": "save_error_data",
        "保持一致性": "keep_consistency",
        "自动加速模型(.dvt)": "auto_deploy",
        "使用OK图训练": "use_ok_images",
        "使用线程读取数据": "use_thread_dataloader",
    }

    def _parse_task_prompt(self, prompt: str) -> Dict[str, Any]:
        hp: Dict[str, Any] = {}
        for line in prompt.splitlines():
            for pat, action in self._PARSE_PATTERNS.items():
                m = pat.search(line)
                if not m:
                    continue
                if action == "img_scale":
                    hp["img_scale"] = {"width": int(m.group(1)), "height": int(m.group(2))}
                elif action in ("check_true", "check_false"):
                    key = self._LABEL_TO_KEY.get(m.group(1))
                    if key:
                        hp[key] = action == "check_true"
                else:
                    key = self._LABEL_TO_KEY.get(m.group(1))
                    if key:
                        val = m.group(2).strip()
                        if val.endswith("%"):
                            val = val[:-1]
                        try:
                            if "." in val:
                                val = float(val)
                            else:
                                val = int(val)
                        except ValueError:
                            pass
                        hp[key] = val
        return hp
