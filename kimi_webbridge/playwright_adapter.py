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
        "slow_mo": 50,
        "screenshot_dir": "./webbridge_screenshots",
        "timeout": 120
      }
    }
"""

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

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.browser_type = config.get("browser", "chromium").lower()
        self.headless = config.get("headless", False)
        self.slow_mo = int(config.get("slow_mo", 50))
        self.launch_args = config.get("launch_args", ["--disable-blink-features=AutomationControlled"])
        Path(self.screenshot_dir).mkdir(parents=True, exist_ok=True)
        self._playwright = None
        self._browser = None
        self._page: Optional[Page] = None

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
            self.session_id = f"playwright-{int(time.time())}"
        except Exception as e:
            raise WebBridgeError(f"Playwright 启动浏览器失败: {e}")

    def navigate(self, url: str) -> WebBridgeResult:
        if not self._browser:
            self.connect()
        try:
            self._page = self._browser.new_page()
            self._page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
            return WebBridgeResult(success=True, message=f"navigated to {url}")
        except PlaywrightTimeoutError as e:
            raise WebBridgeError(f"页面加载超时: {e}")
        except Exception as e:
            raise WebBridgeError(f"导航失败: {e}")

    def execute_task(self, task_description: str) -> WebBridgeResult:
        """兼容接口：将自然语言描述解析为超参后应用。"""
        hp = self._parse_task_prompt(task_description)
        return self.apply_hyperparameters(hp)

    def click(self, element_description: str) -> WebBridgeResult:
        self._ensure_page()
        try:
            self._page.get_by_role("button", name=element_description).click(timeout=10000)
            return WebBridgeResult(success=True)
        except Exception as e:
            raise WebBridgeError(f"点击失败: {e}")

    def fill(self, element_description: str, value: str) -> WebBridgeResult:
        self._ensure_page()
        try:
            self._set_input_by_label(element_description, value)
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
        try:
            if self._page:
                self._page.close()
                self._page = None
            if self._browser:
                self._browser.close()
                self._browser = None
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
        except Exception as e:
            print(f"关闭 Playwright 时出错: {e}")
        self.session_id = None

    # ------------------------------------------------------------------
    # 训练参数自动化
    # ------------------------------------------------------------------
    def apply_hyperparameters(self, hyperparameters: Dict[str, Any]) -> WebBridgeResult:
        """在训练参数页批量设置超参并触发训练。"""
        if not hyperparameters:
            return WebBridgeResult(success=True, message="没有超参需要设置")

        self._ensure_page()
        self._wait_for_parameter_page()
        self._expand_sections()

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
                self._set_field(meta["label"], display_value, meta["type"])
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
        self._click_button("保存")
        time.sleep(1.0)

        # 开始训练
        self._click_button("开始训练")
        time.sleep(2.0)

        # 处理可能出现的确认弹窗
        self._handle_confirm_dialog()

        # 再等待一下，让后端创建 training session
        time.sleep(1.0)

        return WebBridgeResult(success=True, message="训练参数已设置并开始训练")

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _ensure_page(self) -> None:
        if self._page is None or self._page.is_closed():
            raise WebBridgeError("Playwright 页面未初始化，请先调用 navigate()")

    def _wait_for_parameter_page(self, timeout: int = 60) -> None:
        start = time.time()
        last_log = start
        while time.time() - start < timeout:
            try:
                # 页面存在“开始训练”或“批量大小”即认为参数页已加载
                self._page.wait_for_selector("text=开始训练", timeout=500)
                return
            except PlaywrightTimeoutError:
                pass
            try:
                self._page.wait_for_selector("text=批量大小", timeout=500)
                return
            except PlaywrightTimeoutError:
                pass
            now = time.time()
            if now - last_log >= 10:
                print(f"[PlaywrightWebBridge] 等待训练参数页... 已等待 {int(now - start)}s")
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
        time.sleep(0.5)
        try:
            has_dialog = self._page.evaluate(
                """() => !!document.querySelector('.el-message-box, .ant-modal, [role=dialog], .modal, .dialog')"""
            )
            if not has_dialog:
                return
        except Exception:
            return

        for confirm_text in ["确定", "确认", "是", "Yes", "OK"]:
            try:
                self._click_button(confirm_text)
                time.sleep(0.3)
                return
            except WebBridgeError:
                continue

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
