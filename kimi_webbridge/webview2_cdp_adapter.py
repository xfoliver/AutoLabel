"""
基于 WebView2 CDP 的 WebBridge 适配器。

通过 Playwright 的 connect_over_cdp 连接到已运行的 WebView2 壳端应用，
复用 PlaywrightWebBridgeAdapter 的 DOM 操作能力，实现壳端自动化。

录屏策略：
  WebView2 是 Playwright 外部创建的 context，无法使用内置 video。
  本适配器只录制壳端窗口（PrintWindow / BitBlt / ImageGrab），绝不回退到全桌面录制。
  窗口矩形用 DWM 扩展边框 + Per-Monitor DPI 物理像素，避免 150%/125% 缩放下只录到左上角。

  录屏可在 CDP attach 前启动（auto_train2.0 在平台启动后立刻调用 start_recording），
  从而覆盖从壳端启动到项目信息页的完整过程。

配置示例：
    {
      "webbridge": {
        "enabled": true,
        "adapter": "webview2_cdp",
        "cdp_port": 9222,
        "cdp_host": "127.0.0.1",
        "cdp_timeout": 90,
        "frontend_url": "http://localhost:8843",
        "screenshot_dir": "./webbridge_screenshots",
        "timeout": 120,
        "slow_mo": 300,
        "record_video_dir": "./record"
      },
      "platform": {
        "executable_path": "C:/path/to/DLCV AI Training Platform-win_x64.exe"
      }
    }
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from ctypes import wintypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from autotrain4.constants import DEFAULT_FRONTEND_URL
except ImportError:  # pragma: no cover
    DEFAULT_FRONTEND_URL = "http://localhost:8843"

import psutil

from playwright.sync_api import Browser, Page, sync_playwright

from .base import WebBridgeError, WebBridgeResult
from .playwright_adapter import PlaywrightWebBridgeAdapter


# ---------------------------------------------------------------------------
# 进程级 DPI 感知：必须在第一次 GetWindowRect / PrintWindow 之前设置，
# 否则在 125%/150% 缩放下拿到的是逻辑像素，录屏只覆盖左上角一块区域。
# ---------------------------------------------------------------------------
_DPI_AWARENESS_SET = False


def _ensure_process_dpi_awareness() -> None:
    """尽量把当前进程设为 Per-Monitor DPI Aware V2。"""
    global _DPI_AWARENESS_SET
    if _DPI_AWARENESS_SET:
        return
    _DPI_AWARENESS_SET = True
    if sys.platform != "win32":
        return
    try:
        # -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2（Win10 1703+）
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        # 2 = PROCESS_PER_MONITOR_DPI_AWARE（Win8.1+）
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class WebView2CDPWebBridgeAdapter(PlaywrightWebBridgeAdapter):
    """基于 WebView2 CDP attach 的适配器，复用 Playwright DOM 操作。"""

    # 壳端进程名候选（当 config 未带 platform.executable_path 时兜底）
    _SHELL_NAME_CANDIDATES = (
        "dlcv ai training platform-win_x64.exe",
        "dlcv ai training platform.exe",
        "dlcv ai training backend.exe",
    )

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        _ensure_process_dpi_awareness()
        self.cdp_port = int(config.get("cdp_port", 9222))
        self.cdp_host = config.get("cdp_host", "127.0.0.1")
        self.cdp_timeout = int(config.get("cdp_timeout", 90))
        self.frontend_url = config.get("frontend_url", DEFAULT_FRONTEND_URL)
        self.record_video_dir = config.get("record_video_dir")
        self.record_video_size = config.get(
            "record_video_size", {"width": 1920, "height": 1080}
        )
        self.video_path: Optional[str] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_log_fh: Optional[Any] = None
        self._ffmpeg_log_path: Optional[str] = None
        self._recording = False
        self._recording_thread: Optional[threading.Thread] = None
        self._record_rect: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h
        self._record_hwnd: Optional[int] = None
        Path(self.screenshot_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 公开录屏 API：可在 CDP attach 前启动，覆盖项目信息页之前的过程
    # ------------------------------------------------------------------
    def start_recording(self, timeout: float = 20.0) -> bool:
        """
        立刻开始壳端窗口录屏（可在 connect() 之前调用）。

        返回 True 表示录屏线程已启动；False 表示跳过（无 ffmpeg / 无窗口 / 未配置目录）。
        已在录制时幂等返回 True。
        """
        if self._recording and self._ffmpeg_proc is not None:
            return True
        self._start_ffmpeg_recording(timeout=timeout)
        return bool(self._recording and self._ffmpeg_proc is not None)

    def stop_recording(self) -> Optional[str]:
        """停止录屏并返回 video_path（可能为 None）。"""
        self._stop_ffmpeg_recording()
        return self.video_path

    # ------------------------------------------------------------------
    # 连接与关闭（重写父类）
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """通过 CDP attach 到已运行的 WebView2 壳端，并挑选业务 page。"""
        try:
            self._playwright = sync_playwright().start()
            cdp_url = f"http://{self.cdp_host}:{self.cdp_port}"
            self._wait_for_cdp(cdp_url)

            self._browser: Browser = self._playwright.chromium.connect_over_cdp(cdp_url)
            self._page = self._pick_business_page(self._browser)
            if self._page is None:
                raise WebBridgeError("CDP 中未找到业务页面")

            self._page.set_default_timeout(self.timeout * 1000)
            self.session_id = f"cdp-{self.cdp_port}-{id(self._page)}"
            self._log_step(
                f"已 attach 页面: url={self._page.url!r}, "
                f"title={self._page.title()!r}"
            )

            # 如果壳端启动后是空白页，先导航到前端 URL
            original_url = self._page.url
            if original_url in ("about:blank", "") and self.frontend_url:
                self._log_step(f"壳端当前为空白页，导航到 {self.frontend_url}")
                self._page.goto(self.frontend_url, wait_until="networkidle")

            # 等待页面内容出现，避免后续操作命中空白页
            try:
                self._page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass

            # 如果页面有 URL 但标题为空（白屏），尝试用原始 URL 重载一次
            try:
                title = self._page.title()
                if (not title or title.strip() in ("", "about:blank")) and original_url not in (
                    "about:blank",
                    "",
                ):
                    self._log_step(f"壳端白屏，尝试重载原始 URL: {original_url}")
                    self._page.goto(original_url, wait_until="networkidle")
                    self._page.wait_for_load_state("networkidle", timeout=15000)
            except Exception as e:
                self._log_step(f"重载原始 URL 失败: {e}")

            self._take_debug_screenshot("cdp_connect")

            # WebView2 是 Playwright 外部创建的 context，无法使用内置 video 录制；
            # 若上层已提前 start_recording()（覆盖项目信息页之前），这里幂等跳过。
            if not (self._recording and self._ffmpeg_proc is not None):
                self._start_ffmpeg_recording()
            else:
                self._log_step("录屏已在 connect 前启动，继续复用")
        except Exception as e:
            self.close()
            raise WebBridgeError(f"WebView2 CDP 连接失败: {e}")

    def close(self) -> None:
        """断开与 WebView2 的连接，停止录屏，不结束壳端 EXE 进程。"""
        self._stop_ffmpeg_recording()
        try:
            if self._browser:
                self._browser.disconnect()
        except Exception:
            pass
        self._browser = None
        self._context = None
        self._page = None
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def navigate(self, url: str) -> WebBridgeResult:
        """让当前 page 导航到指定 URL；如果已经在目标页则复用。"""
        if self._page is None:
            return WebBridgeResult(False, "尚未连接到 WebView2")
        try:
            # 先把窗口提到最前，避免 WebView2 在后台时渲染空白
            try:
                self._page.bring_to_front()
            except Exception:
                pass
            # 同步把原生窗口置前，便于录屏与点击
            self._bring_shell_to_foreground()

            current = self._page.url
            self._log_step(f"navigate: current={current!r}, target={url!r}")
            self._page.goto(url, wait_until="networkidle")

            # 额外等待页面内容渲染
            try:
                self._page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass

            # 若页面仍是空白，尝试 reload 一次
            try:
                title = self._page.title()
                if not title or title.strip() in ("", "about:blank"):
                    self._log_step("页面标题为空，尝试刷新")
                    self._page.reload(wait_until="networkidle")
                    self._page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass

            self._take_debug_screenshot("navigate")
            return WebBridgeResult(True, f"已导航到 {self._page.url}")
        except Exception as e:
            return WebBridgeResult(False, f"导航失败: {e}")

    # ------------------------------------------------------------------
    # 调试辅助方法
    # ------------------------------------------------------------------
    def _take_debug_screenshot(self, name: str) -> None:
        """保存调试截图到 screenshot_dir。"""
        if not self._page:
            return
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = Path(self.screenshot_dir) / f"{name}_{ts}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._page.screenshot(path=str(path), full_page=True)
            self._log_step(f"调试截图已保存: {path}")
        except Exception as e:
            self._log_step(f"调试截图保存失败: {e}")

    # ------------------------------------------------------------------
    # 录屏辅助方法（只录壳端窗口，绝不录桌面）
    # ------------------------------------------------------------------
    def _start_ffmpeg_recording(self, timeout: float = 20.0) -> None:
        """只录制壳端窗口：PrintWindow/BitBlt 抓帧 → ffmpeg rawvideo pipe。

        可在 CDP connect 之前调用（见 start_recording），以便从壳端启动起录。
        """
        if not self.record_video_dir:
            return
        if self._recording and self._ffmpeg_proc is not None:
            return

        _ensure_process_dpi_awareness()

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._log_step("未找到 ffmpeg，壳端录制将不可用")
            return

        # 等待壳端窗口出现；找不到就跳过，绝不退化为桌面录制
        rect = self._find_shell_window_rect(timeout=timeout)
        if not rect:
            self._log_step("未找到壳端窗口，跳过录屏（不录制桌面）")
            self.video_path = None
            return

        hwnd, x, y, w, h = rect
        # 偶数尺寸，兼容 yuv420p
        w = max(2, w - (w % 2))
        h = max(2, h - (h % 2))
        self._record_hwnd = hwnd
        self._record_rect = (x, y, w, h)
        self._log_step(
            f"录屏窗口矩形(物理像素): hwnd={hwnd} xy=({x},{y}) size={w}x{h}"
        )

        Path(self.record_video_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_path = str(
            Path(self.record_video_dir) / f"webview2_cdp_{timestamp}.mp4"
        )
        self._ffmpeg_log_path = str(
            Path(self.record_video_dir) / f"webview2_cdp_{timestamp}.log"
        )

        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            "10",
            "-i",
            "-",
            "-an",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            self.video_path,
        ]

        try:
            self._ffmpeg_log_fh = open(
                self._ffmpeg_log_path, "w", encoding="utf-8", errors="replace"
            )
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._ffmpeg_log_fh,
                bufsize=0,  # 原始帧立即写入，避免缓冲导致空文件
                creationflags=creationflags,
            )
            time.sleep(0.4)
            if self._ffmpeg_proc.poll() is not None:
                raise RuntimeError(
                    f"ffmpeg 启动后立即退出，日志见 {self._ffmpeg_log_path}"
                )

            self._recording = True
            self._recording_thread = threading.Thread(
                target=self._record_window_loop, daemon=True, name="ShellWindowRecorder"
            )
            self._recording_thread.start()
            self._log_step(f"已启动壳端窗口录屏 [{w}x{h} hwnd={hwnd}]: {self.video_path}")
        except Exception as e:
            self._log_step(f"启动 ffmpeg 窗口录屏失败: {e}")
            self._recording = False
            if self._recording_thread:
                self._recording_thread.join(timeout=2)
                self._recording_thread = None
            if self._ffmpeg_log_fh:
                try:
                    self._ffmpeg_log_fh.close()
                except Exception:
                    pass
                self._ffmpeg_log_fh = None
            if self._ffmpeg_proc:
                try:
                    self._ffmpeg_proc.kill()
                except Exception:
                    pass
            self._ffmpeg_proc = None
            self.video_path = None
            self._record_rect = None
            self._record_hwnd = None

    def _stop_ffmpeg_recording(self) -> None:
        """停止窗口帧捕获线程，关闭 ffmpeg stdin，等待生成 MP4。"""
        if not getattr(self, "_recording", False) and not self._ffmpeg_proc:
            return

        self._recording = False
        if self._recording_thread:
            self._recording_thread.join(timeout=12)
            self._recording_thread = None

        try:
            if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                try:
                    self._ffmpeg_proc.stdin.close()
                except Exception:
                    pass
            if self._ffmpeg_proc:
                try:
                    self._ffmpeg_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._log_step("ffmpeg 未在 10s 内退出，强制结束")
                    self._ffmpeg_proc.terminate()
                    try:
                        self._ffmpeg_proc.wait(timeout=3)
                    except Exception:
                        self._ffmpeg_proc.kill()
        except Exception as e:
            self._log_step(f"停止 ffmpeg 录屏时出错: {e}")
            if self._ffmpeg_proc:
                try:
                    self._ffmpeg_proc.kill()
                except Exception:
                    pass
        finally:
            if self._ffmpeg_log_fh:
                try:
                    self._ffmpeg_log_fh.close()
                except Exception:
                    pass
                self._ffmpeg_log_fh = None
            self._ffmpeg_proc = None
            self._record_rect = None
            self._record_hwnd = None
            if self.video_path and Path(self.video_path).exists():
                size = Path(self.video_path).stat().st_size
                # 空 mp4 壳（仅 ftyp/moov、无视频轨）通常只有几百字节，视为失败并删除
                if size > 4096:
                    self._log_step(f"壳端录屏已保存: {self.video_path} ({size} bytes)")
                else:
                    self._log_step(
                        f"警告: 壳端录屏无效（{size} bytes，可能 0 帧），已删除: {self.video_path}"
                    )
                    try:
                        Path(self.video_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                    self.video_path = None
            else:
                self.video_path = None

    def _record_window_loop(self) -> None:
        """后台线程：持续用 PrintWindow/BitBlt 捕获壳端窗口，写入 ffmpeg rawvideo。"""
        try:
            import numpy as np
            from PIL import Image

            if not self._record_rect:
                return
            out_w, out_h = self._record_rect[2], self._record_rect[3]
            frame_bytes = out_w * out_h * 3
            frame_count = 0
            fail_count = 0
            next_tick = time.perf_counter()

            while self._recording:
                next_tick += 0.1  # 10 fps
                frame = None
                try:
                    # 优先按 hwnd 抓；窗口移动时刷新 rect
                    hwnd = self._record_hwnd
                    rect = self._find_shell_window_rect(timeout=0)
                    if rect:
                        hwnd, x, y, w, h = rect
                        self._record_hwnd = hwnd
                        # 尺寸变了就 resize 到启动时的 out_w/out_h，保证 ffmpeg 输入稳定
                        frame = self._capture_hwnd_rgb(hwnd, out_w, out_h)
                    elif hwnd:
                        frame = self._capture_hwnd_rgb(hwnd, out_w, out_h)
                except Exception as e:
                    fail_count += 1
                    if fail_count <= 3 or fail_count % 50 == 0:
                        self._log_step(f"窗口捕获失败({fail_count}): {e}")

                if frame is not None:
                    try:
                        if frame.shape != (out_h, out_w, 3):
                            img = Image.fromarray(frame, mode="RGB")
                            img = img.resize((out_w, out_h), Image.Resampling.BILINEAR)
                            frame = np.asarray(img, dtype=np.uint8)
                        if not frame.flags["C_CONTIGUOUS"]:
                            frame = np.ascontiguousarray(frame)
                        payload = frame.tobytes()
                        if len(payload) != frame_bytes:
                            # 尺寸异常，丢帧
                            pass
                        elif self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                            self._ffmpeg_proc.stdin.write(payload)
                            # 定期 flush，降低异常中断时空文件概率
                            if frame_count % 10 == 0:
                                try:
                                    self._ffmpeg_proc.stdin.flush()
                                except Exception:
                                    pass
                            frame_count += 1
                            fail_count = 0
                    except BrokenPipeError:
                        self._log_step("ffmpeg pipe 已断开，停止写帧")
                        break
                    except Exception as e:
                        fail_count += 1
                        if fail_count <= 3 or fail_count % 50 == 0:
                            self._log_step(f"写帧失败({fail_count}): {e}")

                # 固定帧率
                sleep_for = next_tick - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # 落后太多就重置时钟，避免狂奔
                    next_tick = time.perf_counter()

            self._log_step(f"窗口帧捕获结束，共写入 {frame_count} 帧")
        except Exception as e:
            self._log_step(f"窗口录屏线程异常: {e}")

    def _get_window_bounds(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """
        取窗口物理像素边界 (left, top, right, bottom)。

        优先 DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS)：
        返回不含阴影的真实可见矩形，且始终是物理像素。
        回退 GetWindowRect。
        """
        import win32gui

        if not hwnd or not win32gui.IsWindow(hwnd):
            return None

        # 1) DWM 扩展边框（物理像素，无阴影）
        try:
            rect = RECT()
            # 9 = DWMWA_EXTENDED_FRAME_BOUNDS
            hr = ctypes.windll.dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(int(hwnd)),
                wintypes.DWORD(9),
                ctypes.byref(rect),
                ctypes.sizeof(rect),
            )
            if hr == 0:
                w = int(rect.right) - int(rect.left)
                h = int(rect.bottom) - int(rect.top)
                if w > 1 and h > 1:
                    return (
                        int(rect.left),
                        int(rect.top),
                        int(rect.right),
                        int(rect.bottom),
                    )
        except Exception:
            pass

        # 2) GetWindowRect（进程已 Per-Monitor DPI Aware 时应为物理像素）
        try:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            if right - left > 1 and bottom - top > 1:
                return int(left), int(top), int(right), int(bottom)
        except Exception:
            pass
        return None

    def _capture_hwnd_rgb(self, hwnd: int, out_w: int, out_h: int):
        """
        抓取壳端窗口 RGB 帧，顺序：
          1) ctypes PrintWindow(PW_RENDERFULLCONTENT)  — 兼容未导出 PrintWindow 的 pywin32
          2) BitBlt 窗口 DC
          3) PIL ImageGrab 仅裁剪该窗口矩形（不是全桌面）
        失败返回 None。

        位图尺寸与裁剪 bbox 均使用 _get_window_bounds 的物理像素，
        避免 125%/150% DPI 下只录到左上角。
        """
        import numpy as np
        import win32con
        import win32gui
        import win32ui
        from PIL import Image

        if not hwnd or not win32gui.IsWindow(hwnd):
            return None

        bounds = self._get_window_bounds(hwnd)
        if not bounds:
            return None
        left, top, right, bottom = bounds
        width = right - left
        height = bottom - top
        if width <= 1 or height <= 1:
            return None

        def _resize_rgb(rgb: np.ndarray) -> np.ndarray:
            h, w = rgb.shape[:2]
            if (w, h) != (out_w, out_h):
                img = Image.fromarray(rgb, mode="RGB")
                img = img.resize((out_w, out_h), Image.Resampling.BILINEAR)
                rgb = np.asarray(img, dtype=np.uint8)
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = np.ascontiguousarray(rgb)
            return rgb

        # --- 1/2: PrintWindow (ctypes) + BitBlt ---
        hwnd_dc = None
        mfc_dc = None
        save_dc = None
        bitmap = None
        try:
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            # 用物理像素尺寸建位图，确保覆盖整个可见窗口
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bitmap)

            hdc_dst = save_dc.GetSafeHdc()
            printed = False
            try:
                # pywin32 的 win32gui 常常没有 PrintWindow 属性，必须走 user32
                user32 = ctypes.windll.user32
                # BOOL PrintWindow(HWND hwnd, HDC hdcBlt, UINT nFlags);
                user32.PrintWindow.argtypes = [
                    wintypes.HWND,
                    wintypes.HDC,
                    wintypes.UINT,
                ]
                user32.PrintWindow.restype = wintypes.BOOL
                # PW_RENDERFULLCONTENT = 2，可抓 DWM/WebView2 合成内容
                printed = bool(user32.PrintWindow(int(hwnd), int(hdc_dst), 2))
                if not printed:
                    # 再试普通 PrintWindow
                    printed = bool(user32.PrintWindow(int(hwnd), int(hdc_dst), 0))
            except Exception:
                printed = False

            if not printed:
                # BitBlt 窗口自身 DC（不是桌面 gdigrab）
                save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

            bmpinfo = bitmap.GetInfo()
            bmpstr = bitmap.GetBitmapBits(True)
            stride_w = int(bmpinfo["bmWidth"])
            stride_h = int(bmpinfo["bmHeight"])
            arr = np.frombuffer(bmpstr, dtype=np.uint8)
            expected = stride_w * stride_h * 4
            if arr.size >= expected and stride_w > 1 and stride_h > 1:
                arr = arr[:expected].reshape((stride_h, stride_w, 4))
                # BGRA -> RGB
                rgb = arr[:, :, 2::-1].copy()
                # 全黑帧视为失败，走 ImageGrab 兜底
                if int(rgb.max()) > 0:
                    return _resize_rgb(rgb)
        except Exception:
            pass
        finally:
            try:
                if bitmap is not None:
                    win32gui.DeleteObject(bitmap.GetHandle())
            except Exception:
                pass
            try:
                if save_dc is not None:
                    save_dc.DeleteDC()
            except Exception:
                pass
            try:
                if mfc_dc is not None:
                    mfc_dc.DeleteDC()
            except Exception:
                pass
            try:
                if hwnd_dc is not None:
                    win32gui.ReleaseDC(hwnd, hwnd_dc)
            except Exception:
                pass

        # --- 3: 仅裁剪壳端窗口矩形（物理像素 bbox），不是全桌面录制 ---
        try:
            from PIL import ImageGrab

            try:
                img = ImageGrab.grab(
                    bbox=(left, top, right, bottom), all_screens=True
                )
            except TypeError:
                # 旧 Pillow 无 all_screens
                img = ImageGrab.grab(bbox=(left, top, right, bottom))
            if img is None:
                return None
            if img.mode != "RGB":
                img = img.convert("RGB")
            rgb = np.asarray(img, dtype=np.uint8)
            if rgb.size == 0 or int(rgb.max()) == 0:
                return None
            return _resize_rgb(rgb)
        except Exception:
            return None

    def _bring_shell_to_foreground(self) -> None:
        """尽量把壳端窗口置前（录屏/点击更稳）。"""
        try:
            import win32con
            import win32gui

            rect = self._find_shell_window_rect(timeout=0)
            if not rect:
                return
            hwnd = rect[0]
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def _iter_shell_pids(self) -> set:
        """收集壳端相关进程 PID。"""
        target_pids: set = set()
        platform_cfg = {}
        if isinstance(self.config, dict):
            platform_cfg = self.config.get("platform") or {}
        platform_exe = (platform_cfg.get("executable_path") or "").strip()
        exe_names = set(self._SHELL_NAME_CANDIDATES)
        if platform_exe:
            exe_names.add(Path(platform_exe).name.lower())
            platform_exe_l = platform_exe.lower()
        else:
            platform_exe_l = ""

        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                proc_name = (proc.info.get("name") or "").lower()
                proc_exe = (proc.info.get("exe") or "").lower()
                if proc_name in exe_names:
                    target_pids.add(proc.pid)
                    continue
                if platform_exe_l and platform_exe_l in proc_exe:
                    target_pids.add(proc.pid)
                    continue
                # 宽松匹配：名字里带 dlcv + platform
                if "dlcv" in proc_name and "platform" in proc_name:
                    target_pids.add(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return target_pids

    def _find_shell_window_rect(
        self, timeout: float = 0
    ) -> Optional[Tuple[int, int, int, int, int]]:
        """
        找到壳端主窗口：返回 (hwnd, x, y, w, h)。
        优先面积最大的可见顶层窗口。
        """
        try:
            import win32gui
            import win32process

            deadline = time.time() + max(0.0, float(timeout))
            while True:
                target_pids = self._iter_shell_pids()
                candidates: List[Tuple[int, int, int, int, int]] = []

                def enum_callback(hwnd, _):
                    if not win32gui.IsWindowVisible(hwnd):
                        return
                    try:
                        # 跳过 tool window
                        if win32gui.GetWindow(hwnd, 4):  # GW_OWNER
                            # 有 owner 的通常是子对话框；也允许，但后面按面积选最大
                            pass
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        if target_pids and pid not in target_pids:
                            return
                        # 若没识别到 pid，则用标题兜底
                        title = ""
                        try:
                            title = win32gui.GetWindowText(hwnd) or ""
                        except Exception:
                            title = ""
                        if not target_pids:
                            # 无 PID 时，只接受看起来像壳端的标题
                            t = title.lower()
                            if not any(
                                k in t
                                for k in (
                                    "dlcv",
                                    "training",
                                    "vite",
                                    "训练",
                                    "platform",
                                )
                            ):
                                return
                        # 用物理像素边界（DWM + DPI aware），避免 150% 缩放下只录左上角
                        bounds = self._get_window_bounds(hwnd)
                        if not bounds:
                            return
                        x, y, x2, y2 = bounds
                        w, h = x2 - x, y2 - y
                        # 物理像素阈值：150% 下逻辑 200x150 ≈ 物理 300x225
                        if w >= 300 and h >= 200:
                            candidates.append((hwnd, x, y, w, h))
                    except Exception:
                        return

                win32gui.EnumWindows(enum_callback, None)
                if candidates:
                    # 面积最大
                    best = max(candidates, key=lambda r: r[3] * r[4])
                    return best

                if time.time() >= deadline:
                    return None
                time.sleep(0.4)
        except Exception as e:
            self._log_step(f"查找壳端窗口失败: {e}")
            return None

    # ------------------------------------------------------------------
    # CDP 辅助方法
    # ------------------------------------------------------------------
    def _wait_for_cdp(self, cdp_url: str) -> None:
        """轮询 /json/version 直到 CDP 服务就绪。"""
        start = time.time()
        last_error = ""
        while time.time() - start < self.cdp_timeout:
            try:
                with urllib.request.urlopen(
                    f"{cdp_url}/json/version", timeout=2
                ) as resp:
                    if resp.status == 200:
                        return
            except Exception as e:
                last_error = str(e)
            time.sleep(1)
        raise WebBridgeError(
            f"CDP 服务在 {self.cdp_timeout}s 内未就绪 "
            f"({self.cdp_host}:{self.cdp_port}): {last_error}"
        )

    def _pick_business_page(self, browser) -> Optional[Page]:
        """从 browser.contexts 中挑选最可能是业务页面的 page。"""
        self._log_step(f"开始挑选业务 page，contexts={len(browser.contexts)}")
        candidates: List[Page] = []
        for idx, context in enumerate(browser.contexts):
            self._log_step(f"  context[{idx}] pages={len(context.pages)}")
            for page in context.pages:
                url = page.url or ""
                title = ""
                try:
                    title = page.title()
                except Exception:
                    pass
                self._log_step(f"    - url={url!r} title={title!r}")
                # 优先业务页面，过滤 devtools/about:blank
                if url.startswith("http") and not url.startswith(
                    "http://localhost:9222"
                ):
                    candidates.append(page)

        if not candidates:
            # fallback：返回第一个非空白页
            for context in browser.contexts:
                for page in context.pages:
                    if page.url not in ("about:blank", ""):
                        candidates.append(page)

        if not candidates:
            return None

        # 按 URL 相关性打分：项目编辑页优先于 train_detail
        def score(page: Page) -> int:
            url = (page.url or "").lower()
            s = 0
            if self.frontend_url and self.frontend_url.lower() in url:
                s += 100
            if "localhost" in url or "127.0.0.1" in url:
                s += 50
            if "8842" in url or "8843" in url:
                s += 30
            if "proj_panel" in url:
                s += 40
            if "parameter_edit" in url or "proj_panel_edit" in url:
                s += 25
            # train_detail 分数低，尽量避免 attach 后卡在详情页
            if "train_detail" in url:
                s -= 40
            return s

        candidates.sort(key=score, reverse=True)
        chosen = candidates[0]
        self._log_step(f"选中 page: url={chosen.url!r}")
        return chosen

    def get_page_state(self) -> WebBridgeResult:
        """获取当前页面状态。"""
        if self._page is None:
            return WebBridgeResult(False, "尚未连接到 WebView2")
        return WebBridgeResult(
            True,
            f"URL: {self._page.url}, Title: {self._page.title()}",
            {"url": self._page.url, "title": self._page.title()},
        )
