#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集数据质量检查 — 简易 tkinter GUI。

.. deprecated::
    本入口已弃用。请改用：

    - PyQt 工作台：``python pyqt_frontend/main.py`` → 数据质检面板
    - CLI：``python check_dataset.py --folder <dir> --task <task>``

仍可 ``python check_dataset_ui.py`` 启动（会打印弃用警告），后续版本可能删除。

依赖: 标准库 tkinter + 本仓库 autotrain4.data_quality（可选 Pillow 做坏图解码）
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import traceback
import warnings
from pathlib import Path
from tkinter import (
    BOTH,
    DISABLED,
    END,
    LEFT,
    NORMAL,
    RIGHT,
    W,
    BooleanVar,
    StringVar,
    Tk,
    filedialog,
    messagebox,
    ttk,
)
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 任务下拉：显示名 -> 传给检查器的值
# 分类放前面，避免默认 det 去扫无 JSON 的分类集而误报 E_PAIR_RATE
TASK_CHOICES = [
    ("分类 (cls)", "cls"),
    ("目标检测 (det)", "det"),
    ("OCR (ocr)", "ocr"),
    ("DETR (detr)", "detr"),
    ("实例分割 (iseg)", "iseg"),
    ("语义分割 (sseg)", "sseg"),
    ("旋转框检测 (rotate)", "rotate"),
]

DECODE_CHOICES = [
    ("抽样解码 (快)", "sample"),
    ("全量解码 (慢)", "full"),
    ("不解码图片", "off"),
]

# 路径片段 -> 任务 value（长关键词优先，避免 det 误伤 detr）
_PATH_TASK_HINTS = [
    ("detr", "detr"),
    ("iseg", "iseg"),
    ("sseg", "sseg"),
    ("rotate", "rotate"),
    ("rotation", "rotate"),
    ("实例分割", "iseg"),
    ("语义分割", "sseg"),
    ("旋转", "rotate"),
    ("ocr", "ocr"),
    ("cls", "cls"),
    ("分类", "cls"),
    ("classification", "cls"),
    ("det", "det"),
    ("目标检测", "det"),
    ("detection", "det"),
]


def _guess_task_value_from_path(path: str) -> str | None:
    """根据目录名猜测任务类型 value；无法判断时返回 None。"""
    if not path:
        return None
    text = str(path).replace("\\", "/").lower()
    parts = [p for p in text.split("/") if p]
    # 先整段路径，再从末级目录往上匹配
    candidates = ["/".join(parts)] + list(reversed(parts))
    for blob in candidates:
        low = blob.lower()
        for key, val in _PATH_TASK_HINTS:
            if key.lower() in low:
                return val
    return None


def _task_label_for_value(value: str) -> str | None:
    for name, val in TASK_CHOICES:
        if val == value:
            return name
    return None


class DatasetQualityApp:
    def __init__(self, master: Tk) -> None:
        self.master = master
        master.title("训练集数据质量检查")
        master.minsize(720, 520)
        master.geometry("900x640")

        self.folder_var = StringVar()
        self.task_var = StringVar(value=TASK_CHOICES[0][0])  # 默认分类
        self.decode_var = StringVar(value=DECODE_CHOICES[0][0])
        self.strict_var = BooleanVar(value=False)
        self.resolve_var = BooleanVar(value=True)
        self.min_images_var = StringVar(value="10")
        self.min_pair_var = StringVar(value="0.9")
        self.status_var = StringVar(value="就绪：选择数据集目录与任务类型后点击「开始检查」")

        self._report = None  # type: ignore
        self._msg_q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

        self._build()
        self._poll_queue()
        self._sync_pair_controls()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self.master, padding=10)
        frm.pack(fill=BOTH, expand=True)

        # 路径
        row0 = ttk.Frame(frm)
        row0.pack(fill="x", **pad)
        ttk.Label(row0, text="数据集目录", width=12).pack(side=LEFT)
        ent = ttk.Entry(row0, textvariable=self.folder_var)
        ent.pack(side=LEFT, fill="x", expand=True, padx=(0, 6))
        ttk.Button(row0, text="浏览…", command=self._browse, width=10).pack(side=LEFT)

        # 任务 + 解码
        row1 = ttk.Frame(frm)
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="任务类型", width=12).pack(side=LEFT)
        self.task_combo = ttk.Combobox(
            row1,
            textvariable=self.task_var,
            values=[c[0] for c in TASK_CHOICES],
            state="readonly",
            width=22,
        )
        self.task_combo.pack(side=LEFT)
        self.task_combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_pair_controls())
        ttk.Label(row1, text="  坏图检查").pack(side=LEFT, padx=(16, 0))
        self.decode_combo = ttk.Combobox(
            row1,
            textvariable=self.decode_var,
            values=[c[0] for c in DECODE_CHOICES],
            state="readonly",
            width=18,
        )
        self.decode_combo.pack(side=LEFT)

        # 选项
        row2 = ttk.Frame(frm)
        row2.pack(fill="x", **pad)
        ttk.Checkbutton(row2, text="自动进入「训练集」子目录", variable=self.resolve_var).pack(
            side=LEFT
        )
        ttk.Checkbutton(row2, text="严格模式（警告也算失败）", variable=self.strict_var).pack(
            side=LEFT, padx=(16, 0)
        )
        ttk.Label(row2, text="  最少图片").pack(side=LEFT, padx=(16, 4))
        ttk.Entry(row2, textvariable=self.min_images_var, width=6).pack(side=LEFT)
        self.pair_lbl = ttk.Label(row2, text="  配对率警告阈值")
        self.pair_lbl.pack(side=LEFT, padx=(12, 4))
        self.pair_entry = ttk.Entry(row2, textvariable=self.min_pair_var, width=6)
        self.pair_entry.pack(side=LEFT)

        # 按钮
        row3 = ttk.Frame(frm)
        row3.pack(fill="x", **pad)
        self.btn_run = ttk.Button(row3, text="开始检查", command=self._start_check)
        self.btn_run.pack(side=LEFT)
        self.btn_export = ttk.Button(
            row3, text="导出 JSON…", command=self._export_json, state=DISABLED
        )
        self.btn_export.pack(side=LEFT, padx=(8, 0))
        self.btn_open = ttk.Button(
            row3, text="打开数据目录", command=self._open_folder, state=DISABLED
        )
        self.btn_open.pack(side=LEFT, padx=(8, 0))
        ttk.Button(row3, text="清空结果", command=self._clear).pack(side=LEFT, padx=(8, 0))
        ttk.Button(row3, text="退出", command=self.master.destroy).pack(side=RIGHT)

        # 摘要条
        self.summary_lbl = ttk.Label(
            frm,
            text="尚未检查",
            font=("Microsoft YaHei UI", 11, "bold"),
            anchor=W,
        )
        self.summary_lbl.pack(fill="x", **pad)

        # 结果文本
        self.text = ScrolledText(frm, wrap="word", height=28, font=("Consolas", 10))
        self.text.pack(fill=BOTH, expand=True, padx=4, pady=4)
        self.text.tag_configure("error", foreground="#b00020")
        self.text.tag_configure("warn", foreground="#b36b00")
        self.text.tag_configure("info", foreground="#1565c0")
        self.text.tag_configure("ok", foreground="#2e7d32")
        self.text.tag_configure("head", foreground="#333333", font=("Microsoft YaHei UI", 10, "bold"))
        self.text.configure(state=DISABLED)

        # 状态栏
        status = ttk.Label(frm, textvariable=self.status_var, relief="sunken", anchor=W)
        status.pack(fill="x", pady=(6, 0))

    def _browse(self) -> None:
        initial = self.folder_var.get().strip() or str(ROOT / "AI训练平台测试")
        if not Path(initial).is_dir():
            initial = str(ROOT)
        path = filedialog.askdirectory(title="选择数据集目录", initialdir=initial)
        if path:
            self.folder_var.set(path)
            self._maybe_guess_task(force=True)

    def _task_value(self) -> str:
        label = self.task_var.get()
        for name, val in TASK_CHOICES:
            if name == label:
                return val
        return label

    def _is_cls_task(self) -> bool:
        return self._task_value() == "cls"

    def _maybe_guess_task(self, force: bool = False) -> None:
        """根据路径自动切换任务类型（分类/检测等），避免默认 det 误报配对率。"""
        path = self.folder_var.get().strip()
        guessed = _guess_task_value_from_path(path)
        if not guessed:
            self._sync_pair_controls()
            return
        label = _task_label_for_value(guessed)
        if not label:
            self._sync_pair_controls()
            return
        # 浏览选文件夹时跟随路径；其它情况不覆盖用户手选
        if force:
            self.task_var.set(label)
        self._sync_pair_controls()

    def _sync_pair_controls(self) -> None:
        """分类无图↔JSON 配对：灰掉配对阈值，避免误解。"""
        is_cls = self._is_cls_task()
        state = DISABLED if is_cls else NORMAL
        try:
            self.pair_entry.configure(state=state)
            if is_cls:
                self.pair_lbl.configure(text="  配对率(分类不适用)")
            else:
                self.pair_lbl.configure(text="  配对率警告阈值")
        except Exception:
            pass

    def _decode_value(self) -> str:
        label = self.decode_var.get()
        for name, val in DECODE_CHOICES:
            if name == label:
                return val
        return "sample"

    def _clear(self) -> None:
        self._report = None
        self.btn_export.configure(state=DISABLED)
        self.summary_lbl.configure(text="尚未检查")
        self._set_text("")
        self.status_var.set("已清空")

    def _set_text(self, content: str, colored: bool = False) -> None:
        self.text.configure(state=NORMAL)
        self.text.delete("1.0", END)
        if not colored:
            self.text.insert(END, content)
        self.text.configure(state=DISABLED)

    def _append_line(self, line: str, tag: str | None = None) -> None:
        self.text.configure(state=NORMAL)
        if tag:
            self.text.insert(END, line + "\n", tag)
        else:
            self.text.insert(END, line + "\n")
        self.text.configure(state=DISABLED)

    def _open_folder(self) -> None:
        path = None
        if self._report is not None:
            path = Path(self._report.dataset_dir)
        else:
            f = self.folder_var.get().strip()
            path = Path(f) if f else None
        if not path or not path.is_dir():
            messagebox.showwarning("提示", "目录无效")
            return
        try:
            import os

            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception as e:
            messagebox.showerror("无法打开", str(e))

    def _export_json(self) -> None:
        if self._report is None:
            messagebox.showinfo("提示", "请先完成一次检查")
            return
        path = filedialog.asksaveasfilename(
            title="导出质量报告",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialfile="data_quality_report.json",
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self._report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.status_var.set(f"已导出: {path}")
            messagebox.showinfo("完成", f"报告已保存到:\n{path}")
        except OSError as e:
            messagebox.showerror("写入失败", str(e))

    # ------------------------------------------------------------------ run
    def _start_check(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("提示", "正在检查中，请稍候…")
            return

        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("提示", "请先选择数据集目录")
            return
        if not Path(folder).is_dir():
            messagebox.showerror("错误", f"目录不存在:\n{folder}")
            return

        try:
            min_images = int(self.min_images_var.get().strip() or "10")
            min_pair = float(self.min_pair_var.get().strip() or "0.9")
        except ValueError:
            messagebox.showerror("错误", "最少图片 / 配对率阈值必须是数字")
            return

        self.btn_run.configure(state=DISABLED)
        self.btn_export.configure(state=DISABLED)
        self.btn_open.configure(state=DISABLED)
        self._report = None
        self.summary_lbl.configure(text="检查中…")
        self._set_text("")
        self.status_var.set("正在扫描，请稍候…")

        args = {
            "folder": folder,
            "task": self._task_value(),
            "decode": self._decode_value(),
            "strict": bool(self.strict_var.get()),
            "resolve": bool(self.resolve_var.get()),
            "min_images": min_images,
            "min_pair_rate": min_pair,
        }
        self._worker = threading.Thread(target=self._run_worker, args=(args,), daemon=True)
        self._worker.start()

    def _run_worker(self, args: dict) -> None:
        try:
            from autotrain4.data_quality import QualityPolicy, check_dataset

            def progress(msg: str) -> None:
                self._msg_q.put(("progress", msg))

            policy = QualityPolicy(
                min_images=args["min_images"],
                min_pair_rate=args["min_pair_rate"],
                image_decode=args["decode"],
                fail_on_warning=args["strict"],
            )
            report = check_dataset(
                args["folder"],
                args["task"],
                policy=policy,
                resolve_train_subdir=args["resolve"],
                progress_cb=progress,
            )
            self._msg_q.put(("done", report))
        except Exception as e:
            self._msg_q.put(("error", f"{e}\n\n{traceback.format_exc()}"))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._msg_q.get_nowait()
                if kind == "progress":
                    self.status_var.set(str(payload))
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self._on_error(str(payload))
        except queue.Empty:
            pass
        self.master.after(120, self._poll_queue)

    def _on_error(self, msg: str) -> None:
        self.btn_run.configure(state=NORMAL)
        self.summary_lbl.configure(text="检查失败")
        self.status_var.set("出错")
        self._set_text(msg)
        messagebox.showerror("检查失败", msg.split("\n", 1)[0])

    def _on_done(self, report) -> None:
        from autotrain4.data_quality import format_report_text

        self._report = report
        self.btn_run.configure(state=NORMAL)
        self.btn_export.configure(state=NORMAL)
        self.btn_open.configure(state=NORMAL)

        d = report.to_dict()
        passed = d["pass"]
        grade = d["grade"]
        score = d["score"]
        n_err = d["summary"].get("blocking_issues", 0)
        n_warn = d["summary"].get("warnings", 0)
        pair = d["summary"].get("pair_rate")
        n_img = d["summary"].get("n_images", 0)
        label_mode = d["summary"].get("label_mode") or (
            "folder" if d.get("task_type") == "分类" else "json"
        )
        is_cls = label_mode == "folder" or d.get("task_type") == "分类"

        verdict = "通过 PASS" if passed else "未通过 FAIL"
        if is_cls:
            pair_txt = "文件夹分类(无JSON配对)"
        elif isinstance(pair, (int, float)):
            pair_txt = f"配对率 {pair:.1%}"
        else:
            pair_txt = "配对率 N/A"
        self.summary_lbl.configure(
            text=(
                f"{verdict}  |  等级 {grade}  分数 {score:.2f}  |  "
                f"图 {n_img}  {pair_txt}  |  "
                f"错误 {n_err}  警告 {n_warn}"
            )
        )
        self.status_var.set(
            f"完成 — {verdict}  grade={grade}  任务={d.get('task_type')}  目录: {d['dataset_dir']}"
        )

        # 带颜色渲染
        self.text.configure(state=NORMAL)
        self.text.delete("1.0", END)
        plain = format_report_text(report)
        for line in plain.splitlines():
            tag = None
            s = line.lstrip()
            if s.startswith("[ERROR]"):
                tag = "error"
            elif s.startswith("[WARN]"):
                tag = "warn"
            elif s.startswith("[INFO]"):
                tag = "info"
            elif "结论:" in line:
                tag = "ok" if passed else "error"
            elif line.startswith("===") or line.startswith("---"):
                tag = "head"
            self.text.insert(END, line + "\n", tag if tag else ())
        self.text.configure(state=DISABLED)
        self.text.see("1.0")


def main() -> int:
    msg = (
        "check_dataset_ui.py 已弃用，请使用 PyQt 工作台质检面板"
        "（python pyqt_frontend/main.py）或 CLI check_dataset.py。"
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    print(f"[DEPRECATED] {msg}", file=sys.stderr, flush=True)

    # Windows 高 DPI 稍微友好一点
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)  # type: ignore
    except Exception:
        pass

    root = Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        try:
            ttk.Style().theme_use("clam")
        except Exception:
            pass

    DatasetQualityApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
