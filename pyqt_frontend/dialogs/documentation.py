"""README / 使用文档弹窗。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import markdown

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def merge_adjacent_pre_blocks(html: str) -> str:
    """合并相邻的 <pre>...</pre> 代码块为一块，避免一行一条黑条。"""

    def merge_match(m: re.Match) -> str:
        inner = m.group(1).strip("\n")
        inner = re.sub(r"</code>\s*</pre>\s*<pre>\s*<code[^>]*>", "\n", inner)
        inner = re.sub(r"</pre>\s*<pre>", "\n", inner)
        return f"<pre><code>{inner}</code></pre>"

    pattern = re.compile(
        r"<pre>\s*<code[^>]*>(.*?)</code>\s*</pre>(?:\s*<pre>\s*<code[^>]*>.*?</code>\s*</pre>)+",
        re.DOTALL,
    )
    return pattern.sub(merge_match, html)


def show_documentation_dialog(parent: Optional[QWidget], html_body: str) -> None:
    """用 QTextEdit 展示渲染后的文档 HTML。"""
    dialog = QDialog(parent)
    dialog.setWindowTitle("AutoTrain 4.0 使用文档")
    dialog.setMinimumSize(760, 600)

    layout = QVBoxLayout(dialog)
    layout.setSpacing(16)
    layout.setContentsMargins(20, 20, 20, 20)

    header = QLabel("📖 使用文档")
    header.setStyleSheet("font-size: 22px; font-weight: bold; color: #192033;")
    layout.addWidget(header)

    doc_edit = QTextEdit()
    doc_edit.setReadOnly(True)
    doc_edit.setStyleSheet("""
        QTextEdit {
            border: 1px solid #e8ebf2;
            border-radius: 16px;
            background: #ffffff;
            padding: 16px;
            font-size: 14px;
            line-height: 1.6;
        }
    """)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            color: #192033;
            line-height: 1.7;
        }}
        h1, h2, h3 {{
            color: #6c5ce7;
            font-weight: 700;
            margin-top: 1.4em;
            margin-bottom: 0.5em;
        }}
        h1 {{ font-size: 28px; border-bottom: 2px solid #f0edff; padding-bottom: 8px; }}
        h2 {{ font-size: 22px; }}
        h3 {{ font-size: 18px; }}
        a {{ color: #6c5ce7; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        code {{
            background: #f4f6fb;
            padding: 2px 6px;
            border-radius: 5px;
            font-family: Consolas, "SFMono-Regular", monospace;
            color: #db5a65;
            font-size: 13px;
        }}
        pre {{
            background: #f4f6fb;
            color: #1f2937;
            padding: 12px 14px;
            border-radius: 12px;
            border: 1px solid #e8ebf2;
            overflow-x: auto;
            font-family: Consolas, "SFMono-Regular", monospace;
            font-size: 13px;
            line-height: 1.6;
            margin: 8px 0;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        pre code {{
            background: transparent;
            color: inherit;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 12px 0;
        }}
        th, td {{
            border: 1px solid #e8ebf2;
            padding: 10px 12px;
            text-align: left;
        }}
        th {{
            background: #f0edff;
            color: #6c5ce7;
            font-weight: 700;
        }}
        tr:nth-child(even) {{ background: #f8f9fc; }}
        ul, ol {{ padding-left: 24px; }}
        li {{ margin: 6px 0; }}
        hr {{
            border: none;
            border-top: 1px solid #e8ebf2;
            margin: 24px 0;
        }}
        blockquote {{
            border-left: 4px solid #c4b5fd;
            margin: 12px 0;
            padding: 8px 16px;
            background: #f7f5ff;
            color: #5347b4;
        }}
    </style>
    </head>
    <body>
    {html_body}
    </body>
    </html>
    """
    doc_edit.setHtml(html)
    layout.addWidget(doc_edit, 1)

    close_btn = QPushButton("关闭")
    close_btn.setObjectName("secondaryBtn")
    close_btn.clicked.connect(dialog.accept)
    layout.addWidget(close_btn, alignment=Qt.AlignRight)

    dialog.exec_()


def open_readme(parent: Optional[QWidget] = None) -> None:
    """读取项目 README.md，转换为 HTML 后在弹窗中展示。"""
    # dialogs/ -> pyqt_frontend/ -> project root
    readme_path = Path(__file__).resolve().parent.parent.parent / "README.md"
    if not readme_path.exists():
        QMessageBox.warning(parent, "查看文档", "未找到 README.md")
        return

    try:
        md_text = readme_path.read_text(encoding="utf-8")
        html_body = markdown.markdown(md_text, extensions=["fenced_code", "tables"])
        html_body = merge_adjacent_pre_blocks(html_body)
    except Exception as e:
        QMessageBox.warning(parent, "查看文档", f"读取文档失败：{e}")
        return

    show_documentation_dialog(parent, html_body)
