"""Preview gallery: show overlay thumbnails and confirm batch."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QWheelEvent
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.styles import apply_window_styles
from ui.widgets.flow_layout import FlowLayout
from ui.widgets.ui_scale import UIScale


class _ClickableImageLabel(QLabel):
    """Thumbnail that opens a zoomable viewer on click."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self._image_path = image_path
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("点击放大查看")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._image_path:
            ImageZoomDialog(self._image_path, self.window()).exec_()
        super().mousePressEvent(event)


class ImageZoomDialog(QDialog):
    """Full-size / zoomable preview for a single overlay image."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"预览 — {Path(image_path).name}")
        self.setMinimumSize(480, 360)
        self.resize(1000, 720)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self._ui_scale = getattr(parent, "_ui_scale", None) or UIScale()

        self._path = image_path
        self._pix = QPixmap(str(image_path))
        self._scale = 1.0
        self._user_zoomed = False

        self._img = QLabel()
        self._img.setAlignment(Qt.AlignCenter)
        self._img.setStyleSheet("background:#111;")
        self._img.setMinimumSize(1, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)
        scroll.setWidget(self._img)
        self._scroll = scroll

        hint = QLabel("滚轮缩放 · 双击复位 · Esc / 关闭 退出")
        hint.setObjectName("Hint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(scroll, 1)
        layout.addWidget(hint)
        layout.addWidget(buttons)
        apply_window_styles(self)

        if self._pix.isNull():
            self._img.setText(f"无法加载：{image_path}")
        else:
            self._fit_to_view()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._pix.isNull() and self._scale == 1.0:
            self._fit_to_view()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if not self._pix.isNull() and not self._user_zoomed:
            self._fit_to_view()

    def _fit_to_view(self):
        vp = self._scroll.viewport().size()
        if vp.width() < 50 or vp.height() < 50:
            return
        sx = vp.width() / max(1, self._pix.width())
        sy = vp.height() / max(1, self._pix.height())
        self._scale = min(sx, sy, 1.0)  # never upscale beyond 1x on fit
        self._apply_scale()

    def _apply_scale(self):
        if self._pix.isNull():
            return
        w = max(1, int(self._pix.width() * self._scale))
        h = max(1, int(self._pix.height() * self._scale))
        scaled = self._pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img.setPixmap(scaled)
        self._img.resize(scaled.size())

    def wheelEvent(self, event: QWheelEvent):
        if self._pix.isNull():
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._scale = max(0.1, min(8.0, self._scale * factor))
        self._user_zoomed = True
        self._apply_scale()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._user_zoomed = False
            self._fit_to_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reject()
            return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._scale = min(8.0, self._scale * 1.15)
            self._user_zoomed = True
            self._apply_scale()
            return
        if event.key() == Qt.Key_Minus:
            self._scale = max(0.1, self._scale / 1.15)
            self._user_zoomed = True
            self._apply_scale()
            return
        if event.key() == Qt.Key_0:
            self._user_zoomed = False
            self._fit_to_view()
            return
        super().keyPressEvent(event)


class PreviewGalleryDialog(QDialog):
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("试标预览（确认后才全量）")
        self.setMinimumSize(480, 360)
        self.resize(840, 520)
        self.setAttribute(Qt.WA_QuitOnClose, False)
        self._result = "cancel"
        self._results = list(results or [])
        self._ui_scale = getattr(parent, "_ui_scale", None) or UIScale()
        self._cards: list[QWidget] = []
        self._last_cols = 0
        self._thumb_w = 180
        self._thumb_h = 135
        self._relayout_timer = QTimer(self)
        self._relayout_timer.setSingleShot(True)
        self._relayout_timer.timeout.connect(self._relayout_cards)

        title = QLabel(f"试标结果：{len(self._results)} 张")
        title.setObjectName("Title")
        title.setWordWrap(True)
        skipped_n = sum(1 for r in self._results if getattr(r, "skipped", False))
        if skipped_n:
            title.setText(f"试标结果：{len(self._results)} 张（其中审核拦截 {skipped_n} 张，已跳过）")

        grid_host = QWidget()
        self._grid = QGridLayout(grid_host)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._build_cards()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(grid_host)
        self._scroll = scroll

        buttons = QDialogButtonBox()
        btn_ok = buttons.addButton("确认并批量标注", QDialogButtonBox.AcceptRole)
        btn_retry = buttons.addButton("修改后重试", QDialogButtonBox.ActionRole)
        btn_cancel = buttons.addButton("取消", QDialogButtonBox.RejectRole)
        btn_ok.clicked.connect(self._ok)
        btn_retry.clicked.connect(self._retry)
        btn_cancel.clicked.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(title)
        hint = QLabel(
            "左图是最终 polygon overlay（LabelMe 导出，带轮廓和类别色）。"
            "右图是同一批实例的像素 mask（青色，无多边形），用来看贴边和锯齿。"
            "两边内容应对齐；若右图多出划痕，说明过滤没生效。"
            "点击缩略图可放大。不满意请修改描述/换 backend，不要直接全量。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(scroll, 1)
        layout.addWidget(buttons)
        apply_window_styles(self)

    def _thumb_size(self) -> tuple[int, int]:
        s = self._ui_scale
        return s.px(180, 140), s.px(135, 100)

    def _build_cards(self) -> None:
        tw, th = self._thumb_size()
        self._thumb_w, self._thumb_h = tw, th
        self._cards = []
        for r in self._results:
            path = r.preview_overlay_path or r.image_path
            img_label = self._thumb(str(path), tw, th, skipped=bool(getattr(r, "skipped", False)))
            mask_path = getattr(r, "preview_mask_path", None)
            thumbs = QWidget()
            thumbs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            thumbs_row = FlowLayout(thumbs, spacing=6)
            thumbs_row.addWidget(img_label)
            if mask_path:
                mask_label = self._thumb(str(mask_path), tw, th, skipped=False)
                thumbs_row.addWidget(mask_label)
            bucket = r.review.bucket if r.review else "-"
            note = ""
            skipped = bool(getattr(r, "skipped", False))
            if skipped:
                note = "\n审核拦截（已跳过，不进 Commit）"
            elif r.review and r.review.notes:
                note = f"\n{r.review.notes[:80]}"
            mask_hint = "  · 右图=青色像素 mask" if mask_path else "  · 无 raw overlay"
            meta = QLabel(
                f"{Path(r.image_path).name}\n"
                f"shapes={len(r.shapes)}  bucket={bucket}"
                f"{note}\n"
                f"左图=overlay{mask_hint} · 点击放大"
            )
            meta.setObjectName("Hint")
            meta.setWordWrap(True)
            meta.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            card = QWidget()
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            card.setMinimumWidth(0)
            col = QVBoxLayout(card)
            col.setContentsMargins(4, 4, 4, 4)
            col.addWidget(thumbs)
            col.addWidget(meta)
            self._cards.append(card)
        self._relayout_cards()

    def _thumb(self, path: str, tw: int, th: int, *, skipped: bool) -> QLabel:
        img_label = _ClickableImageLabel(path)
        img_label.setFixedSize(tw, th)
        img_label.setAlignment(Qt.AlignCenter)
        border = "#c0392b" if skipped else "transparent"
        img_label.setStyleSheet(
            f"background:#111; border-radius:8px;"
            f"border:2px solid {border};"
        )
        pix = QPixmap(path)
        if not pix.isNull():
            img_label.setPixmap(
                pix.scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            img_label.setText(Path(path).name)
        return img_label

    def _column_count(self) -> int:
        avail = self._scroll.viewport().width() if hasattr(self, "_scroll") else self.width()
        cell = self._thumb_w + 28
        return max(1, avail // max(1, cell))

    def _relayout_cards(self) -> None:
        cols = self._column_count()
        if cols == self._last_cols and self._grid.count() == len(self._cards):
            return
        self._last_cols = cols
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    self._grid.removeWidget(w)
        for c in range(max(self._grid.columnCount(), cols + 1)):
            self._grid.setColumnStretch(c, 0)
        for r in range(self._grid.rowCount() + 1):
            self._grid.setRowStretch(r, 0)
        for i, card in enumerate(self._cards):
            self._grid.addWidget(card, i // cols, i % cols)
        self._grid.setColumnStretch(cols, 1)

    def showEvent(self, event):
        super().showEvent(event)
        self._relayout_cards()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._relayout_timer.start(40)

    def _ok(self):
        self._result = "batch"
        self.accept()

    def _retry(self):
        self._result = "retry"
        self.accept()

    @property
    def choice(self) -> str:
        return self._result
