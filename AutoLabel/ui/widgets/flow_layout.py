"""Flow layout: items wrap to the next line when the row is full."""

from __future__ import annotations

from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtWidgets import QBoxLayout, QLayout, QSizePolicy, QStyle, QWidget


class FlowLayout(QLayout):
    """Left-to-right wrapping layout (Qt Flow Layout example)."""

    def __init__(self, parent=None, spacing: int = -1):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _smart_spacing(self, pm) -> int:
        parent = self.parent()
        if parent is None:
            return 8
        if parent.isWidgetType():
            return parent.style().pixelMetric(pm, None, parent)
        return int(parent.spacing())

    def _hspacing(self) -> int:
        sp = super().spacing()
        if sp >= 0:
            return sp
        v = self._smart_spacing(QStyle.PM_LayoutHorizontalSpacing)
        return 8 if v < 0 else v

    def _vspacing(self) -> int:
        sp = super().spacing()
        if sp >= 0:
            return sp
        v = self._smart_spacing(QStyle.PM_LayoutVerticalSpacing)
        return 8 if v < 0 else v

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = effective.x()
        y = effective.y()
        line_height = 0
        space_x = max(0, self._hspacing())
        space_y = max(0, self._vspacing())
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + space_x
            if next_x - space_x > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + space_y
                next_x = x + hint.width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + m.bottom()


def layout_height_for_width(layout, width: int) -> int:
    """Best-effort height of a layout at `width` (walks widgets / nested layouts)."""
    if layout is None:
        return 0
    m = layout.contentsMargins()
    inner = max(0, width - m.left() - m.right())
    spacing = max(0, layout.spacing())
    vertical = True
    try:
        direction = layout.direction()
        vertical = direction in (QBoxLayout.TopToBottom, QBoxLayout.BottomToTop)
    except AttributeError:
        pass
    heights = []
    widths = []
    stretches = []
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None:
            continue
        if item.spacerItem() is not None:
            sp = item.spacerItem()
            if sp.expandingDirections():
                continue
        w = item.widget()
        child = item.layout()
        hint = item.sizeHint()
        h = hint.height()
        share_w = inner if vertical else hint.width()
        if child is not None:
            h = layout_height_for_width(child, share_w if vertical else hint.width())
        elif w is not None:
            wlay = w.layout()
            if w.hasHeightForWidth():
                h = w.heightForWidth(share_w if vertical else max(hint.width(), 1))
            elif wlay is not None:
                h = layout_height_for_width(wlay, share_w if vertical else max(hint.width(), 1))
        heights.append(max(0, h))
        widths.append(max(0, hint.width()))
        try:
            stretches.append(max(0, layout.stretch(i)))
        except Exception:
            stretches.append(0)
    if not heights:
        return m.top() + m.bottom()
    if vertical:
        return m.top() + m.bottom() + sum(heights) + spacing * (len(heights) - 1)
    stretch_sum = sum(max(1, s) for s in stretches) or len(heights)
    avail = max(0, inner - spacing * (len(heights) - 1))
    max_h = 0
    idx = 0
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is None or idx >= len(heights):
            continue
        if item.spacerItem() is not None and item.spacerItem().expandingDirections():
            continue
        share = int(avail * max(1, stretches[idx]) / stretch_sum)
        w = item.widget()
        child = item.layout()
        h = heights[idx]
        if child is not None:
            h = layout_height_for_width(child, share)
        elif w is not None:
            if w.hasHeightForWidth():
                h = w.heightForWidth(share)
            elif w.layout() is not None:
                h = layout_height_for_width(w.layout(), share)
        max_h = max(max_h, h)
        idx += 1
    return m.top() + m.bottom() + max_h


class VScrollHost(QWidget):
    """QScrollArea inner widget that reports height-for-width and can shrink."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setMinimumWidth(0)

    def setLayout(self, layout) -> None:  # noqa: N802
        if layout is not None:
            layout.setSizeConstraint(QLayout.SetNoConstraint)
        super().setLayout(layout)
        self.setMinimumWidth(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        lay = self.layout()
        if lay is None:
            return super().heightForWidth(width)
        return max(lay.minimumSize().height(), layout_height_for_width(lay, width))

    def sizeHint(self):  # noqa: N802
        lay = self.layout()
        if lay is None:
            return super().sizeHint()
        w = self.width() if self.width() > 0 else lay.sizeHint().width()
        h = self.heightForWidth(w)
        return QSize(lay.sizeHint().width(), h)

    def minimumSizeHint(self):  # noqa: N802
        lay = self.layout()
        if lay is None:
            return QSize(0, 0)
        w = self.width() if self.width() > 0 else 0
        h = self.heightForWidth(w) if w else lay.minimumSize().height()
        return QSize(0, h)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        h = self.heightForWidth(max(1, self.width()))
        if h > 0 and self.minimumHeight() != h:
            self.setMinimumHeight(h)
        self.updateGeometry()
