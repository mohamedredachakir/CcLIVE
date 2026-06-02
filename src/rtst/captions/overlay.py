"""Floating always-on-top caption overlay (PyQt6).

Renders a frameless, translucent, click-through-ish window pinned to the bottom
of the screen — suitable as a subtitle overlay on top of Teams / Meet / Zoom
desktop apps without modifying them. PyQt6 is imported lazily so this module
can be imported (and the rest of the package tested) without a GUI environment.

Threading note: Qt widgets must be touched from the GUI thread. The pipeline
runs on a worker thread, so :meth:`update` marshals work onto the Qt event loop
via a queued signal.
"""

from __future__ import annotations

import threading

from rtst.captions.base import CaptionSink
from rtst.types import Caption


class OverlayCaptionSink(CaptionSink):
    def __init__(
        self,
        mode: str = "compact",
        max_lines: int = 2,
        linger_seconds: float = 4.0,
        font_point_size: int = 22,
        opacity: float = 0.85,
    ) -> None:
        self.mode = mode
        self.max_lines = max_lines
        self.linger_seconds = linger_seconds
        self.font_point_size = font_point_size
        self.opacity = opacity
        self._app = None
        self._window = None
        self._ready = threading.Event()

    def start(self) -> None:
        """Build the Qt window. Must be called from the GUI thread."""
        from PyQt6 import QtCore, QtGui, QtWidgets

        self._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self._window = _OverlayWindow(
            QtCore,
            QtGui,
            QtWidgets,
            mode=self.mode,
            max_lines=self.max_lines,
            linger_seconds=self.linger_seconds,
            font_point_size=self.font_point_size,
            opacity=self.opacity,
        )
        self._window.show()
        self._ready.set()

    def run(self) -> None:  # pragma: no cover - requires display
        """Build the window and block on the Qt event loop."""
        if self._window is None:
            self.start()
        self._app.exec()

    def update(self, caption: Caption) -> None:  # pragma: no cover - requires display
        if self._window is None:
            return
        self._window.emit_caption(
            self._format(caption), caption.is_final
        )

    def status(self, message: str) -> None:  # pragma: no cover - requires display
        if self._window is not None:
            self._window.emit_caption(f"[{message}]", False)

    def close(self) -> None:  # pragma: no cover - requires display
        if self._window is not None:
            self._window.emit_close()

    def _format(self, caption: Caption) -> str:
        if self.mode == "dual" and caption.original:
            return f"{caption.original}\n→ {caption.translated}"
        return caption.translated


def _OverlayWindow(QtCore, QtGui, QtWidgets, **kwargs):  # noqa: ANN001, N802
    """Factory building the overlay window class against an imported Qt.

    Defined as a factory (rather than a module-level class) so the module does
    not need PyQt6 at import time.
    """

    class _Window(QtWidgets.QWidget):
        _caption_signal = QtCore.pyqtSignal(str, bool)
        _close_signal = QtCore.pyqtSignal()

        def __init__(
            self,
            mode: str,
            max_lines: int,
            linger_seconds: float,
            font_point_size: int,
            opacity: float,
        ) -> None:
            super().__init__()
            self._linger_ms = int(linger_seconds * 1000)
            self.setWindowFlags(
                QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
                | QtCore.Qt.WindowType.Tool
            )
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setWindowOpacity(opacity)

            self._label = QtWidgets.QLabel("", self)
            self._label.setWordWrap(True)
            self._label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            font = QtGui.QFont()
            font.setPointSize(font_point_size)
            font.setBold(True)
            self._label.setFont(font)
            self._label.setStyleSheet(
                "color: white; background: rgba(0,0,0,160);"
                "padding: 10px 18px; border-radius: 10px;"
            )
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._label)

            self._timer = QtCore.QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._clear)

            self._caption_signal.connect(self._on_caption)
            self._close_signal.connect(self.close)
            self._position()

        def _position(self) -> None:
            screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
            width = int(screen.width() * 0.6)
            self.setFixedWidth(width)
            self.move(
                screen.left() + (screen.width() - width) // 2,
                int(screen.height() * 0.82),
            )

        def emit_caption(self, text: str, final: bool) -> None:
            self._caption_signal.emit(text, final)

        def emit_close(self) -> None:
            self._close_signal.emit()

        def _on_caption(self, text: str, final: bool) -> None:
            self._label.setText(text)
            self.adjustSize()
            self._position()
            if final:
                self._timer.start(self._linger_ms)
            else:
                self._timer.stop()

        def _clear(self) -> None:
            self._label.setText("")

    return _Window(**kwargs)
