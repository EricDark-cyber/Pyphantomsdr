"""Widget spectre + waterfall (pyqtgraph) : clic = accord, molette = zoom."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGraphicsSceneWheelEvent

WF_ROWS = 400
WF_COLS = 1024
DB_FLOOR = -120.0
DB_CEIL = -20.0


class SpectrumWaterfall(pg.GraphicsLayoutWidget):
    tuneRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground("k")
        self.spec_plot = self.addPlot(row=0, col=0)
        self.wf_plot = self.addPlot(row=1, col=0)
        self.spec_plot.setLabel("bottom", "Fréquence", units="Hz")
        self.spec_plot.showGrid(True, True, 0.3)
        self.spec_plot.setMouseEnabled(False, False)
        self.wf_plot.setMouseEnabled(False, False)
        self.wf_plot.hideAxis("left")
        self.wf_plot.setXLink(self.spec_plot)
        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen("#4af", width=1))
        self.wf_img = pg.ImageItem()
        self.wf_plot.addItem(self.wf_img)
        self.wf = np.full((WF_ROWS, WF_COLS), DB_FLOOR, dtype=np.float32)
        self.center = 100e6
        self.fs = 1e6
        self.span = 1e6
        self.zoom = 1.0
        scene = self.spec_plot.scene()
        scene.sigMouseClicked.connect(self._on_click)
        scene.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if isinstance(ev, QGraphicsSceneWheelEvent):
            vb = self.spec_plot.getViewBox()
            pos = vb.mapSceneToView(ev.scenePos())
            factor = 0.8 if ev.delta() > 0 else 1.25
            new_zoom = float(np.clip(self.zoom * factor, 1.0, 128.0))
            self.set_zoom(new_zoom, anchor=pos.x())
            return True
        return super().eventFilter(obj, ev)

    def _on_click(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        vb = self.spec_plot.getViewBox()
        pos = vb.mapSceneToView(ev.scenePos())
        self.tuneRequested.emit(float(pos.x()))

    def set_zoom(self, zoom: float, anchor: float | None = None) -> None:
        self.zoom = float(np.clip(zoom, 1.0, 128.0))
        if anchor is not None:
            frac = (anchor - (self.center - self.span / 2.0)) / self.span
            frac = float(np.clip(frac, 0.0, 1.0))
        else:
            frac = 0.5
        new_span = self.fs / self.zoom
        self.center = anchor - (frac - 0.5) * new_span if anchor is not None else self.center
        self.span = new_span
        self._apply_range()

    def _apply_range(self) -> None:
        lo = self.center - self.span / 2.0
        hi = self.center + self.span / 2.0
        self.spec_plot.setXRange(lo, hi, padding=0)
        self.wf_img.setRect(lo, WF_ROWS, self.span, -WF_ROWS)

    def set_center(self, center: float) -> None:
        self.center = float(center)
        self._apply_range()

    def update_spectrum(self, db: np.ndarray, center: float, fs: float) -> None:
        db = np.asarray(db, dtype=np.float32)
        self.fs = float(fs)
        if center != self.center:
            self.center = float(center)
        n = len(db)
        freqs = self.center + (np.arange(n) - n / 2.0) * (self.fs / n)
        self.spec_curve.setData(freqs, db)
        xs = np.linspace(self.center - self.span / 2.0, self.center + self.span / 2.0, WF_COLS)
        row = np.interp(xs, freqs, db)
        self.wf = np.roll(self.wf, -1, axis=0)
        self.wf[-1, :] = row
        self.wf_img.setImage(self.wf, autoLevels=False, levels=(DB_FLOOR, DB_CEIL))
        self._apply_range()
