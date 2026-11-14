"""Thread d'acquisition : lit la source, émet spectre / IQ brut / audio / S-metre."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal, QThread

from ..dsp import Demodulator, compute_spectrum_db, power_dbfs


class SdrWorker(QThread):
    spectrumReady = Signal(object, float, float)
    iqReady = Signal(object)
    audioReady = Signal(object)
    smeterReady = Signal(float)
    infoReady = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.fs = 1e6
        self.audio_rate = 48_000
        self.calibration_db = 0.0
        self.demod = Demodulator(self.fs, self.audio_rate)
        self.block_s = 0.05
        self._stop = False

    def set_source(self, source, fs: float) -> None:
        self.source = source
        self.fs = float(fs)
        self.demod = Demodulator(self.fs, self.audio_rate, self.demod.mode, self.demod.bw)

    def run(self) -> None:
        block = max(1024, int(self.fs * self.block_s))
        while not self._stop:
            if self.source is None:
                self.msleep(50)
                continue
            try:
                iq = self.source.read(block)
            except Exception as exc:
                self.infoReady.emit(f"Erreur lecture source : {exc}")
                self.msleep(200)
                continue
            if iq is None or len(iq) == 0:
                self.msleep(5)
                continue
            db = compute_spectrum_db(iq, 2048)
            self.spectrumReady.emit(db, float(self.source.center_freq), self.fs)
            self.iqReady.emit(iq)
            audio = self.demod.process(iq)
            if len(audio):
                self.audioReady.emit(audio)
            self.smeterReady.emit(power_dbfs(iq) + self.calibration_db)

    def stop(self) -> None:
        self._stop = True
