"""Thread d'acquisition : lit la source, émet spectre / IQ brut / audio / S-metre / FT8."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal, QThread

from ..dsp import Demodulator, compute_spectrum_db, power_dbfs
from ..ft8.decoder import FT4Decoder, FT8Decoder


class SdrWorker(QThread):
    spectrumReady = Signal(object, float, float)
    iqReady = Signal(object)
    audioReady = Signal(object)
    smeterReady = Signal(float)
    infoReady = Signal(str)
    ft8Ready = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.fs = 1e6
        self.audio_rate = 48_000
        self.calibration_db = 0.0
        self.demod = Demodulator(self.fs, self.audio_rate)
        self.out_rate = self.demod.out_rate
        self.block_s = 0.025
        self.audio_sink = None
        self.emit_iq = False
        self.emit_audio = False
        self.ft8 = None
        self.ft8_enabled = False
        self.ft8_center = 1500.0
        self.ft8_mode = "FT8"
        self._ft8_tick = 0
        self._stop = False

    def set_source(self, source, fs: float) -> None:
        self.source = source
        self.fs = float(fs)
        self.demod = Demodulator(self.fs, self.audio_rate, self.demod.mode, self.demod.bw)
        self.out_rate = self.demod.out_rate
        if self.ft8_enabled:
            self.set_ft8(True, self.ft8_center, self.ft8_mode)

    def set_ft8(self, enabled: bool, center: float = 1500.0, mode: str = "FT8") -> None:
        self.ft8_enabled = bool(enabled)
        self.ft8_center = float(center)
        self.ft8_mode = mode
        self._ft8_tick = 0
        if enabled:
            cls = FT4Decoder if mode == "FT4" else FT8Decoder
            self.ft8 = cls(self.out_rate, emit=None, center=self.ft8_center)
        else:
            self.ft8 = None

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
            if self.emit_iq:
                self.iqReady.emit(iq)
            audio = self.demod.process(iq)
            if len(audio):
                if self.audio_sink is not None:
                    self.audio_sink.push(audio)
                if self.emit_audio:
                    self.audioReady.emit(audio)
                if self.ft8 is not None:
                    self.ft8.process(audio)
                    self._ft8_tick += 1
                    if self._ft8_tick >= 20:
                        self._ft8_tick = 0
                        try:
                            msg = self.ft8.try_decode()
                        except Exception:
                            msg = None
                        if msg:
                            self.ft8Ready.emit(msg)
            self.smeterReady.emit(power_dbfs(iq) + self.calibration_db)

    def stop(self) -> None:
        self._stop = True
