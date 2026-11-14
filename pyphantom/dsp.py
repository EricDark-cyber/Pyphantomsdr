"""Chaine DSP : spectre, demodulation, S-metre (filtres en flux : filters.py)."""
from __future__ import annotations

import numpy as np

from .filters import (
    LinearResampler,
    Mixer,
    PolyDDC,
    StreamFIR,
    exp_taps,
    lowpass_taps,
    taps_for_transition,
)

MODES = ("AM", "NFM", "WFM", "USB", "LSB", "CW")
MODE_BW = {
    "AM": 10_000.0,
    "NFM": 12_000.0,
    "WFM": 200_000.0,
    "USB": 2_700.0,
    "LSB": 2_700.0,
    "CW": 600.0,
}
BFO_CW = 700.0
DEEMPHASIS_TAU = 50e-6


def compute_spectrum_db(iq: np.ndarray, n_fft: int = 2048) -> np.ndarray:
    n = min(n_fft, len(iq))
    x = np.asarray(iq[-n:], dtype=np.complex128) * np.hanning(n)
    spec = np.fft.fftshift(np.fft.fft(x)) / n
    return 20.0 * np.log10(np.abs(spec) + 1e-12)


def s_unit_str(dbm: float, band_mhz: float = 14.0) -> str:
    s9 = -73.0 if band_mhz <= 30.0 else -93.0
    if dbm <= s9 - 54.0:
        return "S0"
    if dbm < s9:
        s = int(round(9.0 + (dbm - s9) / 6.0))
        return f"S{max(1, s)}"
    over = int(round((dbm - s9) / 10.0)) * 10
    return f"S9+{over}" if over > 0 else "S9"


def power_dbfs(iq: np.ndarray) -> float:
    p = float(np.mean(np.abs(np.asarray(iq, dtype=np.complex128)) ** 2))
    return 10.0 * np.log10(p + 1e-20)


class Resampler(LinearResampler):
    """Resampling lineaire en flux (alias de LinearResampler, compat tests)."""


def _lp(cutoff_hz: float, fs: float) -> StreamFIR:
    ntaps = taps_for_transition(fs, max(0.15 * cutoff_hz, 1.0))
    return StreamFIR(lowpass_taps(cutoff_hz, fs, ntaps))


class Demodulator:
    """Chaine : DDC polyphase (filtre de canal + decimation) -> demod -> audio 48 kHz."""

    def __init__(self, fs_in: float, audio_rate: int = 48000, mode: str = "NFM", bw: float | None = None):
        self.fs_in = float(fs_in)
        self.audio_rate = int(audio_rate)
        self.mode = mode
        self.bw = float(bw if bw is not None else MODE_BW[mode])
        self.bfo = BFO_CW
        self.agc_enabled = True
        self.agc_target = 0.05
        self.squelch_db = -120.0
        self._prev = 1.0 + 0.0j
        self._agc_env = 1e-6
        self._agc_gain = 1.0
        self._rebuild()

    def set_mode(self, mode: str, bw: float | None = None) -> None:
        self.mode = mode
        if bw is not None:
            self.bw = float(bw)
        else:
            self.bw = float(MODE_BW[mode])
        self._rebuild()

    def set_bw(self, bw: float) -> None:
        self.bw = float(bw)
        self._rebuild()

    def _rebuild(self) -> None:
        bw = self.bw
        fs1_target = max(2.5 * bw, 2.2 * self.audio_rate)
        m = max(1, int(round(self.fs_in / fs1_target)))
        if self.mode == "CW":
            pass1 = self.bfo + bw / 2.0
        elif self.mode in ("USB", "LSB"):
            pass1 = bw
        else:
            pass1 = bw / 2.0
        self._ddc = PolyDDC(self.fs_in, m, pass1)
        self.fs_demod = self.fs_in / m
        self.out_rate = float(self.audio_rate)
        if self.mode == "CW":
            cutoff = self.bfo + bw / 2.0
        else:
            cutoff = bw / 2.0
        cutoff = min(cutoff, 0.45 * self.fs_demod)
        self._lp = _lp(cutoff, self.fs_demod)
        self._lp_pre = _lp(min(bw / 2.0, 0.45 * self.fs_demod), self.fs_demod)
        self._mix = Mixer(self.fs_demod)
        self._mix_back = Mixer(self.fs_demod)
        if self.mode == "USB":
            self._mix.set_offset(-bw / 2.0)
            self._mix_back.set_offset(bw / 2.0)
        elif self.mode == "LSB":
            self._mix.set_offset(bw / 2.0)
            self._mix_back.set_offset(-bw / 2.0)
        else:  # CW
            self._mix.set_offset(-self.bfo)
        self._resamp = LinearResampler(self.fs_demod, self.audio_rate)
        self._deemph = StreamFIR(exp_taps(DEEMPHASIS_TAU, self.out_rate)) if self.mode in ("NFM", "WFM") else None

    def process(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq, dtype=np.complex64)
        if len(iq) == 0:
            return np.zeros(0, dtype=np.float32)
        x = self._ddc.process(iq)
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        fs = self.fs_demod
        m = self.mode
        if m == "AM":
            env = np.abs(self._lp_pre.process(x)).astype(np.float64)
            base = self._lp.process(env - env.mean())
        elif m in ("NFM", "WFM"):
            ref_dev = 5_000.0 if m == "NFM" else 75_000.0
            y = self._lp_pre.process(x)
            prev = np.concatenate(([self._prev], y[:-1]))
            d = np.angle(y * np.conj(prev)) / (2.0 * np.pi) * fs / ref_dev
            self._prev = y[-1]
            base = self._lp.process(d)
        elif m == "USB":
            z = self._lp.process(self._mix.process(x))
            base = 2.0 * np.real(self._mix_back.process(z))
        elif m == "LSB":
            z = self._lp.process(self._mix.process(x))
            base = 2.0 * np.real(self._mix_back.process(z))
        else:  # CW
            base = 2.0 * np.real(self._lp.process(self._mix.process(x)))
        chan_db = power_dbfs(x)
        audio = self._resamp.process(np.asarray(base, dtype=np.float32))
        if self._deemph is not None and len(audio):
            audio = self._deemph.process(audio.astype(np.float64)).astype(np.float32)
        if len(audio):
            if chan_db < self.squelch_db:
                audio = np.zeros_like(audio)
            elif self.agc_enabled:
                p = float(np.mean(audio.astype(np.float64) ** 2))
                env = 0.95 * self._agc_env + 0.05 * np.sqrt(p + 1e-20)
                self._agc_env = env
                g = float(np.clip(self.agc_target / max(env, 1e-6), 0.1, 100.0))
                self._agc_gain = 0.7 * self._agc_gain + 0.3 * g
                audio = audio * self._agc_gain
        return audio.astype(np.float32)
