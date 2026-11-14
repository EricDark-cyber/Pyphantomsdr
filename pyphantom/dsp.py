"""Chaine DSP : spectre, waterfall, demodulation, S-metre (inspire de PhantomSDR-Plus)."""
from __future__ import annotations

import numpy as np
from scipy import signal

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


class FirLP:
    def __init__(self, cutoff_hz: float, fs: float):
        cutoff = float(min(cutoff_hz, 0.49 * fs))
        numtaps = int(np.clip(63.0 * fs / max(cutoff, 1.0), 33, 1025))
        if numtaps % 2 == 0:
            numtaps += 1
        self.h = signal.firwin(numtaps, cutoff, fs=fs)
        self.zi = signal.lfilter_zi(self.h, [1.0]) * 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.lfilter(self.h, [1.0], x, zi=self.zi)
        return y


class Resampler:
    def __init__(self, fs_in: float, fs_out: int):
        fs_in_i = int(round(fs_in))
        fs_out_i = int(round(fs_out))
        g = int(np.gcd(fs_in_i, fs_out_i))
        self.up = fs_out_i // g
        self.down = fs_in_i // g
        rate_up = fs_in_i * self.up
        cutoff = 0.45 * min(fs_in_i, fs_out_i)
        numtaps = int(np.clip(63.0 * rate_up / cutoff, 33, 4097))
        if numtaps % 2 == 0:
            numtaps += 1
        self.h = signal.firwin(numtaps, cutoff, fs=rate_up)
        self.zi = signal.lfilter_zi(self.h, [1.0]) * 0.0
        self.phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.up == 1 and self.down == 1:
            return x
        z = np.zeros(len(x) * self.up, dtype=np.float64)
        z[:: self.up] = x
        y, self.zi = signal.lfilter(self.h, [1.0], z, zi=self.zi)
        out = y[self.phase :: self.down]
        self.phase = (self.phase + len(out) * self.down - len(y)) % self.down
        return out


class Demodulator:
    def __init__(self, fs_in: float, audio_rate: int = 48000, mode: str = "NFM", bw: float | None = None):
        self.fs_in = float(fs_in)
        self.audio_rate = int(audio_rate)
        self.mode = mode
        self.bw = float(bw if bw is not None else MODE_BW[mode])
        self.bfo = BFO_CW
        self.agc_enabled = True
        self.agc_target = 0.05
        self.squelch_db = -120.0
        self._n = 0
        self._prev = 1.0 + 0.0j
        self._agc_env = 1e-6
        self._agc_gain = 1.0
        self._deemph_zi = np.zeros(1)
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
        if self.mode == "CW":
            cutoff = self.bfo + bw / 2.0
        else:
            cutoff = bw / 2.0
        cutoff = min(cutoff, 0.45 * self.audio_rate)
        self._lp = FirLP(cutoff, self.fs_in)
        self._lp_pre = FirLP(min(bw / 2.0, 0.45 * self.fs_in), self.fs_in)
        self._lp_chan = FirLP(min(bw, 0.45 * self.fs_in), self.fs_in)
        self._resamp = Resampler(self.fs_in, self.audio_rate)
        if self.mode in ("NFM", "WFM"):
            a = np.exp(-1.0 / (self.audio_rate * DEEMPHASIS_TAU))
            self._deemph_b = np.array([1.0 - a])
            self._deemph_a = np.array([1.0, -a])
            self._deemph_zi = np.zeros(1)

    def process(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq, dtype=np.complex64)
        if len(iq) == 0:
            return np.zeros(0, dtype=np.float32)
        m = self.mode
        n = self._n + np.arange(len(iq), dtype=np.float64)
        if m == "AM":
            env = np.abs(self._lp_pre(iq)).astype(np.float64)
            base = self._lp(env - env.mean())
        elif m in ("NFM", "WFM"):
            ref_dev = 5_000.0 if m == "NFM" else 75_000.0
            y = self._lp_pre(iq)
            prev = np.concatenate(([self._prev], y[:-1]))
            d = np.angle(y * np.conj(prev)) / (2.0 * np.pi) * self.fs_in / ref_dev
            self._prev = y[-1]
            base = self._lp(d)
        elif m == "USB":
            y = iq * np.exp(-1j * np.pi * self.bw * n / self.fs_in)
            z = self._lp(y)
            base = 2.0 * np.real(z * np.exp(1j * np.pi * self.bw * n / self.fs_in))
        elif m == "LSB":
            y = iq * np.exp(1j * np.pi * self.bw * n / self.fs_in)
            z = self._lp(y)
            base = 2.0 * np.real(z * np.exp(-1j * np.pi * self.bw * n / self.fs_in))
        else:
            y = iq * np.exp(-2j * np.pi * self.bfo * n / self.fs_in)
            base = 2.0 * np.real(self._lp(y))
        self._n = n[-1] + 1.0
        chan_db = power_dbfs(self._lp_chan(iq))
        audio = self._resamp.process(base)
        if m in ("NFM", "WFM") and len(audio):
            audio, self._deemph_zi = signal.lfilter(self._deemph_b, self._deemph_a, audio, zi=self._deemph_zi)
        if len(audio):
            if chan_db < self.squelch_db:
                audio = np.zeros_like(audio)
            elif self.agc_enabled:
                p = float(np.mean(audio * audio))
                env = 0.95 * self._agc_env + 0.05 * np.sqrt(p + 1e-20)
                self._agc_env = env
                g = float(np.clip(self.agc_target / max(env, 1e-6), 0.1, 100.0))
                self._agc_gain = 0.7 * self._agc_gain + 0.3 * g
                audio = audio * self._agc_gain
        return audio.astype(np.float32)
