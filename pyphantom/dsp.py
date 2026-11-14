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


class FirDecim:
    def __init__(self, down: int, cutoff_hz: float, fs: float, dtype=np.complex128):
        self.down = int(down)
        self.dtype = dtype
        if self.down <= 1:
            self.h = None
            return
        numtaps = min(int(8 * self.down) + 1, 4096)
        if numtaps % 2 == 0:
            numtaps += 1
        cutoff = min(cutoff_hz, 0.49 * fs)
        self.h = signal.firwin(numtaps, cutoff, fs=fs) / self.down
        self.hr = self.h[::-1].copy()
        self.tail = np.zeros(numtaps - 1, dtype=dtype)

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.h is None:
            return np.asarray(x, dtype=self.dtype)
        buf = np.concatenate([self.tail, x])
        L = len(self.h)
        n_out = (len(buf) - L) // self.down
        if n_out <= 0:
            self.tail = buf
            return np.zeros(0, dtype=self.dtype)
        out = np.zeros(n_out, dtype=np.complex128 if self.dtype == np.complex128 else np.float64)
        for k in range(L):
            out += self.hr[k] * buf[k :: self.down][:n_out]
        self.tail = buf[n_out * self.down :]
        return out


class Resampler:
    def __init__(self, fs_in: float, fs_out: int):
        fs_in_i = int(round(fs_in))
        fs_out_i = int(round(fs_out))
        g = int(np.gcd(fs_in_i, fs_out_i))
        self.up = fs_out_i // g
        self.down = fs_in_i // g
        rate_up = fs_in_i * self.up
        cutoff = 0.45 * min(fs_in_i, fs_out_i)
        numtaps = int(np.clip(8 * max(self.up, self.down) + 1, 33, 4096))
        if numtaps % 2 == 0:
            numtaps += 1
        self.h = signal.firwin(numtaps, cutoff, fs=rate_up)
        self.hr = self.h[::-1].copy()
        self.tail = np.zeros(numtaps - 1, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.up == 1 and self.down == 1:
            return x
        z = np.zeros(len(x) * self.up, dtype=np.float64)
        z[:: self.up] = x
        buf = np.concatenate([self.tail, z])
        L = len(self.h)
        n_out = (len(buf) - L) // self.down
        if n_out <= 0:
            self.tail = buf
            return np.zeros(0, dtype=np.float64)
        out = np.zeros(n_out, dtype=np.float64)
        for k in range(L):
            out += self.hr[k] * buf[k :: self.down][:n_out]
        self.tail = buf[n_out * self.down :]
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
        fs1_target = max(2.5 * bw, 2.2 * self.audio_rate)
        k = max(1, int(np.ceil(fs1_target / self.audio_rate)))
        down = max(1, int(round(self.fs_in / (self.audio_rate * k))))
        self._decim = FirDecim(down, min(bw, 0.49 * self.fs_in), self.fs_in)
        self.fs_demod = self.fs_in / down
        down2 = max(1, int(round(self.fs_demod / self.audio_rate)))
        self.out_rate = self.fs_demod / down2
        self._decim2 = FirDecim(down2, 0.45 * self.out_rate, self.fs_demod, dtype=np.float64)
        if self.mode == "CW":
            cutoff = self.bfo + bw / 2.0
        else:
            cutoff = bw / 2.0
        cutoff = min(cutoff, 0.45 * self.fs_demod)
        self._lp = FirLP(cutoff, self.fs_demod)
        self._lp_pre = FirLP(min(bw / 2.0, 0.45 * self.fs_demod), self.fs_demod)
        self._lp_chan = FirLP(min(bw, 0.45 * self.fs_demod), self.fs_demod)
        if self.mode in ("NFM", "WFM"):
            a = np.exp(-1.0 / (self.out_rate * DEEMPHASIS_TAU))
            self._deemph_b = np.array([1.0 - a])
            self._deemph_a = np.array([1.0, -a])
            self._deemph_zi = np.zeros(1)

    def process(self, iq: np.ndarray) -> np.ndarray:
        iq = np.asarray(iq, dtype=np.complex64)
        if len(iq) == 0:
            return np.zeros(0, dtype=np.float32)
        x = self._decim.process(iq)
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        fs = self.fs_demod
        m = self.mode
        n = self._n + np.arange(len(x), dtype=np.float64)
        if m == "AM":
            env = np.abs(self._lp_pre(x)).astype(np.float64)
            base = self._lp(env - env.mean())
        elif m in ("NFM", "WFM"):
            ref_dev = 5_000.0 if m == "NFM" else 75_000.0
            y = self._lp_pre(x)
            prev = np.concatenate(([self._prev], y[:-1]))
            d = np.angle(y * np.conj(prev)) / (2.0 * np.pi) * fs / ref_dev
            self._prev = y[-1]
            base = self._lp(d)
        elif m == "USB":
            y = x * np.exp(-1j * np.pi * self.bw * n / fs)
            z = self._lp(y)
            base = 2.0 * np.real(z * np.exp(1j * np.pi * self.bw * n / fs))
        elif m == "LSB":
            y = x * np.exp(1j * np.pi * self.bw * n / fs)
            z = self._lp(y)
            base = 2.0 * np.real(z * np.exp(-1j * np.pi * self.bw * n / fs))
        else:
            y = x * np.exp(-2j * np.pi * self.bfo * n / fs)
            base = 2.0 * np.real(self._lp(y))
        self._n = n[-1] + 1.0
        chan_db = power_dbfs(self._lp_chan(x))
        audio = self._decim2.process(base)
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
