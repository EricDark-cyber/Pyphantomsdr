"""Tests hors-ligne du DSP sur signaux synthétiques. Lancer : python tests/test_dsp.py"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyphantom.dsp import (  # noqa: E402
    Demodulator,
    Resampler,
    compute_spectrum_db,
    power_dbfs,
    s_unit_str,
)

FS = 1_000_000.0
DUR = 0.5
AUDIO_RATE = 48_000


def make_t(n: int = int(FS * DUR)) -> np.ndarray:
    return np.arange(n) / FS


def audio_peak(audio: np.ndarray, fs: float) -> float:
    a = audio[len(audio) // 10 : -len(audio) // 10]
    if len(a) < 256:
        raise AssertionError("audio trop court")
    win = np.hanning(len(a))
    spec = np.abs(np.fft.rfft(a * win))
    freqs = np.fft.rfftfreq(len(a), 1.0 / fs)
    return float(freqs[np.argmax(spec)])


def test_am() -> None:
    t = make_t()
    r = (1.0 + 0.6 * np.cos(2 * np.pi * 1000 * t)).astype(np.complex64)
    d = Demodulator(FS, AUDIO_RATE, "AM", 10_000)
    audio = np.concatenate([d.process(r[i : i + 50_000]) for i in range(0, len(r), 50_000)])
    peak = audio_peak(audio, AUDIO_RATE)
    assert abs(peak - 1000) < 60, f"AM pic à {peak} Hz, attendu 1000"
    print(f"AM ok (pic {peak:.0f} Hz)")


def test_nfm() -> None:
    t = make_t()
    inst = 5000.0 * np.cos(2 * np.pi * 1000 * t)
    phase = 2 * np.pi * np.cumsum(inst) / FS
    r = np.exp(1j * phase).astype(np.complex64)
    d = Demodulator(FS, AUDIO_RATE, "NFM", 12_000)
    audio = np.concatenate([d.process(r[i : i + 50_000]) for i in range(0, len(r), 50_000)])
    peak = audio_peak(audio, AUDIO_RATE)
    assert abs(peak - 1000) < 60, f"NFM pic à {peak} Hz, attendu 1000"
    print(f"NFM ok (pic {peak:.0f} Hz)")


def test_usb_lsb_cw() -> None:
    t = make_t()
    n = len(t)
    d = Demodulator(FS, AUDIO_RATE, "USB", 2_700)
    r = np.exp(2j * np.pi * 1500 * t).astype(np.complex64)
    audio = np.concatenate([d.process(r[i : i + 50_000]) for i in range(0, n, 50_000)])
    peak = audio_peak(audio, AUDIO_RATE)
    assert abs(peak - 1500) < 60, f"USB pic à {peak} Hz, attendu 1500"
    print(f"USB ok (pic {peak:.0f} Hz)")

    d = Demodulator(FS, AUDIO_RATE, "LSB", 2_700)
    r = np.exp(-2j * np.pi * 1500 * t).astype(np.complex64)
    audio = np.concatenate([d.process(r[i : i + 50_000]) for i in range(0, n, 50_000)])
    peak = audio_peak(audio, AUDIO_RATE)
    assert abs(peak - 1500) < 60, f"LSB pic à {peak} Hz, attendu 1500"
    print(f"LSB ok (pic {peak:.0f} Hz)")

    d = Demodulator(FS, AUDIO_RATE, "CW", 600)
    r = np.ones(n, dtype=np.complex64)
    audio = np.concatenate([d.process(r[i : i + 50_000]) for i in range(0, n, 50_000)])
    peak = audio_peak(audio, AUDIO_RATE)
    assert abs(peak - 700) < 60, f"CW pic à {peak} Hz, attendu 700 (BFO)"
    print(f"CW ok (pic {peak:.0f} Hz)")


def test_resampler() -> None:
    t = make_t()
    x = np.cos(2 * np.pi * 1000 * t)
    r = Resampler(FS, AUDIO_RATE)
    y = np.concatenate([r.process(x[i : i + 50_000]) for i in range(0, len(x), 50_000)])
    peak = audio_peak(y, AUDIO_RATE)
    assert abs(peak - 1000) < 30, f"resampler pic à {peak} Hz, attendu 1000"
    assert abs(len(y) - len(x) * AUDIO_RATE / FS) < 100, "mauvaise longueur de sortie"
    print(f"Resampler ok (pic {peak:.0f} Hz, {len(y)} éch.)")


def test_spectrum_and_smeter() -> None:
    t = make_t(int(FS * 0.05))
    r = (0.1 * np.exp(2j * np.pi * 100_000 * t)).astype(np.complex64)
    db = compute_spectrum_db(r, 2048)
    peak_bin = int(np.argmax(db))
    peak_freq = (peak_bin - 1024) * FS / 2048
    assert abs(peak_freq - 100_000) < FS / 2048, f"pic spectre à {peak_freq} Hz"
    p = power_dbfs(r)
    assert abs(p - 20 * np.log10(0.1)) < 0.5, f"puissance {p} dBFS"
    assert s_unit_str(-73.0, 14.0) == "S9"
    assert s_unit_str(-53.0, 14.0) == "S9+20"
    assert s_unit_str(-121.0, 14.0) == "S1"
    print(f"Spectre/S-mètre ok (pic {peak_freq:.0f} Hz, {p:.1f} dBFS)")


def test_squelch() -> None:
    t = make_t()
    r = (1e-4 * np.exp(2j * np.pi * 1000 * t)).astype(np.complex64)
    d = Demodulator(FS, AUDIO_RATE, "NFM", 12_000)
    d.agc_enabled = False
    d.squelch_db = -30.0
    audio = np.concatenate([d.process(r[i : i + 50_000]) for i in range(0, len(r), 50_000)])
    assert np.max(np.abs(audio)) < 1e-6, "squelch devrait couper le signal faible"
    print("Squelch ok")


if __name__ == "__main__":
    test_am()
    test_nfm()
    test_usb_lsb_cw()
    test_resampler()
    test_spectrum_and_smeter()
    test_squelch()
    print("Tous les tests DSP passent.")
