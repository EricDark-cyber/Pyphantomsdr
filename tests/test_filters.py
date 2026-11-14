"""Tests des filtres en flux (pyphantom/filters.py) : equivalence morcelé vs d'un seul coup.
Lancer : python tests/test_filters.py"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyphantom.filters import (  # noqa: E402
    LinearResampler,
    Mixer,
    PolyDDC,
    StreamFIR,
    bandpass_taps,
    exp_taps,
    lowpass_taps,
    taps_for_transition,
)



def peak_freq_complex(x, fs):
    spec = np.abs(np.fft.fftshift(np.fft.fft(x.astype(np.complex128) * np.hanning(len(x)))))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs))
    return float(freqs[np.argmax(spec)])

def chunked(block, x, sizes):
    out = []
    i = 0
    k = 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        out.append(block.process(x[i:i + n]))
        i += n
        k += 1
    return np.concatenate(out) if out else np.zeros(0, dtype=x.dtype)


def test_taps() -> None:
    h = lowpass_taps(3000.0, 48000.0, 101)
    assert abs(h.sum() - 1.0) < 1e-9, "gain DC != 1"
    assert len(h) % 2 == 1
    spec = np.abs(np.fft.rfft(h, 8192))
    freqs = np.fft.rfftfreq(8192, 1 / 48000.0)
    at_cut = np.interp(3000.0, freqs, spec)
    at_stop = np.max(spec[freqs > 2 * 3000])
    assert at_cut < 0.6 and at_stop < 0.02, f"reponse {at_cut:.3f} / {at_stop:.4f}"
    hb = bandpass_taps(1000, 2000, 48000, 255)
    assert np.iscomplexobj(hb)
    e = exp_taps(50e-6, 48000)
    assert abs(e.sum() - 1.0) < 1e-6
    n = taps_for_transition(48000, 1000)
    assert n % 2 == 1 and 31 <= n <= 1501
    print("taps ok (gain DC=1, attenuation, passe-bande complexe, RC, transition)")


def test_streamfir_streaming() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000).astype(np.float32)
    h = lowpass_taps(3000.0, 48000.0, 101)
    ref = StreamFIR(h).process(x)
    got = chunked(StreamFIR(h), x, [1, 7, 1000, 333, 64])
    assert len(got) == len(ref)
    assert np.allclose(got, ref, atol=1e-5), f"max diff {np.max(np.abs(got - ref))}"
    hd = lowpass_taps(2000.0, 48000.0, 81)
    refd = StreamFIR(hd, decim=4).process(x)
    gotd = chunked(StreamFIR(hd, decim=4), x, [13, 1024, 5, 777])
    assert abs(len(gotd) - len(refd)) <= 1
    assert np.allclose(gotd, refd[:len(gotd)], atol=1e-5)
    print(f"StreamFIR ok (flux identique au coup par coup, decim=4 : {len(refd)} ech.)")


def test_polyddc() -> None:
    fs_in, m, pass_hz = 1_000_000.0, 10, 20_000.0
    n = 200_000
    t = np.arange(n) / fs_in
    tone = np.exp(2j * np.pi * 5_000.0 * t).astype(np.complex64)
    ddc = PolyDDC(fs_in, m, pass_hz)
    ddc.set_offset(5_000.0)  # ramene le ton de 5 kHz a 0 Hz
    ref = ddc.process(tone)
    assert abs(len(ref) - n / m) < m + 2, f"len {len(ref)} vs {n/m}"
    steady = ref[len(ref) // 4:]
    peak = peak_freq_complex(steady, fs_in / m)
    assert abs(peak) < 3 * (fs_in / m) / len(steady), f"pic DDC a {peak:.1f} Hz, attendu ~0"
    ddc2 = PolyDDC(fs_in, m, pass_hz)
    ddc2.set_offset(5_000.0)
    got = chunked(ddc2, tone, [4096, 1000, 65536])
    assert np.max(np.abs(got[:len(ref)] - ref)) < 1e-3, "DDC non equivalent en flux"
    print(f"PolyDDC ok (decim x{m}, ton 5 kHz -> {peak:.2f} Hz, flux identique)")


def test_mixer() -> None:
    fs = 48_000.0
    n = 48_000
    t = np.arange(n) / fs
    x = np.exp(2j * np.pi * 3000.0 * t).astype(np.complex64)
    mix = Mixer(fs)
    mix.set_offset(1000.0)
    ref = mix.process(x)
    steady = ref[len(ref) // 4:]
    peak = peak_freq_complex(steady, fs)
    assert abs(peak - 2000.0) < fs / len(steady), f"pic mixer a {peak} Hz, attendu 2000"
    mix2 = Mixer(fs)
    mix2.set_offset(1000.0)
    got = chunked(mix2, x, [100, 4096, 17])
    assert np.allclose(got, ref, atol=1e-5)
    print(f"Mixer ok (3000-1000 -> {peak:.0f} Hz, phase continue en flux)")


def test_linear_resampler() -> None:
    fs_in, fs_out = 44_100.0, 48_000.0
    n = 44_100
    t = np.arange(n) / fs_in
    x = np.cos(2 * np.pi * 1000.0 * t).astype(np.float32)
    ref = LinearResampler(fs_in, fs_out).process(x)
    assert abs(len(ref) - n * fs_out / fs_in) < 2
    got = chunked(LinearResampler(fs_in, fs_out), x, [500, 8192, 33])
    assert np.allclose(got[:len(ref)], ref, atol=1e-4)
    steady = ref[len(ref) // 4:]
    spec = np.abs(np.fft.rfft(steady * np.hanning(len(steady))))
    freqs = np.fft.rfftfreq(len(steady), 1 / fs_out)
    peak = freqs[np.argmax(spec)]
    assert abs(peak - 1000.0) < 20, f"pic resample a {peak} Hz"
    print(f"LinearResampler ok ({len(ref)} ech., pic {peak:.0f} Hz, flux identique)")


if __name__ == "__main__":
    test_taps()
    test_streamfir_streaming()
    test_polyddc()
    test_mixer()
    test_linear_resampler()
    print("Tous les tests filtres passent.")
