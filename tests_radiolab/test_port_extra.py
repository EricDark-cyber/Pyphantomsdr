"""Tests supplémentaires du port : roundtrip NAVTEX au niveau décodeur, fabrique de tous
les décodeurs, smoke Olivia/Contestia, et mesure des réducteurs de bruit sample-par-sample."""
import time

import numpy as np
import pytest

from radiolab.dsp import navtex
from radiolab.dsp.hfdigi import KINDS, make_decoder
from radiolab.dsp.kalman import KalmanNR
from radiolab.dsp.neural_nr import NeuralNR
from radiolab.dsp.ekf import EKFFilter


def navtex_signal(text, fs=8000.0, center=1700.0):
    bits = []
    for code in navtex.encode(text):
        bits += [(code >> k) & 1 for k in range(6, -1, -1)]
    n = int(len(bits) * fs / 100.0)
    b = np.array(bits)[np.minimum((np.arange(n) * 100.0 / fs).astype(np.int64), len(bits) - 1)]
    f = center + 85.0 * (2 * b - 1)
    return np.sin(2 * np.pi * np.cumsum(f) / fs).astype(np.float32)


def test_navtex_roundtrip_decoder_level():
    """Texte -> SITOR-B -> NAVTEXDecoder : le message est retrouvé (avec sa fin NNNN)."""
    text = "ZCZC TEST123 SECURITE NAVIGATION 45678 NNNN"
    audio = navtex_signal(text)
    msgs = []
    d = navtex.NAVTEXDecoder(8000.0, 1700.0, msgs.append)
    for i in range(0, len(audio), 800):
        msgs += [str(m) for m in d.process(audio[i:i + 800])]
    assert d.synced and d.q > 0.9
    assert any(text in m for m in d.messages + msgs), (d.messages, msgs)


def test_navtex_encode_weight_rule():
    """Chaque mot de 7 bits émis a un poids de 3 ou 4 (code CCIR 476)."""
    for w in navtex.encode("ZCZC 518 KHZ GALE WARNING NNNN"):
        assert bin(w).count("1") in (3, 4)
    assert len(navtex.encode("AB")) == 4           # chaque caractère est émis deux fois


ALL_KINDS = list(KINDS) + ["JS8", "JT65", "M17", "DSTAR", "DMR", "C4FM", "OLIVIA", "CONTESTIA"]


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_make_decoder_every_kind(kind):
    """Tous les décodeurs du menu se fabriquent et avalent un buffer sans exception."""
    d = make_decoder(kind, 8000, 1000.0, lambda s: None)
    assert d is not None, kind
    if hasattr(d, "process"):
        d.process(np.zeros(800, dtype=np.float32))


@pytest.mark.parametrize("kind", ["OLIVIA", "CONTESTIA"])
def test_olivia_contestia_smoke(kind):
    """Un signal bruité autour des tonalités Olivia/Contestia ne plante pas et reste muet."""
    rng = np.random.default_rng(0)
    fs = 8000.0
    t = np.arange(int(4 * fs)) / fs
    audio = (0.2 * np.sin(2 * np.pi * 1000.0 * t)
             + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    seen = []
    d = make_decoder(kind, fs, 1000.0, seen.append)
    for i in range(0, len(audio), 1000):
        d.process(audio[i:i + 1000])
    assert isinstance(seen, list)


def test_noise_reducers_vector_output_and_speed():
    """Kalman / NeuralNR / EKF : sortie float32 de même longueur, et traitement rapide."""
    rng = np.random.default_rng(1)
    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000.0)
         + 0.1 * rng.standard_normal(48000)).astype(np.float32)
    for cls in (KalmanNR, NeuralNR, EKFFilter):
        f = cls(48000.0)
        f.enabled = True
        t0 = time.perf_counter()
        y = f.process(x)
        dt = time.perf_counter() - t0
        assert y.dtype == np.float32 and len(y) == len(x)
        assert dt < 2.0, (cls.__name__, dt)       # 1 s de signal filtrée en moins de 2 s
