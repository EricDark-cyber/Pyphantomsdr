"""Tests du decodeur FT8 : aller-retour complet message -> CRC -> LDPC -> audio 8-FSK -> tons -> bits -> LDPC -> CRC.
Lancer : python tests/test_ft8.py"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyphantom.ft8.decoder import FT8Decoder, GRAY, FT8_COSTAS  # noqa: E402
from pyphantom.ft8.ft8_crc import add_crc14, check_crc14  # noqa: E402
from pyphantom.ft8.ft8_ldpc import FT8LDPC, build_H  # noqa: E402
from pyphantom.ft8.ft8_unpack import unpack_callsign, unpack_ft8  # noqa: E402

FS = 48_000
CENTER = 1500.0
TONE = 6.25
TSYM = 0.16


def gf2_rref(M):
    A = M.copy() % 2
    rows, cols = A.shape
    pivots = []
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return A, pivots


def null_space_G(H):
    R, pivots = gf2_rref(H)
    free = [c for c in range(H.shape[1]) if c not in pivots]
    n = H.shape[1]
    G = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        G[i, f] = 1
        for r, p in enumerate(pivots):
            if R[r, f]:
                G[i, p] = 1
    return G


def make_message_bits(call_n=123456, grid_n=1761):
    val = (0 << 74) | (call_n << 46) | (grid_n << 31)
    return [(val >> (76 - i)) & 1 for i in range(77)]


def synth_frame(codeword174) -> np.ndarray:
    syms = []
    for pos in range(79):
        if pos < 7 or 36 <= pos < 43 or pos >= 72:
            k = pos if pos < 7 else (pos - 36 if pos < 43 else pos - 72)
            syms.append(FT8_COSTAS[k])
        else:
            di = pos - 7 if pos < 36 else pos - 43 + 29
            v = (codeword174[di * 3] << 2) | (codeword174[di * 3 + 1] << 1) | codeword174[di * 3 + 2]
            syms.append(GRAY.index(int(v)))
    n = int(FS * TSYM)
    out = np.zeros(79 * n, dtype=np.float64)
    phase = 0.0
    for s, t in enumerate(syms):
        w = 2 * np.pi * (CENTER + t * TONE) / FS
        for k in range(n):
            out[s * n + k] = np.sin(phase)
            phase += w
    rng = np.random.default_rng(7)
    out += 0.05 * rng.standard_normal(len(out))
    return out.astype(np.float32)


def test_unpack_crc() -> None:
    msg77 = make_message_bits()
    expected_call = unpack_callsign(123456)
    assert expected_call in unpack_ft8(msg77)
    assert check_crc14(add_crc14(msg77))
    print(f"unpack/CRC ok (ex. 'CQ {expected_call} ...')")


def test_ldpc_roundtrip() -> None:
    H = build_H()
    G = null_space_G(H)
    ldpc = FT8LDPC()
    rng = np.random.default_rng(3)
    m = rng.integers(0, 2, size=91, dtype=np.uint8)
    c = (m @ G) % 2
    assert np.all((H @ c) % 2 == 0)
    soft = np.where(c == 0, 1.0, -1.0)
    hard = ldpc.decode(soft)
    assert np.all((H @ hard) % 2 == 0), "syndrome non nul apres BP"
    assert np.array_equal(hard, c), "BP n'a pas retrouve le mot de code"
    print("LDPC ok (mot de code retrouve, syndrome nul)")


def test_audio_roundtrip() -> None:
    H = build_H()
    G = null_space_G(H)
    rng = np.random.default_rng(11)
    m = rng.integers(0, 2, size=91, dtype=np.uint8)
    cwd = (m @ G) % 2
    audio = synth_frame(cwd)
    dec = FT8Decoder(FS, emit=None, mode="FT8", center=CENTER)
    step = int(FS * 0.5)
    for i in range(0, len(audio), step):
        dec.process(audio[i:i + step])
    frame_len = int(FS * dec.period)
    chunk = dec.buf[-frame_len:]
    tones = dec._detect_tones(chunk)
    sync = dec._find_costas(tones)
    assert sync is not None, "Costas non trouve"
    bits = dec._tones_to_bits(tones, sync)
    match = np.mean(np.asarray(bits[:174]) == cwd)
    print(f"tons->bits : {match*100:.1f}% de bits corrects (attendu ~100%)")
    assert match > 0.95
    soft = np.array([1.0 if b == 0 else -1.0 for b in bits[:174]])
    hard = dec.ldpc.decode(soft)
    assert np.all((H @ hard) % 2 == 0)
    print("chaine audio -> bits -> LDPC ok (syndrome nul)")


if __name__ == "__main__":
    test_unpack_crc()
    test_ldpc_roundtrip()
    test_audio_roundtrip()
    print("Tous les tests FT8 passent.")
