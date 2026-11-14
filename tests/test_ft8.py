"""Tests du decodeur FT8 : vecteur CRC officiel WSJT-X, unpack/pack, LDPC (encode
systematique + BP), puis aller-retour complet audio 8-FSK -> tons -> bits -> LDPC ->
CRC -> message, en passant par process()/try_decode comme en reception reelle.
Lancer : python tests/test_ft8.py"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyphantom.ft8.decoder import FT8Decoder, FT8_COSTAS, GRAY, GRAY_ENC  # noqa: E402
from pyphantom.ft8.ft8_crc import add_crc14, check_crc14, crc14  # noqa: E402
from pyphantom.ft8.ft8_ldpc import FT8LDPC, build_H  # noqa: E402
from pyphantom.ft8.ft8_unpack import grid4_to_int, pack_standard, unpack_ft8  # noqa: E402

FS = 48_000
CENTER = 1500.0
TONE = 6.25
TSYM = 0.16


def gf2_solve(A, b):
    """Resout A x = b sur GF(2) (A m x n, rang plein en colonnes)."""
    M = np.concatenate([A % 2, (b % 2).reshape(-1, 1)], axis=1).astype(np.uint8)
    rows, cols = M.shape
    r = 0
    piv = []
    for c in range(cols - 1):
        p = next((i for i in range(r, rows) if M[i, c]), None)
        if p is None:
            continue
        M[[r, p]] = M[[p, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    for i in range(r, rows):
        if M[i, -1] and not M[i, :-1].any():
            return None
    x = np.zeros(cols - 1, dtype=np.uint8)
    for i, c in enumerate(piv):
        x[c] = M[i, -1]
    return x


def encode_systematic(m91, H):
    """Encode WSJT-X : mot de code = [m91 | parite], H @ cwd = 0."""
    m91 = np.asarray(m91, dtype=np.uint8)
    rhs = (H[:, :91] @ m91) % 2
    p = gf2_solve(H[:, 91:], rhs)
    assert p is not None, "H[:, 91:] non inversible"
    cwd = np.concatenate([m91, p]).astype(np.uint8)
    assert np.all((H @ cwd) % 2 == 0)
    return cwd


def synth_frame(codeword174) -> np.ndarray:
    """Audio 8-FSK d'une trame FT8 (S7 D29 S7 D29 S7), graymap WSJT-X."""
    syms = []
    for pos in range(79):
        if pos < 7 or 36 <= pos < 43 or pos >= 72:
            k = pos if pos < 7 else (pos - 36 if pos < 43 else pos - 72)
            syms.append(FT8_COSTAS[k])
        else:
            di = pos - 7 if pos < 36 else pos - 43 + 29
            v = (int(codeword174[di * 3]) << 2) | (int(codeword174[di * 3 + 1]) << 1) | int(codeword174[di * 3 + 2])
            syms.append(GRAY_ENC[v])
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


def test_crc_vector() -> None:
    # Vecteur officiel (weakmon, teste contre la routine C de WSJT-X)
    msg = [0] * 82
    for p in (3, 7, 44, 45, 46, 51, 61, 71):
        msg[p] = 1
    expected = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
    assert crc14(msg[:77]) == expected, f"CRC-14 wrong: {crc14(msg[:77])}"
    m77 = pack_standard("CQ", "F1ABC", grid="JN18")
    assert check_crc14(add_crc14(m77))
    print("CRC-14 ok (vecteur officiel WSJT-X + aller-retour)")


def test_unpack_pack() -> None:
    m77 = pack_standard("CQ", "F1ABC", grid="JN18")
    assert unpack_ft8(m77) == "CQ F1ABC JN18", unpack_ft8(m77)
    m77b = pack_standard("F1ABC", "K1ABC", ir=1, report=-12)
    assert unpack_ft8(m77b) == "F1ABC K1ABC R-12", unpack_ft8(m77b)
    m77c = pack_standard("F1ABC", "K1ABC", ir=0, report=7)
    assert unpack_ft8(m77c) == "F1ABC K1ABC +7", unpack_ft8(m77c)
    m77d = pack_standard("F1ABC", "K1ABC", ir=0, report=-41)
    assert unpack_ft8(m77d) == "F1ABC K1ABC -41", unpack_ft8(m77d)
    print("unpack/pack ok (CQ grid, report R+/- avec wrap, report negatif)")


def test_ldpc_roundtrip() -> None:
    H = build_H()
    ldpc = FT8LDPC()
    rng = np.random.default_rng(3)
    for trial in range(5):
        m91 = rng.integers(0, 2, size=91, dtype=np.uint8)
        cwd = encode_systematic(m91, H)
        assert np.array_equal(cwd[:91], m91)
        soft = np.where(cwd == 0, 4.0, -4.0)
        hard = ldpc.decode(soft)
        assert np.all((H @ hard) % 2 == 0)
        assert np.array_equal(hard, cwd), f"BP sans bruit trial {trial}"
        noisy = soft + rng.normal(0.0, 1.6, size=174)
        hard2 = ldpc.decode(noisy)
        assert np.all((H @ hard2) % 2 == 0), f"syndrome non nul trial {trial}"
        assert np.array_equal(hard2[:91], m91), f"message non retrouve trial {trial}"
    print("LDPC ok (encode systematique, BP sans bruit exact, BP bruite corrige)")


def test_audio_roundtrip() -> None:
    H = build_H()
    m77 = pack_standard("CQ", "F1ABC", grid="JN18")
    m91 = np.array(add_crc14(m77), dtype=np.uint8)
    cwd = encode_systematic(m91, H)
    audio = synth_frame(cwd)

    dec = FT8Decoder(FS, emit=None, mode="FT8", center=CENTER)
    step = int(FS * 0.025)  # blocs de 25 ms comme le worker
    rng = np.random.default_rng(9)
    stream = np.concatenate([audio, 0.02 * rng.standard_normal(int(FS * 3)).astype(np.float32)])
    got = []
    for i in range(0, len(stream), step):
        dec.process(stream[i:i + step])
        if (i // step) % 20 == 19:
            msg = dec.try_decode()
            if msg:
                got.append(msg)
    assert got, "aucun message decode en streaming"
    assert any("CQ F1ABC JN18" in g for g in got), f"messages: {got}"
    print(f"streaming ok : {got[0]!r} decode (attendu 'CQ F1ABC JN18 ...')")

    # verification intermediaire : tons -> bits (decodeur frais, trame alignee)
    dec0 = FT8Decoder(FS, emit=None, mode="FT8", center=CENTER)
    dec0.process(audio)
    frame_len = int(FS * dec0.period)
    chunk = dec0.buf[-frame_len:]
    tones, _confs = dec0._detect_tones(chunk)
    sync = dec0._find_costas(tones)
    assert sync is not None, "Costas non trouve"
    bits = dec0._tones_to_bits(tones, sync)
    match = np.mean(np.asarray(bits[:174]) == cwd)
    print(f"tons->bits : {match*100:.1f}% de bits corrects (attendu ~100%)")
    assert match > 0.95
    hard = dec0.ldpc.decode([1.0 if b == 0 else -1.0 for b in bits[:174]])
    assert np.all((H @ hard) % 2 == 0)
    print("chaine audio -> bits -> LDPC ok (syndrome nul)")


if __name__ == "__main__":
    test_crc_vector()
    test_unpack_pack()
    test_ldpc_roundtrip()
    test_audio_roundtrip()
    print("Tous les tests FT8 passent.")
