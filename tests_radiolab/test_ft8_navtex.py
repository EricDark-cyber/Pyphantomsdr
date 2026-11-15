"""FT8 / FT4 sur toute la bande et NAVTEX conforme CCIR 476 : essais sur signaux de synthèse.

Les signaux FT8/FT4 sont fabriqués par ft8.encode_tones + ft8.synth.  Pour comparer à de
vrais enregistrements (fichiers .wav de 15 s accompagnés de la liste .txt des décodages de
WSJT-X, par exemple le dossier test/wav du projet ft8_lib), indiquer leur dossier :

    set FT8_WAV_DIR=C:\\chemin\\vers\\wav      puis      python -m pytest tests_radiolab/test_ft8_navtex.py
"""
import glob
import os
import wave

import numpy as np
import pytest

from radiolab.dsp import ft8, navtex
from radiolab.dsp.ft8_unpack import forget_calls, hash_call, pack_standard, unpack_ft8
from radiolab.dsp.hfdigi import make_decoder

FS = 12500.0


def scene(mode, signals, snr_db, secs, seed=0, fs=FS):
    """signals : (indicatif 1, indicatif 2, locator, Hz, début en s, niveau relatif en dB).
    Le bruit est réglé pour un rapport signal/bruit `snr_db` dans 2500 Hz (convention WSJT-X)."""
    rng = np.random.default_rng(seed)
    n = int(fs * secs)
    x = np.zeros(n)
    for c1, c2, grid, f0, t0, rel in signals:
        s = ft8.synth(ft8.encode_tones(pack_standard(c1, c2, grid=grid), mode), fs, f0, mode)
        a = int(t0 * fs)
        s = s[:max(n - a, 0)]
        x[a:a + len(s)] += 10 ** (rel / 20) * s
    return x + np.sqrt(0.5 * 10 ** (-snr_db / 10) * (fs / 2) / 2500.0) * rng.standard_normal(n)


def decode(mode, x, fs=FS):
    d = ft8.FT8Decoder(fs, emit=None, mode=mode)
    for i in range(0, len(x), 4096):
        d.process(x[i:i + 4096].astype(np.float32))
        d.try_decode()
    d.flush()
    return d


# --------------------------------------------------------------------------------------------- FT8 / FT4
def test_ft8_whole_band_several_stations():
    """Cinq stations réparties de 400 à 2900 Hz, dont deux à 3 Hz l'une de l'autre : toutes décodées,
    avec la bonne fréquence et un niveau affiché juste à 3 dB près."""
    sig = [("CQ", "F1ABC", "JN18", 412.0, 0.6, 0), ("CQ", "DL1XYZ", "JO62", 1503.1, 1.0, -6),
           ("K1ABC", "F4XYZ", "JN09", 1506.0, 1.4, -14), ("CQ", "EA3AB", "JN11", 2891.7, 2.1, -3),
           ("G4ABC", "ON4XY", "JO20", 2200.0, 0.2, -10)]
    d = decode("FT8", scene("FT8", sig, 5, 17.0))
    got = {m[0]: m for m in d.decodes}
    assert set(got) == {"CQ F1ABC JN18", "CQ DL1XYZ JO62", "K1ABC F4XYZ JN09", "CQ EA3AB JN11", "G4ABC ON4XY JO20"}
    for c1, c2, grid, f0, t0, rel in sig:
        text, snr, freq, start = got["%s %s %s" % (c1, c2, grid)]
        assert abs(freq - f0) < 1.5 and abs(snr - (5 + rel)) <= 3
    assert d.messages[0].endswith("Hz") and "dB" in d.messages[0]


@pytest.mark.parametrize("mode,snr,secs", [("FT8", -17, 17.0), ("FT4", -12, 9.0)])
def test_weak_signal(mode, snr, secs):
    ok = 0
    for seed in range(3):
        d = decode(mode, scene(mode, [("CQ", "F1ABC", "JN18", 1234.5, 1.3, 0)], snr, secs, seed))
        ok += any(m[0] == "CQ F1ABC JN18" and abs(m[1] - snr) <= 3 for m in d.decodes)
    assert ok >= 2, ok


def test_ft4_is_really_ft4():
    """Trame FT4 : 103 symboles, 4 tonalités, quatre motifs de synchronisation différents."""
    tones = ft8.encode_tones(pack_standard("CQ", "F1ABC", grid="JN18"), "FT4")
    assert len(tones) == 103 and set(tones) <= {0, 1, 2, 3}
    for pos, pat in zip((0, 33, 66, 99), ft8.FT4_COSTAS):
        assert list(tones[pos:pos + 4]) == pat
    d = decode("FT4", scene("FT4", [("CQ", "F1ABC", "JN18", 800.0, 0.5, 0), ("CQ", "DL1XYZ", "JO62", 2100.0, 1.2, -5)], 0, 9.0))
    assert {m[0] for m in d.decodes} == {"CQ F1ABC JN18", "CQ DL1XYZ JO62"}
    # un signal FT8 n'est pas pris pour du FT4
    assert not decode("FT4", scene("FT8", [("CQ", "F1ABC", "JN18", 800.0, 0.5, 0)], 0, 17.0)).decodes


def test_continuous_listening_no_duplicates_and_nothing_on_noise():
    """Plusieurs périodes de suite : chaque message sort une fois par période, jamais deux."""
    parts = [scene("FT8", [("CQ", c, "JN18", 1000.0 + 100 * k, 0.8, 0), ("CQ", "DL1XYZ", "JO62", 2000.0, 1.0, 0)], -5, 15.0, seed=k)
             for k, c in enumerate(("F1ABC", "F4XYZ", "F5KLM"))]
    out = []
    d = make_decoder("FT8", FS, 1500.0, out.append)            # comme dans l'interface : calcul dans un fil à part
    x = np.concatenate(parts)
    for i in range(0, len(x), 4096):
        d.process(x[i:i + 4096].astype(np.float32))
        d.on_timer()
        d.wait()                                               # l'essai va plus vite que le temps réel : on attend le calcul
    d.flush()
    texts = sorted(m[0] for m in d.decodes)
    assert texts == sorted(["CQ DL1XYZ JO62"] * 3 + ["CQ F1ABC JN18", "CQ F4XYZ JN18", "CQ F5KLM JN18"]), texts
    assert all(o.endswith("\n") for o in out)
    assert ["sep" if o.startswith("──") else "msg" for o in out] == ["msg", "msg", "sep", "msg", "msg", "sep", "msg", "msg"]
    assert "message" in d.status()
    assert not decode("FT8", scene("FT8", [], 0, 90.0, seed=7)).decodes        # du bruit seul : rien


def test_message_types():
    forget_calls()
    assert unpack_ft8(pack_standard("F1ABC", "K1ABC", ir=1, report=-12)) == "F1ABC K1ABC R-12"
    assert unpack_ft8(pack_standard("F1ABC", "K1ABC", report=7)) == "F1ABC K1ABC +07"
    assert unpack_ft8(pack_standard("CQ", "F1ABC", grid="JN18")) == "CQ F1ABC JN18"
    assert unpack_ft8([0] * 77) is None or unpack_ft8([0] * 77) == ""          # trame vide
    # indicatif remplacé par son condensé de 22 bits : inconnu d'abord, traduit une fois l'indicatif entendu
    h = hash_call("F4XYZ/P")
    val = ((2063592 + h) << 49) | (int("".join(map(str, pack_standard("CQ", "K1ABC", grid="FN42"))), 2) & ((1 << 49) - 1))
    bits = [(val >> (76 - i)) & 1 for i in range(77)]
    assert unpack_ft8(bits) == "<...> K1ABC FN42"
    from radiolab.dsp.ft8_unpack import remember
    remember("F4XYZ/P")
    assert unpack_ft8(bits) == "<F4XYZ/P> K1ABC FN42"
    forget_calls()


def test_real_recordings():
    """Facultatif : au moins 70 % des décodages de WSJT-X retrouvés sur de vrais enregistrements."""
    folder = os.environ.get("FT8_WAV_DIR")
    if not folder:
        pytest.skip("FT8_WAV_DIR non défini")
    expected = found = 0
    for path in sorted(glob.glob(os.path.join(folder, "*.wav"))):
        txt = path[:-4] + ".txt"
        if not os.path.exists(txt):
            continue
        with wave.open(path, "rb") as w:
            fs = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
        d = decode("FT8", np.concatenate([np.zeros(int(2 * fs)), x]), fs)
        got = {m[0].split("<")[0] for m in d.decodes}
        for line in open(txt, errors="replace"):
            if "~" in line:
                msg = " ".join(line.split("~", 1)[1].split())
                expected += 1
                found += any(msg == g or msg.startswith(g + " ") or msg.split("<")[0] == g for g in got if g)
    assert expected and found >= 0.7 * expected, (found, expected)


# --------------------------------------------------------------------------------------------- NAVTEX
def navtex_audio(text, fs=FS, center=1700.0):
    bits = [(code >> k) & 1 for code in navtex.encode(text) for k in range(6, -1, -1)]
    n = int(len(bits) * fs / 100.0)
    b = np.array(bits)[np.minimum((np.arange(n) * 100.0 / fs).astype(np.int64), len(bits) - 1)]
    return np.sin(2 * np.pi * np.cumsum(center + 85.0 * (2 * b - 1)) / fs)


def test_navtex_table_is_ccir476():
    """26 lettres, retour chariot, saut de ligne, espace, 2 inversions et 3 signaux de service :
    34 mots distincts de poids 4 ; le 35e mot possible (0101011, « blanc ») n'est pas utilisé."""
    words = set(navtex.LETTERS) | {navtex.LTRS, navtex.FIGS, navtex.ALPHA, navtex.BETA, navtex.REP}
    assert len(navtex.LETTERS) == 29 and len(words) == 34 and all(navtex.valid(w) for w in words)
    assert [w for w in range(128) if navtex.valid(w) and w not in words] == [0b0101011]
    assert navtex.FIGURES[navtex.ENCODE_LETTERS["Q"]] == "1" and navtex.FIGURES[navtex.ENCODE_LETTERS["P"]] == "0"
    known = {"A": 0b1110001, "E": 0b0110101, "N": 0b1001101, "Z": 0b1100011, "T": 0b0010111}
    assert all(navtex.ENCODE_LETTERS[k] == v for k, v in known.items())
    assert navtex.ALPHA == 0b1111000 and navtex.REP == 0b0110011 and navtex.SPACE == 0b0011101


def test_navtex_in_noise_and_nothing_on_noise():
    text = "ZCZC SA12\r\n151530 UTC NOV 26\r\nCROSS CORSEN GALE WARNING NR 345\r\nWIND SW 8 TO 9, GUSTS 55 KT.\r\nNNNN"
    sig = navtex_audio(text)
    rng = np.random.default_rng(3)
    sigma = np.sqrt(0.5 * 10 ** (-10 / 10) * (FS / 2) / 500.0)             # +10 dB dans 500 Hz
    x = np.concatenate([np.zeros(int(FS)), sig]) + sigma * rng.standard_normal(len(sig) + int(FS))
    out = []
    d = make_decoder("NAVTEX", FS, 1700.0, out.append)
    for i in range(0, len(x), 4096):
        d.process(x[i:i + 4096].astype(np.float32))
    got = "".join(out)
    assert "GALE WARNING NR 345" in got and "GUSTS 55 KT." in got and got.rstrip().endswith("NNNN")
    out = []
    d = make_decoder("NAVTEX", FS, 1700.0, out.append)
    noise = rng.standard_normal(int(FS * 60)).astype(np.float32)
    for i in range(0, len(noise), 4096):
        d.process(noise[i:i + 4096])
    assert not out                                                         # une minute de bruit : aucun caractère
