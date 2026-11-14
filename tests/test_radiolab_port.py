"""Tests du port complet de RadioLab 0.15 (package `radiolab/`) dans PyPhantom SDR.

Lancer depuis la racine du depot :  python tests/test_radiolab_port.py
Les tests pytest d'origine de RadioLab sont aussi ports dans tests_radiolab/
(lancer avec : pip install pytest ; python -m pytest tests_radiolab).
"""
from __future__ import annotations

import importlib
import os
import socket
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FS = 48_000

DSP_MODULES = [
    "acars", "adsb", "adsb_cpr", "ais", "ais_pos", "audiofx", "c4fm", "contestia",
    "crc16", "crc24", "ctcss", "digital", "dmr", "dsp_optimized", "dstar", "ekf",
    "fax", "fec", "fec_advanced", "filters", "fmradio", "ft8", "ft8_crc",
    "ft8_ldpc", "ft8_unpack", "gmsk", "golay", "hell", "hfdigi", "js8", "jt65",
    "kalman", "m17", "metar", "modes", "msk", "navtex", "neural_nr", "noise",
    "olivia", "packet", "pictures", "psk31", "receiver", "rtty", "sigmet",
    "spectrum", "sstv", "tetra", "tetra_acelp", "tetra_channel", "tetra_demod",
    "tetra_viterbi", "tetra_voice", "timesignals", "wspr",
]


def test_imports() -> None:
    import radiolab

    assert radiolab.__version__.startswith("0.15")
    for name in DSP_MODULES:
        importlib.import_module(f"radiolab.dsp.{name}")
    for name in ("audio", "bandplan", "cli", "netcontrol", "planner", "radio", "scanner", "sources.airspy", "sources.iqfile"):
        importlib.import_module(f"radiolab.{name}")
    print(f"imports ok ({len(DSP_MODULES)} modules dsp + app, gui/widgets compiles seulement sans PySide6)")


def test_bandplan_scanner() -> None:
    from radiolab.bandplan import BANDS, band_at
    from radiolab.scanner import Scanner

    assert band_at(7_100_000) == "40 m amateur"
    assert band_at(98_000_000) == "Radiodiffusion FM"
    assert band_at(125_000_000) == "Aviation (AM)"
    assert band_at(300_000_000) == ""
    assert all(lo < hi for lo, hi, _, _ in BANDS)

    class FakeRx:
        mode, level_db = "NFM", -100.0

    class FakeRadio:
        def __init__(self):
            self.rx, self.vfo_hz, self.center_hz, self.tuned = FakeRx(), 0, 0.0, []

        def tune(self, hz):
            if hz == 145_512_500:
                raise ValueError("hors plage")
            self.vfo_hz = hz
            self.tuned.append(hz)

        def set_mode(self, mode):
            self.rx.mode = mode

    radio, now = FakeRadio(), [0.0]
    sc = Scanner(radio, clock=lambda: now[0])
    sc.start_range(145_500_000, 145_550_000, 12_500, threshold_db=-60)
    assert radio.vfo_hz == 145_500_000 and len(sc.channels) == 5
    now[0] += 0.3
    sc.tick()
    assert radio.vfo_hz == 145_525_000          # canal vide -> suivant (145,5125 sauté : pas sur la grille)
    radio.rx.level_db = -40.0
    for _ in range(20):
        now[0] += 0.3
        sc.tick()
    assert radio.vfo_hz == 145_525_000 and sc.holding   # signal présent : on reste
    radio.rx.level_db = -100.0
    now[0] += 1.0
    sc.tick()
    assert radio.vfo_hz == 145_525_000          # maintien 2 s après la fin du signal
    now[0] += 1.5
    sc.tick()
    assert radio.vfo_hz == 145_537_500
    sc.stop()
    print("bandplan + scanner ok (arrêt sur signal, maintien 2 s, reprise)")


def test_new_decoders() -> None:
    from radiolab.dsp import navtex
    from radiolab.dsp.dstar import DStarDecoder
    from radiolab.dsp.fax import FAXDecoder
    from radiolab.dsp.golay import GolayDecoder
    from radiolab.dsp.hell import HellDecoder
    from radiolab.dsp.hfdigi import MORSE, VARICODE, Baseband, make_decoder
    from radiolab.dsp.js8 import JS8Decoder
    from radiolab.dsp.jt65 import JT65Decoder
    from radiolab.dsp.m17 import M17Decoder
    from radiolab.dsp.metar import MetarCollector, format_metar, parse_metar
    from radiolab.dsp.msk import MSKDemodulator, demod_msk

    zeros = np.zeros(FS, dtype=np.float32)
    # NAVTEX : encode SITOR-B + decodeur alimenté
    dec = navtex.NAVTEXDecoder(8000)
    assert isinstance(dec.process(np.zeros(8000, dtype=np.float32)), list)
    assert navtex.NavtexDecoder is navtex.NAVTEXDecoder   # alias hfdigi.make_decoder
    # METAR : parse + collecteur + formatage
    m = parse_metar("METAR LFPG 251200Z 27008KT 9999 FEW030 12/07 Q1013")
    assert m is not None and m["icao"] == "LFPG" and m["qnh"] == "1013 hPa"
    col = MetarCollector()
    found = col.feed("METAR LFPG 251200Z 27008KT 9999 FEW030 12/07 Q1013\n")
    assert len(found) == 1 and len(col.messages) == 1
    assert "LFPG" in format_metar(m)
    # Golay(24,12) : mot de 24 bits ; mot nul ok, 3 erreurs dans les bits de parité corrigées
    g = GolayDecoder()
    assert g.decode([0] * 24) == [0] * 12
    assert g.decode([0] * 12 + [1, 1, 1] + [0] * 9) == [0] * 12
    assert g.decode([0] * 10) is None
    # MSK + décodeurs divers sur des zéros (pas d'exception, types de retour)
    assert len(demod_msk(zeros, FS, 2400.0)) >= 0
    MSKDemodulator(FS).process(zeros)
    for cls, fs in ((DStarDecoder, FS), (JS8Decoder, 12000), (JT65Decoder, 12000),
                    (M17Decoder, FS), (HellDecoder, 8000), (FAXDecoder, 8000)):
        cls(fs).process(np.zeros(fs // 10, dtype=np.float32))
    # hfdigi reel : fabrique les décodeurs HF + Baseband décime
    for kind in ("PSK31", "PSK63", "RTTY45", "RTTY50", "CW", "NAVTEX", "WSPR"):
        d = make_decoder(kind, 8000, 1000.0, lambda s: None)
        assert d is not None, kind
        d.process(np.zeros(800, dtype=np.float32)) if hasattr(d, "process") else None
    assert len(VARICODE) > 0 and MORSE["..."] == "S"
    bb = Baseband(48000, 700.0, 40.0, 60.0)
    out = bb.process(np.zeros(48000, dtype=np.complex64))
    assert len(out) > 0
    print("navtex/metar/golay/msk/dstar/js8/jt65/m17/hell/fax/hfdigi ok")


def test_ft8_port_corrige() -> None:
    # Le port embarque la version CORRIGEE du FT8 (pyphantom/ft8), pas celle boguée du zip.
    from radiolab.dsp.ft8 import FT8Decoder, FT4Decoder
    from radiolab.dsp.ft8_crc import add_crc14, check_crc14, crc14
    from radiolab.dsp.ft8_unpack import pack_standard

    msg = [0] * 82
    for p in (3, 7, 44, 45, 46, 51, 61, 71):
        msg[p] = 1
    assert crc14(msg[:77]) == [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1]  # vecteur officiel WSJT-X
    m77 = pack_standard("CQ", "F1ABC", grid="JN18")
    assert check_crc14(add_crc14(m77))
    d = FT8Decoder(8000, emit=lambda m: None, center=1500.0)
    d.process(np.zeros(8000, dtype=np.float32))
    d.try_decode()
    FT4Decoder(8000, emit=lambda m: None)
    print("FT8 du port ok (CRC-14 vecteur WSJT-X, pack/unpack, FT8Decoder/FT4Decoder)")


def test_files_radio_cli_netcontrol_planner() -> None:
    from radiolab import cli, planner
    from radiolab.audio import NullAudio, WavWriter, as_stereo
    from radiolab.netcontrol import RigCtlServer
    from radiolab.radio import MODES, Radio, allowed_decimations
    from radiolab.sources.iqfile import IQFileSource

    assert as_stereo(np.zeros(100, dtype=np.float32)).shape == (100, 2)
    assert "USB" in MODES and allowed_decimations(10_000_000)

    tmp = tempfile.mkdtemp(prefix="pyphantom_port_")
    # fichier I/Q (WAV stéréo) -> lecture + CLI de démodulation
    fs_iq, seconds = 250_000, 2.0
    n = int(fs_iq * seconds)
    iq = (0.2 * np.exp(2j * np.pi * 21_000.0 * np.arange(n) / fs_iq)).astype(np.complex64)
    iq_path = os.path.join(tmp, "PyPhantom_20260101_000000Z_7100000Hz_IQ.wav")
    w = WavWriter(iq_path, fs_iq, iq=True)
    w.write(iq)
    w.close()
    src = IQFileSource(iq_path, loop=False, realtime=False)
    assert src.sample_rate == fs_iq and src.center_hz == 7_100_000.0
    src.rewind()
    got = []
    while (b := src.read_block()) is not None:
        got.append(b)
    got = np.concatenate(got)
    assert len(got) == len(iq) and float(np.max(np.abs(got - iq))) < 1e-4
    src.close()

    out_wav = os.path.join(tmp, "audio.wav")
    cli.main([iq_path, "-o", out_wav, "--freq", "7121000", "--mode", "USB"])
    assert os.path.exists(out_wav) and os.path.getsize(out_wav) > 44

    # Radio : construction sans source, audio NullAudio par défaut
    r = Radio(audio_factory=lambda: NullAudio())
    assert r.source is None and not r.running

    # rigctl (Hamlib NET) sur 127.0.0.1, port éphémère : f -> fréquence, q -> quit
    srv = RigCtlServer(lambda: (7_100_000.0, "USB", 2400.0, -42.0), host="127.0.0.1", port=0)
    try:
        with socket.create_connection(("127.0.0.1", srv.port), timeout=5) as s:
            s.sendall(b"f\n")
            rep = s.makefile("r").readline()
            assert rep.strip() == "7100000", rep
            s.sendall(b"m\n")
            assert s.makefile("r").readline().strip() == "USB"
            s.sendall(b"q\n")
    finally:
        srv.close()

    # planner : export/import CSV de mémoires, journal, job programmé
    csv_path = os.path.join(tmp, "memories.csv")
    planner.export_memories(csv_path, [{"name": "FT8 40m", "hz": 7_074_000, "mode": "USB", "group": "HF"}])
    mems = planner.import_memories(csv_path, sorted(MODES))
    assert mems and mems[0]["hz"] == 7_074_000 and mems[0]["mode"] == "USB"
    jpath = planner.log_activity(tmp, 144_800_000, "NFM", -55.0, tone="88.5", now=__import__("datetime").datetime(2026, 1, 1, 12, 0, 0))
    assert os.path.exists(jpath)
    job = planner.parse_job("2026-11-12 21:00;30;7100000;LSB;audio", sorted(MODES))
    assert job["hz"] == 7_100_000 and job["mode"] == "LSB" and job["iq"] is False
    assert isinstance(planner.job_label(job), str)
    sched = planner.Scheduler([])
    assert sched is not None
    print(f"fichiers I/Q + CLI + Radio + rigctl(127.0.0.1:{srv.port}) + planner ok (tmp={tmp})")


if __name__ == "__main__":
    test_imports()
    test_bandplan_scanner()
    test_new_decoders()
    test_ft8_port_corrige()
    test_files_radio_cli_netcontrol_planner()
    print("Tous les tests du port RadioLab passent.")
