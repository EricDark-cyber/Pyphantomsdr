"""Tests .cf32 (enregistrement/lecture, format SDR#) et CAT série.

Lancer depuis la racine du depot :  python tests/test_cf32_cat.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FS_IQ = 250_000


def test_cf32_writer_reader() -> None:
    from radiolab.audio import Cf32Writer, recording_name
    from radiolab.sources.iqfile import IQFileSource

    tmp = tempfile.mkdtemp(prefix="pyphantom_cf32_")
    rng = np.random.default_rng(3)
    iq = (0.3 * (rng.standard_normal(20_000) + 1j * rng.standard_normal(20_000))).astype(np.complex64)

    # nom de fichier avec la fréquence centrale (convention SDR#) + extension .cf32
    path = recording_name(tmp, 7_100_000, True, ext="cf32")
    assert path.endswith(".cf32") and "7100000Hz" in path and "PyPhantom_" in path
    w = Cf32Writer(path, FS_IQ)
    w.write(iq[:10_000])
    w.write(iq[10_000:])
    w.close()
    assert os.path.getsize(path) == 20_000 * 8           # 8 octets par échantillon complexe

    # lecture .cf32 : débit fourni (pas d'en-tête), fréquence lue dans le nom
    src = IQFileSource(path, loop=False, realtime=False, sample_rate=FS_IQ)
    assert src.sample_rate == FS_IQ and src.center_hz == 7_100_000.0
    assert src.dtype.kind == "c" and src.frame_bytes == 8
    src.rewind()
    got = []
    while (b := src.read_block()) is not None:
        got.append(b)
    got = np.concatenate(got)
    assert len(got) == len(iq)
    assert np.array_equal(got, iq)                        # cf32 = brut, bit à bit
    src.close()

    # sans --rate : défaut 1 MS/s
    src2 = IQFileSource(path, loop=False, realtime=False)
    assert src2.sample_rate == 1_000_000
    src2.close()

    # le WAV I/Q continue de marcher (régression) — tonalé à |x|=0.2 pour ne pas clipper en 16 bits
    from radiolab.audio import WavWriter

    iq_wav = (0.2 * np.exp(2j * np.pi * 21_000.0 * np.arange(20_000) / FS_IQ)).astype(np.complex64)
    wpath = recording_name(tmp, 14_200_000, True, ext="wav")
    ww = WavWriter(wpath, FS_IQ, iq=True)
    ww.write(iq_wav)
    ww.close()
    src3 = IQFileSource(wpath, loop=False, realtime=False)
    assert src3.sample_rate == FS_IQ and src3.center_hz == 14_200_000.0
    src3.rewind()
    got3 = []
    while (b := src3.read_block()) is not None:
        got3.append(b)
    got3 = np.concatenate(got3)
    assert len(got3) == len(iq_wav) and float(np.max(np.abs(got3 - iq_wav))) < 1e-4
    src3.close()
    print(f"cf32 ok (écriture/lecture bit à bit, nom SDR#, WAV toujours OK) tmp={tmp}")


def test_radio_record_cf32() -> None:
    from radiolab.audio import NullAudio
    from radiolab.radio import Radio
    from radiolab.sources.iqfile import IQFileSource

    tmp = tempfile.mkdtemp(prefix="pyphantom_rec_")

    class FakeSrc:
        sample_rate = FS_IQ
        kind = "file"

    r = Radio(audio_factory=lambda: NullAudio())
    r.source = FakeSrc()                                   # sample_rate = source.sample_rate / decimation
    r.decimation = 1
    r.center_hz = 7_074_000.0

    path_wav = r.start_recording(tmp, True, fmt="wav")
    path_cf32 = r.start_recording(tmp, True, fmt="cf32")
    assert path_wav.endswith(".wav") and path_cf32.endswith(".cf32")
    iq = (0.2 * np.exp(2j * np.pi * 1000.0 * np.arange(5000) / FS_IQ)).astype(np.complex64)
    r.iq_rec.write(iq)
    r.stop_recording(True)
    r.start_recording(tmp, True, fmt="cf32")
    r.iq_rec.write(iq)
    r.stop_recording(True)

    src = IQFileSource(path_cf32, loop=False, realtime=False, sample_rate=FS_IQ)
    src.rewind()
    got = []
    while (b := src.read_block()) is not None:
        got.append(b)
    got = np.concatenate(got)
    assert len(got) == len(iq) and np.array_equal(got, iq)
    src.close()
    print("radio.start_recording ok (wav + cf32, relecture cf32 exacte)")


def test_serial_cat() -> None:
    from radiolab.netcontrol import SerialCatServer

    # sans matériel : autostart=False, on teste le moteur de commandes (même answer() que rigctl)
    srv = SerialCatServer(lambda: (7_100_000.0, "USB", 2400.0, -42.0), port="COM99", baud=9600, autostart=False)
    assert srv.handle_line("f\n").strip() == "7100000"
    assert srv.handle_line("\\get_freq\n").strip() == "7100000"
    assert srv.handle_line("m\n").splitlines()[0].strip() == "USB"
    assert srv.handle_line("F 14074000\n").strip() == "RPRT 0"
    kind, value = srv.commands.get_nowait()
    assert kind == "freq" and value == 14_074_000.0
    srv.handle_line("M FM\n")
    assert srv.commands.get_nowait() == ("mode", "NFM")
    assert srv.handle_line("l STRENGTH\n").strip() == "-42"
    assert srv.handle_line("q\n") is None                  # quit ferme la connexion
    assert srv.handle_line("V\n").strip() == "RPRT 0"
    srv.close()                                            # sans thread : ne plante pas
    print("CAT série ok (commandes rigctl texte sur COM : f/F/m/M/l/q, file commands)")


if __name__ == "__main__":
    test_cf32_writer_reader()
    test_radio_record_cf32()
    test_serial_cat()
    print("Tous les tests cf32 + CAT série passent.")
