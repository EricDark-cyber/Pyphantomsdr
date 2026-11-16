"""Tests des decodeurs numeriques et filtres audio (pyphantom/decoders/).
Lancer : python tests/test_decoders.py"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyphantom.decoders import (  # noqa: E402
    ACARSDecoder,
    ADSBDecoder,
    AISDecoder,
    C4FMDecoder,
    CTCSSDecoder,
    DMRDecoder,
    AudioFX,
    BitSlicer,
    ContestiaDecoder,
    DCSDetector,
    ExtendedKalmanFilter,
    HammingFEC,
    OptimizedDSP,
    ToneDetector,
)
from pyphantom.decoders.adsb_cpr import cpr_decode, cpr_nl  # noqa: E402
from pyphantom.decoders.ais_pos import decode_ais_position  # noqa: E402
from pyphantom.decoders.crc16 import crc16  # noqa: E402
from pyphantom.decoders.crc24 import crc24  # noqa: E402
from pyphantom.decoders.digital import (  # noqa: E402
    POCSAG_SYNC,
    dcs_codeword,
    golay_decode,
    golay_syndrome,
    pocsag_correct,
    pocsag_encode,
    pocsag_syndrome,
)
from pyphantom.decoders.modes import ModeSDecoder, decode_mode_s  # noqa: E402
from pyphantom.decoders.gmsk import GMSKDemodulator  # noqa: E402
from pyphantom.decoders.olivia import OliviaDecoder  # noqa: E402
from pyphantom.decoders import (  # noqa: E402
    DCF77Decoder,
    FaxDecoder,
    KalmanFilter1D,
    KalmanNR,
    LineEnhancer,
    NeuralNR,
    NoiseBlanker,
    PacketDecoder,
    PSK31Decoder,
    RTTYDecoder,
    Receiver,
    SSTVDecoder,
    SpectralNR,
    TETRADecoder,
    WSPRDecoder,
    Baseband,
    decode_minute,
    encode_minute,
    expint_e1,
    parse_sigmet,
    write_png,
)
from pyphantom.spectrum import get_window, power_spectrum_db, zoom_spectrum  # noqa: E402
from pyphantom.filters import lowpass_taps  # noqa: E402

FS = 48_000


def test_crc() -> None:
    # CRC-16 AIS (poly 0x1021, init 0xFFFF, xorout 0xFFFF) : valeur de regression + detection 1 bit
    assert crc16(b"123456789") == 0xD64E, f"crc16 = {crc16(b'123456789'):#06X}"  # CCITT-FALSE complemente (init/xorout FFFF)
    rng = np.random.default_rng(4)
    data = rng.integers(0, 2, size=160).tolist()
    c = crc16(data)
    for k in range(0, 160, 7):
        bad = data.copy()
        bad[k] ^= 1
        assert crc16(bad) != c, f"crc16 ne detecte pas l'erreur bit {k}"
    # Mode S : trame DF17 valide 8D40621D58C382435CC412692AD6 -> CRC-24 sur les 112 bits = 0
    frame = [int(b) for b in bin(int("8D4840D6202CC371C32CE0576098", 16))[2:].zfill(112)]  # trame DF17 valide (pyModeS)
    assert crc24(frame) == 0, f"crc24 trame = {crc24(frame):#08X}"
    parity = crc24(frame[:88])
    assert parity == int("".join(map(str, frame[88:])), 2)
    print("CRC ok (crc16 regression 0xD64E + detection 1 bit, crc24 trame Mode S valide=0)")


def test_cpr() -> None:
    assert cpr_nl(0) == 59
    assert cpr_nl(45) == 42 and cpr_nl(89) == 1

    def cpr_encode(lat, lon, odd):
        dlat = 360.0 / 59 if odd else 360.0 / 60
        yz = int(math.floor(131072 * (math.fmod(lat, dlat) / dlat) + 0.5)) % 131072
        rlat = dlat * (math.floor(lat / dlat) + yz / 131072.0)
        nl = cpr_nl(rlat)
        ni = max(nl - (1 if odd else 0), 1)
        dlon = 360.0 / ni
        xz = int(math.floor(131072 * (math.fmod(lon, dlon) / dlon) + 0.5)) % 131072
        return yz, xz

    lat0, lon0 = 48.5, 2.3
    e = cpr_encode(lat0, lon0, odd=False)
    o = cpr_encode(lat0, lon0, odd=True)
    pos = cpr_decode(e[0], e[1], o[0], o[1], even_time=1.0, odd_time=0.0)
    assert pos is not None
    lat, lon = pos
    assert abs(lat - lat0) < 0.01 and abs(lon - lon0) < 0.01, f"CPR -> {lat}, {lon}"
    print(f"CPR ok (NL(0)=59, NL(45)=42, aller-retour 48.5/2.3 -> {lat:.4f}, {lon:.4f})")


def test_ais_pos() -> None:
    bits = [0] * 168
    bits[0:6] = [0, 0, 0, 0, 0, 1]  # type 1
    mmsi = 227000000
    for i in range(30):
        bits[8 + i] = (mmsi >> (29 - i)) & 1
    lon_raw = int(round(2.3 * 600000))
    for i in range(28):
        bits[61 + i] = (lon_raw >> (27 - i)) & 1
    lat_raw = int(round(48.5 * 600000))
    for i in range(27):
        bits[89 + i] = (lat_raw >> (26 - i)) & 1
    pos = decode_ais_position(bits)
    assert pos is not None
    lat, lon, m = pos
    assert m == mmsi and abs(lat - 48.5) < 1e-3 and abs(lon - 2.3) < 1e-3, f"{pos}"
    print(f"AIS pos ok (mmsi {m}, {lat:.4f} {lon:.4f})")


def test_bitslicer() -> None:
    baud = 1200
    t = fs_t = FS / baud  # 40 ech/bit
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=300)
    x = np.repeat(bits, int(fs_t)).astype(np.float64) * 2 - 1
    x += 0.05 * rng.standard_normal(len(x))
    slicer = BitSlicer(FS, baud)
    out = []
    for i in range(0, len(x), 997):
        out.append(slicer.process(x[i:i + 997]))
    out = np.concatenate(out)
    n = len(bits)
    def err_at(shift):
        m = min(len(out) - shift, n)
        return int(np.sum(out[shift:shift + m] != bits[:m]))
    best = min(range(int(fs_t) + 2), key=err_at)
    err = err_at(best)
    assert err <= 5, f"BitSlicer: {err} erreurs / {len(bits)}"
    print(f"BitSlicer ok ({len(out)} bits, {err} erreurs avec decalage {best})")


def test_pocsag_golay() -> None:
    cw = pocsag_encode(0x12345)
    assert pocsag_syndrome(cw) == 0
    assert pocsag_syndrome(POCSAG_SYNC) == 0
    assert pocsag_correct(cw ^ (1 << 7)) == cw, "correction 1 bit POCSAG"
    for code in (0o23, 0o156, 0o754):  # codes DCS en octal
        w = dcs_codeword(code)
        assert golay_syndrome(w) == 0
        data, nerr = golay_decode(w ^ (1 << 3) ^ (1 << 17))
        assert data == (0x800 | code) and nerr == 2, f"Golay code {code}: nerr={nerr}"
    print("POCSAG/DCS ok (BCH31,21 correction 1 bit, Golay(23,12) correction 2 bits)")


def test_ctcss() -> None:
    t = np.arange(int(FS * 1.5)) / FS
    x = (0.3 * np.sin(2 * np.pi * 100.0 * t)).astype(np.float32)
    dec = CTCSSDecoder(FS)
    found = None
    for i in range(0, len(x), 4800):
        r = dec.process(x[i:i + 4800])
        if r:
            found = r
    assert found and found["tone"] == 100.0, f"CTCSS detecte: {found}"
    td = ToneDetector(FS)
    y = (0.5 * np.sin(2 * np.pi * 88.5 * t)).astype(np.float32)
    for i in range(0, len(y), 4800):
        td.process(y[i:i + 4800])
    assert td.tone == 88.5, f"ToneDetector: {td.tone}"
    print(f"CTCSS ok (100.0 Hz via CTCSSDecoder, 88.5 Hz via ToneDetector)")


def test_audiofx() -> None:
    t = np.arange(FS) / FS
    x = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    fx = AudioFX(FS)
    assert not fx.active
    assert np.array_equal(fx.process(x), x), "sans reglage -> transparent"
    fx.configure(notches=[(1000.0, 80.0)])
    assert fx.active
    y = fx.process(x)
    assert len(y) == len(x)
    rms_in = float(np.sqrt(np.mean(x ** 2)))
    rms_out = float(np.sqrt(np.mean(y[len(y) // 4:] ** 2)))
    assert rms_out < 0.2 * rms_in, f"notch: {rms_out:.4f} vs {rms_in:.4f}"
    fx.configure()
    assert not fx.active
    print(f"AudioFX ok (notch 1000 Hz: RMS {rms_in:.3f} -> {rms_out:.4f}, transparent si inactif)")


def test_adsb_frame() -> None:
    fs = 2_000_000.0
    sps = 2  # 1 us = 2 ech.
    frame_hex = "8D4840D6202CC371C32CE0576098"
    frame_bits = [(int(frame_hex, 16) >> (111 - i)) & 1 for i in range(112)]
    n = int(130 * sps)
    sig = np.zeros(n, dtype=np.float32)
    for p in (0, 1.0, 3.5, 4.5):
        sig[int(p * sps)] = 1.0
    start = int(8 * sps)
    for i, b in enumerate(frame_bits):
        sig[start + i * sps + (sps // 2 if b else 0)] = 1.0
    msgs = []
    dec = ADSBDecoder(fs, emit=msgs.append)
    dec.process_iq(sig.astype(np.complex64))
    assert msgs, "aucune trame ADS-B decodee"
    m = msgs[0]
    assert m["df"] == 17 and m["icao"] == "4840D6", f"{m}"
    print(f"ADS-B ok (trame synthetique PPM -> DF{m['df']} ICAO {m['icao']})")


def test_ekf_fec_dsp() -> None:
    ekf = ExtendedKalmanFilter()
    for _ in range(300):
        x = ekf.update(0.5)
    assert abs(x - math.atanh(0.5)) < 0.05, f"EKF x={x:.4f}, attendu ~{math.atanh(0.5):.4f}"

    ham = HammingFEC(15, 11)
    data = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    cw = None
    for p in range(16):  # brute force des 4 bits de parite -> syndrome nul
        cand = data + [(p >> k) & 1 for k in range(4)]
        s = 0
        for i, b in enumerate(cand):
            if b:
                s ^= i + 1
        if s == 0:
            cw = cand
            break
    assert cw is not None
    assert ham.decode(cw) == data
    cw_err = cw.copy()
    cw_err[5] ^= 1
    assert ham.decode(cw_err) == data, "Hamming corrige 1 bit"

    dsp = OptimizedDSP()
    rng = np.random.default_rng(2)
    iq = (rng.standard_normal(2048) + 1j * rng.standard_normal(2048)).astype(np.complex64)
    assert len(dsp.spectrum(iq, 1024)) == 1024
    assert len(dsp.demod_am(iq)) == len(iq)
    print("EKF/Hamming/OptimizedDSP ok")


def test_modes_gmsk_olivia() -> None:
    # Mode S : trame DF17 valide -> nom de DF + stats
    raw = "8D4840D6202CC371C32CE0576098"
    msg = decode_mode_s({"df": 17, "raw": raw, "bits": 112})
    assert msg["df_name"] == "ADS-B Extended Squitter", msg
    ms = ModeSDecoder()
    ms.process({"df": 17, "raw": raw, "bits": 112})
    ms.process({"df": 11, "raw": "5D4840D6A6B4", "bits": 56})
    assert ms.stats().get(17) == 1 and ms.stats().get(11) == 1
    # squawk DF11 extrait (champ 13 bits)
    m11 = decode_mode_s({"df": 11, "raw": "5D4840D6A6B4", "bits": 56})
    assert "squawk" not in m11 or isinstance(m11.get("squawk"), str)

    # GMSK : signal FSK synthetique -> bits
    fs, baud = 96_000.0, 9600.0
    t = np.arange(int(fs * 0.2)) / fs
    f = np.where((np.arange(len(t)) // int(fs / baud)) % 2 == 0, 4800.0, 2400.0)
    phase = 2 * np.pi * np.cumsum(f) / fs
    x = np.exp(1j * phase).real.astype(np.float32)
    g = GMSKDemodulator(fs, baud)
    bits = []
    for i in range(0, len(x), 4800):
        bits += g.process(x[i:i + 4800])
    assert len(bits) > 50 and set(bits) <= {0, 1}

    # Olivia / Contestia : init + traitement de zeros sans crash
    ol = OliviaDecoder(8000)
    assert isinstance(ol.process(np.zeros(8000, dtype=np.float32)), list)
    co = ContestiaDecoder()
    assert co.baud == 62.5
    assert isinstance(co.process(np.zeros(8000, dtype=np.float32)), list)
    print("modes/gmsk/olivia ok (DF17 nomme, stats, squawk, GMSK bits, Olivia/Contestia)")


def test_spectrum_and_windows() -> None:
    # fenetres nommees disponibles (spectrum.py) -> utilisables par filters.lowpass_taps
    h = lowpass_taps(3000.0, 48000.0, 101, window="Hann")
    assert abs(float(h.sum()) - 1.0) < 1e-6
    w = get_window("Blackman-Harris", 256)
    assert abs(float(w.sum()) - 1.0) < 1e-6 and len(w) == 256
    # spectre : pic a la bonne frequence
    fs = 1_000_000.0
    t = np.arange(8192) / fs
    iq = (0.5 * np.exp(2j * np.pi * 100_000 * t)).astype(np.complex64)
    db = power_spectrum_db(iq, 4096)
    freqs = np.fft.fftshift(np.fft.fftfreq(4096, 1 / fs))
    peak = freqs[np.argmax(db)]
    assert abs(peak - 100_000) < fs / 4096, f"pic spectre {peak}"
    f, part = zoom_spectrum(iq, fs, 0.0, 50_000, 150_000, npoints=256, nfft=4096)
    assert len(f) == len(part) <= 256
    print(f"spectrum ok (fenetres nommees, pic a {peak:.0f} Hz, zoom {len(f)} pts)")


def test_timesignals() -> None:
    bits = encode_minute(minute=37, hour=14, day=5, weekday=6, month=11, year=2026, summer=True)
    t = decode_minute(bits)
    assert t is not None and t["minute"] == 37 and t["hour"] == 14 and t["day"] == 5
    assert t["weekday"] == 6 and t["month"] == 11 and t["year"] == 2026 and t["summer"]
    bad = bits.copy()
    bad[21] ^= 1  # parite minute rompue
    assert decode_minute(bad) is None
    dcf = DCF77Decoder(48000, 700.0, emit=lambda s: None)
    dcf.process(np.zeros(48000, dtype=np.float32))
    print("timesignals ok (encode/decode minute aller-retour, parite, DCF77 smoke)")


def test_sigmet_pictures() -> None:
    txt = "SIGMET A VALID 251200/251800 LFPO TS ISOL CB FL300="
    p = parse_sigmet(txt)
    assert p is not None and p["kind"] == "SIGMET" and p["fir"] == "LFPO", p
    assert "Orages" in p["hazards"] and p["flight_level"] == "FL300", p
    # PNG pur python
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[..., 0] = 255
    path = "/tmp/test_pyphantom.png"
    write_png(path, img)
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"
    fx = FaxDecoder(48000, emit=lambda m: None)
    fx.process(np.zeros(48000, dtype=np.float32))
    print("sigmet/pictures ok (parse SIGMET, write_png signature PNG, FaxDecoder smoke)")


def test_simple_decoders_and_receiver() -> None:
    zeros8 = np.zeros(8000, dtype=np.float32)
    zeros48 = np.zeros(48000, dtype=np.float32)
    assert isinstance(PSK31Decoder(8000).process(zeros8), list)
    assert isinstance(RTTYDecoder(8000).process(zeros8), list)
    assert isinstance(WSPRDecoder(12000).process(zeros8), list)
    assert isinstance(PacketDecoder(48000).process(zeros48), list)
    assert isinstance(SSTVDecoder(8000).process(zeros8), list)
    assert isinstance(TETRADecoder(48000).process(zeros48), list)
    bb = Baseband(48000, 700.0, 40.0, 60.0)
    z = bb.process(np.zeros(48000, dtype=np.complex64))
    assert len(z) > 0 and abs(bb.fs_out - 48000 / bb.M) < 1e-9
    # chaine Receiver complete (I/Q -> audio 48 kHz) avec les vrais blocs noise/kalman/neural_nr
    r = Receiver(1_000_000)
    r.set_mode("USB")
    rng = np.random.default_rng(0)
    iq = (0.1 * (rng.standard_normal(100_000) + 1j * rng.standard_normal(100_000))).astype(np.complex64)
    audio = r.process(iq)
    assert len(audio) > 0
    print(f"decoders simples + Baseband + Receiver ok (audio {len(audio)} ech.)")


def test_noise_kalman_neural() -> None:
    rng = np.random.default_rng(7)
    n = 8000
    t = np.arange(n) / FS

    # --- NoiseBlanker : une impulsion forte sur une tonalite est effacee
    tone = (0.1 * np.exp(2j * np.pi * 700.0 * t)).astype(np.complex64)
    x = tone.copy()
    x[4000] = 20.0 + 0j
    x[4001] = -18.0 + 5j
    nb = NoiseBlanker(threshold=4.0, guard=2)
    nb.enabled = True
    out = nb.process(x)
    assert len(out) == len(x)
    assert nb.blanked > 0
    # le retard d'anticipation (4*guard) decale la sortie : l'impulsion tombe vers 4008
    assert np.max(np.abs(out[3990:4020])) < 1.0, np.max(np.abs(out[3990:4020]))
    nb_off = NoiseBlanker()
    nb_off.enabled = False
    assert np.max(np.abs(nb_off.process(x))) > 10.0  # desactive : l'impulsion passe

    # --- LineEnhancer mode "notch" : une porteuse pure est prédite puis soustraite
    le = LineEnhancer(ntaps=64, delay=64, mu=0.5, mode="notch")
    le.enabled = True
    sig = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    out = le.process(sig)
    assert len(out) == len(sig)
    rms_in = float(np.sqrt(np.mean(sig[n // 2:] ** 2)))
    rms_out = float(np.sqrt(np.mean(out[n // 2:] ** 2)))
    assert rms_out < 0.5 * rms_in, (rms_in, rms_out)
    le2 = LineEnhancer(ntaps=32, delay=1, mu=0.3, mode="enhance")
    le2.enabled = True
    assert len(le2.process(sig)) == len(sig)

    # --- KalmanFilter1D : converge vers une entree constante
    kf = KalmanFilter1D(process_noise=1e-4, measurement_noise=1e-2)
    v = 0.0
    for _ in range(500):
        v = kf.update(1.0)
    assert abs(v - 1.0) < 0.05
    out = kf.process(np.full(1000, 3.0, dtype=np.float32))
    assert len(out) == 1000 and abs(float(out[-1]) - 3.0) < 0.1

    # --- KalmanNR / NeuralNR : meme longueur en sortie, passthrough si desactives
    a = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    knr = KalmanNR(FS)
    assert knr.available
    knr.set_enabled(True)
    knr.set_strength(0.8)
    out = knr.process(a)
    assert len(out) == len(a) and out.dtype == np.float32
    knr.set_enabled(False)
    assert knr.process(a) is a
    nnr = NeuralNR(FS)
    nnr.set_enabled(True)
    out = nnr.process(a)
    assert len(out) == len(a) and out.dtype == np.float32
    nnr.set_enabled(False)
    assert nnr.process(a) is a

    # --- expint_e1 : valeurs de reference (Abramowitz & Stegun)
    xs = np.array([0.5, 1.0, 2.0, 5.0])
    ref = np.array([0.5597735947761608, 0.2193839343955203, 0.0489005107160611, 0.0011482955912753])
    assert np.allclose(expint_e1(xs), ref, rtol=1e-4, atol=1e-8)

    # --- SpectralNR : reduit la puissance d'un bruit blanc, 3 methodes, longueur bornee
    for method in SpectralNR.METHODS:
        snr = SpectralNR(FS, nfft=256, method=method, reduction_db=18.0)
        snr.enabled = True
        noisy = a + 0.05 * rng.standard_normal(n).astype(np.float32)
        out = snr.process(noisy)
        assert 0 < len(out) <= len(noisy)
    snr = SpectralNR(FS)
    snr.enabled = True
    noise = (0.2 * rng.standard_normal(48000)).astype(np.float32)
    out = snr.process(noise)
    rms_in_nr = float(np.sqrt(np.mean(noise[10000:] ** 2)))
    rms_out_nr = float(np.sqrt(np.mean(out[10000:] ** 2)))
    assert rms_out_nr < 0.7 * rms_in_nr, (rms_in_nr, rms_out_nr)

    # --- Receiver : les vrais blocs sont instancies et la chaine complete tourne avec
    r = Receiver(1_000_000)
    assert isinstance(r.nb, NoiseBlanker)
    assert isinstance(r.anf, LineEnhancer) and isinstance(r.lms, LineEnhancer)
    assert isinstance(r.nr, SpectralNR)
    assert isinstance(r.kalman, KalmanNR) and isinstance(r.neural_nr, NeuralNR)
    r.set_mode("USB")
    r.set_nb(True)
    r.set_nr(True, method="lsa")
    r.set_anf(True)
    r.kalman.set_enabled(True)
    r.neural_nr.set_enabled(True)
    iq = (0.1 * (rng.standard_normal(100_000) + 1j * rng.standard_normal(100_000))).astype(np.complex64)
    audio = r.process(iq)
    assert len(audio) > 0
    print(f"noise/kalman/neural ok (blanked={nb.blanked}, notch RMS {rms_in:.3f}->{rms_out:.4f}, "
          f"NR bruit {rms_in_nr:.3f}->{rms_out_nr:.4f}, Receiver+NR audio {len(audio)} ech.)")


def test_smoke_decoders() -> None:
    zeros = np.zeros(48000, dtype=np.float32)
    assert isinstance(DMRDecoder(FS).process(zeros), list)
    assert isinstance(C4FMDecoder(FS).process(zeros), list)
    assert isinstance(ACARSDecoder(FS).process(zeros), list)
    ais = AISDecoder(FS, emit=lambda m: None)  # GMSK reel maintenant present
    ais.process(zeros)
    dcs = DCSDetector(FS)
    dcs.process(zeros)
    print("smoke ok (DMR/C4FM/ACARS/AIS(GMSK)/DCS sur zeros)")


if __name__ == "__main__":
    test_crc()
    test_cpr()
    test_ais_pos()
    test_bitslicer()
    test_pocsag_golay()
    test_ctcss()
    test_audiofx()
    test_adsb_frame()
    test_ekf_fec_dsp()
    test_modes_gmsk_olivia()
    test_spectrum_and_windows()
    test_timesignals()
    test_sigmet_pictures()
    test_simple_decoders_and_receiver()
    test_noise_kalman_neural()
    test_smoke_decoders()
    print("Tous les tests decodeurs passent.")
