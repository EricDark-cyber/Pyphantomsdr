"""ADS-B / Mode S, AIS et ACARS : messages de référence publiés, signaux de synthèse bruités, bruit seul.

Références : les messages ADS-B et Mode S sont les exemples classiques de la littérature
(« The 1090 MHz Riddle ») ; le message AIS est l'exemple de la documentation AIVDM de gpsd.
"""
import numpy as np

from radiolab.dsp import acars, adsb, ais
from radiolab.dsp.crc24 import crc24_bytes
from radiolab.dsp.modes import decode_mode_s
from radiolab.dsp.receiver import Receiver

IDENT = "8D4840D6202CC371C32CE0576098"       # indicatif KLM1023
EVEN = "8D40621D58C382D690C8AC2863A7"        # position, trame paire
ODD = "8D40621D58C386435CC412692AD6"         # position, trame impaire
SPEED = "8D485020994409940838175B284F"       # vitesse


def noisy(env, snr_db, rng):
    sigma = 10 ** (-snr_db / 20) / np.sqrt(2)
    return (env + sigma * (rng.standard_normal(len(env)) + 1j * rng.standard_normal(len(env)))).astype(np.complex64)


# --------------------------------------------------------------------------------------------- ADS-B
def test_adsb_reference_messages():
    for raw in (IDENT, EVEN, ODD, SPEED):
        assert crc24_bytes(bytes.fromhex(raw)) == 0                       # le contrôle d'un vrai message est nul
    assert crc24_bytes(bytes.fromhex(IDENT[:10] + "3" + IDENT[11:])) != 0
    for fs in (2e6, 2.5e6, 10e6):
        rng = np.random.default_rng(1)
        env = np.concatenate([adsb.modulate(raw, fs, lead_us=float(rng.uniform(20, 60))) for raw in (IDENT, EVEN, ODD, SPEED) * 6])
        d = adsb.ADSBDecoder(fs)
        got = []
        iq = noisy(env, 20, rng)
        for i in range(0, len(iq), 65536):
            got += d.process_iq(iq[i:i + 65536], t=i / fs)
        assert len(got) >= 18, (fs, len(got))                             # 24 messages émis
        assert d.aircraft["4840D6"]["callsign"] == "KLM1023"
        pos = d.aircraft["40621D"]
        assert pos["altitude"] == 38000 and abs(pos["lat"] - 52.2572) < 1e-3 and abs(pos["lon"] - 3.91937) < 1e-3
        v = d.aircraft["485020"]
        assert abs(v["speed"] - 159.2) < 0.1 and abs(v["heading"] - 182.9) < 0.1 and v["vrate"] == -832


def test_adsb_rejects_errors_and_noise():
    fs = 10e6
    rng = np.random.default_rng(3)
    bad = IDENT[:12] + ("%X" % (int(IDENT[12], 16) ^ 4)) + IDENT[13:]     # un seul bit faux
    d = adsb.ADSBDecoder(fs)
    assert d.process_iq(noisy(adsb.modulate(bad, fs), 25, rng)) == [] and d.rejected >= 1
    d = adsb.ADSBDecoder(fs)
    noise = noisy(np.zeros(int(fs)), 0, rng)
    assert sum(len(d.process_iq(noise[i:i + 65536])) for i in range(0, len(noise), 65536)) == 0


def test_mode_s_radar_replies_need_a_known_address():
    assert decode_mode_s({"df": 5, "raw": "2A00516D492B80", "bits": 56})["squawk"] == "0356"
    assert decode_mode_s({"df": 4, "raw": "2000171806A983", "bits": 56})["altitude"] == 36000
    head = bytes.fromhex("20001718")                                      # réponse radar (format 4), 36000 pieds
    ap = crc24_bytes(head) ^ 0x4840D6                                     # contrôle mélangé à l'adresse 4840D6
    reply = (head + ap.to_bytes(3, "big")).hex().upper()
    fs = 10e6
    rng = np.random.default_rng(5)
    d = adsb.ADSBDecoder(fs)
    block = lambda raw: noisy(adsb.modulate(raw, fs, lead_us=500.0, tail_us=500.0), 25, rng)          # noqa: E731
    assert d.process_iq(block(reply), t=0.0) == []                                      # adresse jamais entendue : rejeté
    assert len(d.process_iq(block(IDENT), t=1.0)) == 1
    got = d.process_iq(block(reply), t=2.0)
    assert len(got) == 1 and got[0]["icao"] == "4840D6" and got[0]["altitude"] == 36000
    assert d.process_iq(block(reply), t=200.0) == []                                    # plus d'une minute après : oubliée


def test_adsb_worker_thread():
    fs = 2e6
    d = adsb.ADSBDecoder(fs)
    d.start()
    d.feed(noisy(adsb.modulate(IDENT, fs), 25, np.random.default_rng(0)))
    d.feed(np.zeros(4096, dtype=np.complex64))
    import time
    for _ in range(100):
        got = d.take()
        if got:
            break
        time.sleep(0.02)
    d.stop()
    assert got and got[0]["callsign"] == "KLM1023" and "ADS-B" in d.status()


# --------------------------------------------------------------------------------------------- AIS
REF = ais.payload_from_nmea("177KQJ5000G?tO`K>RA1wUbN0TKH")


def position(mmsi, lat, lon, sog, cog):
    f = lambda v, n: format(int(round(v)) & ((1 << n) - 1), "0%db" % n)   # noqa: E731
    bits = (f(1, 6) + "00" + f(mmsi, 30) + f(0, 4) + f(0, 8) + f(sog * 10, 10) + "1" + f(lon * 600000, 28)
            + f(lat * 600000, 27) + f(cog * 10, 12) + f(511, 9) + f(30, 6) + "0" * 25)
    return int(bits, 2).to_bytes(21, "big")


def test_ais_reference_message():
    assert ais.crc16_x25(b"123456789") == 0x906E
    m = ais.decode_message(REF)
    assert m == {"type": 1, "mmsi": 477553000, "status": 5, "lon": -122.345833, "lat": 47.582833,
                 "sog": 0.0, "cog": 51.0, "heading": 181, "second": 15}
    assert ais.decode_message(REF[:-1]) is None                           # mauvaise longueur pour ce type
    assert "477553000" in ais.format_ais(m) and "à quai" in ais.format_ais(m)


def test_ais_bursts_in_noise():
    fs = 50000.0
    rng = np.random.default_rng(4)
    msgs = [REF, position(227006760, 49.4421, 0.1073, 12.3, 215.0), position(228123400, -33.5, -70.25, 0.0, 0.0)]
    parts = []
    for rep in range(8):
        for m in msgs:
            parts.append(1.5 * rng.standard_normal(int(fs * rng.uniform(0.004, 0.02))).astype(np.float32))   # entre deux salves : bruit fort
            s = ais.modulate(m, fs, offset=rng.uniform(-0.25, 0.25))                                         # récepteur mal calé en fréquence
            parts.append(s + 0.48 * 10 ** (-15 / 20) * rng.standard_normal(len(s)).astype(np.float32))
    x = np.concatenate(parts)
    got = []
    d = ais.AISDecoder(fs, got.append)
    for i in range(0, len(x), 3000):
        d.process(x[i:i + 3000])
    assert len(got) >= 22 and set(d.vessels) == {477553000, 227006760, 228123400}
    assert abs(d.vessels[227006760]["lat"] - 49.4421) < 1e-4 and d.vessels[227006760]["sog"] == 12.3
    assert d.vessels[228123400]["lat"] == -33.5 and d.vessels[228123400]["lon"] == -70.25
    d = ais.AISDecoder(fs)
    noise = rng.standard_normal(int(fs * 30)).astype(np.float32)
    for i in range(0, len(noise), 5000):
        d.process(noise[i:i + 5000])
    assert d.count == 0                                                   # 30 s de bruit : aucun navire inventé
    assert "débit trop faible" in ais.AISDecoder(12500.0).status()


def test_ais_through_receiver():
    """I/Q modulé en fréquence -> récepteur en NFM -> décodeur AIS branché avant la réduction de débit."""
    fs, off = 250e3, 30e3
    rng = np.random.default_rng(2)
    f = np.concatenate([np.concatenate([np.zeros(int(fs * 0.01)), ais.modulate(m, fs, deviation=2400.0).astype(np.float64)])
                        for m in (REF, position(227006760, 49.4421, 0.1073, 12.3, 215.0)) * 4])
    iq = 0.02 * np.exp(2j * np.pi * np.cumsum(off + f) / fs) * (f != 0)
    iq = iq + 0.0006 * (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))
    rx = Receiver(fs)
    rx.set_mode("NFM"); rx.set_offset(off); rx.set_digi("AIS", 1000.0)
    for i in range(0, len(iq), 32768):
        rx.process(iq[i:i + 32768].astype(np.complex64))
    text = "".join(rx.digi_text)
    assert rx.digi.count >= 6 and "MMSI 477553000" in text and "MMSI 227006760" in text and "12.3 nd" in text
    rx.set_mode("USB")
    assert rx.digi is None                                                # l'AIS ne se décode qu'en NFM


# --------------------------------------------------------------------------------------------- ACARS
def test_acars_frame_parity_and_block_check():
    fs = 12500.0
    bits = acars.encode(".F-GKXA", "H1", "3", "M02AAF1234 #DFB/REPORT TEST 123 OIL PRESS OK")
    rng = np.random.default_rng(1)
    audio = np.concatenate([np.zeros(600, dtype=np.float32), acars.modulate(bits, fs), np.zeros(600, dtype=np.float32)])
    x = audio + 0.1 * rng.standard_normal(len(audio)).astype(np.float32)
    got = []
    d = acars.ACARSDecoder(fs, got.append)
    for i in range(0, len(x), 2048):
        d.process(x[i:i + 2048])
    assert len(got) == 1 and got[0]["crc_ok"]
    assert got[0]["registration"] == "F-GKXA" and got[0]["label"] == "H1" and got[0]["block"] == "3"
    assert got[0]["text"] == "M02AAF1234 #DFB/REPORT TEST 123 OIL PRESS OK"
    assert acars.format_acars(got[0]).startswith("ACARS  F-GKXA")

    # contrôle de bloc faux (deux bits du texte retournés : les parités restent justes) : compté, pas affiché
    bad = list(bits)
    k = 128 + 8 * 5 + 8 * 20
    bad[k] ^= 1; bad[k + 1] ^= 1
    d = acars.ACARSDecoder(fs)
    assert d.process(acars.modulate(bad, fs)) == [] or d.count == 0
    d.process(np.zeros(2048, dtype=np.float32))
    assert d.count == 0 and d.unverified == 1
    # une seule parité fausse : la trame est abandonnée
    bad = list(bits)
    bad[k] ^= 1
    d = acars.ACARSDecoder(fs)
    d.process(acars.modulate(bad, fs)); d.process(np.zeros(2048, dtype=np.float32))
    assert d.count == 0 and d.unverified == 0 and d.rejected >= 1
    d = acars.ACARSDecoder(fs)
    noise = rng.standard_normal(int(fs * 60)).astype(np.float32)
    for i in range(0, len(noise), 4096):
        d.process(noise[i:i + 4096])
    assert d.count == 0


def test_acars_through_receiver():
    """Porteuse modulée en amplitude par l'audio ACARS -> récepteur en AM -> décodeur."""
    fs, off = 250e3, 30e3
    bits = acars.encode(".F-HBNK", "Q0", "7", "")
    audio = acars.modulate(bits, fs)
    audio = np.concatenate([np.zeros(int(fs * 0.3), dtype=np.float32), audio, np.zeros(int(fs * 0.2), dtype=np.float32)])
    rng = np.random.default_rng(6)
    iq = 0.02 * (1 + 0.6 * audio) * np.exp(2j * np.pi * off * np.arange(len(audio)) / fs)
    iq = iq + 0.0004 * (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))
    rx = Receiver(fs)
    rx.set_mode("AM"); rx.set_offset(off); rx.set_digi("ACARS", 1000.0)
    for i in range(0, len(iq), 32768):
        rx.process(iq[i:i + 32768].astype(np.complex64))
    text = "".join(rx.digi_text)
    assert rx.digi.count == 1 and "F-HBNK" in text and "étiquette Q0" in text, (text, rx.digi.status())
