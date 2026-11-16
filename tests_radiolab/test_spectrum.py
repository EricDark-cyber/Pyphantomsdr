"""Analyse spectrale fine : résolution, fenêtres, zoom."""
import numpy as np
import pytest
from radiolab.dsp.spectrum import WINDOWS, get_window, power_spectrum_db, zoom_spectrum, auto_nfft
from radiolab.radio import Radio

FS = 2.5e6


def two_tones(n, f1, f2, a2=1.0):
    t = np.arange(n) / FS
    return (0.1 * np.exp(2j * np.pi * f1 * t) + 0.1 * a2 * np.exp(2j * np.pi * f2 * t)).astype(np.complex64)


@pytest.mark.parametrize("name", list(WINDOWS))
def test_windows_are_normalised(name):
    w = get_window(name, 4096)
    assert len(w) == 4096 and abs(w.sum() - 1) < 1e-4 and w.min() >= -0.01
    iq = (0.1 * np.exp(2j * np.pi * (FS / 4096 * 100) * np.arange(8192) / FS)).astype(np.complex64)   # pile sur un point
    assert abs(power_spectrum_db(iq, 4096, window=name).max() - (-20.0)) < 0.1      # 0,1 d'amplitude = -20 dB


def test_finer_fft_separates_close_signals():
    iq = two_tones(1 << 18, 100_000.0, 100_150.0)                # deux porteuses à 150 Hz d'écart
    def peaks(nfft):
        f, db = zoom_spectrum(iq, FS, 0.0, 99_000.0, 101_000.0, npoints=4000, nfft=nfft)
        strong = db > db.max() - 20
        return int(np.sum(np.diff(strong.astype(int)) == 1) + strong[0]), f, db
    assert peaks(4096)[0] == 1                                   # 610 Hz par point : une seule bosse
    n, f, db = peaks(1 << 18)                                    # 9,5 Hz par point : deux raies nettes
    assert n == 2 and abs(f[np.argmax(db)] - 100_075) < 100 and abs(db.max() + 20) < 1.0


def test_window_tradeoff_weak_signal_next_to_strong():
    iq = two_tones(1 << 16, 100_010.0, 100_600.0, a2=1e-3)       # un signal 60 dB plus faible, 590 Hz à côté
    def visible(name):
        f, db = zoom_spectrum(iq, FS, 0.0, 100_300.0, 100_900.0, npoints=4000, nfft=1 << 16, window=name)
        k = np.argmin(np.abs(f - 100_600.0))
        return db[k] - np.median(db)
    assert visible("Blackman-Harris") > 20                       # bien dégagé
    assert visible("Rectangulaire") < 10                         # noyé dans les bavures du signal fort


def test_zoom_reduces_to_screen_points_keeping_peaks():
    iq = two_tones(1 << 18, 400_000.0, -700_000.0)
    f, db = zoom_spectrum(iq, FS, 14_000_000.0, 14_000_000 - FS / 2, 14_000_000 + FS / 2, npoints=1000, nfft=1 << 18)
    assert len(f) == 1000 and abs(db.max() + 20) < 1.0
    assert abs(f[np.argmax(db)] - 14_400_000) < FS / 1000 or abs(f[np.argmax(db)] - 13_300_000) < FS / 1000
    f, db = zoom_spectrum(iq, FS, 14_000_000.0, 14_399_000.0, 14_401_000.0, npoints=1000, nfft=1 << 18)
    assert len(f) < 1000 and f[0] <= 14_399_010 and f[-1] >= 14_400_990 and abs(f[np.argmax(db)] - 14_400_000) < 10


def test_auto_nfft_follows_zoom():
    assert auto_nfft(FS, FS, 1536, 1 << 20) == 4096
    assert auto_nfft(FS, 90_000, 1536, 1 << 20) == 65536
    assert auto_nfft(FS, 2_000, 1536, 1 << 20) == 1 << 20
    assert auto_nfft(FS, 2_000, 1536, 100_000) == 65536          # pas encore assez d'échantillons reçus


def test_radio_ring_and_view():
    class Src:
        kind, sample_rate, center_hz, sample_rates = "file", int(FS), 14_000_000.0, [int(FS)]
        def close(self): pass
        def stop(self): pass
    radio = Radio(); radio.set_source(Src())
    iq = two_tones(1 << 21, 100_000.0, 100_150.0)
    from radiolab.audio import NullAudio
    radio.audio = NullAudio()
    radio.running = True
    import threading
    t = threading.Thread(target=radio._work, daemon=True); t.start()
    for i in range(0, len(iq), 65536):
        radio.blocks.put(iq[i:i + 65536])
    import time; time.sleep(1.0); radio.running = False; t.join()
    ring = radio.RING_SAMPLES                                    # la mémoire garde les derniers échantillons, pas plus
    assert radio.error is None and min(len(iq), ring) <= radio.ring_len < ring + 65536
    assert np.array_equal(radio.recent_iq(100_000), iq[-100_000:])
    f, db, nfft = radio.spectrum_view(14_099_000, 14_101_000)
    assert radio.zoom_div >= 32 and FS / nfft < 1.5 and abs(f[np.argmax(db)] - 14_100_075) < 100   # loupe : finesse sous 1,5 Hz
    strong = db > db.max() - 20
    assert int(np.sum(np.diff(strong.astype(int)) == 1) + strong[0]) == 2
    f, db, nfft = radio.spectrum_view(14_000_000 - FS / 2, 14_000_000 + FS / 2, nfft=8192, window="Hamming")
    assert nfft == 8192 and len(f) == 1536


def test_input_decimation(tmp_path):
    """Décimation par 8 : bande 8 fois plus étroite, signaux hors bande rejetés, écoute inchangée."""
    import time
    from radiolab.audio import WavWriter, NullAudio
    from radiolab.sources.iqfile import IQFileSource
    from radiolab.radio import allowed_decimations
    fs = 1_000_000
    tt = np.arange(fs * 3) / fs
    iq = (0.1 * np.exp(2j * np.pi * 21_000 * tt) + 0.3 * np.exp(2j * np.pi * 300_000 * tt)).astype(np.complex64)
    p = tmp_path / "d_7100000Hz_IQ.wav"
    w = WavWriter(str(p), fs, iq=True); w.write(iq); w.close()
    assert allowed_decimations(2_500_000) == [1, 2, 4, 8, 16, 32] and allowed_decimations(250_000) == [1, 2, 4]

    class Cap(NullAudio):
        def __init__(self): self.parts = []
        def write(self, a): self.parts.append(a)
    cap = Cap()
    radio = Radio(audio_factory=lambda: cap)
    radio.decimation = 8
    radio.set_source(IQFileSource(str(p), loop=False, realtime=True))
    assert radio.sample_rate == 125_000 and radio.rx.fs_in == 125_000
    radio.set_mode("USB"); radio.tune(7_120_000)
    radio.start(); time.sleep(2.0)
    f, db, nfft = radio.spectrum_view(7_100_000 - 62_500, 7_100_000 + 62_500, nfft=4096)
    assert abs(f[np.argmax(db)] - 7_121_000) < 100 and abs(db.max() + 20) < 1.0           # 30 Hz par point au lieu de 244
    assert np.sort(db)[-40] < -85                                                        # le signal à +300 kHz ne se replie pas
    radio.tune(7_170_000); assert radio.vfo_hz == 7_100_000 + 0.40 * 125_000              # accord borné à la bande utile
    radio.tune(7_120_000)
    rec = radio.start_recording(str(tmp_path), iq=True); time.sleep(0.5)
    radio.stop()
    assert radio.error is None and radio.lost_blocks == 0
    audio = np.concatenate(cap.parts)[24000:60000]
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    assert abs(np.argmax(spec) * 48000 / len(audio) - 1000) < 5                           # la note de 1 kHz est intacte
    assert IQFileSource(rec, loop=False, realtime=False).sample_rate == 125_000           # enregistrement I/Q allégé
    with pytest.raises(RuntimeError):
        radio.running = True; radio.set_decimation(2)
    radio.running = False
    radio.set_decimation(2); assert radio.sample_rate == 500_000 and radio.vfo_hz == 7_120_000
    radio.set_decimation(1); assert radio.decimator is None and radio.sample_rate == fs


# --------------------------------------------------------------------------------------------- loupe (zoom FFT)
from radiolab.dsp.spectrum import zoom_fft, zoom_plan, zoom_raw_samples   # noqa: E402


def _tone(n, f, a, fs=FS):
    return (a * np.exp(2j * np.pi * f * np.arange(n) / fs)).astype(np.complex64)


def test_zoom_plan():
    assert zoom_plan(FS, FS) is None and zoom_plan(FS, 400_000) is None          # bande large : FFT classique
    assert zoom_plan(FS, 100_000) == (1, 10)
    d1, d2 = zoom_plan(FS, 3_000)
    assert d2 == 16 and FS / (d1 * d2) >= 2.5 * 3_000                            # débit réduit suffisant pour la bande


@pytest.mark.parametrize("fs,span,expected_hz", [(2.5e6, 3_000.0, 1.0), (2.5e6, 500.0, 0.65), (10e6, 3_000.0, 2.5)])
def test_zoom_is_finer_and_levels_are_right(fs, span, expected_hz):
    """Deux porteuses très proches sont séparées, et les niveaux sont justes jusqu'aux bords de la bande regardée."""
    n = 1 << 22
    f0 = 314_159.0
    sep = 6 * expected_hz
    rng = np.random.default_rng(0)
    iq = (_tone(n, f0, 0.1, fs) + _tone(n, f0 + sep, 0.1, fs) + _tone(n, f0 - 0.45 * span, 0.01, fs) + _tone(n, f0 + 0.48 * span, 0.01, fs)
          + (1e-4 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64))
    raw = zoom_raw_samples(fs, span, 1536, n)
    f, db, used = zoom_fft(iq[-raw:], fs, 0.0, f0 - span / 2, f0 + span / 2, 1536)
    res = fs / used
    assert res <= expected_hz and res < fs / (1 << 20) / 2                       # au moins deux fois plus fin que la FFT d'un million de points
    assert len(f) <= 1536 and f[0] >= f0 - span / 2 - res and f[-1] <= f0 + span / 2 + res
    peak = lambda fq: db[np.abs(f - fq) <= 2 * res].max()                        # noqa: E731
    assert abs(peak(f0) + 20) < 0.3 and abs(peak(f0 + sep) + 20) < 0.3
    assert abs(peak(f0 - 0.45 * span) + 40) < 0.3 and abs(peak(f0 + 0.48 * span) + 40) < 0.3
    assert db[(f > f0 + 0.35 * sep) & (f < f0 + 0.65 * sep)].min() < -30         # un vrai creux entre les deux porteuses


def test_zoom_rejects_signals_outside_the_view():
    """Un signal fort placé là où il se replierait dans la bande regardée ne doit pas y apparaître."""
    n = 1 << 21
    f0, span = 200_000.0, 5_000.0
    d1, d2 = zoom_plan(FS, span)
    fs_d = FS / (d1 * d2)
    rng = np.random.default_rng(1)
    noise = (1e-5 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    for alias in (fs_d, -fs_d + 0.3 * span, 5 * fs_d - 0.4 * span, FS / d1 + 0.2 * span):
        iq = noise + _tone(n, f0 + alias, 0.5)                                   # -6 dB, hors de la vue
        f, db, _ = zoom_fft(iq[-zoom_raw_samples(FS, span, 1536, n):], FS, 0.0, f0 - span / 2, f0 + span / 2)
        assert db.max() < -86, (alias, db.max())                                 # plus de 80 dB sous le brouilleur


def test_radio_view_uses_the_zoom():
    r = Radio()

    class Src:
        sample_rate = FS
        center_hz = 7_100_000.0
    r.source = Src()
    r.center_hz = 7_100_000.0
    n = 1 << 21
    iq = _tone(n, 12_345.0, 0.1) + _tone(n, 12_355.0, 0.1)                       # 10 Hz d'écart
    for i in range(0, n, 65536):
        r.ring.append(iq[i:i + 65536]); r.ring_len += 65536
    f, db, used = r.spectrum_view(7_100_000 + 12_345 - 1_000, 7_100_000 + 12_345 + 1_000, 1536)
    assert r.zoom_div >= 32 and FS / used < 1.3
    k1, k2 = np.argmin(np.abs(f - 7_112_345.0)), np.argmin(np.abs(f - 7_112_355.0))
    assert abs(db[k1 - 2:k1 + 3].max() + 20) < 0.3 and abs(db[k2 - 2:k2 + 3].max() + 20) < 0.3 and db[k1:k2].min() < -40
    f, db, nfft = r.spectrum_view(7_100_000 - FS / 2, 7_100_000 + FS / 2, 1536)  # toute la bande : FFT classique
    assert r.zoom_div == 0 and nfft == 4096
    f, db, nfft = r.spectrum_view(7_112_345 - 1_000, 7_112_345 + 1_000, 1536, nfft=65536)   # taille imposée : respectée
    assert r.zoom_div == 0 and nfft == 65536


def test_sharp_waterfall_follows_time_and_keeps_thin_lines():
    """Cascade nette : une porteuse qui vient de s'arrêter disparaît tout de suite, alors que
    l'analyse longue seule la montre encore ; une porteuse stable garde son niveau et sa finesse."""
    from radiolab.dsp.spectrum import ZOOM_T_SHARP
    fs, span, f0 = 250e3, 2_000.0, 40_000.0
    n = int(fs * 3)
    t = np.arange(n) / fs
    stops = n / fs - ZOOM_T_SHARP / 2 - 0.12                  # s'arrête 0,12 s avant l'instant analysé
    rng = np.random.default_rng(2)
    iq = (0.1 * np.exp(2j * np.pi * (f0 - 400) * t) + 0.1 * (t < stops) * np.exp(2j * np.pi * (f0 + 300) * t)
          + 1e-4 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(np.complex64)
    raw = zoom_raw_samples(fs, span, 1536, n, ZOOM_T_SHARP)
    args = (iq[-raw:], fs, 0.0, f0 - span / 2, f0 + span / 2, 1536)
    f, plain, used = zoom_fft(*args, window="Blackman-Harris", sharp=False)
    f2, sharp, used2 = zoom_fft(*args, window="Blackman-Harris", sharp=True)
    assert abs(fs / used2 - 1 / ZOOM_T_SHARP) < 0.1           # analyse longue de 0,6 s
    near = lambda db, fq, w=3.0: db[np.abs(f2 - fq) <= w].max()     # noqa: E731
    assert abs(near(sharp, f0 - 400) + 20) < 0.5              # porteuse stable : niveau juste
    assert (sharp[np.abs(f2 - (f0 - 400)) < 60] > -40).sum() <= 8      # et raie fine : moins de 11 Hz de large à 20 dB sous le sommet
    gone_plain, gone_sharp = plain[np.abs(f - (f0 + 300)) <= 3].max(), near(sharp, f0 + 300)
    assert gone_plain > -40 and gone_sharp < gone_plain - 25  # la porteuse arrêtée ne traîne plus
