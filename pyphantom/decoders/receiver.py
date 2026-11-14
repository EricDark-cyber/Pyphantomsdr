import numpy as np
np.seterr(all='ignore')
from .dsp_optimized import OptimizedDSP
"""Chaîne de réception complète : I/Q du récepteur -> audio 48 kHz.

    I/Q (2,5 MS/s)
      -> PolyDDC        : sélection de la fréquence + décimation      (~50 kHz ou ~250 kHz)
      -> NoiseBlanker   : effacement des impulsions
      -> filtre de canal + démodulation (AM / BLU / CW / NFM / WFM)   (~12,5 kHz ou ~50 kHz)
      -> LineEnhancer   : notch automatique
      -> SpectralNR ou LineEnhancer("enhance") : réduction de bruit
      -> squelch, AGC, volume
      -> LinearResampler: 48 kHz pour la carte son
"""
import collections
import threading
from ..filters import (StreamFIR, PolyDDC, Mixer, LinearResampler, lowpass_taps,
                      bandpass_taps, taps_for_transition, exp_taps)
from .ekf import EKFFilter
from .tetra_voice import TetraVoiceDecoder


from .kalman import KalmanNR
from .neural_nr import NeuralNR
from .noise import NoiseBlanker, LineEnhancer, SpectralNR
from .fmradio import StereoPilot, RDSDecoder, ToneDetector
from .digital import DCSDetector, Pocsag
from .audiofx import AudioFX, Contour, APF, DeepNotch
from .hfdigi import make_decoder
from .tetra import TetraDecoder

# mode : (filtre par défaut bas, haut en Hz ; largeur de transition du filtre)
MODES = {
    "AM":  (-4500.0, 4500.0, 400.0),
    "LSB": (-2800.0, -100.0, 200.0),
    "USB": (100.0, 2800.0, 200.0),
    "CW":  (450.0, 950.0, 100.0),      # centré sur la note de 700 Hz
    "DSB": (-3000.0, 3000.0, 300.0),   # double bande latérale : les deux côtés de la porteuse
    "RAW": (-5000.0, 5000.0, 500.0),   # I/Q brut : I sur la voie gauche, Q sur la voie droite
    "TETRA": (-12500.0, 12500.0, 0.0), # analyse d'un canal TETRA de 25 kHz (pas de son)
    "NFM": (-6000.0, 6000.0, 1500.0),
    "WFM": (-90000.0, 90000.0, 0.0),
}
CW_PITCH = 700.0


class AGC:
    """Commande automatique de gain : attaque immédiate, remontée lente."""

    def __init__(self, fs, target=0.25, decay_s=0.5, max_gain=1e4):
        self.fs, self.target, self.max_gain = fs, target, max_gain
        self.step = max(int(fs * 0.004), 1)                  # le gain est recalculé toutes les 4 ms
        self.rise = 1.0 - np.exp(-self.step / (decay_s * fs))
        self.gain = 1.0
        self.enabled = True

    def process(self, x):
        if not self.enabled or len(x) == 0:
            return x
        n = len(x)
        edges = np.arange(0, n, self.step)
        peaks = np.maximum.reduceat(np.abs(x), edges)
        gains = np.empty(len(edges) + 1)
        gains[0] = g = self.gain
        for i, pk in enumerate(peaks):
            want = min(self.target / max(float(pk), 1e-9), self.max_gain)
            g = want if want < g else g + (want - g) * self.rise
            gains[i + 1] = g
        self.gain = g
        pos = np.append(edges, n)
        return (x * np.interp(np.arange(n), pos, gains)).astype(np.float32)


class Receiver:
    """Un récepteur : reçoit des blocs I/Q complexes, rend de l'audio mono 48 kHz."""

    def __init__(self, fs_in, audio_rate=48000):
        self.lock = threading.RLock()
        self.fs_in, self.audio_rate = float(fs_in), int(audio_rate)
        self.mode = "USB"
        self.offset = 0.0
        self.lo, self.hi = MODES["USB"][:2]
        self.volume = 0.5
        self.squelch_db = -150.0
        self.level_db = -150.0
        self.nr_method = "lsa"
        self.nr_enabled = self.anf_enabled = False
        self.nr_reduction_db = 18.0
        self.agc_enabled = True
        self.stereo_enabled = True
        self.rds_enabled = True
        self.is_stereo = False
        self.rds = None
        self.ctcss_hz = None
        self.dcs_label = None
        self.pocsag_enabled = False
        self.pocsag_messages = collections.deque(maxlen=200)
        self.raw_output = False            # FM étroite : sortie directe du discriminateur, pour décodeur externe
        self.notches, self.bass_db, self.treble_db = [], 0.0, 0.0
        self.digi_kind, self.digi_center, self.digi_reverse = None, 1000.0, False
        self.cw_pitch = CW_PITCH           # note entendue quand on est accordé pile sur une émission CW
        self.filter_window, self.filter_order = None, None   # forme du filtre de canal (None = automatique)
        self.binaural = False              # BLU / CW : signal en quadrature sur les deux oreilles
        self.anti_fading = False           # AM synchrone : choix de la meilleure bande latérale
        self.fade_choice = ""
        self.am_sync = False               # AM synchrone : démodulation par la porteuse reconstituée
        self.digi = None
        self.cw_wpm, self.cw_threshold_db = None, None        # None = réglage automatique
        self.digi_text = collections.deque(maxlen=5000)       # caractères décodés, lus par l'interface
        self.nb = NoiseBlanker()
        self._build()

    # ------------------------------------------------------------ réglages
    def set_mode(self, mode, reset_filter=True):
        with self.lock:
            if mode not in MODES:
                raise ValueError("mode inconnu : %s" % mode)
            if mode == "WFM" and self.fs_in < 200e3:
                raise ValueError("la FM large demande au moins 200 kHz de bande")
            if mode == "TETRA" and self.fs_in < 54e3:
                raise ValueError("l'analyse TETRA demande au moins 54 kHz de bande")
            self.mode = mode
            if reset_filter:
                self.lo, self.hi = MODES[mode][:2]
                if mode == "CW":
                    self.lo, self.hi = self.cw_pitch - 250.0, self.cw_pitch + 250.0
            self._build()

    def set_offset(self, offset_hz):
        """Écart entre la fréquence écoutée et la fréquence centrale du récepteur."""
        with self.lock:
            self.offset = float(offset_hz)
            self._apply_offset()

    def set_filter(self, lo, hi):
        with self.lock:
            if hi - lo < 50:
                return
            self.lo, self.hi = float(lo), float(hi)
            self._build_channel()

    def set_nb(self, enabled, threshold=None):
        with self.lock:
            self.nb.enabled = bool(enabled)
            if threshold is not None:
                self.nb.threshold = float(threshold)

    def set_nr(self, enabled, method=None, reduction_db=None):
        with self.lock:
            self.nr_enabled = bool(enabled)
            if method is not None:
                self.nr_method = method
            if reduction_db is not None:
                self.nr_reduction_db = float(reduction_db)
            self._apply_audio_options()

    def set_anf(self, enabled):
        with self.lock:
            self.anf_enabled = bool(enabled)
            self._apply_audio_options()

    def set_digi(self, kind, center=None, reverse=None):
        """Décodeur HF : None, "RTTY45", "RTTY50", "PSK31", "PSK63" ou "CW" ; `center` = fréquence audio du signal."""
        with self.lock:
            self.digi_kind = kind
            if center is not None:
                self.digi_center = float(center)
            if reverse is not None:
                self.digi_reverse = bool(reverse)
            self._make_digi()

    def set_cw_options(self, wpm=None, threshold_db=None):
        """Décodeur CW : vitesse et seuil imposés, ou None pour les laisser automatiques."""
        with self.lock:
            self.cw_wpm, self.cw_threshold_db = wpm, threshold_db
            if self.digi is not None and self.digi_kind == "CW":
                self.digi.set_speed(wpm)
                self.digi.set_threshold_db(threshold_db)

    def digi_audio_center(self):
        """Fréquence audio où le décodeur attend le signal (la note CW en mode CW)."""
        return self.cw_pitch if self.mode == "CW" else self.digi_center

    def set_cw_pitch(self, hz):
        """Hauteur de la note CW (300 à 1200 Hz) ; le filtre CW reste centré sur elle."""
        with self.lock:
            hz = float(min(max(hz, 300.0), 1200.0))
            if self.mode == "CW":
                half = (self.hi - self.lo) / 2
                self.lo, self.hi = hz - half, hz + half
            self.cw_pitch = hz
            if self.mode == "CW":
                self._build_channel()
                self._apply_offset()
                self._make_digi()

    def set_am_sync(self, enabled, anti_fading=None):
        with self.lock:
            self.am_sync = bool(enabled)
            if anti_fading is not None:
                self.anti_fading = bool(anti_fading)

    def set_filter_shape(self, window=None, order=None):
        """Fenêtre et nombre de coefficients du filtre de canal (None = choix automatique)."""
        with self.lock:
            self.filter_window = window
            self.filter_order = None if not order else int(min(max(order, 11), 3001)) | 1
            self._build_channel()

    def set_binaural(self, enabled):
        with self.lock:
            self.binaural = bool(enabled)

    def _make_digi(self):
        self.digi = None
        wanted = ("NFM",) if self.digi_kind == "APRS1200" else ("USB", "LSB", "CW")
        if self.digi_kind == "SSTV":                          # la SSTV existe aussi en FM (station spatiale, relais)
            wanted += ("NFM",)
        if self.mode == "TETRA":
            self.digi = TetraDecoder(self.fs1, self.digi_text.append)
        elif self.digi_kind and self.mode in wanted:
            # en RTTY la fréquence de repos (mark) est la plus haute à l'antenne : en BLI elle devient la plus basse
            mark_high = (self.mode != "LSB") != self.digi_reverse
            self.digi = make_decoder(self.digi_kind, self.fs2, self.digi_audio_center(),
                                     self.digi_text.append, mark_high)
            if self.digi is not None and self.digi_kind == "CW":
                self.digi.set_speed(self.cw_wpm)
                self.digi.set_threshold_db(self.cw_threshold_db)

    def set_pocsag(self, enabled):
        with self.lock:
            self.pocsag_enabled = bool(enabled)

    def set_raw_output(self, enabled):
        with self.lock:
            if bool(enabled) != self.raw_output:
                self.raw_output = bool(enabled)
                self._build()

    def set_audio_fx(self, notches=None, bass_db=None, treble_db=None):
        with self.lock:
            if notches is not None:
                self.notches = list(notches)
            if bass_db is not None:
                self.bass_db = float(bass_db)
            if treble_db is not None:
                self.treble_db = float(treble_db)
            self._apply_audio_options()

    def set_agc(self, enabled):
        with self.lock:
            self.agc_enabled = bool(enabled)
            self.agc.enabled = self.agc_enabled and self.mode not in ("NFM", "WFM")

    def set_contour(self, enabled, freq=1500.0, gain_db=0.0, width_hz=800.0):
        with self.lock:
            self.contour_enabled = bool(enabled)
            self.contour.configure(enabled, freq, gain_db, width_hz)

    def set_apf(self, enabled, center=700.0, width_hz=100.0, gain_db=12.0):
        with self.lock:
            self.apf_enabled = bool(enabled)
            self.apf.configure(enabled, center, width_hz, gain_db)

    def set_deep_notch(self, enabled, freq=1000.0, q=30.0):
        with self.lock:
            self.deep_notch_enabled = bool(enabled)
            self.deep_notch.configure(enabled, freq, q)

    # -------------------------------------------------------- construction
    def _build(self):
        wide = self.mode == "WFM"
        tetra = self.mode == "TETRA"
        target = 250e3 if wide else (90e3 if tetra else 50e3)
        self.M1 = max(int(self.fs_in // target), 1)
        self.fs1 = self.fs_in / self.M1
        if self.M1 >= 2:
            self.ddc = PolyDDC(self.fs_in, self.M1, 100e3 if wide else (min(13e3, 0.3 * self.fs1) if tetra else min(8e3, 0.3 * self.fs1)))
        else:
            self.ddc = Mixer(self.fs_in)
        raw = self.raw_output and self.mode == "NFM"
        self.M2 = max(int(round(self.fs1 / (50e3 if wide else (25e3 if raw else 12.5e3)))), 1)
        if tetra:
            self.M2 = 1
        self.fs2 = self.fs1 / self.M2                       # débit audio interne
        self.prev = np.zeros(1, dtype=np.complex64)         # dernier échantillon (démodulation FM)
        self.dc = 0.0
        self._build_channel()
        self.anf = LineEnhancer(ntaps=64, delay=max(int(self.fs2 * 0.005), 8), mu=0.5, mode="notch")
        self.lms = LineEnhancer(ntaps=128, delay=1, mu=0.3, leak=1e-4, mode="enhance")
        self.nr = SpectralNR(self.fs2)
        self.neural_nr = NeuralNR(self.fs2)
        self.kalman = KalmanNR(self.fs2)
        self.ekf = EKFFilter(self.fs2)
        self.tetra_voice = TetraVoiceDecoder()
        self.agc = AGC(self.fs2)
        self.resampler = LinearResampler(self.fs2, self.audio_rate)
        self.resampler_r = LinearResampler(self.fs2, self.audio_rate)      # voie droite (FM stéréo)
        self.is_stereo, self.ctcss_hz = False, None
        self.rds = RDSDecoder(self.fs1) if wide else None
        self.pilot = StereoPilot(self.fs1) if wide else None
        self.tones = ToneDetector(self.fs2) if self.mode == "NFM" else None
        self.dcs = DCSDetector(self.fs2) if self.mode == "NFM" else None
        self.pocsag = Pocsag(self.fs2, self.pocsag_messages.append) if self.mode == "NFM" else None
        self.dcs_label = None
        self.fx = AudioFX(self.fs2, sharp=not wide)
        self.fx_r = AudioFX(self.fs2, sharp=False)
        self.contour = Contour(self.fs2)
        self.apf = APF(self.fs2)
        self.deep_notch = DeepNotch(self.fs2)
        self.contour_enabled = False
        self.apf_enabled = False
        self.deep_notch_enabled = False
        self._make_digi()
        self._apply_offset()
        self._apply_audio_options()
        self.set_agc(self.agc_enabled)

    def _build_channel(self):
        m, fs1, fs2 = self.mode, self.fs1, self.fs2
        trans = MODES[m][2]
        self.deemph = None
        if m == "TETRA":
            self.chan = self.post = None
            return
        if m == "WFM":
            self.chan = None                                 # le filtre du DDC suffit
            pass_hz = 15e3
            ntaps = taps_for_transition(fs1, (fs2 - pass_hz) - pass_hz) if self.M2 > 1 else 63
            taps = lowpass_taps(fs2 / 2 if self.M2 > 1 else pass_hz, fs1, ntaps)
            self.post, self.post_s = StreamFIR(taps, self.M2), StreamFIR(taps, self.M2)
            # coupe nette entre l'audio (15 kHz) et le pilote (19 kHz), une fois le débit réduit
            sharp = lowpass_taps(17000.0, fs2, taps_for_transition(fs2, 3500.0)) if fs2 > 40e3 else np.ones(1)
            self.sharp, self.sharp_s = StreamFIR(sharp), StreamFIR(sharp)
            self.deemph, self.deemph_s = StreamFIR(exp_taps(50e-6, fs2)), StreamFIR(exp_taps(50e-6, fs2))
        elif m == "NFM":
            half = min(max(abs(self.lo), abs(self.hi)), 0.45 * fs1)
            self.chan = StreamFIR(lowpass_taps(half, fs1, taps_for_transition(fs1, trans)))
            if self.raw_output:              # bande large et plate jusqu'à ~8 kHz pour les décodeurs numériques
                self.post = StreamFIR(lowpass_taps(0.36 * fs2, fs1, taps_for_transition(fs1, 0.2 * fs2)), self.M2)
            else:
                self.post = StreamFIR(lowpass_taps(3400.0, fs1, taps_for_transition(fs1, 1200.0)), self.M2)
            self.deemph = StreamFIR(exp_taps(530e-6, fs2))
        else:                                                # AM, LSB, USB, CW
            edge = min(max(abs(self.lo), abs(self.hi)) + trans, 0.46 * fs2)
            if self.M2 > 1:
                self.pre = StreamFIR(lowpass_taps(fs2 / 2, fs1,
                                                  taps_for_transition(fs1, max(fs2 - 2 * edge, 500.0))), self.M2)
            else:
                self.pre = None
            ntaps = self.filter_order or taps_for_transition(fs2, trans)
            self.chan = StreamFIR(bandpass_taps(self.lo, self.hi, fs2, ntaps, window=self.filter_window))
            self.post = None
            # anti-fading : une voie par bande latérale, de part et d'autre de la porteuse
            n_sb = taps_for_transition(fs2, 150.0)
            self.sb_up = StreamFIR(bandpass_taps(80.0, max(self.hi, 500.0), fs2, n_sb))
            self.sb_lo = StreamFIR(bandpass_taps(min(self.lo, -500.0), -80.0, fs2, n_sb))
            self.p_up = self.p_lo = 0.0
            # AM synchrone : filtre très étroit qui isole la porteuse, et retard égal pour le signal
            self.carrier = StreamFIR(lowpass_taps(25.0, fs2, taps_for_transition(fs2, 60.0)).astype(np.complex64))
            self.carrier_delay = np.zeros((len(self.carrier.taps) - 1) // 2, dtype=np.complex64)
            self.sync_locked = False

    def _apply_offset(self):
        shift = self.offset - (self.cw_pitch if self.mode == "CW" else 0.0)
        self.ddc.set_offset(shift)

    def _apply_audio_options(self):
        narrow = self.mode != "WFM"
        self.anf.enabled = self.anf_enabled and self.mode in ("AM", "LSB", "USB", "DSB", "NFM")
        use_lms = self.nr_method == "lms"
        was = self.nr.enabled
        self.lms.enabled = self.nr_enabled and narrow and use_lms
        self.nr.enabled = self.nr_enabled and narrow and not use_lms
        if not use_lms:
            self.nr.method = self.nr_method
        self.nr.reduction_db = self.nr_reduction_db
        if self.nr.enabled and not was:
            self.nr.reset()
        self.fx.configure(self.notches if narrow else (), self.bass_db, self.treble_db)
        self.fx_r.configure((), self.bass_db, self.treble_db)

    # ----------------------------------------------------------- traitement
    def process(self, iq):
        """iq : tableau complex64 au débit fs_in.  Retour : audio float32 à 48 kHz."""
        with self.lock:
            z = self.ddc.process(iq)
            if len(z) == 0:
                return np.zeros(0, dtype=np.float32)
            m = self.mode
            if m == "TETRA":                                 # analyse seule : pas de son
                self._measure(z)
                self.digi.process(z)
                return self.resampler.process(np.zeros(len(z), dtype=np.float32))
            if m != "WFM":
                z = self.nb.process(z)
            if m in ("NFM", "WFM"):
                if self.chan is not None:
                    z = self.chan.process(z)
                self._measure(z)
                zz = np.concatenate([self.prev, z])
                self.prev = z[-1:]
                dev = 75e3 if m == "WFM" else 5e3
                a = np.angle(zz[1:] * np.conj(zz[:-1])).astype(np.float32) * (self.fs1 / (2 * np.pi * dev))
                if m == "WFM":
                    return self._wfm_audio(a)
                a = self.post.process(a)
                self.tones.process(a)
                self.ctcss_hz = self.tones.tone
                self.dcs.process(a)
                self.dcs_label = self.dcs.label()
                if self.pocsag_enabled:
                    self.pocsag.process(a)
                if self.digi is not None:                    # APRS / packet 1200 bauds
                    self.digi.process(a)
                if self.raw_output:
                    return self.resampler.process(np.clip(a * self.volume, -1.0, 1.0).astype(np.float32))
                a = self.deemph.process(a)
            else:
                if self.pre is not None:
                    z = self.pre.process(z)
                z = self.chan.process(z)
                self._measure(z)
                if m == "RAW":
                    return self._raw_audio(z)
                if m == "AM":
                    env = self._am_sync(z) if self.am_sync else np.abs(z)
                    if len(env):
                        self.dc += 0.1 * (float(env.mean()) - self.dc)
                    a = (env - self.dc).astype(np.float32)
                else:                                        # BLU / CW : la partie réelle suffit,
                    a = (z.real * 2.0).astype(np.float32)    # le filtre a déjà supprimé l'autre bande
                    if self.digi is not None:
                        self.digi.process(a)
                    if self.binaural:                        # partie réelle à gauche, partie imaginaire à droite
                        return self._raw_audio(z * 2.0)
            a = self.anf.process(a)
            a = self.lms.process(a)
            a = self.nr.process(a)
            if self.ekf.enabled:
                a = self.ekf.process(a)
            if self.kalman.enabled:
                a = self.kalman.process(a)
            if self.neural_nr.enabled:
                a = self.neural_nr.process(a)
            a = self.fx.process(a)
            a = self.contour.process(a)
            if self.mode == "CW":
                a = self.apf.process(a)
            a = self.deep_notch.process(a)          # notch profond avant AGC
            if self.level_db < self.squelch_db:
                a = a * 0.0
            a = self.agc.process(a)
            a = np.clip(a * self.volume * (1.0 if self.agc.enabled else 0.5), -1.0, 1.0)
            return self.resampler.process(a.astype(np.float32))

    def _am_sync(self, z):
        """Détection synchrone : on multiplie le signal par sa propre porteuse, isolée et « nettoyée ».

        Le détecteur d'enveloppe classique déforme le son quand la porteuse faiblit (évanouissement
        sélectif en ondes courtes) ; la détection synchrone n'utilise que la PHASE de la porteuse et
        reste propre.  Si la porteuse est introuvable, on revient à l'enveloppe.
        """
        c = self.carrier.process(z)
        buf = np.concatenate([self.carrier_delay, z])
        zd = buf[:len(z)]
        self.carrier_delay = buf[len(z):]
        mag = np.abs(c)
        self.sync_locked = bool(len(z)) and float(np.mean(mag)) > 0.25 * float(np.sqrt(np.mean(np.abs(zd) ** 2)) + 1e-30)
        if not self.sync_locked:
            self.fade_choice = ""
            return np.abs(zd)
        w = zd * np.conj(c) / (mag + 1e-30)                        # porteuse ramenée sur l'axe réel
        if not self.anti_fading:
            return w.real
        # Anti-fading : l'évanouissement sélectif touche rarement les deux bandes latérales à la fois.
        # Chacune porte tout le message ; on garde la plus forte, ou les deux si elles se valent.
        up, lo = 2.0 * self.sb_up.process(w).real, 2.0 * self.sb_lo.process(w).real
        self.p_up += 0.2 * (float(np.mean(up * up)) - self.p_up)
        self.p_lo += 0.2 * (float(np.mean(lo * lo)) - self.p_lo)
        if self.p_up > 2.0 * self.p_lo:
            self.fade_choice = "haute"
            return up
        if self.p_lo > 2.0 * self.p_up:
            self.fade_choice = "basse"
            return lo
        self.fade_choice = "les deux"
        return 0.5 * (up + lo)

    def _raw_audio(self, z):
        """Mode RAW : le signal I/Q filtré, I à gauche et Q à droite (pour un logiciel externe)."""
        gain = self.volume * (0.0 if self.level_db < self.squelch_db else 1.0)
        if self.agc.enabled:
            self.agc.process(z.real.astype(np.float32))            # sert seulement à faire évoluer le gain
            gain *= self.agc.gain
        z = z * gain
        left = self.resampler.process(np.clip(z.real, -1.0, 1.0).astype(np.float32))
        right = self.resampler_r.process(np.clip(z.imag, -1.0, 1.0).astype(np.float32))
        return np.stack([left, right], axis=1)

    def _wfm_audio(self, mpx):
        """Multiplex FM -> audio 48 kHz, mono (1 colonne) ou stéréo (tableau n x 2)."""
        if self.rds_enabled:
            self.rds.process(mpx)
        carrier = self.pilot.process(mpx)                     # porteuse 38 kHz reconstruite (zéros sans pilote)
        mono = self.deemph.process(self.sharp.process(self.post.process(mpx)))               # G + D
        diff = self.deemph_s.process(self.sharp_s.process(self.post_s.process(mpx * carrier)))  # G - D
        gain = self.volume * 0.5 * (0.0 if self.level_db < self.squelch_db else 1.0)
        self.is_stereo = self.stereo_enabled and self.pilot.locked
        left = self.fx.process(mono + diff if self.is_stereo else mono)
        right = self.fx_r.process(mono - diff if self.is_stereo else mono)
        left = np.clip(left * gain, -1.0, 1.0).astype(np.float32)
        right = np.clip(right * gain, -1.0, 1.0).astype(np.float32)
        out_l, out_r = self.resampler.process(left), self.resampler_r.process(right)
        return np.stack([out_l, out_r], axis=1) if self.is_stereo else out_l

    def rds_info(self):
        with self.lock:
            return self.rds.info() if self.rds is not None else None

    def _measure(self, z):
        if len(z):
            p = float(np.mean(z.real ** 2 + z.imag ** 2))
            self.level_db = 10 * np.log10(p + 1e-20)
