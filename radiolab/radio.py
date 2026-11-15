"""Chef d'orchestre : relie la source I/Q, le récepteur, la carte son et les enregistrements.

L'interface graphique ne parle qu'à cette classe ; on peut donc aussi la piloter
sans écran (essais automatiques, ligne de commande).
"""
import collections
import json
import os
import queue
import time
import threading
import numpy as np
from .dsp.receiver import Receiver, MODES
from .dsp.filters import PolyDDC
from .dsp.spectrum import (power_spectrum_db, zoom_spectrum, auto_nfft, zoom_fft, zoom_plan, zoom_raw_samples,
                           DEFAULT_WINDOW)
from .audio import Cf32Writer, NullAudio, WavWriter, recording_name, as_stereo

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".radiolab.json")
DEFAULTS = {"vfo_hz": 7_100_000, "mode": "LSB", "gain": 12, "gain_mode": "linearity", "spyverter": True,
            "sample_rate": 2_500_000, "volume": 0.5, "dll_path": "", "record_dir": os.path.expanduser("~"),
            "jobs": [], "audio_device": "", "sv_ppm": 0.0, "decimation": 1,
            "iq_format": "wav", "file_rate": 0, "cat_port": "COM3", "cat_baud": 9600,
            "memories": [{"name": "France Inter FM", "hz": 87_800_000, "mode": "WFM"},
                         {"name": "40 m BLU", "hz": 7_100_000, "mode": "LSB"},
                         {"name": "20 m BLU", "hz": 14_200_000, "mode": "USB"}]}


def load_config(path=CONFIG_PATH):
    cfg = json.loads(json.dumps(DEFAULTS))
    try:
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    cfg.setdefault("bias_tee", cfg["spyverter"])      # anciennes configurations : une seule case pour les deux
    return cfg


def save_config(cfg, path=CONFIG_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1, ensure_ascii=False)
    except OSError:
        pass


DECIMATIONS = (1, 2, 4, 8, 16, 32)
MIN_RATE = 48000.0                                        # débit minimal après décimation


def allowed_decimations(source_rate):
    return [d for d in DECIMATIONS if source_rate / d >= MIN_RATE]


class Radio:
    RING_SAMPLES = 1 << 22                                # 1,7 s à 2,5 MS/s, 0,42 s à 10 MS/s : mémoire de la loupe du spectre
    HISTORY_BYTES = 512 * 1024 * 1024                     # mémoire maximale du TimeShift

    def __init__(self, audio_factory=None):
        self.audio_factory = audio_factory or NullAudio
        self.source = None
        self.rx = None
        self.audio = None
        self.decimation = 1               # division du débit du récepteur avant tout le reste (1 = aucune)
        self.rx2 = None                   # second récepteur (VFO B) sur le même flux I/Q
        self.vfo2_hz = 0.0
        self.rx2_in_band = False
        self.split_ears = False           # VFO A à gauche, VFO B à droite (sinon mélangés)
        self._mix_a = np.zeros((0, 2), dtype=np.float32)
        self._mix_b = np.zeros((0, 2), dtype=np.float32)
        self.timeshift = False            # écoute différée : on traite un bloc reçu `delay_s` secondes plus tôt
        self.delay_s = 0.0
        self.history = collections.deque()
        self.history_samples = 0
        self.invert_iq = False            # spectre inversé (certains convertisseurs retournent la bande)
        self.remove_dc = False            # supprime la raie parasite au centre du spectre
        self._dc = 0j
        self.decimator = None
        self.center_hz = 0.0
        self.vfo_hz = 0.0
        self.running = False
        self.blocks = queue.Queue(maxsize=24)
        self.lost_blocks = 0
        self.zoom_div = 0                           # réduction de débit de la loupe du spectre (0 = loupe inactive)
        self.iq_tap = None                          # fonction appelée avec chaque bloc I/Q (ADS-B)
        self.last_block = None
        self.ring = collections.deque()                   # derniers blocs I/Q, pour l'analyse fine du spectre
        self.ring_len = 0
        self.last_audio = np.zeros(0, dtype=np.float32)
        self.audio_rec = self.iq_rec = None
        self._thread = None
        self.error = None
        self.load = 0.0                                   # part du temps passée à traiter le signal (1 = saturation)

    # -------------------------------------------------------------- source
    def set_source(self, source):
        self.stop()
        if self.source is not None:
            self.source.close()
        self.source = source
        if self.decimation not in allowed_decimations(source.sample_rate):
            self.decimation = 1
        # Décimation d'entrée : on ne garde que la bande centrale (filtrée), à un débit divisé par N.
        # La même FFT montre alors une bande N fois plus étroite, donc N fois plus de détail, et le
        # bruit hors bande est éliminé avant toute la suite.
        self.decimator = None
        if self.decimation > 1:
            self.decimator = PolyDDC(source.sample_rate, self.decimation, 0.4 * source.sample_rate / self.decimation)
        old = self.rx
        self.rx = Receiver(self.sample_rate)
        if old is not None:                               # on conserve les réglages de l'auditeur
            for k in ("volume", "squelch_db", "nr_method", "nr_enabled", "anf_enabled",
                      "nr_reduction_db", "agc_enabled", "stereo_enabled", "pocsag_enabled",
                      "raw_output", "notches", "bass_db", "treble_db", "pocsag_messages",
                      "digi_kind", "digi_center", "digi_reverse", "digi_text", "cw_wpm", "cw_threshold_db",
                      "cw_pitch", "am_sync", "anti_fading", "binaural", "filter_window", "filter_order"):
                setattr(self.rx, k, getattr(old, k))
            self.rx.nb.enabled, self.rx.nb.threshold = old.nb.enabled, old.nb.threshold
            try:
                self.rx.set_mode(old.mode)
                self.rx.set_filter(old.lo, old.hi)
            except ValueError:
                self.rx.set_mode("USB")
        self.center_hz = float(source.center_hz)
        self.vfo_hz = self.center_hz
        self.clear_history()
        if self.rx2 is not None:
            mode2 = self.rx2.mode
            self.enable_second(True)
            try:
                self.rx2.set_mode(mode2)
            except ValueError:
                pass
        self.last_block = None
        self.ring.clear()
        self.ring_len = 0
        while not self.blocks.empty():                    # blocs restants de l'ancienne source
            self.blocks.get_nowait()

    @property
    def sample_rate(self):
        """Débit des échantillons traités et affichés (débit du récepteur divisé par la décimation)."""
        if not self.source:
            return 0
        rate = self.source.sample_rate / self.decimation
        return int(rate) if rate == int(rate) else rate

    def set_decimation(self, decimation):
        """Change la décimation (réception arrêtée) et reconstruit la chaîne de traitement."""
        if self.running:
            raise RuntimeError("arrêter la réception avant de changer la décimation")
        self.decimation = int(decimation)
        if self.source is not None:
            src, self.source = self.source, None
            center, vfo = self.center_hz, self.vfo_hz
            self.set_source(src)
            self.center_hz, self.vfo_hz = center, vfo
            self.rx.set_offset(vfo - center)

    def start(self):
        if self.running or self.source is None:
            return
        self.audio = self.audio_factory()
        self.running = True
        self.error = None
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()
        try:
            self.source.start(self._on_block)
        except Exception:
            self.running = False
            self._thread.join(timeout=2)
            self.audio.close()
            self.audio = None
            raise

    def stop(self):
        if not self.running:
            return
        self.source.stop()
        self.running = False
        self._thread.join(timeout=2)
        self.stop_recording(True)
        self.stop_recording(False)
        self.audio.close()
        self.audio = None

    # ------------------------------------------------------ second récepteur
    def enable_second(self, enabled):
        """Active le VFO B : un second récepteur complet, accordable n'importe où dans la bande reçue."""
        if not enabled or self.source is None:
            self.rx2 = None
            return
        rx2 = Receiver(self.sample_rate)
        rx2.volume = self.rx.volume
        self._mix_a = np.zeros((0, 2), dtype=np.float32)
        self._mix_b = np.zeros((0, 2), dtype=np.float32)
        if not self.vfo2_hz:
            self.vfo2_hz = self.vfo_hz
        self.rx2 = rx2
        self.tune2(self.vfo2_hz)

    def tune2(self, hz):
        self.vfo2_hz = float(hz)
        rx2 = self.rx2
        if rx2 is not None:
            offset = self.vfo2_hz - self.center_hz
            self.rx2_in_band = abs(offset) <= 0.48 * self.sample_rate
            if self.rx2_in_band:
                rx2.set_offset(offset)

    def _mix(self, a, b):
        """Réunit l'audio des deux récepteurs (ils ne livrent pas toujours le même nombre d'échantillons)."""
        a = as_stereo(a)
        b = np.zeros_like(a) if b is None else as_stereo(b)
        self._mix_a = np.concatenate([self._mix_a, a])
        self._mix_b = np.concatenate([self._mix_b, b])
        n = min(len(self._mix_a), len(self._mix_b))
        a, self._mix_a = self._mix_a[:n], self._mix_a[n:]
        b, self._mix_b = self._mix_b[:n], self._mix_b[n:]
        if len(self._mix_a) > 48000 or len(self._mix_b) > 48000:      # sécurité : jamais plus d'une seconde d'écart
            self._mix_a, self._mix_b = self._mix_a[:0], self._mix_b[:0]
        if self.split_ears:
            return np.stack([a.mean(axis=1), b.mean(axis=1)], axis=1).astype(np.float32)
        return np.clip(a + b, -1.0, 1.0).astype(np.float32)

    # -------------------------------------------------------------- TimeShift
    def set_timeshift(self, enabled, delay_s=None):
        self.timeshift = bool(enabled)
        if delay_s is not None:
            self.delay_s = max(float(delay_s), 0.0)
        if not enabled:
            self.clear_history()

    def clear_history(self):
        self.history.clear()
        self.history_samples = 0

    def history_seconds(self):
        fs = self.sample_rate
        return self.history_samples / fs if fs else 0.0

    def restart_audio(self):
        """Change de sortie audio sans interrompre la réception."""
        if self.running:
            old, self.audio = self.audio, self.audio_factory()
            old.close()

    def close(self):
        self.stop()
        if self.source is not None:
            self.source.close()
            self.source = None

    # --------------------------------------------------------- flux de données
    def _on_block(self, block):
        """Appelé par la source (autre fil d'exécution) : on dépose le bloc et on rend la main."""
        try:
            self.blocks.put_nowait(block)
        except queue.Full:
            self.lost_blocks += 1                         # l'ordinateur ne suit pas : on saute un bloc

    def _work(self):
        while self.running:
            try:
                block = self.blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            began, received = time.perf_counter(), len(block)
            try:
                if self.invert_iq:
                    block = np.conj(block)
                if self.remove_dc:                        # composante continue : moyenne lente, retranchée
                    self._dc += 0.05 * (complex(block.mean()) - self._dc)
                    block = block - np.complex64(self._dc)
                if self.decimator is not None:
                    block = self.decimator.process(block)
                    if len(block) == 0:
                        continue
                if self.timeshift:
                    self.history.append(block)
                    self.history_samples += len(block)
                    while self.history_samples * 8 > self.HISTORY_BYTES and len(self.history) > 1:
                        self.history_samples -= len(self.history.popleft())
                    if self.delay_s > 0:                  # on traite un bloc plus ancien : spectre et son sont différés
                        back = int(self.delay_s * self.sample_rate / max(len(block), 1))
                        block = self.history[max(len(self.history) - 1 - back, 0)]
                self.last_block = block
                tap = self.iq_tap
                if tap is not None:
                    try:
                        tap(block)
                    except Exception:
                        pass
                self.ring.append(block)
                self.ring_len += len(block)
                while self.ring_len - len(self.ring[0]) >= self.RING_SAMPLES:
                    self.ring_len -= len(self.ring.popleft())
                rec = self.iq_rec
                if rec is not None:
                    rec.write(block)
                audio = self.rx.process(block)
                rx2 = self.rx2
                if rx2 is not None:
                    audio = self._mix(audio, rx2.process(block) if self.rx2_in_band else None)
                self.audio.write(audio)
                mono = audio if audio.ndim == 1 else audio.mean(axis=1)
                self.last_audio = np.concatenate([self.last_audio, mono])[-4096:]
                rec = self.audio_rec
                if rec is not None:
                    rec.write(audio)
            except Exception as e:                        # ne jamais tuer le fil en silence
                self.error = "%s : %s" % (type(e).__name__, e)
            if self.source is not None and received:      # charge : temps de calcul / durée du signal traité
                spent = (time.perf_counter() - began) / (received / float(self.source.sample_rate))
                self.load += 0.1 * (spent - self.load)

    # ------------------------------------------------------------- accord
    def tune(self, vfo_hz):
        """Écoute la fréquence `vfo_hz` ; déplace la fréquence centrale si nécessaire."""
        fs = self.sample_rate
        usable = 0.48 if self.decimation == 1 else 0.40       # la décimation filtre les bords de la bande
        if self.source.kind == "file":
            vfo_hz = float(np.clip(vfo_hz, self.center_hz - usable * fs, self.center_hz + usable * fs))
        elif abs(vfo_hz - self.center_hz) > (usable - 0.06) * fs:
            # on garde la fréquence écoutée à l'écart du centre du spectre, où le
            # récepteur laisse souvent une raie parasite (résidu de composante continue)
            center = vfo_hz - 0.125 * fs
            self.source.set_center(center)
            self.center_hz = float(center)
            self.clear_history()                          # l'ancien enregistrement n'est plus à la bonne fréquence
        self.vfo_hz = float(vfo_hz)
        self.rx.set_offset(self.vfo_hz - self.center_hz)
        if self.rx2 is not None:
            self.tune2(self.vfo2_hz)

    def set_center(self, center_hz):
        if self.source.kind != "file":
            self.source.set_center(center_hz)
            self.center_hz = float(center_hz)
        self.tune(self.vfo_hz)

    def set_mode(self, mode):
        self.rx.set_mode(mode)

    # -------------------------------------------------------------- mesures
    def spectrum(self, nfft=4096):
        """(fréquences en Hz, niveaux en dB) du dernier bloc reçu, ou None."""
        block = self.last_block
        if block is None:
            return None
        db = power_spectrum_db(block, nfft)
        fs = self.sample_rate
        freqs = self.center_hz + (np.arange(nfft) - nfft // 2) * (fs / nfft)
        return freqs, db

    def recent_iq(self, n):
        """Les n derniers échantillons I/Q reçus (moins s'il n'y en a pas encore assez)."""
        blocks, got = [], 0
        for block in reversed(list(self.ring)):
            blocks.append(block)
            got += len(block)
            if got >= n:
                break
        if not blocks:
            return np.zeros(0, dtype=np.complex64)
        data = blocks[0] if len(blocks) == 1 else np.concatenate(blocks[::-1])
        return data[-n:]

    def spectrum_view(self, f_lo, f_hi, npoints=1536, nfft=None, window=DEFAULT_WINDOW):
        """Spectre de la bande affichée [f_lo, f_hi]. nfft=None : finesse adaptée au zoom.

        Retourne (fréquences, niveaux en dB, taille de FFT utilisée), ou None.
        """
        fs = self.sample_rate
        if not fs or self.ring_len < 4096:
            return None
        available = min(self.ring_len, self.RING_SAMPLES)
        self.zoom_div = 0
        if nfft is None:                                      # finesse automatique : la loupe dès que la bande regardée est étroite
            n_raw = zoom_raw_samples(fs, f_hi - f_lo, npoints, available)
            if n_raw:
                view = zoom_fft(self.recent_iq(n_raw), fs, self.center_hz, f_lo, f_hi, npoints, window)
                if view is not None:
                    plan = zoom_plan(fs, f_hi - f_lo)
                    self.zoom_div = plan[0] * plan[1]
                    return view
            nfft = auto_nfft(fs, f_hi - f_lo, npoints, min(available, 1 << 20))
        while nfft > 4096 and nfft > available:
            nfft >>= 1
        iq = self.recent_iq(max(nfft, min(available, 1 << 18)))
        freqs, db = zoom_spectrum(iq, fs, self.center_hz, f_lo, f_hi, npoints, nfft, window)
        return freqs, db, nfft

    def measure_carrier(self, center_hz, span_hz=3000.0):
        """Fréquence exacte de la raie la plus forte à ±span_hz de center_hz : (Hz, dB au-dessus du bruit), ou None."""
        view = self.spectrum_view(center_hz - span_hz, center_hz + span_hz, 3000, None)
        if view is None:
            return None
        freqs, db, _ = view
        k = int(np.argmax(db))
        if k == 0 or k == len(db) - 1:
            return None
        a, b, c = db[k - 1], db[k], db[k + 1]              # sommet d'une parabole passant par les trois points voisins
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        return float(freqs[k] + shift * (freqs[1] - freqs[0])), float(b - np.median(db))

    def audio_spectrum(self, nfft=2048):
        """(fréquences en Hz, niveaux en dB) du son en cours d'écoute, ou None."""
        a = self.last_audio
        nfft = min(nfft, 4096)
        if len(a) < nfft:
            return None
        spec = np.abs(np.fft.rfft(a[-nfft:] * np.hanning(nfft))) / (nfft / 4)
        return np.arange(len(spec)) * (self.rx.audio_rate / nfft), 20 * np.log10(spec + 1e-9)

    def if_spectrum(self, npoints=1024, window=DEFAULT_WINDOW):
        """Spectre autour de la fréquence écoutée : (écarts en Hz par rapport à elle, niveaux en dB, demi-largeur).

        C'est la « fréquence intermédiaire » : on y voit le signal écouté, ses voisins proches
        et la position du filtre.
        """
        rx = self.rx
        shift = -rx.cw_pitch if rx.mode == "CW" else 0.0
        half = max(abs(rx.lo + shift), abs(rx.hi + shift)) * 1.6 + 400.0
        view = self.spectrum_view(self.vfo_hz - half, self.vfo_hz + half, npoints, None, window)
        if view is None:
            return None
        return view[0] - self.vfo_hz, view[1], half

    def passband(self):
        """Bande écoutée (Hz absolus), pour la dessiner sur le spectre."""
        shift = -self.rx.cw_pitch if self.rx.mode == "CW" else 0.0
        return self.vfo_hz + self.rx.lo + shift, self.vfo_hz + self.rx.hi + shift

    # ------------------------------------------------------ enregistrements
    def start_recording(self, folder, iq, fmt=None):
        fmt = (fmt or "wav").lower()
        ext = "cf32" if (iq and fmt == "cf32") else "wav"
        path = recording_name(folder, self.center_hz if iq else self.vfo_hz, iq, ext=ext)
        if iq and ext == "cf32":
            writer = Cf32Writer(path, self.sample_rate)
        else:
            writer = WavWriter(path, self.sample_rate if iq else self.rx.audio_rate, iq=iq)
        if iq:
            self.iq_rec = writer
        else:
            self.audio_rec = writer
        return path

    def stop_recording(self, iq):
        writer = self.iq_rec if iq else self.audio_rec
        if iq:
            self.iq_rec = None
        else:
            self.audio_rec = None
        if writer is not None:
            writer.close()


__all__ = ["Radio", "MODES", "load_config", "save_config"]
