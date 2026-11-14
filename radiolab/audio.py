"""Sortie audio (carte son) et enregistrements WAV / CF32 (I/Q brut, format SDR#)."""
import collections
import datetime
import threading
import wave
import numpy as np


def as_stereo(audio):
    """Tableau n x 2 : l'audio mono est recopié sur les deux voies."""
    return audio if audio.ndim == 2 else np.repeat(audio[:, None], 2, axis=1)


class AudioOutput:
    """Joue l'audio 48 kHz (deux voies) via la bibliothèque sounddevice.

    Le traitement produit l'audio par paquets ; la carte son en réclame à son
    propre rythme.  Entre les deux, une file d'attente absorbe les irrégularités.
    Si elle dépasse `max_seconds` (les horloges du récepteur et de la carte son
    ne sont jamais parfaitement égales), on jette le plus ancien.
    """

    def __init__(self, rate=48000, max_seconds=0.6, device=None):
        import sounddevice as sd                      # importé ici : le reste fonctionne sans carte son
        self.rate, self.max_samples = rate, int(rate * max_seconds)
        self.queue = collections.deque()
        self.count = 0
        self.lock = threading.Lock()
        self.underruns = 0
        self.stream = sd.OutputStream(samplerate=rate, channels=2, dtype="float32",
                                      callback=self._callback, device=device, latency="high")
        self.stream.start()

    def write(self, audio):
        if len(audio) == 0:
            return
        audio = as_stereo(audio)
        with self.lock:
            self.queue.append(audio)
            self.count += len(audio)
            while self.count > self.max_samples and len(self.queue) > 1:
                self.count -= len(self.queue.popleft())

    def _callback(self, outdata, frames, time_info, status):
        out = outdata
        filled = 0
        with self.lock:
            while filled < frames and self.queue:
                chunk = self.queue[0]
                n = min(len(chunk), frames - filled)
                out[filled:filled + n] = chunk[:n]
                filled += n
                if n == len(chunk):
                    self.queue.popleft()
                else:
                    self.queue[0] = chunk[n:]
                self.count -= n
        if filled < frames:
            out[filled:] = 0.0
            self.underruns += 1

    def close(self):
        self.stream.stop()
        self.stream.close()


class NullAudio:
    """Remplace la carte son quand il n'y en a pas (essais, ligne de commande)."""
    underruns = 0

    def write(self, audio):
        pass

    def close(self):
        pass


class WavWriter:
    """Écrit un WAV 16 bits à deux voies : audio (gauche, droite) ou I/Q (I à gauche, Q à droite)."""

    def __init__(self, path, rate, iq=False):
        self.path, self.iq = path, iq
        self.w = wave.open(path, "wb")
        self.w.setnchannels(2)
        self.w.setsampwidth(2)
        self.w.setframerate(int(rate))
        self.lock = threading.Lock()
        self.bytes = 0

    def write(self, x):
        data = x.view(np.float32) if self.iq else as_stereo(x).reshape(-1)
        pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        with self.lock:
            if self.w is not None and self.bytes + len(pcm) < 4_000_000_000:   # limite du format WAV
                self.w.writeframes(pcm)
                self.bytes += len(pcm)

    def close(self):
        with self.lock:
            if self.w is not None:
                self.w.close()
                self.w = None


class Cf32Writer:
    """Écrit un fichier .cf32 : I/Q complexe 32 bits little-endian, brut (format SDR#).

    Pas d'en-tête, pas de limite des 4 Go du WAV : on peut enregistrer longtemps.
    """

    def __init__(self, path, rate):
        self.path, self.rate = path, int(rate)
        self.f = open(path, "wb")
        self.lock = threading.Lock()
        self.bytes = 0

    def write(self, x):
        data = np.ascontiguousarray(x, dtype="<c8").tobytes()
        with self.lock:
            if self.f is not None:
                self.f.write(data)
                self.bytes += len(data)

    def close(self):
        with self.lock:
            if self.f is not None:
                self.f.close()
                self.f = None


def recording_name(folder, center_hz, iq, ext="wav"):
    """Nom compatible avec la convention de SDR# (la fréquence est relue à l'ouverture)."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return "%s/RadioLab_PyPhantom_%s_%dHz_%s.%s" % (folder, stamp, int(round(center_hz)), "IQ" if iq else "AF", ext)
