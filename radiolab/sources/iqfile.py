"""Lecture d'un enregistrement I/Q au format WAV stéréo (I = gauche, Q = droite).

Formats acceptés : PCM 8 ou 16 bits et flottant 32 bits, comme les fichiers
produits par SDR# ou par RadioLab.  La fréquence centrale est lue dans le nom
du fichier quand il contient « _7100000Hz ».
"""
import re
import struct
import threading
import time
import numpy as np


def read_wav_header(f):
    riff, _, wave = struct.unpack("<4sI4s", f.read(12))
    if riff != b"RIFF" or wave != b"WAVE":
        raise ValueError("ce n'est pas un fichier WAV")
    fmt = None
    while True:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError("fichier WAV sans données")
        cid, size = struct.unpack("<4sI", head)
        if cid == b"fmt ":
            data = f.read(size + (size & 1))
            tag, channels, rate, _, _, bits = struct.unpack("<HHIIHH", data[:16])
            if tag == 0xFFFE and len(data) >= 26:                 # WAVE_FORMAT_EXTENSIBLE
                tag = struct.unpack("<H", data[24:26])[0]
            fmt = (tag, channels, rate, bits)
        elif cid == b"data":
            if fmt is None:
                raise ValueError("fichier WAV mal formé")
            return fmt + (f.tell(), size)
        else:
            f.seek(size + (size & 1), 1)


class IQFileSource:
    kind = "file"

    def __init__(self, path, loop=True, realtime=True, block=32768):
        self.path, self.loop, self.realtime, self.block = path, loop, realtime, block
        self.f = open(path, "rb")
        tag, channels, rate, bits, self.data_start, self.data_size = read_wav_header(self.f)
        if channels != 2:
            raise ValueError("un fichier I/Q doit être stéréo (I à gauche, Q à droite)")
        if (tag, bits) == (1, 16):
            self.dtype, self.scale, self.bias = np.dtype("<i2"), 1 / 32768.0, 0.0
        elif (tag, bits) == (1, 8):
            self.dtype, self.scale, self.bias = np.dtype("u1"), 1 / 128.0, -1.0
        elif (tag, bits) == (3, 32):
            self.dtype, self.scale, self.bias = np.dtype("<f4"), 1.0, 0.0
        else:
            raise ValueError("format WAV non géré (%d bits, type %d)" % (bits, tag))
        self.sample_rate = int(rate)
        self.sample_rates = [self.sample_rate]
        m = re.search(r"_(\d{4,11})Hz", path)
        self.center_hz = float(m.group(1)) if m else 0.0
        self.dropped = 0
        self.streaming = False
        self._thread = None
        self.paused = False
        self._lock = threading.Lock()                     # la lecture et les déplacements viennent de deux fils différents
        self.frame_bytes = 2 * self.dtype.itemsize
        self.f.seek(0, 2)
        self.data_size = max(min(self.data_size, self.f.tell() - self.data_start), 0)
        self.f.seek(self.data_start)
        self.frames = self.data_size // self.frame_bytes
        self.duration_s = self.frames / float(self.sample_rate)

    def position(self):
        """Avancement de la lecture, de 0 à 1."""
        with self._lock:
            return min(max((self.f.tell() - self.data_start) / float(max(self.data_size, 1)), 0.0), 1.0)

    def seek(self, fraction):
        """Reprend la lecture à l'endroit voulu (0 = début, 1 = fin)."""
        frame = int(min(max(fraction, 0.0), 1.0) * max(self.frames - 1, 0))
        with self._lock:
            self.f.seek(self.data_start + frame * self.frame_bytes)

    def set_center(self, hz):                # un fichier a une fréquence centrale fixe
        pass

    def read_block(self):
        with self._lock:
            left = self.data_start + self.data_size - self.f.tell()
            raw = self.f.read(max(min(self.block * self.frame_bytes, left), 0))
        raw = raw[:len(raw) - len(raw) % (2 * self.dtype.itemsize)]
        if not raw:
            return None
        x = np.frombuffer(raw, dtype=self.dtype).astype(np.float32) * self.scale + self.bias
        return x.view(np.complex64)

    def rewind(self):
        with self._lock:
            self.f.seek(self.data_start)

    def start(self, on_block):
        self.streaming = True
        self.rewind()

        def run():
            t_next = time.perf_counter()
            while self.streaming:
                if self.paused:                           # pause : on attend sans avancer dans le fichier
                    time.sleep(0.05)
                    t_next = time.perf_counter()
                    continue
                blk = self.read_block()
                if blk is None:
                    if not self.loop:
                        break
                    self.rewind()
                    continue
                on_block(blk)
                if self.realtime:
                    t_next += len(blk) / self.sample_rate
                    delay = t_next - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -0.5:
                        t_next = time.perf_counter()
            self.streaming = False
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self.streaming = False
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def close(self):
        self.stop()
        self.f.close()
