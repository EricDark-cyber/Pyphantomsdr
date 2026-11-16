"""Décodage vocal TETRA pour RadioLab.

Utilise OpenACELP (codec libre compatible ETSI EN 300 395-2)
si disponible, sinon tente un décodage simplifié.
"""
import numpy as np
import subprocess
import os
import tempfile


class TetraVoiceDecoder:
    """Décodeur vocal TETRA (ACELP)."""

    def __init__(self, fs=8000):
        self.fs = fs
        self.enabled = False
        self.decoder_path = None
        self.last_error = None
        self._find_decoder()
        self.audio_buf = []
        self.frame_bits = 137  # bits par trame ACELP
        self.frame_samples = 240  # échantillons par trame (30 ms @ 8 kHz)

    def _find_decoder(self):
        """Cherche un décodeur ACELP disponible."""
        candidates = [
            "sdecoder",
            "sdecoder.exe",
            os.path.expanduser("~/tetra/bin/sdecoder"),
            "/usr/local/bin/sdecoder",
            "/tetra/bin/sdecoder",
        ]
        # WSL path
        wsl_path = "wsl /tetra/bin/sdecoder"
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                self.decoder_path = c
                return
        self.decoder_path = None

    @property
    def available(self):
        return self.decoder_path is not None

    def set_enabled(self, on):
        self.enabled = bool(on)

    def decode_frame(self, bits):
        """Décode une trame de 137 bits en audio.

        bits : liste ou array de 137 bits (0/1)
        Retourne un array de 240 échantillons float32, ou None.
        """
        if not self.enabled or len(bits) < self.frame_bits:
            return None

        if self.decoder_path:
            return self._decode_external(bits[:self.frame_bits])
        else:
            return self._decode_simple(bits[:self.frame_bits])

    def _decode_external(self, bits):
        """Utilise le décodeur externe (sdecoder)."""
        try:
            bit_bytes = bytes([int(b) for b in bits])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bits") as f:
                f.write(bit_bytes)
                bit_path = f.name
            out_path = bit_path + ".raw"
            # Appeler via WSL si le chemin commence par /tetra
            if self.decoder_path and self.decoder_path.startswith("/tetra"):
                cmd = ["wsl", self.decoder_path, bit_path, out_path]
            else:
                cmd = [self.decoder_path, bit_path, out_path]
            subprocess.run(cmd, capture_output=True, timeout=2)
            if os.path.exists(out_path):
                audio = np.fromfile(out_path, dtype=np.int16)
                os.unlink(out_path)
                os.unlink(bit_path)
                return audio.astype(np.float32) / 32768.0
            os.unlink(bit_path)
        except Exception:
            pass
        return None

    def _decode_simple(self, bits):
        """Pas de codec réel embarqué : on ne fabrique aucun faux audio."""
        self.last_error = "codec ACELP non disponible (sdecoder/OpenACELP requis)"
        return None

    def decode(self, bits):
        """Alias pour compatibilité."""
        return self.process_bits(bits)

    def process_bits(self, bit_stream):
        """Traite un flux de bits et retourne l'audio décodé."""
        if not self.enabled:
            return np.zeros(0, dtype=np.float32)
        audio_chunks = []
        for i in range(0, len(bit_stream) - self.frame_bits + 1, self.frame_bits):
            frame = bit_stream[i:i + self.frame_bits]
            audio = self.decode_frame(frame)
            if audio is not None:
                audio_chunks.append(audio)
        if audio_chunks:
            return np.concatenate(audio_chunks)
        return np.zeros(0, dtype=np.float32)
