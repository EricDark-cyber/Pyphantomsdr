"""Décodeur CTCSS/DCS pour NFM.

CTCSS : Continuous Tone-Coded Squelch System (67-254 Hz)
DCS : Digital Coded Squelch (codes numériques)
"""
import numpy as np


# Fréquences CTCSS standard (Hz)
CTCSS_TONES = [
    67.0, 69.3, 71.9, 74.4, 77.0, 79.7, 82.5, 85.4, 88.5, 91.5,
    94.8, 97.4, 100.0, 103.5, 107.2, 110.9, 114.8, 118.8, 123.0, 127.3,
    131.8, 136.5, 141.3, 146.2, 151.4, 156.7, 162.2, 167.9, 173.8, 179.9,
    186.2, 192.8, 203.5, 206.5, 210.7, 218.1, 225.7, 229.1, 233.6, 241.8, 250.3, 254.1
]


class CTCSSDecoder:
    """Décodeur CTCSS/DCS."""

    def __init__(self, fs=48000):
        self.fs = fs
        self.detected_tone = None
        self.detected_dcs = None
        self.buf = np.zeros(0, dtype=np.float32)
        self.confidence = 0.0

    def process(self, audio):
        """Analyse l'audio et détecte CTCSS/DCS."""
        if audio is None or len(audio) == 0:
            return None
        x = np.asarray(audio).astype(np.float32).flatten()  # signal brut (pas abs : le redressement doublerait la frequence)
        self.buf = np.concatenate([self.buf, x])
        if len(self.buf) > self.fs:
            self.buf = self.buf[-self.fs:]
        return self._detect()

    def _detect(self):
        """Détecte le ton CTCSS."""
        if len(self.buf) < self.fs // 4:
            return None
        
        # FFT sur la bande sub-audio (50-300 Hz)
        n = min(len(self.buf), int(self.fs))  # 1 s -> resolution 1 Hz, assez pour le seuil de 1.5 Hz
        spec = np.abs(np.fft.rfft(self.buf[-n:] * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1/self.fs)
        
        # Chercher le pic dans la bande CTCSS
        mask = (freqs >= 60) & (freqs <= 260)
        if not np.any(mask):
            return None
        
        peak_idx = np.argmax(spec[mask])
        peak_freq = freqs[mask][peak_idx]
        peak_power = spec[mask][peak_idx]
        
        # Vérifier si c'est un ton CTCSS valide
        for tone in CTCSS_TONES:
            if abs(peak_freq - tone) < 1.5:
                self.detected_tone = tone
                self.confidence = min(1.0, peak_power / 1000)
                return {'type': 'CTCSS', 'tone': tone, 'confidence': self.confidence}
        
        self.detected_tone = None
        return None

    def get_tone(self):
        """Retourne le ton détecté."""
        return self.detected_tone

    def get_status(self):
        """Retourne le statut."""
        if self.detected_tone:
            return f"CTCSS {self.detected_tone:.1f} Hz"
        return "Pas de ton"

# Codes DCS standard
DCS_CODES = [
    23, 25, 26, 31, 32, 36, 43, 47, 51, 53, 54, 65, 71, 72, 73, 74,
    114, 115, 116, 122, 125, 131, 132, 134, 143, 145, 152, 155, 156, 162, 165, 172,
    174, 205, 212, 223, 225, 226, 243, 244, 245, 246, 251, 252, 255, 261, 263, 265,
    266, 271, 274, 306, 311, 315, 325, 331, 332, 343, 346, 351, 356, 364, 365, 371,
    411, 412, 413, 423, 431, 432, 445, 446, 452, 454, 455, 462, 464, 465, 466, 503,
    506, 516, 523, 526, 532, 546, 565, 606, 612, 624, 627, 631, 632, 654, 662, 664,
    703, 712, 723, 731, 732, 734, 743, 754
]


class DCSDecoder:
    """Décodeur DCS (Digital Coded Squelch)."""

    def __init__(self, fs=48000):
        self.fs = fs
        self.detected_code = None
        self.buf = np.zeros(0, dtype=np.float32)

    def process(self, audio):
        """Détecte le code DCS."""
        if audio is None or len(audio) == 0:
            return None
        x = np.asarray(audio).astype(np.float32).flatten()  # signal brut (pas abs : le redressement doublerait la frequence)
        self.buf = np.concatenate([self.buf, x])
        if len(self.buf) > self.fs // 2:
            self.buf = self.buf[-self.fs // 2:]
        return self._detect()

    def _detect(self):
        """Détecte le code DCS par analyse du signal."""
        if len(self.buf) < 1000:
            return None
        # DCS utilise une modulation FSK sub-audio
        # Détection simplifiée par passages par zéro
        zc = np.sum(np.abs(np.diff(np.sign(self.buf[-1000:])))) / 2
        rate = zc * self.fs / 1000
        # Codes DCS typiques autour de 134 Hz
        if 100 < rate < 200:
            # Estimer le code
            code_idx = int((rate - 100) / 2) % len(DCS_CODES)
            self.detected_code = DCS_CODES[code_idx]
            return {'type': 'DCS', 'code': self.detected_code}
        self.detected_code = None
        return None

    def get_code(self):
        return self.detected_code

    def get_status(self):
        if self.detected_code:
            return f"DCS {self.detected_code:03d}"
        return "Pas de DCS"
