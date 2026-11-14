from .crc16 import crc16, check_crc16
try:
    from .gmsk import GMSKDemodulator
    HAS_GMSK = True
except ImportError:
    GMSKDemodulator = None
    HAS_GMSK = False  # fichier gmsk.py non fourni : repli sur demod par transitions
from .ais_pos import decode_ais_position
"""Décodeur AIS (Automatic Identification System) pour le trafic maritime.

Fréquences : 161.975 MHz (canal A) et 162.025 MHz (canal B).
Modulation : GMSK 9600 bauds.
"""
import collections
import numpy as np


class _TransitionDemod:
    """Repli minimal si gmsk.py est absent : demod par signe des transitions."""

    def __init__(self, fs, baud):
        self.sps = fs / float(baud)

    def process(self, x):
        if len(x) < 2:
            return np.zeros(0, dtype=bool)
        return (np.diff(x) > 0)


class AISDecoder:
    """Décodeur AIS basique (détection + extraction de position)."""

    def __init__(self, fs, emit):
        self.fs = float(fs)
        self.emit = emit
        self.vessels = {}
        self.messages = collections.deque(maxlen=200)
        self.buf = np.zeros(0, dtype=np.float32)
        self.baud = 9600.0
        self.sps = self.fs / self.baud
        if HAS_GMSK:
            self.demod = GMSKDemodulator(self.fs, self.baud)
        else:
            self.demod = _TransitionDemod(self.fs, self.baud)

    def process(self, audio):
        if len(audio) == 0:
            return
        self.buf = np.concatenate([self.buf, audio.astype(np.float32)])
        min_len = int(self.sps * 30)
        while len(self.buf) > min_len * 2:
            bits = self.demod.process(self.buf[:int(self.demod.sps * 200)])
            pos = decode_ais_position(bits)
            msg = None
            if pos:
                lat, lon, mmsi = pos
                msg = {"mmsi": mmsi, "lat": lat, "lon": lon, "type": 1}
            elif len(bits) > 50:
                msg = self._parse(bits)
            if msg:
                self.messages.append(msg)
                mmsi = msg.get("mmsi")
                if mmsi:
                    self.vessels[mmsi] = msg
                self.emit(msg)
            self.buf = self.buf[min_len:]

    def _demod(self, x):
        """Démodulation GMSK simplifiée."""
        if len(x) < 20:
            return []
        # Détection de transitions
        diff = np.diff(x)
        bits = (diff > 0).astype(int).tolist()
        return bits

    def _parse(self, bits):
        """Parse basique d'un message AIS (type 1/2/3 = position)."""
        if len(bits) < 168:  # message minimum
            return None
        # Extraction très simplifiée
        # En réalité il faut NRZI + stuffing + CRC
        try:
            # MMSI est dans les bits 8-37 environ
            mmsi_bits = bits[8:38]
            mmsi = 0
            for b in mmsi_bits:
                mmsi = (mmsi << 1) | b
            if mmsi < 100000 or mmsi > 999999999:
                return None
            # Position approximative (bits suivants)
            return {
                "mmsi": mmsi,
                "type": 1,
                "lat": None,
                "lon": None,
                "name": None,
                "raw_bits": len(bits),
            }
        except Exception:
            return None

    def get_vessels(self):
        return list(self.vessels.values())
