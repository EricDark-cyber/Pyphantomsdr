from .crc24 import crc24, check_crc24
from .modes import ModeSDecoder, decode_mode_s
from .adsb_cpr import CPRTracker
"""Décodeur ADS-B / Mode S (1090 MHz) simple.

Détection de préambule + démodulation PPM + extraction des messages Extended Squitter.
Basé sur les principes de dump1090 / pyModeS.
"""
import numpy as np
import collections


# Préambule Mode S (impulsions à 0, 1.0, 3.5, 4.5 µs)
PREAMBLE = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)


class ADSBDecoder:
    """Décode les trames ADS-B à partir d'échantillons I/Q ou d'amplitude."""

    def __init__(self, fs, emit):
        """fs : fréquence d'échantillonnage (idéalement >= 2 MS/s).
        emit(dict) : callback appelé pour chaque avion décodé.
        """
        self.fs = float(fs)
        self.emit = emit
        self.sps = self.fs / 1e6          # échantillons par µs (1 MHz symbol rate)
        self.history = np.zeros(0, dtype=np.float32)
        self.aircraft = {}                # icao -> dernier message
        self.messages = collections.deque(maxlen=200)
        self.cpr = CPRTracker()
        self._time = 0.0
        self.modes = ModeSDecoder()

    def process_iq(self, iq):
        """Traite un bloc d'échantillons complexes."""
        if len(iq) == 0:
            return
        mag = np.abs(iq).astype(np.float32)
        self._process_magnitude(mag)

    def _process_magnitude(self, mag):
        buf = np.concatenate([self.history, mag])
        # Recherche de préambules
        step = max(int(self.sps * 0.5), 1)
        for i in range(0, len(buf) - int(self.sps * 120), step):
            if self._check_preamble(buf, i):
                msg = self._decode_message(buf, i)
                if msg:
                    self.messages.append(msg)
                    icao = msg.get("icao")
                    if icao:
                        self.aircraft[icao] = msg
                        self.emit(msg)
        # Garder la fin pour le prochain bloc
        keep = int(self.sps * 130)
        self.history = buf[-keep:] if len(buf) > keep else buf

    def _check_preamble(self, buf, idx):
        """Vérifie la présence du préambule Mode S."""
        if idx + int(self.sps * 8) >= len(buf):
            return False
        # Positions des impulsions (en µs)
        positions = [0, 1.0, 3.5, 4.5]
        peaks = []
        for p in positions:
            s = int(idx + p * self.sps)
            e = s + max(int(self.sps * 0.5), 1)
            if e >= len(buf):
                return False
            peaks.append(float(np.max(buf[s:e])))
        # Les 4 pics doivent être nettement au-dessus du bruit local
        noise = float(np.median(buf[idx:idx + int(self.sps * 8)]))
        return all(pk > noise * 3.5 for pk in peaks) and min(peaks) > 0.05

    def _decode_message(self, buf, idx):
        """Décode les bits PPM après le préambule."""
        start = idx + int(self.sps * 8)   # après le préambule
        bits = []
        for b in range(112):              # on tente un long message
            center = start + int((b + 0.5) * self.sps)
            if center + 2 >= len(buf):
                break
            # PPM : impulsion dans la première ou seconde moitié du bit
            half = max(int(self.sps * 0.4), 1)
            p0 = float(np.max(buf[center - half:center]))
            p1 = float(np.max(buf[center:center + half]))
            bits.append(1 if p1 > p0 else 0)
        if len(bits) < 56:
            return None
        # Convertir en hex
        data = 0
        for bit in bits[:112]:
            data = (data << 1) | bit
        df = (data >> (112 - 5)) & 0x1F if len(bits) >= 112 else (data >> (56 - 5)) & 0x1F
        icao = None
        if len(bits) >= 32:
            icao = (data >> (len(bits) - 32)) & 0xFFFFFF
        return {
            "df": int(df),
            "icao": f"{icao:06X}" if icao else None,
            "bits": len(bits),
            "raw": f"{data:X}",
            "time": None,
        }

    def get_aircraft(self):
        return list(self.aircraft.values())


def decode_adsb_payload(msg, cpr_tracker=None, timestamp=0.0):
    """Extrait callsign, position (CPR), altitude d'un message Extended Squitter (DF17/18)."""
    raw = msg.get("raw", "")
    if len(raw) < 14:
        return msg
    try:
        data = int(raw, 16)
    except ValueError:
        return msg
    if msg.get("df") not in (17, 18):
        return msg
    me = (data >> 56) & 0xFFFFFFFFFFFFFF
    tc = (me >> 51) & 0x1F
    msg["tc"] = int(tc)

    if 1 <= tc <= 4:
        # Callsign
        chars = []
        for i in range(8):
            c = (me >> (42 - i * 6)) & 0x3F
            if 1 <= c <= 26:
                chars.append(chr(ord('A') + c - 1))
            elif 48 <= c <= 57:
                chars.append(chr(c))
            elif c == 32:
                chars.append(' ')
        msg["callsign"] = "".join(chars).strip()

    elif 9 <= tc <= 18 or 20 <= tc <= 22:
        # Position
        msg["altitude"] = _decode_altitude(me)
        cpr_lat = (me >> 17) & 0x1FFFF
        cpr_lon = me & 0x1FFFF
        is_odd = bool((me >> 34) & 1)  # F flag? actually the odd/even is in a specific bit
        # Correction : le bit odd/even est le bit 21 du ME pour airborne (simplifié ici)
        is_odd = bool((me >> 21) & 1)
        icao = msg.get("icao")
        if cpr_tracker and icao and cpr_lat and cpr_lon:
            pos = cpr_tracker.update(icao, cpr_lat, cpr_lon, is_odd, timestamp, airborne=True)
            if pos:
                msg["lat"], msg["lon"] = pos
    return msg


def _decode_altitude(me):
    """Altitude barométrique approximative (Q-bit)."""
    alt = (me >> 36) & 0xFFF
    q = (alt >> 4) & 1
    if q:
        n = ((alt & 0xFE0) >> 1) | (alt & 0xF)
        return n * 25 - 1000
    return None
