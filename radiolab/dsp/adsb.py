"""ADS-B / Mode S (1090 MHz) : les avions annoncent leur identité, leur position et leur vitesse.

Le signal : des impulsions de 0,5 µs.  Un préambule de 8 µs (impulsions à 0 ; 1 ; 3,5 et 4,5 µs),
puis 56 ou 112 bits d'une microseconde chacun : impulsion dans la première moitié = 1, dans la
seconde moitié = 0 (modulation par position d'impulsion).

Les 24 derniers bits sont un contrôle (CRC, polynôme FFF409).  C'EST LUI QUI GARANTIT LE MESSAGE :
  - formats 17 et 18 (ADS-B) et 11 : le reste de la division doit être nul ;
  - formats 0, 4, 5, 16, 20, 21 (réponses à un radar) : le contrôle est mélangé à l'adresse de
    l'avion ; on n'accepte le message que si l'adresse obtenue est celle d'un avion déjà entendu
    en ADS-B dans la minute.
Tout le reste est jeté : du bruit ne peut plus passer pour un avion.

Interface :
    d = ADSBDecoder(fs, emit)        # fs : 2 MHz ou plus (2,5 ou 10 MHz avec l'Airspy)
    d.process_iq(iq)                 # emit(message) pour chaque message valide, dans l'appel
    d.start() ; d.feed(iq) ; d.take()    # variante : calcul dans un fil à part, messages lus par take()
    d.aircraft                       # adresse -> dernières informations connues de l'avion
Un message est un dictionnaire : df, icao, raw, bits, et selon le cas callsign, altitude (pieds),
lat, lon, speed (noeuds), heading (degrés), vrate (pieds/min), squawk.
"""
import collections
import math
import queue
import threading
import time

import numpy as np

from .adsb_cpr import CPRTracker
from .crc24 import crc24_bytes
from .modes import DF_NAMES, ModeSDecoder, decode_ac13, decode_id13, decode_mode_s  # noqa: F401  (réexportés)

CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"
HIGH = (0, 2, 7, 9)                      # demi-microsecondes du préambule où il y a une impulsion
NEAR = (1, 3, 6, 8)                      # creux voisins d'une impulsion : seulement plus faibles qu'elle
FAR = (4, 5, 11, 12, 13)                 # creux francs : nettement plus faibles
KNOWN_S = 60.0                           # durée pendant laquelle une adresse entendue reste « connue »


class ADSBDecoder:
    def __init__(self, fs, emit=None):
        self.fs = float(fs)
        self.emit = emit
        self.U = int(math.ceil(8e6 / self.fs)) if self.fs < 4e6 else 1      # débit trop faible : on interpole
        self.D = max(int(round(self.fs * self.U / 2e6)), 1)  # échantillons par demi-microseconde
        self.phases = tuple(range(0, self.D, 1 if self.U > 1 else 2))       # plusieurs calages possibles des impulsions
        self.spc = self.fs * self.U / self.D / 2e6           # demi-microseconde, en échantillons réduits (environ 1)
        k = np.arange(16 + 224)
        self.pos = np.round(k * self.spc).astype(np.int64)
        self.span = int(self.pos[-1]) + 2                    # longueur d'un message long, en échantillons réduits
        self.tail = np.zeros(0, dtype=np.float32)
        self.cpr = CPRTracker()
        self.modes = ModeSDecoder()
        self.aircraft = {}
        self.known = {}                                      # adresse -> dernier instant où elle a été entendue
        self.messages = collections.deque(maxlen=200)
        self.count = 0                                       # messages valides
        self.rejected = 0                                    # préambules vus dont le contrôle était faux
        self._base = 0                                       # rang absolu du premier échantillon gardé
        self._done = -1                                      # rang absolu de la fin du dernier message valide
        self._time = 0.0
        self._jobs = queue.Queue(maxsize=8)
        self._out = collections.deque(maxlen=2000)
        self._thread = None
        self._run = False

    # ------------------------------------------------------------------ fil d'exécution (facultatif)
    def start(self):
        """Calcul dans un fil à part : feed() dépose les blocs, take() rend les messages."""
        if self._thread is None or not self._thread.is_alive():
            self._run = True
            self._thread = threading.Thread(target=self._worker, name="adsb", daemon=True)
            self._thread.start()

    def stop(self):
        self._run = False

    def feed(self, iq):
        """Dépose un bloc I/Q sans attendre ; s'il y a trop de retard, le bloc est abandonné."""
        try:
            self._jobs.put_nowait((np.asarray(iq), time.monotonic()))
        except queue.Full:
            pass

    def _worker(self):
        while self._run:
            try:
                iq, t = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._out.extend(self._process(np.abs(iq).astype(np.float32), t))
            except Exception:
                pass

    def take(self):
        """Messages décodés depuis le dernier appel."""
        out = []
        while self._out:
            out.append(self._out.popleft())
        return out

    def status(self):
        return "ADS-B : %d message(s) valides, %d rejeté(s), %d avion(s)" % (self.count, self.rejected, len(self.aircraft))

    # ------------------------------------------------------------------ traitement
    def process_iq(self, iq, t=None):
        """Traite un bloc d'échantillons complexes ; retourne les messages valides du bloc."""
        if iq is None or len(iq) == 0:
            return []
        msgs = self._process(np.abs(np.asarray(iq)).astype(np.float32), time.monotonic() if t is None else t)
        if self.emit is not None:
            for m in msgs:
                self.emit(m)
        return msgs

    def _process(self, mag, t):
        self._time = t
        buf = np.concatenate([self.tail, mag])
        need = int(math.ceil((self.span + 2) * self.D / self.U))             # à garder pour le bloc suivant
        if len(buf) <= need:
            self.tail = buf
            return []
        out = []
        x = buf
        if self.U > 1:                                       # interpolation linéaire entre les échantillons
            x = np.interp(np.arange(len(buf) * self.U) / self.U, np.arange(len(buf)), buf).astype(np.float32)
        if self.D > 1 and self.U == 1:                       # somme glissante sur une demi-microseconde
            c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
            smooth = (c[self.D:] - c[:-self.D]).astype(np.float32)
        else:
            smooth = x
        found = []
        for ph in self.phases:
            m = smooth[ph::self.D]
            for i in self._preambles(m):
                at = self._base + int(i) * self.D + ph           # rang absolu du début du préambule
                if at <= self._done or any(a <= at < b for a, b, _ in found):
                    continue                                     # déjà décodé (bloc précédent ou autre phase)
                msg = self._message(m, int(i))
                if msg is not None:
                    found.append((at - 2 * self.D, at + (int(self.pos[16 + 2 * msg["bits"] - 1]) + 1) * self.D, msg))
        for a, b, msg in sorted(found, key=lambda f: f[0]):
            self._done = max(self._done, b - 4 * self.D)
            out.append(self._finish(msg))
        self._base += (len(buf) - need) * self.U
        self.tail = buf[-need:]
        return out

    def _preambles(self, m):
        """Positions où l'on voit les quatre impulsions du préambule, et rien entre elles."""
        n = len(m) - self.span
        if n <= 0:
            return []
        p = self.pos
        hi = m[p[HIGH[0]]:p[HIGH[0]] + n].copy()
        for k in HIGH[1:]:
            np.minimum(hi, m[p[k]:p[k] + n], out=hi)
        near = m[p[NEAR[0]]:p[NEAR[0]] + n].copy()
        for k in NEAR[1:]:
            np.maximum(near, m[p[k]:p[k] + n], out=near)
        far = m[p[FAR[0]]:p[FAR[0]] + n].copy()
        for k in FAR[1:]:
            np.maximum(far, m[p[k]:p[k] + n], out=far)
        floor = float(np.median(m[::16])) + 1e-12
        ok = (hi > 1.5 * far) & (hi > 2.5 * floor)
        if self.U == 1:                                      # à faible débit les impulsions bavent sur leurs voisines
            ok &= hi > near
        return np.flatnonzero(ok)

    def _message(self, m, i):
        first = m[i + self.pos[16::2]]
        second = m[i + self.pos[17::2]]
        bits = (first > second).astype(np.uint8)
        df = int(bits[0]) << 4 | int(bits[1]) << 3 | int(bits[2]) << 2 | int(bits[3]) << 1 | int(bits[4])
        nbits = 112 if df >= 16 else 56
        data = np.packbits(bits[:nbits]).tobytes()
        rem = crc24_bytes(data)
        if df in (11, 17, 18):
            if rem != 0:
                self.rejected += 1
                return None
            icao = int.from_bytes(data[1:4], "big")
            self.known[icao] = self._time
        elif df in (0, 4, 5, 16, 20, 21):                    # le contrôle est mélangé à l'adresse de l'avion
            icao = crc24_bytes(data[:-3]) ^ int.from_bytes(data[-3:], "big")
            seen = self.known.get(icao)
            if seen is None or self._time - seen > KNOWN_S:
                self.rejected += 1
                return None
        else:
            self.rejected += 1
            return None
        return {"df": df, "icao": "%06X" % icao, "bits": nbits, "raw": data.hex().upper(), "crc_ok": True, "time": self._time}

    def _finish(self, msg):
        decode_mode_s(msg)
        decode_adsb_payload(msg, self.cpr, self._time)
        self.count += 1
        self.messages.append(msg)
        self.modes.messages.append(msg)
        self.modes.by_df[msg["df"]] += 1
        ac = self.aircraft.setdefault(msg["icao"], {"icao": msg["icao"]})
        for key in ("callsign", "altitude", "lat", "lon", "speed", "heading", "vrate", "squawk"):
            if msg.get(key) not in (None, ""):
                ac[key] = msg[key]
        ac["seen"] = self._time
        if len(self.known) > 2000:
            limit = self._time - KNOWN_S
            self.known = {k: v for k, v in self.known.items() if v >= limit}
        return msg

    def get_aircraft(self):
        return list(self.aircraft.values())


def decode_adsb_payload(msg, cpr_tracker=None, timestamp=0.0):
    """Complète un message ADS-B (formats 17 et 18) : indicatif, altitude, position, vitesse."""
    raw = msg.get("raw", "")
    if msg.get("df") not in (17, 18) or len(raw) < 28:
        return msg
    try:
        data = int(raw, 16)
    except ValueError:
        return msg
    me = (data >> 24) & 0xFFFFFFFFFFFFFF                      # les 56 bits utiles, entre l'adresse et le contrôle
    tc = (me >> 51) & 0x1F
    msg["tc"] = int(tc)
    if 1 <= tc <= 4:                                          # identification : 8 caractères de 6 bits
        chars = [CHARSET[(me >> (42 - 6 * i)) & 0x3F] for i in range(8)]
        msg["callsign"] = "".join(chars).replace("#", "").strip()
    elif 9 <= tc <= 18 or 20 <= tc <= 22:                     # position en vol
        if tc <= 18:
            msg["altitude"] = _decode_altitude(me)
        cpr_lat = (me >> 17) & 0x1FFFF
        cpr_lon = me & 0x1FFFF
        is_odd = bool((me >> 34) & 1)                         # bit F : trame paire (0) ou impaire (1)
        icao = msg.get("icao")
        if cpr_tracker is not None and icao:
            pos = cpr_tracker.update(icao, cpr_lat, cpr_lon, is_odd, timestamp, airborne=True)
            if pos:
                msg["lat"], msg["lon"] = pos
    elif tc == 19 and ((me >> 48) & 7) in (1, 2):             # vitesse par rapport au sol
        v_ew, v_ns = (me >> 32) & 0x3FF, (me >> 21) & 0x3FF
        if v_ew and v_ns:
            scale = 4 if ((me >> 48) & 7) == 2 else 1         # sous-type 2 : avions supersoniques
            vx = (v_ew - 1) * scale * (-1 if (me >> 42) & 1 else 1)        # vers l'est
            vy = (v_ns - 1) * scale * (-1 if (me >> 31) & 1 else 1)        # vers le nord
            msg["speed"] = round(math.hypot(vx, vy), 1)
            msg["heading"] = round(math.degrees(math.atan2(vx, vy)) % 360.0, 1)
        vr = (me >> 10) & 0x1FF
        if vr:
            msg["vrate"] = (vr - 1) * 64 * (-1 if (me >> 19) & 1 else 1)
    return msg


def _decode_altitude(me):
    """Altitude barométrique en pieds (pas de 25 pieds), ou None si le codage est l'ancien (pas de 100 pieds)."""
    alt = (me >> 36) & 0xFFF
    if alt == 0 or not (alt >> 4) & 1:
        return None
    n = ((alt & 0xFE0) >> 1) | (alt & 0xF)
    return n * 25 - 1000


def modulate(raw_hex, fs, amplitude=1.0, lead_us=20.0, tail_us=20.0):
    """Enveloppe (amplitude) d'un message Mode S au débit fs : sert aux essais."""
    bits = [int(b) for b in bin(int(raw_hex, 16))[2:].zfill(4 * len(raw_hex))]
    chips = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    for b in bits:
        chips += [1, 0] if b else [0, 1]
    n = int(round((lead_us + len(chips) * 0.5 + tail_us) * 1e-6 * fs))
    t_us = np.arange(n) / fs * 1e6 - lead_us
    k = np.floor(t_us / 0.5).astype(np.int64)
    inside = (k >= 0) & (k < len(chips))
    return (amplitude * np.where(inside, np.array(chips)[np.clip(k, 0, len(chips) - 1)], 0)).astype(np.float32)
