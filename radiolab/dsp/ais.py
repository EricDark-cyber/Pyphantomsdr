"""AIS : les navires annoncent leur identité, leur position, leur route (161,975 et 162,025 MHz).

Le signal : salves de 26,7 ms, 9600 bits/s, modulation de fréquence GMSK (écart de ± 2,4 kHz).
Dans une salve :  24 bits alternés (réglage du récepteur), un drapeau 01111110, le message,
16 bits de contrôle (CRC-16), un drapeau de fin.

Trois règles de transmission, défaites ici dans l'ordre inverse :
  - NRZI : un 0 est émis comme un CHANGEMENT de fréquence, un 1 comme une absence de changement ;
  - bourrage : après cinq 1 de suite, l'émetteur glisse un 0 (pour qu'on ne confonde jamais le
    message avec un drapeau) ; le récepteur le retire ;
  - chaque octet est émis bit de poids faible en premier.

UN MESSAGE N'EST AFFICHÉ QUE SI SON CRC-16 EST JUSTE et si sa longueur est celle de son type :
du bruit ne peut pas passer pour un navire.

Entrée : le signal en sortie du démodulateur FM (avant réduction de débit : il faut au moins
3 échantillons par bit, soit 28,8 kHz ; le récepteur fournit environ 50 kHz).

Interface :
    d = AISDecoder(fs, emit)     # emit(message) : dictionnaire type, mmsi, lat, lon, sog, cog, name...
    d.process(signal)
    d.vessels                    # mmsi -> dernières informations connues
"""
import collections
import re

import numpy as np

BAUD = 9600.0
FLAG = b"01111110"
# longueur du message en bits selon son type (les types absents sont de longueur variable)
LENGTHS = {1: (168,), 2: (168,), 3: (168,), 4: (168,), 5: (424,), 9: (168,), 10: (72,), 11: (168,),
           18: (168,), 19: (312,), 22: (168,), 23: (160,), 24: (160, 168), 27: (96,)}
STATUS = {0: "en route (moteur)", 1: "au mouillage", 2: "non maître de sa manoeuvre", 3: "manoeuvre restreinte",
          5: "à quai", 7: "en pêche", 8: "en route (voile)"}


def crc16_x25(data):
    """CRC-16 des trames HDLC (polynôme 1021 pris à l'envers, départ FFFF, résultat inversé)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


class _Bits:
    """Lecture de champs dans un message, bit de poids fort en premier."""

    def __init__(self, data):
        self.v = int.from_bytes(data, "big")
        self.n = 8 * len(data)

    def u(self, start, length):
        if start + length > self.n:
            return 0
        return (self.v >> (self.n - start - length)) & ((1 << length) - 1)

    def s(self, start, length):
        x = self.u(start, length)
        return x - (1 << length) if x >> (length - 1) else x

    def text(self, start, length):
        out = []
        for i in range(length // 6):
            c = self.u(start + 6 * i, 6)
            out.append(chr(c + 64) if c < 32 else chr(c))
        return "".join(out).split("@")[0].strip()


def _position(b, lon_at, out):
    lon, lat = b.s(lon_at, 28) / 600000.0, b.s(lon_at + 28, 27) / 600000.0
    if abs(lon) <= 180.0 and abs(lat) <= 90.0:               # 181° et 91° veulent dire « inconnu »
        out["lon"], out["lat"] = round(lon, 6), round(lat, 6)


def decode_message(data):
    """Octets d'un message AIS -> dictionnaire, ou None si le type ou la longueur ne conviennent pas."""
    if len(data) < 5:
        return None
    b = _Bits(data)
    kind = b.u(0, 6)
    if not 1 <= kind <= 27:
        return None
    if kind in LENGTHS and b.n not in LENGTHS[kind]:
        return None
    mmsi = b.u(8, 30)
    if not 0 < mmsi <= 999999999:
        return None
    out = {"type": kind, "mmsi": mmsi}
    if kind in (1, 2, 3):                                    # position d'un navire (classe A)
        out["status"] = b.u(38, 4)
        sog, cog, hdg = b.u(50, 10), b.u(116, 12), b.u(128, 9)
        _position(b, 61, out)
        if sog < 1023:
            out["sog"] = sog / 10.0
        if cog < 3600:
            out["cog"] = cog / 10.0
        if hdg < 360:
            out["heading"] = hdg
        out["second"] = b.u(137, 6)
    elif kind == 4:                                          # station de base : date et heure, position
        out["utc"] = "%04d-%02d-%02d %02d:%02d:%02d" % (b.u(38, 14), b.u(52, 4), b.u(56, 5), b.u(61, 5), b.u(66, 6), b.u(72, 6))
        _position(b, 79, out)
    elif kind == 5:                                          # identité et voyage
        out["imo"] = b.u(40, 30)
        out["callsign"] = b.text(70, 42)
        out["name"] = b.text(112, 120)
        out["shiptype"] = b.u(232, 8)
        out["destination"] = b.text(302, 120)
    elif kind in (18, 19):                                   # position d'un navire de plaisance (classe B)
        sog, cog, hdg = b.u(46, 10), b.u(112, 12), b.u(124, 9)
        _position(b, 57, out)
        if sog < 1023:
            out["sog"] = sog / 10.0
        if cog < 3600:
            out["cog"] = cog / 10.0
        if hdg < 360:
            out["heading"] = hdg
        if kind == 19:
            out["name"] = b.text(143, 120)
    elif kind == 21:                                         # balise, bouée, phare
        out["name"] = b.text(43, 120)
        _position(b, 164, out)
    elif kind == 24:                                         # identité, classe B
        if b.u(38, 2) == 0:
            out["name"] = b.text(40, 120)
        else:
            out["shiptype"] = b.u(40, 8)
            out["callsign"] = b.text(90, 42)
    return out


def format_ais(m):
    """Une ligne de texte pour un message AIS."""
    parts = ["AIS %2d  MMSI %09d" % (m["type"], m["mmsi"])]
    if m.get("name"):
        parts.append(m["name"])
    if m.get("callsign"):
        parts.append("indicatif " + m["callsign"])
    if "lat" in m:
        parts.append("%.4f°%s %.4f°%s" % (abs(m["lat"]), "N" if m["lat"] >= 0 else "S", abs(m["lon"]), "E" if m["lon"] >= 0 else "O"))
    if "sog" in m:
        parts.append("%.1f nd" % m["sog"])
    if "cog" in m:
        parts.append("route %.0f°" % m["cog"])
    if m.get("destination"):
        parts.append("vers " + m["destination"])
    if m.get("status") in STATUS:
        parts.append(STATUS[m["status"]])
    if "utc" in m:
        parts.append(m["utc"])
    return "  ".join(parts)


class AISDecoder:
    def __init__(self, fs, emit=None):
        self.fs = float(fs)
        self.emit = emit
        self.spb = self.fs / BAUD                            # échantillons par bit
        self.ok = self.spb >= 3.0
        self.nphase = max(min(int(round(self.spb)), 8), 2)   # on essaie plusieurs instants de lecture du bit
        self.win = max(int(round(14 * self.spb)), 4)         # 14 bits : fenêtre de mesure des deux niveaux
        self.box = max(int(round(0.6 * self.spb)), 1)        # petit lissage, plus court qu'un bit
        self.buf = np.zeros(0, dtype=np.float32)
        self.t0 = 0.0                                        # position (en échantillons du tampon) du prochain bit, phase 0
        self.bits = [b"" for _ in range(self.nphase)]
        self.last = [0] * self.nphase
        self.vessels = {}
        self.messages = collections.deque(maxlen=200)
        self.history = collections.deque(maxlen=400)
        self.count = 0
        self.bitpos = [0] * self.nphase                      # rang, depuis le début, du premier bit gardé (par phase)
        self._recent = collections.deque(maxlen=16)          # (octets, rang du bit) : la même salve vue par plusieurs phases

    def status(self):
        if not self.ok:
            return "AIS : débit trop faible (il faut au moins 28,8 kHz)"
        return "AIS : %d message(s), %d navire(s)" % (self.count, len(self.vessels))

    def get_vessels(self):
        return list(self.vessels.values())

    def process(self, signal):
        if signal is None or len(signal) == 0 or not self.ok:
            return []
        x = np.asarray(signal, dtype=np.float32).ravel()
        self.history.append(20 * np.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-9))
        self.buf = np.concatenate([self.buf, x])
        n = len(self.buf) - self.win - self.box
        if n < 4 * self.spb:
            return []
        # lissage court, puis niveau milieu = (max + min) / 2 des 14 derniers bits
        c = np.concatenate([[0.0], np.cumsum(self.buf, dtype=np.float64)])
        y = ((c[self.box:] - c[:-self.box]) / self.box).astype(np.float32)
        w = np.lib.stride_tricks.sliding_window_view(y, self.win)
        mid = 0.5 * (w.max(axis=1) + w.min(axis=1))          # mid[k] : pour l'échantillon y[k + win - 1]
        level = y[self.win - 1:] > mid
        limit = len(level)
        out = []
        nb = int((limit - 1 - self.t0 - (self.nphase - 1) * self.spb / self.nphase) // self.spb)
        if nb > 0:
            for ph in range(self.nphase):
                idx = np.floor(self.t0 + ph * self.spb / self.nphase + np.arange(nb) * self.spb).astype(np.int64)
                lv = level[idx].astype(np.uint8)
                prev = np.concatenate([[self.last[ph]], lv[:-1]])
                self.last[ph] = int(lv[-1])
                data = (lv == prev).astype(np.uint8) + 48    # NRZI : pas de changement = 1 ; en caractères « 0 » / « 1 »
                self.bits[ph] += data.tobytes()
                out += self._frames(ph)
            self.t0 += nb * self.spb
        drop = int(self.t0) - 1
        if drop > 0:
            self.buf = self.buf[drop:]
            self.t0 -= drop
        return out

    def _frames(self, ph):
        """Cherche dans la suite de bits les trames complètes : drapeau, message, drapeau."""
        s = self.bits[ph]
        out = []
        flags = [m.start() for m in re.finditer(b"(?=01111110)", s)]
        for a, b in zip(flags, flags[1:]):
            body = s[a + 8:b]
            if not 56 <= len(body) <= 1300:
                continue
            body = body.replace(b"111110", b"11111")         # on retire les bits de bourrage
            if len(body) % 8 or len(body) < 56:
                continue
            arr = np.frombuffer(body, dtype=np.uint8) - 48
            data = np.packbits(arr.reshape(-1, 8)[:, ::-1], axis=1).ravel().tobytes()   # poids faible en premier
            if crc16_x25(data[:-2]) != data[-2] | (data[-1] << 8):
                continue
            msg = decode_message(data[:-2])
            at = self.bitpos[ph] + a
            if msg is None or any(d == data and abs(at - t) < 16 for d, t in self._recent):
                continue
            self._recent.append((data, at))
            out.append(self._accept(msg))
        keep = flags[-1] if flags else len(s) - 7            # on repart du dernier drapeau vu
        if len(s) - keep > 1400:
            keep = len(s) - 7
        self.bitpos[ph] += max(keep, 0)
        self.bits[ph] = s[max(keep, 0):]
        return out

    def _accept(self, msg):
        self.count += 1
        self.messages.append(msg)
        v = self.vessels.setdefault(msg["mmsi"], {"mmsi": msg["mmsi"]})
        v.update({k: val for k, val in msg.items() if k != "type" and val not in (None, "")})
        full = dict(msg)
        for key in ("name", "callsign"):                     # on rappelle le nom déjà connu du navire
            if key not in full and v.get(key):
                full[key] = v[key]
        if self.emit is not None:
            self.emit(full)
        return full


def frame_bits(payload):
    """Bits émis pour un message (sert aux essais) : réglage, drapeau, message bourré, contrôle, drapeau."""
    crc = crc16_x25(payload)
    data = bytes(payload) + bytes([crc & 0xFF, crc >> 8])
    raw = [(byte >> k) & 1 for byte in data for k in range(8)]
    stuffed, run = [], 0
    for bit in raw:
        stuffed.append(bit)
        run = run + 1 if bit else 0
        if run == 5:
            stuffed.append(0)
            run = 0
    flag = [0, 1, 1, 1, 1, 1, 1, 0]
    return [0, 1] * 12 + flag + stuffed + flag + [0] * 8


def modulate(payload, fs, deviation=0.48, offset=0.0):
    """Signal en sortie d'un démodulateur FM pour un message (sert aux essais).
    `deviation` : excursion (2,4 kHz rapportés aux 5 kHz de l'échelle du récepteur)."""
    level, levels = 1, []
    for bit in frame_bits(payload):
        if bit == 0:                                         # NRZI : un 0 change la fréquence
            level = -level
        levels.append(level)
    spb = fs / BAUD
    n = int(len(levels) * spb)
    x = np.array(levels, dtype=np.float64)[np.minimum((np.arange(n) / spb).astype(np.int64), len(levels) - 1)]
    sigma = np.sqrt(np.log(2.0)) / (2 * np.pi * 0.4) * spb   # filtre gaussien BT = 0,4
    half = int(4 * sigma) + 1
    g = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    x = np.convolve(np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])]), g / g.sum(), mode="valid")
    return (deviation * x + offset).astype(np.float32)


def payload_from_nmea(text):
    """Charge utile d'une phrase « !AIVDM » (caractères de 6 bits) -> octets du message."""
    bits = ""
    for ch in text:
        c = ord(ch) - 48
        if c > 40:
            c -= 8
        bits += format(c & 0x3F, "06b")
    bits = bits[:len(bits) - len(bits) % 8]
    return int(bits, 2).to_bytes(len(bits) // 8, "big")
