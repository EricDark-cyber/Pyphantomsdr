"""ACARS : messages techniques échangés entre les avions et le sol (en Europe 131,725 et 131,525 MHz, en AM).

Le signal, dans l'audio AM : 2400 bits/s, deux notes.  Une note de 2400 Hz veut dire « même bit
que le précédent », une note de 1200 Hz « bit différent du précédent » (modulation MSK).

La trame, en caractères de 7 bits + 1 bit de parité IMPAIRE, bit de poids faible en premier :
    2400 Hz seul (réglage), « + », « * », SYN, SYN, SOH,
    mode (1), immatriculation (7), accusé (1), étiquette (2), numéro de bloc (1),
    STX, texte, ETX (ou ETB), contrôle de bloc sur 16 bits, DEL.

DEUX CONTRÔLES avant d'afficher quoi que ce soit : la parité de chaque caractère, puis le
contrôle de bloc (CRC-16).  Une trame dont le contrôle de bloc ne tombe pas juste est jetée et
seulement comptée (d.unverified) ; avec d.show_unverified = True elle est affichée précédée
d'un « ? », ce qui sert à diagnostiquer une réception faible.

Interface :
    d = ACARSDecoder(fs, emit)   # fs : débit de l'audio AM (12,5 kHz ou plus)
    d.process(audio)             # retourne les nouveaux messages ; emit(message) si fourni
Un message est un dictionnaire : mode, registration, ack, label, block, text, crc_ok.
"""
import collections

import numpy as np

from .filters import StreamFIR, lowpass_taps

BAUD = 2400.0
F_CENTER = 1800.0                                            # entre les deux notes
SYN, SOH, STX, ETX, ETB, DEL = 0x16, 0x01, 0x02, 0x03, 0x17, 0x7F


def _char_bits(c):
    """Caractère de 7 bits -> ses 8 bits émis (poids faible d'abord, puis la parité impaire)."""
    bits = [(c >> k) & 1 for k in range(7)]
    return bits + [1 - sum(bits) % 2]


SYNC = bytes(48 + b for c in (SYN, SYN, SOH) for b in _char_bits(c))
SYNC_INV = bytes(97 - b for b in SYNC)                       # la même suite, bits inversés


def crc_block(data):
    """Contrôle de bloc ACARS : CRC-16 (polynôme 1021 pris à l'envers, départ 0)."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc


def format_acars(m):
    return "%sACARS  %s  étiquette %s  bloc %s  %s" % (
        "" if m.get("crc_ok") else "? ", m.get("registration") or "-------", m.get("label", ""), m.get("block", ""), m.get("text", ""))


class ACARSDecoder:
    def __init__(self, fs=48000, emit=None):
        self.fs = float(fs)
        self.emit = emit
        self.baud = BAUD
        self.spb = self.fs / BAUD
        self.ok = self.spb >= 3.0
        # la porteuse de l'avion n'apparaît qu'avec le message : l'audio AM fait alors une marche,
        # qu'on retire en soustrayant la moyenne sur 1/300 s (les deux notes sont bien au-dessus)
        n = int(self.fs / 300.0) | 1
        hp = -np.ones(n) / n
        hp[n // 2] += 1.0
        self.hp = StreamFIR(hp)
        ntaps = int(8 * self.spb) | 1
        self.lp = StreamFIR(lowpass_taps(1500.0, self.fs, ntaps))
        self.phi = 0.0
        self.prev = np.zeros(1, dtype=np.complex64)
        self.nphase = max(min(int(round(self.spb)), 8), 2)
        self.box = max(int(round(self.spb)), 1)              # moyenne de la fréquence sur un bit
        self.fbuf = np.zeros(0, dtype=np.float32)
        self.t0 = 0.0
        self.tones = [b"" for _ in range(self.nphase)]       # par phase : « 1 » = 2400 Hz (même bit), « 0 » = 1200 Hz
        self.state = [1] * self.nphase
        self.bits = [b"" for _ in range(self.nphase)]
        self.messages = []
        self.frames = collections.deque(maxlen=200)
        self.history = collections.deque(maxlen=400)
        self.count = 0
        self.rejected = 0                                    # trames commencées puis jetées (parité fausse)
        self.unverified = 0                                  # trames complètes dont le contrôle de bloc était faux
        self.show_unverified = False
        self.bitpos = [0] * self.nphase                      # rang, depuis le début, du premier bit gardé (par phase)
        self._recent = collections.deque(maxlen=8)

    def status(self):
        return "ACARS : %d message(s), %d trame(s) au contrôle faux" % (self.count, self.unverified)

    def process(self, audio):
        if audio is None or len(audio) == 0 or not self.ok:
            return []
        x = np.asarray(audio, dtype=np.float32).ravel()
        self.history.append(20 * np.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-9))
        # 1. on ramène 1800 Hz à 0 : les deux notes deviennent -600 Hz et +600 Hz
        w = 2 * np.pi * F_CENTER / self.fs
        lo = np.exp(-1j * (self.phi + w * np.arange(len(x)))).astype(np.complex64)
        self.phi = float((self.phi + w * len(x)) % (2 * np.pi))
        x = self.hp.process(x)
        z = self.lp.process(x * lo)
        if len(z) == 0:
            return []
        # 2. fréquence instantanée (rotation d'un échantillon au suivant), positive = 2400 Hz
        zz = np.concatenate([self.prev, z])
        self.prev = z[-1:]
        f = np.angle(zz[1:] * np.conj(zz[:-1])).astype(np.float32)
        self.fbuf = np.concatenate([self.fbuf, f])
        if len(self.fbuf) < self.box + 4 * self.spb:
            return []
        c = np.concatenate([[0.0], np.cumsum(self.fbuf, dtype=np.float64)])
        y = c[self.box:] - c[:-self.box]                     # somme sur un bit
        out = []
        nb = int((len(y) - 1 - self.t0 - (self.nphase - 1) * self.spb / self.nphase) // self.spb)
        if nb > 0:
            for ph in range(self.nphase):
                idx = np.floor(self.t0 + ph * self.spb / self.nphase + np.arange(nb) * self.spb).astype(np.int64)
                same = y[idx] > 0
                # 3. chaque bit vaut le précédent (2400 Hz) ou son contraire (1200 Hz)
                flips = np.cumsum(~same) & 1
                data = (self.state[ph] ^ flips).astype(np.uint8)
                self.state[ph] = int(data[-1])
                self.bits[ph] += (data + 48).tobytes()
                out += self._frames(ph)
            self.t0 += nb * self.spb
        drop = int(self.t0) - 1
        if drop > 0:
            self.fbuf = self.fbuf[drop:]
            self.t0 -= drop
        return out

    def _frames(self, ph):
        out = []
        s = self.bits[ph]
        n0 = len(s)
        while True:
            a, b = s.find(SYNC), s.find(SYNC_INV)
            if a < 0 and b < 0:
                s = s[-(len(SYNC) - 1):]
                break
            inverted = a < 0 or (0 <= b < a)
            start = (b if inverted else a) + len(SYNC)
            got = self._frame(s[start:], inverted, self.bitpos[ph] + n0 - len(s) + start)
            if got is None:                                  # trame incomplète : on attend la suite
                if len(s) - start > 8 * 260:
                    s = s[start:]
                    continue
                s = s[(b if inverted else a):]
                break
            used, msg = got
            s = s[start + used:]
            if msg is not None:
                out.append(msg)
        self.bitpos[ph] += n0 - len(s)
        self.bits[ph] = s
        return out

    def _frame(self, s, inverted, at=0):
        """Lit une trame à partir du caractère qui suit SOH.  Retourne None s'il manque des bits,
        sinon (bits consommés, message ou None)."""
        raw = []
        for i in range(0, min(len(s), 8 * 250) - 7, 8):
            v = 0
            for k in range(8):
                v |= ((s[i + k] - 48) ^ inverted) << k
            if not bin(v).count("1") & 1:                    # parité impaire fausse : ce n'est pas une trame
                self.rejected += 1
                return 8, None
            raw.append(v)
            if v & 0x7F in (ETX, ETB) and len(raw) >= 13:
                if len(s) < i + 8 + 16:
                    return None
                bcs = 0
                for k in range(16):
                    bcs |= ((s[i + 8 + k] - 48) ^ inverted) << k
                return i + 24, self._message(raw, bcs, at)
        if len(s) >= 8 * 250:
            self.rejected += 1
            return 8, None
        return None

    def _message(self, raw, bcs, at=0):
        chars = [chr(v & 0x7F) for v in raw]
        body = chars[12:-1]
        if body and body[0] == chr(STX):
            body = body[1:]
        text = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in body).strip()
        msg = {"mode": chars[0], "registration": "".join(chars[1:8]).strip(". "),
               "ack": "NAK" if raw[8] & 0x7F == 0x15 else chars[8], "label": "".join(chars[9:11]), "block": chars[11],
               "text": text, "crc_ok": crc_block(bytes(raw)) == bcs}
        key = (msg["registration"], msg["label"], msg["block"], text)
        if any(k == key and abs(at - t) < 16 for k, t in self._recent):
            return None                                      # la même trame, vue par une autre phase de lecture
        self._recent.append((key, at))
        if not msg["crc_ok"]:
            self.unverified += 1
            if not self.show_unverified:
                return None
        self.count += 1
        self.frames.append(msg)
        self.messages.append(text)
        del self.messages[:-100]
        if self.emit is not None:
            self.emit(msg)
        return msg


def encode(registration, label, block, text, mode="2", ack="\x15"):
    """Bits émis pour un message (sert aux essais)."""
    head = mode + (registration.rjust(7, "."))[:7] + ack + label[:2] + block[:1]
    body = [ord(c) for c in head] + [STX] + [ord(c) & 0x7F for c in text] + [ETX]
    with_parity = [sum(b << k for k, b in enumerate(_char_bits(c))) for c in body]
    bcs = crc_block(bytes(with_parity))
    bits = [1] * 128                                         # réglage : 2400 Hz
    for c in (ord("+"), ord("*"), SYN, SYN, SOH):
        bits += _char_bits(c)
    for v in with_parity:
        bits += [(v >> k) & 1 for k in range(8)]
    bits += [(bcs >> k) & 1 for k in range(16)]
    bits += _char_bits(DEL) + [1] * 16
    return bits


def modulate(bits, fs):
    """Audio AM démodulé pour une suite de bits (sert aux essais) : 2400 Hz = même bit, 1200 Hz = bit différent."""
    prev, freqs = 1, []
    for b in bits:
        freqs.append(2400.0 if b == prev else 1200.0)
        prev = b
    spb = fs / BAUD
    n = int(len(freqs) * spb)
    f = np.array(freqs)[np.minimum((np.arange(n) / spb).astype(np.int64), len(freqs) - 1)]
    return np.sin(2 * np.pi * np.cumsum(f) / fs).astype(np.float32)
