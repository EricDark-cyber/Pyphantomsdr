"""Images reçues par radio : fax météo HF et télévision à balayage lent (SSTV).

Les deux procédés transmettent la luminosité par la FRÉQUENCE d'une note audio :
1500 Hz pour le noir, 2300 Hz pour le blanc.  Le décodage consiste donc à mesurer
cette fréquence instant par instant, puis à ranger les mesures ligne par ligne.
"""
import collections
import struct
import zlib
import numpy as np
from .filters import StreamFIR, lowpass_taps, taps_for_transition

BLACK_HZ, WHITE_HZ = 1500.0, 2300.0


def write_png(path, image):
    """Enregistre un tableau uint8 (H, L) en gris ou (H, L, 3) en couleurs au format PNG, sans bibliothèque externe."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim == 2:
        color = 0
    elif image.ndim == 3 and image.shape[2] == 3:
        color = 2
    else:
        raise ValueError("image attendue : (hauteur, largeur) ou (hauteur, largeur, 3)")
    height, width = image.shape[:2]
    rows = image.reshape(height, -1)
    raw = np.concatenate([np.zeros((height, 1), dtype=np.uint8), rows], axis=1).tobytes()   # filtre 0 devant chaque ligne

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


class FrequencyTrack:
    """Fréquence instantanée d'un signal audio autour de `center` (discriminateur)."""

    def __init__(self, fs, center, half_width):
        self.fs, self.center = float(fs), float(center)
        self.fir = StreamFIR(lowpass_taps(half_width, fs, taps_for_transition(fs, 0.5 * half_width)))
        self.phi = 0.0
        self.prev = 0j

    def process(self, audio):
        """Retourne (fréquences en Hz, amplitudes), un point par échantillon reçu."""
        n = len(audio)
        w = 2 * np.pi * self.center / self.fs
        lo = np.exp(-1j * (self.phi + w * np.arange(n))).astype(np.complex64)
        self.phi = float((self.phi + w * n) % (2 * np.pi))
        z = self.fir.process(audio.astype(np.float32, copy=False) * lo).astype(np.complex128)
        if len(z) == 0:
            return np.zeros(0), np.zeros(0)
        d = z * np.conj(np.concatenate([[self.prev], z[:-1]]))
        self.prev = z[-1]
        return self.center + np.angle(d) * self.fs / (2 * np.pi), np.abs(z)


def _tone_strength(v, fs, hz):
    """Part de la variation de v qui bat à la fréquence hz (0 = aucune, ~1 = créneau parfait)."""
    x = v - v.mean()
    p = float(np.sum(x * x)) + 1e-12
    c = np.sum(x * np.exp(-2j * np.pi * hz * np.arange(len(x)) / fs))
    return float(2 * abs(c) ** 2 / (len(x) * p))


class FaxDecoder:
    """Fax météo HF : 120 lignes par minute, module (IOC) 576, soit 1810 points par ligne.

    Une émission complète comporte une tonalité de départ (noir/blanc alternés 300 fois par
    seconde), des lignes de mise en phase (une impulsion étroite marque le bord de l'image),
    l'image, puis une tonalité d'arrêt (450 fois par seconde).  Le décodeur trace les lignes
    en continu, même pris en cours de route, et se recale quand il reconnaît ces repères.
    """
    WIDTH = 1810
    START_HZ, STOP_HZ = 300.0, 450.0

    def __init__(self, fs, emit, lpm=120.0):
        self.fs, self.emit = float(fs), emit
        self.track = FrequencyTrack(fs, 1900.0, 700.0)
        self.period = self.fs * 60.0 / lpm                  # échantillons par ligne
        self.buf = np.zeros(0)
        self.pos = 0.0                                      # début de la prochaine ligne, dans buf
        self.rows = collections.deque()                     # lignes prêtes (uint8), prises par l'interface
        self.image_id = 0                                   # augmente à chaque nouvelle image
        self.finished = 0                                   # augmente à chaque image terminée (tonalité d'arrêt)
        self.lines = 0
        self.phased = False
        self._phasing = []                                  # positions de l'impulsion sur les dernières lignes
        self._tone = [0, 0]                                 # lignes consécutives de départ / d'arrêt
        self.state = "attente"
        self.history = collections.deque(maxlen=1500)
        self.thr_db = None

    def status(self):
        return "Fax : %s, %d lignes%s" % (self.state, self.lines, "" if self.phased else " (cadrage non calé)")

    def take_rows(self):
        out = []
        while self.rows:
            out.append(self.rows.popleft())
        return out

    def process(self, audio):
        f, amp = self.track.process(audio)
        if len(f) == 0:
            return
        self.history.extend(20 * np.log10(amp[::25] + 1e-12))
        self.buf = np.concatenate([self.buf, np.clip((f - BLACK_HZ) / (WHITE_HZ - BLACK_HZ), 0.0, 1.0)])
        while self.pos + self.period + 1 <= len(self.buf):
            start = int(self.pos)
            line = self.buf[start:start + int(self.period) + 1]
            self._line(line)
            self.pos += self.period
        drop = min(int(self.pos), len(self.buf))            # après un recalage, pos peut dépasser ce qui est déjà reçu
        if drop > 0:
            self.buf = self.buf[drop:]
            self.pos -= drop

    def _line(self, v):
        start, stop = _tone_strength(v, self.fs, self.START_HZ), _tone_strength(v, self.fs, self.STOP_HZ)
        self._tone[0] = self._tone[0] + 1 if start > 0.5 else 0
        self._tone[1] = self._tone[1] + 1 if stop > 0.5 else 0
        if self._tone[0] == 3:                              # tonalité de départ : une nouvelle image commence
            self.image_id += 1
            self.lines, self.phased, self._phasing = 0, False, []
            self.rows.clear()
            self.state = "départ"
            self.emit("\n[Fax : début d'une image]\n")
        if self._tone[1] == 3 and self.state != "attente":
            self.finished += 1
            self.state = "attente"
            self.emit("[Fax : fin de l'image, %d lignes]\n" % self.lines)
        if start > 0.5 or stop > 0.5:
            return                                          # ces tonalités ne font pas partie de l'image
        x = (np.arange(self.WIDTH) + 0.5) * (len(v) - 1) / self.WIDTH
        row = np.interp(x, np.arange(len(v)), v)
        if (not self.phased or self.state == "mise en phase") and self._phase_on(row):
            return                                          # ligne de mise en phase : elle ne fait pas partie de l'image
        if self.state in ("départ", "mise en phase"):
            self.state = "image"
        elif self.state == "attente" and self.lines == 0:
            self.state = "image (prise en cours)"
        self.lines += 1
        self.rows.append(np.round(row * 255).astype(np.uint8))

    def _phase_on(self, row):
        """Ligne de mise en phase : un seul niveau, sauf une impulsion étroite qui marque le bord de l'image."""
        white = row > 0.5
        share = float(white.mean())
        minority = white if share < 0.5 else ~white
        width = float(minority.mean())
        if not 0.02 <= width <= 0.12:
            self._phasing = []
            return False
        ang = 2 * np.pi * np.flatnonzero(minority) / self.WIDTH
        m = np.exp(1j * ang).mean()
        if abs(m) < 0.9:                                    # points dispersés : ce n'est pas une impulsion unique
            self._phasing = []
            return False
        center = (np.angle(m) / (2 * np.pi)) % 1.0          # position de l'impulsion, en fraction de ligne
        self._phasing.append(center)
        self.state = "mise en phase"
        if not self.phased and len(self._phasing) >= 4:
            last = np.exp(2j * np.pi * np.array(self._phasing[-4:]))
            if abs(last.mean()) > 0.995:                    # quatre lignes d'accord : on cale le bord sur l'impulsion
                shift = (np.angle(last.mean()) / (2 * np.pi)) % 1.0
                self.pos += shift * self.period
                self.phased = True
                self._phasing = []
        return True


def _mode(name, vis, width, lines, period_ms, sync_ms, sync_at_ms, scans, kind="rgb", first_ms=0.0):
    return dict(name=name, vis=vis, width=width, lines=lines, period=period_ms, sync=sync_ms, sync_at=sync_at_ms,
                scans=scans, kind=kind, first=first_ms)


def _martin(name, vis, scan):
    g = 4.862 + 0.572
    return _mode(name, vis, 320, 256, 4.862 + 0.572 + 3 * (scan + 0.572), 4.862, 0.0,
                 [("g", g, scan), ("b", g + scan + 0.572, scan), ("r", g + 2 * (scan + 0.572), scan)])


def _scottie(name, vis, scan):
    b = 1.5 + scan + 1.5
    return _mode(name, vis, 320, 256, 3 * scan + 3 * 1.5 + 9.0, 9.0, b + scan,
                 [("g", 1.5, scan), ("b", b, scan), ("r", b + scan + 9.0 + 1.5, scan)], first_ms=9.0)


def _pd(name, vis, width, pairs, scan):
    y = 20.0 + 2.08
    return _mode(name, vis, width, pairs, y + 4 * scan, 20.0, 0.0,
                 [("y1", y, scan), ("cr", y + scan, scan), ("cb", y + 2 * scan, scan), ("y2", y + 3 * scan, scan)], kind="pd")


# Durées en millisecondes, d'après les descriptions publiées de chaque mode.
SSTV_MODES = [
    _martin("Martin M1", 44, 146.432), _martin("Martin M2", 40, 73.216),
    _scottie("Scottie S1", 60, 138.240), _scottie("Scottie S2", 56, 88.064), _scottie("Scottie DX", 76, 345.600),
    _pd("PD90", 99, 320, 128, 170.240), _pd("PD120", 95, 640, 248, 121.600), _pd("PD180", 96, 640, 248, 183.040),
    _mode("Robot 36", 8, 320, 240, 150.0, 9.0, 0.0, [("y", 12.0, 88.0), ("c", 106.0, 44.0)], kind="robot36"),
]
SSTV_BY_VIS = {m["vis"]: m for m in SSTV_MODES}


def ycc_to_rgb(y, cr, cb):
    y, cr, cb = y.astype(np.float64), cr.astype(np.float64) - 128.0, cb.astype(np.float64) - 128.0
    rgb = np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


class SSTVDecoder:
    """SSTV : images fixes en couleurs envoyées par les radioamateurs (et parfois depuis la station spatiale).

    Chaque émission commence par un en-tête « VIS » qui annonce le mode ; le décodeur l'attend,
    puis découpe le signal ligne par ligne selon les durées du mode annoncé, en se recalant
    sur l'impulsion de synchronisation (1200 Hz) de chaque ligne.
    """
    BIN_MS = 5.0

    def __init__(self, fs, emit):
        self.fs, self.emit = float(fs), emit
        self.track = FrequencyTrack(fs, 1700.0, 1000.0)
        self.f = np.zeros(0)                                # fréquence instantanée, tampon glissant
        self.base = 0                                       # numéro absolu du premier échantillon du tampon
        self.mode = None
        self.t0 = 0.0                                       # début (absolu, en échantillons) de la ligne 0
        self.line = 0
        self.rows = collections.deque()
        self.image_id = 0
        self.finished = 0
        self.lines = 0
        self._scan_from = 0                                 # reprise de la recherche d'en-tête
        self._chroma = None
        self._drift = 0.0
        self.history = collections.deque(maxlen=1500)
        self.thr_db = None

    def status(self):
        if self.mode is None:
            return "SSTV : attente d'un en-tête (VIS)"
        return "SSTV : %s, ligne %d / %d" % (self.mode["name"], self.line, self.mode["lines"])

    def take_rows(self):
        out = []
        while self.rows:
            out.append(self.rows.popleft())
        return out

    def _ms(self, ms):
        return ms * 1e-3 * self.fs

    def process(self, audio):
        f, amp = self.track.process(audio)
        if len(f) == 0:
            return
        self.history.extend(20 * np.log10(amp[::25] + 1e-12))
        self.f = np.concatenate([self.f, f])
        if self.mode is None:
            self._find_header()
        if self.mode is not None:
            self._decode_lines()
        if self.mode is not None:                           # on ne garde que ce qui reste à décoder
            keep_from = int(self.t0 + self.line * self._ms(self.mode["period"]) - self._ms(10))
        else:
            keep_from = len(self.f) + self.base - int(self._ms(1200))
        drop = min(max(keep_from - self.base, 0), len(self.f))
        if drop > 0:
            self.f = self.f[drop:]
            self.base += drop

    # ---- en-tête : 1900 Hz, puis bit de départ (1200 Hz), 7 bits + parité (1100 Hz = 1, 1300 Hz = 0), bit d'arrêt
    def _find_header(self):
        size = max(int(round(self._ms(self.BIN_MS))), 1)
        first = max(self._scan_from - self.base, 0) // size * size
        nb = (len(self.f) - first) // size
        if nb < 70 + 66:
            return
        med = np.median(self.f[first:first + nb * size].reshape(nb, size), axis=1)
        leader = np.abs(med - 1900.0) < 70.0
        csum = np.concatenate([[0], np.cumsum(leader)])
        per_bit = int(round(30.0 / self.BIN_MS))                               # 6 cases de 5 ms par bit
        for i in range(60, nb - 10 * per_bit - 1):
            if med[i] > 1350.0 or csum[i] - csum[i - 60] < 40:
                continue
            cells = med[i:i + 10 * per_bit].reshape(10, per_bit)[:, 1:per_bit - 1]
            tone = np.median(cells, axis=1)
            if abs(tone[0] - 1200.0) > 80 or abs(tone[9] - 1200.0) > 80 or np.any(np.abs(tone[1:9] - 1200.0) > 180):
                continue
            bits = [int(t < 1200.0) for t in tone[1:9]]
            code = sum(b << k for k, b in enumerate(bits[:7]))
            if sum(bits) % 2 != 0 or code not in SSTV_BY_VIS:
                continue
            a = first + (i - 1) * size                                         # front 1900 -> 1200 Hz, au plus près
            seg = self.f[a:a + 3 * size]
            below = np.flatnonzero(seg < 1550.0)
            edge = a + (int(below[0]) if len(below) else size)
            self.mode = SSTV_BY_VIS[code]
            self.t0 = self.base + edge + self._ms(300.0 + self.mode["first"])
            self.line, self.lines, self._chroma, self._drift = 0, 0, None, 0.0
            self.image_id += 1
            self.emit("\n[SSTV : image en mode %s]\n" % self.mode["name"])
            return
        self._scan_from = self.base + first + max(nb - 70 - 66, 0) * size

    # ---- lignes
    def _pixels(self, start, duration, count):
        """`count` points de 0 à 255, lus entre start et start + duration (positions absolues en échantillons)."""
        step = duration / count
        k = max(int(round(step)), 1)
        a = int(start) - self.base - k
        b = int(start + duration) - self.base + k + 2
        seg = self.f[max(a, 0):b]
        if k > 1:
            seg = np.convolve(seg, np.ones(k) / k, "same")                    # moyenne sur la durée d'un point
        x = start - self.base - max(a, 0) + (np.arange(count) + 0.5) * step
        v = np.interp(x, np.arange(len(seg)), seg)
        return np.clip(np.round((v - BLACK_HZ) / (WHITE_HZ - BLACK_HZ) * 255.0), 0, 255).astype(np.uint8)

    def _resync(self, start):
        """Écart (en échantillons) entre l'impulsion de synchronisation attendue et celle reçue, ou 0."""
        m = self.mode
        n = int(self._ms(m["sync"]))
        w = int(self._ms(3.0))
        a = int(start + self._ms(m["sync_at"])) - self.base - w
        if a < 0 or a + n + 2 * w > len(self.f):
            return 0.0
        low = np.concatenate([[0], np.cumsum(self.f[a:a + n + 2 * w] < 1350.0)])
        score = low[n:] - low[:-n]
        if score.max() < 0.6 * n:
            return 0.0
        best = np.flatnonzero(score >= score.max() - 1)
        return float(best.mean() - w)

    def _decode_lines(self):
        m = self.mode
        period = self._ms(m["period"])
        while self.line < m["lines"]:
            start = self.t0 + self.line * period
            if start + period + self._ms(5) > self.base + len(self.f):
                return
            gap = self._resync(start)
            if self.line == 0:
                self.t0 += gap                                                 # première ligne : calage complet
            else:                                                              # ensuite : correction douce, plus une
                self._drift += 0.05 * gap                                      # estimation de l'écart d'horloge de l'émetteur
                self.t0 += 0.3 * gap + self._drift
            start = self.t0 + self.line * period
            scans = {name: self._pixels(start + self._ms(at), self._ms(dur), m["width"] if name != "c" else m["width"] // 2)
                     for name, at, dur in m["scans"]}
            if m["kind"] == "rgb":
                self._push(np.stack([scans["r"], scans["g"], scans["b"]], axis=-1))
            elif m["kind"] == "pd":                                            # deux lignes d'image par ligne émise
                self._push(ycc_to_rgb(scans["y1"], scans["cr"], scans["cb"]))
                self._push(ycc_to_rgb(scans["y2"], scans["cr"], scans["cb"]))
            else:                                                              # Robot 36 : la couleur alterne d'une ligne à l'autre
                chroma = np.repeat(scans["c"], 2)
                if self.line % 2 == 0:
                    self._chroma = (scans["y"], chroma)                        # ligne paire : rouge - luminance
                elif self._chroma is not None:
                    self._push(ycc_to_rgb(self._chroma[0], self._chroma[1], chroma))
                    self._push(ycc_to_rgb(scans["y"], self._chroma[1], chroma))
            self.line += 1
        self.emit("[SSTV : image terminée]\n")
        self.finished += 1
        self._scan_from = int(self.t0 + m["lines"] * period)
        self.mode = None

    def _push(self, row):
        self.lines += 1
        self.rows.append(row)


class HellDecoder:
    """Hellschreiber (« Feld Hell ») : du texte transmis comme une image, colonne par colonne.

    L'émetteur allume et éteint une note au rythme des points noirs de chaque caractère :
    17,5 colonnes par seconde, 14 demi-points par colonne, de bas en haut.  Le récepteur ne
    « décode » rien : il trace l'intensité reçue sur une bande qui défile, et c'est l'œil qui
    lit.  Chaque colonne est tracée deux fois l'une sous l'autre, ce qui garde le texte lisible
    même sans aucune synchronisation avec l'émetteur.
    """
    COLUMNS_PER_S = 17.5
    HALF_DOTS = 14
    WIDTH = 420                                             # colonnes visibles : 24 secondes de texte
    ZOOM = 3

    def __init__(self, fs, center, emit):
        from .hfdigi import Baseband
        self.emit = emit
        self.bb = Baseband(fs, center, 260.0, 200.0)
        self.step = self.bb.fs_out / (self.COLUMNS_PER_S * self.HALF_DOTS)      # échantillons par demi-point
        self.pos = 0.0
        self.tail = np.zeros(1)
        self.pending = []
        self.columns = collections.deque(maxlen=self.WIDTH)
        self.level = 1e-9
        self.count = 0
        self.image_id = 0
        self.finished = 0
        self.history = collections.deque(maxlen=1500)
        self.thr_db = None

    def status(self):
        return "Hellschreiber : %d colonnes reçues — le texte se lit sur la bande, sous la cascade" % self.count

    def take_rows(self):
        return []

    def process(self, audio):
        z = self.bb.process(audio)
        if len(z) == 0:
            return
        env = np.abs(z).astype(np.float64)
        self.history.extend(20 * np.log10(env[::5] + 1e-12))
        buf = np.concatenate([self.tail, env])
        last = len(buf) - 1
        n = int(np.floor((last - self.pos) / self.step)) + 1 if last >= self.pos else 0
        if n > 0:
            x = self.pos + self.step * np.arange(n)
            self.pending += list(np.interp(x, np.arange(len(buf)), buf))
            self.pos = x[-1] + self.step
        self.pos -= len(buf) - 1
        self.tail = buf[-1:]
        while len(self.pending) >= self.HALF_DOTS:
            col = np.array(self.pending[:self.HALF_DOTS])
            self.pending = self.pending[self.HALF_DOTS:]
            self.level = max(0.995 * self.level, float(col.max()))             # niveau du « noir », suivi lentement
            self.columns.append(np.clip(col / self.level, 0.0, 1.0))
            self.count += 1

    def tape(self):
        """La bande reçue : image uint8 (noir sur blanc), le texte le plus récent à droite."""
        strip = np.ones((2 * self.HALF_DOTS, self.WIDTH))
        if self.columns:
            cols = np.array(self.columns).T[::-1]                               # première valeur reçue = bas de la colonne
            strip[:, :cols.shape[1]] = 1.0 - np.vstack([cols, cols])
        return np.repeat(np.round(strip * 255).astype(np.uint8), self.ZOOM, axis=0)
