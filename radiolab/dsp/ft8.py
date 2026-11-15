"""Décodeur FT8 / FT4 : toute la bande audio, plusieurs stations par période.

Rappel des deux modes (protocoles de WSJT-X) :

  FT8 : 79 symboles de 0,16 s (12,64 s), 8 tonalités espacées de 6,25 Hz, périodes de 15 s.
        Trois motifs de synchronisation « Costas » de 7 symboles (positions 0, 36 et 72) ;
        58 symboles de données x 3 bits = 174 bits.
  FT4 : 103 symboles de 0,048 s (4,94 s), 4 tonalités espacées de 20,83 Hz, périodes de 7,5 s.
        Quatre motifs de 4 symboles (positions 0, 33, 66, 99) ; 87 symboles x 2 bits = 174 bits ;
        les 77 bits du message sont de plus brouillés par une suite fixe.

Dans les deux cas : 174 bits -> code correcteur LDPC(174,91) -> 91 bits = 77 bits de message
+ 14 bits de contrôle (CRC-14) -> texte (indicatifs, carré locator, report).

Comment on s'y prend, toutes les 2 secondes, sur les 18 dernières secondes reçues (9,6 s en FT4) :

  1. Spectrogramme : une FFT par quart de symbole, résolution d'une demi-tonalité.
  2. Recherche des motifs de synchronisation à TOUTES les fréquences et à tous les instants
     de départ possibles : chaque pic donne un candidat (fréquence, instant).
  3. Pour chaque candidat : on ramène son petit bout de spectre autour de 0 Hz (32 échantillons
     par symbole), on affine l'instant et la fréquence, puis on mesure l'énergie de chaque
     tonalité pour chaque symbole.
  4. Ces énergies donnent pour chaque bit une vraisemblance « souple » (à quel point le bit
     est sûr), que le décodeur LDPC (propagation de croyance) exploite.
  5. On ne garde que les mots dont le CRC-14 est juste : c'est ce contrôle qui interdit
     d'afficher un message inventé.
  6. Chaque signal décodé est reconstruit et retiré de l'audio, puis on recommence : on
     découvre ainsi les stations plus faibles cachées sous les plus fortes.

Aucune horloge n'est nécessaire : le décodeur fonctionne aussi sur un enregistrement rejoué.

Interface (inchangée) :
    d = FT8Decoder(fs, emit=None, mode="FT8", center=1500.0)
    d.process(audio)      # à chaque bloc audio
    d.on_timer()          # appelé régulièrement par l'interface (ou d.try_decode())
    d.messages            # derniers messages, « TEXTE  -12dB  1234Hz »
    d.decodes             # (texte, dB, Hz, instant de début en s)
    d.wait(), d.flush()   # attendre la fin des calculs ; flush() analyse aussi la fin d'un enregistrement
`center` n'est plus qu'un repère : toute la bande de 100 à 3200 Hz est décodée.

Avec un `emit`, le calcul se fait dans un fil d'exécution à part (l'interface ne se fige
pas) ; sans `emit`, il se fait dans try_decode().  `flush()` termine les calculs en attente.
"""
import collections
import queue
import threading
import time

import numpy as np

from .ft8_crc import add_crc14, check_crc14
from .ft8_ldpc import build_H
from .ft8_unpack import unpack_ft8

FT8_COSTAS = [3, 1, 4, 0, 6, 5, 2]
FT8_SYMBOLS = 79
FT8_SYMBOL_S = 0.16
FT8_TONE_HZ = 6.25
# valeur de 3 bits -> tonalité (code de Gray : deux tonalités voisines ne diffèrent que d'un bit)
GRAY_ENC = [0, 1, 3, 2, 5, 6, 4, 7]
GRAY = [0] * 8
for _v, _t in enumerate(GRAY_ENC):
    GRAY[_t] = _v

FT4_SYMBOLS = 103
FT4_SYMBOL_S = 0.048
FT4_TONE_HZ = 1.0 / FT4_SYMBOL_S
FT4_COSTAS = [[0, 1, 3, 2], [1, 0, 2, 3], [2, 3, 1, 0], [3, 2, 0, 1]]
FT4_GRAY_ENC = [0, 1, 3, 2]
# suite de brouillage des 77 bits du message FT4 (octets 4A 5E 89 B4 B0 8A 79 55 BE 28)
FT4_RVEC = [int(b) for b in "".join(format(x, "08b") for x in
                                    (0x4A, 0x5E, 0x89, 0xB4, 0xB0, 0x8A, 0x79, 0x55, 0xBE, 0x28))[:77]]

SPS = 32                      # échantillons par symbole après retour autour de 0 Hz
HOP_S = 2.0                   # une analyse toutes les 2 s
MAX_CANDIDATES = 80
MAX_PASSES = 3                # passes successives, en retirant à chaque fois les signaux décodés
SYNC_MIN = 1.5                # pic de synchronisation minimal, par rapport au niveau courant de la bande


def _mode_params(name):
    if name == "FT4":
        sync = {0: FT4_COSTAS[0], 33: FT4_COSTAS[1], 66: FT4_COSTAS[2], 99: FT4_COSTAS[3]}
        return dict(name="FT4", nsym=FT4_SYMBOLS, tsym=FT4_SYMBOL_S, ntones=4, bps=2, gray=FT4_GRAY_ENC,
                    sync=sync, slot=7.5, window=9.6, min_sync=8, rvec=np.array(FT4_RVEC, dtype=np.uint8), bt=1.0)
    sync = {0: FT8_COSTAS, 36: FT8_COSTAS, 72: FT8_COSTAS}
    return dict(name="FT8", nsym=FT8_SYMBOLS, tsym=FT8_SYMBOL_S, ntones=8, bps=3, gray=GRAY_ENC,
                sync=sync, slot=15.0, window=18.0, min_sync=7, rvec=None, bt=2.0)


# ---------------------------------------------------------------------------------------------
# Code correcteur LDPC(174,91)
# ---------------------------------------------------------------------------------------------
class LDPCDecoder:
    """Propagation de croyance (somme-produit), plusieurs mots à la fois.

    Chaque équation de parité relie 6 ou 7 bits ; chaque bit figure dans 3 équations.  À chaque
    tour, chaque équation dit à chacun de ses bits ce qu'il devrait valoir d'après les autres ;
    on s'arrête dès que toutes les équations sont satisfaites."""

    def __init__(self, H=None):
        H = build_H() if H is None else H
        self.H = H
        self.M, self.N = H.shape
        rows = [np.flatnonzero(H[i]) for i in range(self.M)]
        self.W = max(len(r) for r in rows)
        self.idx = np.full((self.M, self.W), self.N, dtype=np.int64)      # N = bit fictif (cases vides)
        self.mask = np.zeros((self.M, self.W), dtype=bool)
        for i, r in enumerate(rows):
            self.idx[i, :len(r)] = r
            self.mask[i, :len(r)] = True
        flat = self.idx.ravel()
        self.edges = np.array([np.flatnonzero(flat == n) for n in range(self.N)])    # (N, 3)

    def decode(self, llr, iters=30):
        """llr : (B, 174), positif = bit à 1.  Retourne (réussi (B,), bits (B, 174))."""
        llr = -np.atleast_2d(np.asarray(llr, dtype=np.float64))      # en interne : log(P0 / P1)
        B = llr.shape[0]
        c2v = np.zeros((B, self.M * self.W))
        done = np.zeros(B, dtype=bool)
        out = np.zeros((B, self.N), dtype=np.uint8)
        pad_l = np.full((B, 1), 1e3)
        pad_b = np.zeros((B, 1), dtype=np.int64)
        for _ in range(iters):
            tot = llr + c2v[:, self.edges].sum(axis=2)
            hard = (tot < 0).astype(np.int64)
            synd = np.concatenate([hard, pad_b], axis=1)[:, self.idx].sum(axis=2) & 1
            ok = ~synd.any(axis=1)
            new = ok & ~done
            out[new] = hard[new]
            done |= ok
            if done.all():
                break
            v2c = np.concatenate([tot, pad_l], axis=1)[:, self.idx] - c2v.reshape(B, self.M, self.W)
            t = np.tanh(np.clip(v2c, -40.0, 40.0) / 2.0)
            t = np.where(self.mask, t, 1.0)
            t = np.where(np.abs(t) < 1e-12, 1e-12, t)
            ex = np.clip(t.prod(axis=2, keepdims=True) / t, -0.999999999999, 0.999999999999)
            c2v = (2.0 * np.arctanh(ex) * self.mask).reshape(B, self.M * self.W)
        return done, out


_GEN = None


def _generator():
    """Matrice (83 x 91) donnant les 83 bits de parité à partir des 91 bits utiles (déduite de H)."""
    global _GEN
    if _GEN is None:
        H = build_H().astype(np.uint8)
        A = np.concatenate([H[:, 91:], H[:, :91]], axis=1)             # [B | A] -> on réduit B à l'identité
        m = A.shape[0]
        for c in range(m):
            p = c + int(np.flatnonzero(A[c:, c])[0])
            if p != c:
                A[[c, p]] = A[[p, c]]
            for r in np.flatnonzero(A[:, c]):
                if r != c:
                    A[r] ^= A[c]
        _GEN = A[:, m:].copy()
    return _GEN


def encode_tones(msg77, mode="FT8"):
    """77 bits de message -> suite des tonalités de la trame (sert aux essais)."""
    p = _mode_params(mode)
    m = np.array(list(msg77)[:77], dtype=np.uint8)
    if p["rvec"] is not None:
        m = m ^ p["rvec"]
    m91 = np.array(add_crc14([int(b) for b in m]), dtype=np.uint8)
    cw = np.concatenate([m91, (_generator() @ m91) % 2])
    bps = p["bps"]
    vals = cw.reshape(-1, bps) @ (1 << np.arange(bps - 1, -1, -1))
    tones = np.zeros(p["nsym"], dtype=np.int64)
    is_sync = np.zeros(p["nsym"], dtype=bool)
    for pos, pat in p["sync"].items():
        tones[pos:pos + len(pat)] = pat
        is_sync[pos:pos + len(pat)] = True
    tones[~is_sync] = np.array(p["gray"])[vals]
    return tones


def synth(tones, fs, f0, mode="FT8", gfsk=True):
    """Signal audio d'une suite de tonalités, la plus basse à f0 (sert aux essais)."""
    p = _mode_params(mode)
    return np.sin(_phase(tones, fs, f0, p["tsym"], p["bt"] if gfsk else None))


# ---------------------------------------------------------------------------------------------
# Décodeur
# ---------------------------------------------------------------------------------------------
class FT8Decoder:
    def __init__(self, fs, emit=None, mode="FT8", center=1500.0, threaded=None):
        self.fs = float(fs)
        self.emit = emit
        self.mode = "FT4" if mode == "FT4" else "FT8"
        self.center = float(center)
        p = self.p = _mode_params(self.mode)
        self.nsym, self.tsym, self.ntones, self.bps = p["nsym"], p["tsym"], p["ntones"], p["bps"]
        self.tone = 1.0 / self.tsym
        self.period = self.nsym * self.tsym
        self.slot_s = p["slot"]
        self.window_s = p["window"]
        self.costas = FT4_COSTAS if self.mode == "FT4" else FT8_COSTAS

        # découpage de l'audio
        self.n_win = int(round(p["window"] * self.fs))                 # fenêtre d'analyse (une période et un peu plus)
        self.hop = int(round(HOP_S * self.fs))
        self.nsps = int(round(self.tsym * self.fs))                    # échantillons par symbole
        self.nfft = 2 * self.nsps                                      # cases d'une demi-tonalité
        self.bin_hz = self.fs / self.nfft
        self.step = self.tsym * self.fs / 4.0                          # un quart de symbole
        self.nframes = int((self.n_win - self.nsps) // self.step) + 1
        self.frame_idx = (np.round(np.arange(self.nframes) * self.step).astype(np.int64)[:, None]
                          + np.arange(self.nsps)[None, :])
        self.nstarts = self.nframes - 4 * (self.nsym - 1)              # instants de départ possibles
        self.f_lo = 100.0
        self.f_hi = min(3200.0, self.fs / 2 - (self.ntones + 1) * self.tone)
        self.i_lo = int(np.ceil(self.f_lo / self.bin_hz))
        self.i_hi = int(np.floor(self.f_hi / self.bin_hz))

        # signal ramené autour de 0 Hz : 32 échantillons par symbole
        self.fs_d = SPS / self.tsym
        self.n_d = int(round(p["window"] * self.fs_d))
        self.df = self.fs / self.n_win                                 # pas de la grande FFT
        self.q_lo = -int(round(1.5 * self.tone / self.df))
        self.q_hi = int(round((self.ntones + 0.5) * self.tone / self.df))
        nt = int(round(self.tone / self.df))
        taper = np.ones(self.q_hi - self.q_lo + 1)
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(nt) + 0.5) / nt)
        taper[:nt], taper[-nt:] = ramp, ramp[::-1]
        self.taper = taper

        # symboles de synchronisation et de données
        pos, tones = [], []
        for start, pat in sorted(p["sync"].items()):
            pos += list(range(start, start + len(pat)))
            tones += list(pat)
        self.sync_pos = np.array(pos)
        self.sync_tone = np.array(tones)
        self.data_pos = np.array([k for k in range(self.nsym) if k not in set(pos)])
        n = self._n = np.arange(SPS)
        self.sync_ref = np.exp(-2j * np.pi * self.sync_tone[:, None] * n[None, :] / SPS)
        self.sym_idx = np.arange(self.nsym)[:, None] * SPS + n[None, :]
        vals = np.arange(1 << self.bps)
        self.gray = np.array(p["gray"])
        self.bit_sets = [((vals >> (self.bps - 1 - b)) & 1).astype(bool) for b in range(self.bps)]
        self.ldpc = LDPCDecoder()

        # état
        self.buf = np.zeros(0, dtype=np.float32)
        self.samples_received = 0
        self._due = self.n_win                                         # prochaine analyse (en échantillons reçus)
        self.threaded = (emit is not None) if threaded is None else bool(threaded)
        self._jobs = queue.Queue()
        self._thread = None
        self._outbox = collections.deque(maxlen=200)
        self._last_emit_at = {}
        self._known = []                                               # signaux déjà décodés : (début absolu en s, Hz, tonalités, texte)
        self.messages = []
        self.decodes = collections.deque(maxlen=200)                   # (texte, dB, Hz, instant de début en s)
        self.last_snr = -99
        self.activity = False
        self.history = collections.deque(maxlen=200)
        self.last_pass_s = 0.0
        self.last_candidates = 0
        self.passes = 0

    # ---------------------------------------------------------------- arrivée de l'audio
    def process(self, audio):
        if audio is None or len(audio) == 0:
            return
        y = np.asarray(audio, dtype=np.float32).ravel()
        self.history.append(20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-8))
        for a in range(0, len(y), self.hop):
            part = y[a:a + self.hop]
            self.buf = np.concatenate([self.buf, part])[-(self.n_win + self.hop):]
            self.samples_received += len(part)
            while self.samples_received >= self._due:
                end = len(self.buf) - (self.samples_received - self._due)
                if end >= self.n_win:
                    self._submit(self.buf[end - self.n_win:end].copy(), self._due / self.fs)
                self._due += self.hop

    def _submit(self, x, t_end):
        while self._jobs.qsize() >= 3:                                 # en retard : on laisse tomber le plus ancien
            try:
                self._jobs.get_nowait()
                self._jobs.task_done()
            except queue.Empty:
                break
        self._jobs.put((x, t_end))
        if self.threaded and (self._thread is None or not self._thread.is_alive()):
            self._thread = threading.Thread(target=self._worker, name="ft8", daemon=True)
            self._thread.start()

    def _worker(self):
        while True:
            try:
                x, t_end = self._jobs.get(timeout=30.0)
            except queue.Empty:
                return
            try:
                self._run(x, t_end)
            except Exception:
                pass
            finally:
                self._jobs.task_done()

    def _drain(self):
        while True:
            try:
                x, t_end = self._jobs.get_nowait()
            except queue.Empty:
                return
            try:
                self._run(x, t_end)
            finally:
                self._jobs.task_done()

    def flush(self):
        """Analyse aussi la fin d'un enregistrement (complétée par du silence) et attend la fin des calculs."""
        if self.samples_received > 0:
            missing = self._due - self.samples_received
            if 0 < missing < self.n_win and self.samples_received >= int(self.period * self.fs):
                self.process(np.zeros(missing, dtype=np.float32))
        self.wait()
        return list(self.messages)

    def wait(self):
        """Attend la fin des analyses en cours (utile quand on rejoue un fichier plus vite que le temps réel)."""
        if self.threaded and self._thread is not None and self._thread.is_alive():
            self._jobs.join()
        else:
            self._drain()

    def try_decode(self):
        """Compatibilité : lance les analyses en attente (mode sans fil d'exécution) et rend un message nouveau, ou None."""
        if not self.threaded:
            self._drain()
        return self._outbox.popleft() if self._outbox else None

    def on_timer(self):
        self.try_decode()

    def status(self):
        secs = self.samples_received / self.fs if self.fs else 0
        if secs < self.window_s:
            return "%s : écoute, première analyse dans %.0f s" % (self.mode, self.window_s - secs)
        return "%s : %d message(s) | %d candidat(s) | analyse en %.2f s" % (
            self.mode, len(self.messages), self.last_candidates, self.last_pass_s)

    # ---------------------------------------------------------------- une analyse
    def _run(self, x, t_end):
        t0 = time.perf_counter()
        w0 = t_end - self.window_s                                     # instant du début de la fenêtre
        self._known = [k for k in self._known if k[0] + self.period > w0]
        known = [(k[0] - w0, k[1], k[2], k[3]) for k in self._known]
        found = self.analyse(x, known)
        self.passes += 1
        self.activity = bool(found)
        for text, snr, freq, dt, tones in found:
            start = w0 + dt
            self._known.append((start, freq, tones, text))
            last = self._last_emit_at.get(text)
            if last is not None and abs(start - last) < 0.8 * self.slot_s:
                continue                                               # déjà vu dans cette période
            self._last_emit_at[text] = start
            self.last_snr = snr
            line = "%s  %+03ddB  %4.0fHz" % (text, snr, freq)
            self.decodes.append((text, snr, freq, start))
            self.messages.append(line)
            if len(self.messages) > 60:
                self.messages = self.messages[-40:]
            self._outbox.append(line)
            if self.emit:
                self.emit(line + "\n")
        if len(self._last_emit_at) > 400:
            limit = t_end - 4 * self.slot_s
            self._last_emit_at = {k: v for k, v in self._last_emit_at.items() if v > limit}
        self.last_pass_s = time.perf_counter() - t0

    def analyse(self, x, known=()):
        """Une fenêtre d'audio -> liste de (texte, dB dans 2500 Hz, fréquence en Hz, début en s, tonalités).

        Plusieurs passes : les signaux décodés sont reconstruits puis RETIRÉS de l'audio, ce qui
        découvre les stations plus faibles qu'ils masquaient.  `known` donne les signaux déjà
        décodés lors d'une analyse précédente (début en s dans cette fenêtre, Hz, tonalités,
        texte) : on les retire d'emblée, sans les décoder une seconde fois."""
        x = np.array(x, dtype=np.float64)
        x -= x.mean()
        if not np.any(x):
            return []
        out, seen = [], set()
        for start, freq, tones, text in known:
            self._subtract(x, tones, freq, start)
            if start >= -0.5:
                seen.add(text)
        self.last_candidates = 0
        for npass in range(MAX_PASSES):
            found = self._one_pass(x, seen)
            if not found:
                break
            out += found
            if npass < MAX_PASSES - 1:
                for text, snr, freq, start, tones in found:
                    self._subtract(x, tones, freq, start)
        out.sort(key=lambda r: r[2])
        return out

    def _one_pass(self, x, seen):
        cands = self._candidates(x)
        self.last_candidates += len(cands)
        if not cands:
            return []
        X = np.fft.rfft(x)
        band = np.abs(X[int(300 / self.df):int(min(2700.0, self.fs / 2 - 100) / self.df)]) ** 2
        noise_x = np.percentile(band, 25) / 0.2877 + 1e-30             # puissance moyenne du bruit par case
        noise_sym = SPS * noise_x / self.n_d                           # ... et par case de symbole
        metrics, info = [], []
        for f0, j0 in cands:
            r = self._demodulate(X, f0, j0)
            if r is None:
                continue
            for llr in self._bit_metrics(r[0]):
                metrics.append(llr)
                info.append(r)
        if not metrics:
            return []
        ok, bits = self.ldpc.decode(np.array(metrics))
        out = []
        for k in np.flatnonzero(ok):
            b = bits[k]
            if not b[:77].any():
                continue
            if not check_crc14([int(v) for v in b[:91]]):
                continue
            m77 = b[:77] ^ self.p["rvec"] if self.p["rvec"] is not None else b[:77]
            text = unpack_ft8([int(v) for v in m77])
            if not text or text in seen:
                continue
            seen.add(text)
            cs, freq, start = info[k]
            tones = self._tones_of(b)
            out.append((text, self._snr(cs, tones, noise_sym), freq, start, tones))
        return out

    # ---------------------------------------------------------------- 1 et 2 : candidats
    def _candidates(self, x):
        spec = np.fft.rfft(x[self.frame_idx], n=self.nfft, axis=1)
        P = (spec.real ** 2 + spec.imag ** 2)[:, :self.i_hi + 2 * self.ntones + 1]
        ni = self.i_hi - self.i_lo + 1
        nj = self.nstarts
        if ni <= 0 or nj <= 0:
            return []
        band = np.zeros((self.nframes, ni))
        for m in range(self.ntones):
            band += P[:, self.i_lo + 2 * m:self.i_lo + 2 * m + ni]
        T = np.zeros((nj, ni))
        T0 = np.zeros((nj, ni))
        for pos, tone in zip(self.sync_pos, self.sync_tone):
            T += P[4 * pos:4 * pos + nj, self.i_lo + 2 * tone:self.i_lo + 2 * tone + ni]
            T0 += band[4 * pos:4 * pos + nj]
        sync = T / ((T0 - T) / (self.ntones - 1) + 1e-30)
        jbest = np.argmax(sync, axis=0)
        red = sync[jbest, np.arange(ni)]
        red = red / (np.percentile(red, 40) + 1e-30)
        peak = red >= SYNC_MIN
        peak[1:] &= red[1:] >= red[:-1]
        peak[:-1] &= red[:-1] > red[1:]
        order = np.flatnonzero(peak)
        order = order[np.argsort(-red[order])][:MAX_CANDIDATES]
        return [((self.i_lo + i) * self.bin_hz, int(jbest[i])) for i in order]

    # ---------------------------------------------------------------- 3 : un candidat
    def _sync_power(self, z, starts, deltas=0.0):
        """Énergie retrouvée sur les symboles de synchronisation, pour chaque instant de départ
        `starts` (en échantillons) et chaque décalage de fréquence `deltas` (en tonalités).
        Retourne un tableau (instants, décalages)."""
        starts = np.atleast_1d(starts)
        deltas = np.atleast_1d(np.asarray(deltas, dtype=np.float64))
        idx = starts[:, None, None] + self.sync_pos[None, :, None] * SPS + self._n[None, None, :]
        good = (idx >= 0) & (idx < len(z))
        seg = np.where(good, z[np.clip(idx, 0, len(z) - 1)], 0.0) * self.sync_ref[None, :, :]
        rot = np.exp(-2j * np.pi * deltas[:, None] * self._n[None, :] / SPS)          # (décalages, 32)
        c = seg @ rot.T                                                # (instants, symboles, décalages)
        return (c.real ** 2 + c.imag ** 2).sum(axis=1)

    def _demodulate(self, X, f0, j0):
        k0 = int(round(f0 / self.df))
        if k0 + self.q_lo < 0 or k0 + self.q_hi >= len(X):
            return None
        Z = np.zeros(self.n_d, dtype=np.complex128)
        q = np.arange(self.q_lo, self.q_hi + 1)
        Z[q % self.n_d] = X[k0 + q] * self.taper
        z = np.fft.ifft(Z)
        start = int(round(j0 * SPS / 4.0))
        shifts = np.arange(-SPS // 4, SPS // 4 + 1)
        deltas = np.linspace(-0.4, 0.4, 11)
        delta = 0.0
        for _ in range(2):                                             # instant, puis fréquence, deux fois
            start += int(shifts[np.argmax(self._sync_power(z, start + shifts, delta)[:, 0])])
            delta = float(deltas[np.argmax(self._sync_power(z, start, deltas)[0])])
        idx = start + self.sym_idx
        good = (idx >= 0) & (idx < len(z))
        seg = np.where(good, z[np.clip(idx, 0, len(z) - 1)], 0.0)
        seg = seg * np.exp(-2j * np.pi * delta * np.arange(SPS) / SPS)[None, :]
        cs = np.fft.fft(seg, axis=1)[:, :self.ntones]                  # (symboles, tonalités)
        hits = int((np.argmax(np.abs(cs[self.sync_pos]), axis=1) == self.sync_tone).sum())
        if hits < self.p["min_sync"]:
            return None
        # instant et fréquence au plus juste (sommet d'une parabole passant par trois mesures)
        step = deltas[1] - deltas[0]
        fine_t = start + _parabola(self._sync_power(z, start + np.array([-1, 0, 1]), delta)[:, 0])
        fine_d = delta + step * _parabola(self._sync_power(z, start, delta + step * np.array([-1.0, 0.0, 1.0]))[0])
        return cs, k0 * self.df + fine_d * self.tone, fine_t / self.fs_d

    # ---------------------------------------------------------------- 4 : vraisemblances des bits
    def _bit_metrics(self, cs):
        """Pour chaque bit : (meilleure tonalité où il vaut 1) - (meilleure tonalité où il vaut 0)."""
        mag = np.abs(cs[self.data_pos])[:, self.gray]                  # rangé par valeur des bits
        out = []
        for m in (mag, mag ** 2):
            bm = np.stack([m[:, s].max(axis=1) - m[:, ~s].max(axis=1) for s in self.bit_sets], axis=1).ravel()
            sd = bm.std()
            if sd > 0:
                out.append(2.83 * bm / sd)
        return out

    def _tones_of(self, bits):
        """174 bits décodés -> les tonalités de toute la trame (synchronisation comprise)."""
        vals = bits.reshape(-1, self.bps).astype(np.int64) @ (1 << np.arange(self.bps - 1, -1, -1))
        tones = np.zeros(self.nsym, dtype=np.int64)
        tones[self.sync_pos] = self.sync_tone
        tones[self.data_pos] = self.gray[vals]
        return tones

    def _snr(self, cs, tones, noise_sym):
        """Rapport signal/bruit ramené à 2500 Hz de bande, comme l'affiche WSJT-X."""
        sig = float(np.mean(np.abs(cs[np.arange(self.nsym), tones]) ** 2))
        ratio = max(sig / noise_sym - 1.0, 1e-3)
        return int(round(max(10 * np.log10(ratio) + 10 * np.log10(self.tone / 2500.0), -30.0)))

    # ---------------------------------------------------------------- soustraction d'un signal décodé
    def _subtract(self, x, tones, freq, start):
        """Retire de `x` (sur place) le signal dont on connaît maintenant toutes les tonalités.

        On fabrique une copie de phase parfaite du signal ; le produit de l'audio par son
        conjugué donne l'amplitude et la phase réelles, qui varient lentement (propagation) :
        on les lisse sur une fraction de symbole, puis on retranche la copie ainsi ajustée."""
        ref = np.exp(1j * _phase(tones, self.fs, freq, self.tsym, self.p["bt"]))
        n0 = int(round(start * self.fs))
        a, b = max(n0, 0), min(n0 + len(ref), len(x))
        if b - a < self.nsps:
            return
        ref = ref[a - n0:b - n0]
        amp = _smooth(x[a:b] * np.conj(ref), int(0.7 * self.nsps))
        x[a:b] -= 2.0 * np.real(amp * ref)


def _parabola(y):
    """Position (entre -0,5 et 0,5) du sommet de la parabole passant par trois points voisins."""
    y0, y1, y2 = (float(v) for v in y)
    d = y0 - 2 * y1 + y2
    return float(np.clip(0.5 * (y0 - y2) / d, -0.5, 0.5)) if d < 0 else 0.0


def _smooth(v, n):
    """Moyenne glissante sur n échantillons (bords compris)."""
    n = max(int(n) | 1, 1)
    c = np.concatenate([[0], np.cumsum(v)])
    k = np.arange(len(v))
    lo, hi = np.maximum(k - n // 2, 0), np.minimum(k + n // 2 + 1, len(v))
    return (c[hi] - c[lo]) / (hi - lo)


def _phase(tones, fs, f0, tsym, bt=None):
    """Phase (en radians) du signal émis.  Avec `bt`, les sauts de fréquence sont arrondis par
    un filtre gaussien (modulation GFSK des vrais émetteurs : BT = 2 en FT8, 1 en FT4)."""
    tones = np.asarray(tones, dtype=np.float64)
    nsps = tsym * fs
    n = int(round(len(tones) * nsps))
    k = np.minimum((np.arange(n) / nsps).astype(np.int64), len(tones) - 1)
    f = tones[k]
    if bt:
        sigma = np.sqrt(np.log(2.0)) / (2 * np.pi * bt) * nsps
        half = int(4 * sigma) + 1
        g = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
        edge = np.cumsum(g / g.sum()) - (np.arange(-half, half + 1) >= 0)      # saut arrondi - saut brusque
        for b in np.flatnonzero(np.diff(k)) + 1:                               # premier échantillon de chaque symbole
            jump = tones[k[b]] - tones[k[b - 1]]
            if jump:
                lo, hi = max(b - half, 0), min(b + half + 1, n)
                f[lo:hi] += jump * edge[lo - (b - half):hi - (b - half)]
    return 2 * np.pi * np.cumsum(f0 + f / tsym) / fs


class FT4Decoder(FT8Decoder):
    def __init__(self, fs, emit=None, center=1500.0, threaded=None):
        super().__init__(fs, emit, mode="FT4", center=center, threaded=threaded)
