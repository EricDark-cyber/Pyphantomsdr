"""Scanner : parcourt des fréquences et s'arrête sur celles où un signal dépasse le seuil."""
import time


class Scanner:
    """À appeler régulièrement par `tick()` (l'interface le fait 20 fois par seconde).

    Sur chaque canal : attendre `dwell` secondes que la mesure se stabilise ; si
    le niveau dépasse le seuil, rester tant que le signal dure puis encore `hang`
    secondes ; sinon passer au canal suivant.
    """

    MAX_CHANNELS = 20000

    def __init__(self, radio, clock=time.monotonic):
        self.radio, self.clock = radio, clock
        self.channels = []
        self.active = False
        self.index = 0
        self.holding = False
        self.threshold_db = -60.0
        self.dwell, self.hang = 0.25, 2.0

    def start(self, channels, threshold_db):
        """channels : liste de (fréquence en Hz, mode ou None pour garder le mode courant)."""
        self.channels = list(channels)[:self.MAX_CHANNELS]
        if not self.channels:
            raise ValueError("aucune fréquence à parcourir")
        self.threshold_db = threshold_db
        self.active, self.holding = True, False
        self.index = -1
        self._next()

    def start_range(self, lo_hz, hi_hz, step_hz, threshold_db):
        if hi_hz <= lo_hz or step_hz <= 0:
            raise ValueError("plage de fréquences invalide")
        count = int((hi_hz - lo_hz) / step_hz) + 1
        if count > self.MAX_CHANNELS:
            raise ValueError("plage trop grande pour ce pas (%d canaux)" % count)
        self.start([(int(round(lo_hz + k * step_hz)), None) for k in range(count)], threshold_db)

    def stop(self):
        self.active = self.holding = False

    def _next(self):
        for _ in range(len(self.channels)):                    # saute les canaux impossibles à accorder
            self.index = (self.index + 1) % len(self.channels)
            hz, mode = self.channels[self.index]
            try:
                if mode and mode != self.radio.rx.mode:
                    self.radio.set_mode(mode)
                center = self.radio.center_hz
                self.radio.tune(hz)
                # un déplacement de la fréquence centrale demande plus de temps avant une mesure fiable
                self.t_ready = self.clock() + (2 * self.dwell if self.radio.center_hz != center else self.dwell)
                return
            except Exception:
                continue
        self.stop()

    def tick(self):
        if not self.active:
            return
        now = self.clock()
        if now < self.t_ready:
            return
        if self.radio.rx.level_db > self.threshold_db:
            self.holding, self.t_signal = True, now
            return
        if self.holding and now - self.t_signal < self.hang:
            return
        self.holding = False
        self._next()
