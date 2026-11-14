"""Récepteur Airspy R2 / Mini, piloté par la bibliothèque officielle libairspy (ctypes).

Sous Windows il faut `airspy.dll` en 64 bits (comme Python) avec les DLL dont
elle dépend (libusb-1.0.dll et la DLL pthread) dans le même dossier.
"""
import ctypes
import os
import sys
import numpy as np

SAMPLE_FLOAT32_IQ = 0
SPYVERTER_OFFSET_HZ = 120_000_000        # le SpyVerter R2 décale la HF de +120 MHz


class _Transfer(ctypes.Structure):
    _fields_ = [("device", ctypes.c_void_p), ("ctx", ctypes.c_void_p), ("samples", ctypes.c_void_p),
                ("sample_count", ctypes.c_int), ("dropped_samples", ctypes.c_uint64),
                ("sample_type", ctypes.c_int)]


_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(_Transfer))


class AirspyError(RuntimeError):
    pass


def load_library(dll_path=None):
    names = [dll_path] if dll_path else []
    here = os.path.dirname(os.path.abspath(sys.argv[0] or "."))
    if sys.platform == "win32":
        names += [os.path.join(here, "airspy.dll"), os.path.join(os.getcwd(), "airspy.dll"), "airspy.dll"]
    elif sys.platform == "darwin":
        names += ["libairspy.dylib", "/opt/homebrew/lib/libairspy.dylib", "/usr/local/lib/libairspy.dylib"]
    else:
        names += ["libairspy.so.0", "libairspy.so"]
    errors = []
    for name in names:
        try:
            if sys.platform == "win32" and os.path.isabs(name) and os.path.isfile(name):
                os.add_dll_directory(os.path.dirname(name))       # pour trouver libusb et pthread à côté
            return ctypes.CDLL(name)
        except OSError as e:
            errors.append("%s : %s" % (name, e))
    raise AirspyError("Bibliothèque Airspy introuvable.\n" + "\n".join(errors))


class AirspySource:
    """Flux I/Q complexe (float32) venant d'un Airspy."""

    kind = "airspy"

    def __init__(self, dll_path=None):
        self.lib = lib = load_library(dll_path)
        p, u8, u32 = ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint32
        for name, args in [("airspy_open", [ctypes.POINTER(p)]), ("airspy_close", [p]),
                           ("airspy_set_sample_type", [p, ctypes.c_int]), ("airspy_set_samplerate", [p, u32]),
                           ("airspy_get_samplerates", [p, ctypes.POINTER(u32), u32]),
                           ("airspy_set_freq", [p, u32]), ("airspy_set_linearity_gain", [p, u8]),
                           ("airspy_set_sensitivity_gain", [p, u8]), ("airspy_set_rf_bias", [p, u8]),
                           ("airspy_start_rx", [p, _CALLBACK, p]), ("airspy_stop_rx", [p])]:
            f = getattr(lib, name)
            f.argtypes, f.restype = args, ctypes.c_int
        self.dev = ctypes.c_void_p()
        self._check(lib.airspy_open(ctypes.byref(self.dev)), "ouverture de l'Airspy (est-il branché ? "
                    "un autre logiciel l'utilise-t-il ?)")
        self._check(lib.airspy_set_sample_type(self.dev, SAMPLE_FLOAT32_IQ), "choix du format I/Q")
        count = ctypes.c_uint32(0)
        lib.airspy_get_samplerates(self.dev, ctypes.byref(count), 0)
        rates = (ctypes.c_uint32 * max(count.value, 1))()
        if count.value:
            lib.airspy_get_samplerates(self.dev, rates, count.value)
        self.sample_rates = sorted(int(r) for r in rates[:count.value]) or [2_500_000, 10_000_000]
        self.sample_rate = None
        self.center_hz = 0
        self.converter_offset = 0
        self.spyverter = False
        self.sv_ppm = 0.0
        self.shift_hz = 0.0
        self.dropped = 0
        self.streaming = False
        self._cb = None
        self.set_sample_rate(self.sample_rates[0])

    def _check(self, code, what):
        if code != 0:
            raise AirspyError("Airspy : échec de %s (code %d)" % (what, code))

    def set_sample_rate(self, rate):
        if self.streaming:
            raise AirspyError("arrêter la réception avant de changer le débit")
        self._check(self.lib.airspy_set_samplerate(self.dev, int(rate)), "réglage du débit")
        self.sample_rate = int(rate)

    def set_center(self, hz):
        """Fréquence centrale AFFICHÉE ; le décalage du convertisseur est ajouté ici."""
        hw = int(round(hz + self.converter_offset))
        if not 24_000_000 <= hw <= 1_800_000_000:
            raise AirspyError("fréquence hors de la plage de l'Airspy (24 à 1800 MHz)%s"
                              % ("" if self.converter_offset else " : cocher SpyVerter pour la HF"))
        self._check(self.lib.airspy_set_freq(self.dev, hw), "réglage de la fréquence")
        self.center_hz = float(hz)

    def set_spyverter(self, enabled):
        """Convertisseur HF en service : on ajoute son décalage de 120 MHz à la fréquence demandée."""
        self.spyverter = bool(enabled)
        self._update_offset()

    def set_sv_ppm(self, ppm):
        """Erreur de l'oscillateur 120 MHz du SpyVerter, en parties par million (1 ppm = 120 Hz).

        Valeur positive : l'oscillateur est un peu trop haut, donc tout est reçu
        trop haut ; on compense en décalant d'autant la fréquence demandée à l'Airspy.
        """
        self.sv_ppm = float(ppm)
        self._update_offset()

    def set_shift(self, hz):
        """Décalage d'affichage pour un autre convertisseur : fréquence affichée = fréquence de l'Airspy + décalage.

        Exemple : un convertisseur HF qui ajoute 125 MHz se compense avec -125 000 000.
        """
        self.shift_hz = float(hz)
        self._update_offset()

    def _update_offset(self):
        sv = SPYVERTER_OFFSET_HZ * (1.0 + self.sv_ppm * 1e-6) if self.spyverter else 0
        self.converter_offset = sv - self.shift_hz

    def set_bias_tee(self, enabled):
        """Tension continue (4,5 V) sur la prise d'antenne, pour alimenter un SpyVerter ou un préamplificateur."""
        self._check(self.lib.airspy_set_rf_bias(self.dev, 1 if enabled else 0), "commande du bias-tee")

    def set_gain(self, value, mode="linearity"):
        """Gain global de 0 à 21 : « linearity » résiste mieux aux signaux forts, « sensitivity » privilégie les signaux faibles."""
        value = int(np.clip(value, 0, 21))
        f = self.lib.airspy_set_sensitivity_gain if mode == "sensitivity" else self.lib.airspy_set_linearity_gain
        self._check(f(self.dev, value), "réglage du gain")

    def start(self, on_block):
        def cb(tp):
            t = tp.contents
            n = int(t.sample_count)
            self.dropped += int(t.dropped_samples)
            buf = (ctypes.c_float * (2 * n)).from_address(t.samples)
            on_block(np.frombuffer(buf, dtype=np.float32).copy().view(np.complex64))
            return 0
        self._cb = _CALLBACK(cb)                      # garder une référence : sinon Python le détruit
        self._check(self.lib.airspy_start_rx(self.dev, self._cb, None), "démarrage de la réception")
        self.streaming = True

    def stop(self):
        if self.streaming:
            self.lib.airspy_stop_rx(self.dev)
            self.streaming = False

    def close(self):
        self.stop()
        if self.dev:
            self.lib.airspy_close(self.dev)
            self.dev = ctypes.c_void_p()
