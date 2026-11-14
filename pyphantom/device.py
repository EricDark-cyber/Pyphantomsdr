"""Sources SDR : materiel via SoapySDR (Windows : PothosSDR) ou fichier IQ .cf32."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

POTHOS_ROOTS = (r"C:\Program Files\PothosSDR", r"C:\Program Files (x86)\PothosSDR")


def _try_pothos_paths() -> None:
    for root in POTHOS_ROOTS:
        bin_dir = os.path.join(root, "bin")
        if not os.path.isdir(bin_dir):
            continue
        try:
            os.add_dll_directory(bin_dir)
        except Exception:
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        lib = os.path.join(root, "lib")
        if os.path.isdir(lib):
            for name in os.listdir(lib):
                if name.startswith("python3"):
                    sp = os.path.join(lib, name, "site-packages")
                    if os.path.isdir(sp) and sp not in sys.path:
                        sys.path.insert(0, sp)


def import_soapy():
    try:
        import SoapySDR

        return SoapySDR
    except Exception:
        pass
    _try_pothos_paths()
    try:
        import SoapySDR

        return SoapySDR
    except Exception as exc:
        raise RuntimeError(
            "Module SoapySDR introuvable. Installez PothosSDR (https://downloads.pothosware.com/) "
            "dans C:\\Program Files\\PothosSDR, ou posez PATH=...\\PothosSDR\\bin et "
            "PYTHONPATH=...\\PothosSDR\\lib\\python3.X\\site-packages (voir README.md)."
        ) from exc


def list_devices() -> list[dict]:
    SoapySDR = import_soapy()
    out = []
    for d in SoapySDR.Device.enumerate():
        label = d.get("driver", "?")
        for key in ("label", "serial", "product"):
            if d.get(key):
                label = f"{label}: {d[key]}"
                break
        out.append({"label": label, "args": d})
    return out


class SdrDevice:
    def __init__(self, args: dict):
        SoapySDR = import_soapy()
        self.SoapySDR = SoapySDR
        self.dev = SoapySDR.Device(args)
        self.driver = str(args.get("driver", ""))
        self.stream = None
        self.center_freq = 100_000_000.0
        self.sample_rate = 1_000_000.0
        self.lo_offset = 0.0
        self._gain_range = None

    def supported_sample_rates(self, presets: list[float]) -> list[float]:
        try:
            rng = self.dev.getSampleRateRange(self.SoapySDR.SOAPY_SDR_RX, 0)
            lo, hi = float(rng.minimum()), float(rng.maximum())
            rates = [float(r) for r in presets if lo - 1 <= r <= hi + 1]
            if rates:
                return rates
        except Exception:
            pass
        return [float(r) for r in presets]

    def list_gain_elements(self) -> list[str]:
        try:
            return list(self.dev.listGains(self.SoapySDR.SOAPY_SDR_RX, 0))
        except Exception:
            return []

    def set_gain_fraction(self, frac: float) -> None:
        frac = float(min(max(frac, 0.0), 1.0))
        RX = self.SoapySDR.SOAPY_SDR_RX
        elements = self.list_gain_elements()
        try:
            if elements:
                for el in elements:
                    r = self.dev.getGainRange(RX, 0, el)
                    self.dev.setGain(RX, 0, el, float(r.start()) + frac * (float(r.stop()) - float(r.start())))
            else:
                r = self.dev.getGainRange(RX, 0)
                self.dev.setGain(RX, 0, float(r.start()) + frac * (float(r.stop()) - float(r.start())))
        except Exception:
            pass

    def set_bias_tee(self, on: bool) -> bool:
        try:
            self.dev.writeSetting("bias_tee", "true" if on else "false")
            return True
        except Exception:
            return False

    def set_gain_mode(self, mode: str) -> bool:
        RX = self.SoapySDR.SOAPY_SDR_RX
        try:
            if mode == "linearity":
                self.dev.writeSetting("linearity", "true")
                self.dev.writeSetting("sensitivity", "false")
            elif mode == "sensitivity":
                self.dev.writeSetting("sensitivity", "true")
                self.dev.writeSetting("linearity", "false")
            return True
        except Exception:
            return False

    def supports_bias_tee(self) -> bool:
        try:
            return "airspy" in self.driver or "hackrf" in self.driver
        except Exception:
            return False

    def gain_range(self) -> tuple[float, float]:
        if self._gain_range is None:
            try:
                r = self.dev.getGainRange(self.SoapySDR.SOAPY_SDR_RX, 0)
                self._gain_range = (float(r.start()), float(r.stop()))
            except Exception:
                self._gain_range = (0.0, 50.0)
        return self._gain_range

    def setup(self, sample_rate: float, center_freq: float, gain: float | None = None) -> None:
        RX = self.SoapySDR.SOAPY_SDR_RX
        self.dev.setSampleRate(RX, 0, float(sample_rate))
        self.dev.setFrequency(RX, 0, float(center_freq) + self.lo_offset)
        if gain is not None:
            self.set_gain_fraction(gain)
        self.sample_rate = float(self.dev.getSampleRate(RX, 0))
        self.center_freq = float(center_freq)
        self.stream = self.dev.setupStream(RX, "CF32")
        self.dev.activateStream(self.stream)

    def set_center_freq(self, freq: float) -> None:
        self.dev.setFrequency(self.SoapySDR.SOAPY_SDR_RX, 0, float(freq) + self.lo_offset)
        self.center_freq = float(freq)

    def set_gain(self, gain: float) -> None:
        lo, hi = self.gain_range()
        self.set_gain_fraction((float(gain) - lo) / max(hi - lo, 1e-9))

    def set_agc(self, on: bool) -> None:
        try:
            self.dev.setGainMode(self.SoapySDR.SOAPY_SDR_RX, 0, bool(on))
        except Exception:
            pass

    def read(self, n: int) -> np.ndarray:
        buf = np.empty(n, dtype=np.complex64)
        total = 0
        deadline = time.monotonic() + 2.0
        while total < n and time.monotonic() < deadline:
            res = self.dev.readStream(self.stream, [buf[total:]], n - total, 1_000_000)
            sr = res[0] if isinstance(res, tuple) else int(getattr(res, "ret", 0))
            if sr <= 0:
                time.sleep(0.002)
                continue
            total += sr
        return buf[:total]

    def close(self) -> None:
        try:
            if self.stream is not None:
                self.dev.deactivateStream(self.stream)
                self.dev.closeStream(self.stream)
                self.stream = None
        except Exception:
            pass


class IqFileSource:
    def __init__(self, path: str, sample_rate: float, loop: bool = True, center_freq: float = 100_000_000.0):
        self.path = path
        self.sample_rate = float(sample_rate)
        self.record_center = float(center_freq)
        self.center_freq = float(center_freq)
        self.loop = loop
        self._f = open(path, "rb")
        self._mix_n = 0
        import os

        self._size = os.fstat(self._f.fileno()).st_size
        self._pos = 0

    def set_center_freq(self, freq: float) -> None:
        self.center_freq = float(freq)

    def set_gain(self, gain: float) -> None:
        pass

    def set_agc(self, on: bool) -> None:
        pass

    def gain_range(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def read(self, n: int) -> np.ndarray:
        want = n * 8
        buf = bytearray()
        while len(buf) < want:
            chunk = self._f.read(want - len(buf))
            if not chunk:
                if not self.loop or self._size == 0:
                    break
                self._f.seek(0)
                self._pos = 0
                continue
            buf += chunk
        arr = np.frombuffer(bytes(buf[: (len(buf) // 8) * 8]), dtype=np.complex64).copy()
        offset = self.center_freq - self.record_center
        if offset != 0.0 and len(arr):
            n = self._mix_n + np.arange(len(arr), dtype=np.float64)
            arr = (arr * np.exp(-2j * np.pi * offset * n / self.sample_rate)).astype(np.complex64)
            self._mix_n = n[-1] + 1.0
        time.sleep(len(arr) / self.sample_rate)
        return arr

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
