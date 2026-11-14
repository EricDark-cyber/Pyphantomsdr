"""Persistance des reglages utilisateur (JSON)."""
from __future__ import annotations

import json
import os

DEFAULTS = {
    "frequency_hz": 100_000_000.0,
    "mode": "NFM",
    "bandwidth_hz": 12_000.0,
    "sample_rate": 1_000_000.0,
    "gain": 30.0,
    "agc": True,
    "volume": 0.8,
    "muted": False,
    "squelch_db": -120.0,
    "zoom": 1,
    "calibration_db": 0.0,
    "device_args": "",
    "audio_rate": 48_000,
    "spyverter": False,
    "bias_tee": False,
    "gain_mode": "Auto",
    "ft8_enabled": False,
    "ft8_offset_hz": 1500,
    "ft8_mode": "FT8",
}


def default_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "PyPhantom", "config.json")


def load(path: str | None = None) -> dict:
    path = path or default_path()
    cfg = dict(DEFAULTS)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    return cfg


def save(cfg: dict, path: str | None = None) -> None:
    path = path or default_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
