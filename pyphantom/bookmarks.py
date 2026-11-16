"""Liste de frequences favorites (bookmarks), format JSON {frequency, name, mode}."""
from __future__ import annotations

import json
import os

DEFAULT_BOOKMARKS = [
    {"frequency": 77_500, "name": "DCF77 (Allemagne)", "mode": "CW"},
    {"frequency": 60_000, "name": "MSF (UK)", "mode": "CW"},
    {"frequency": 10_000_000, "name": "WWV 10 MHz", "mode": "USB"},
    {"frequency": 15_000_000, "name": "WWV 15 MHz", "mode": "USB"},
    {"frequency": 3_650_000, "name": "Bande 80 m", "mode": "LSB"},
    {"frequency": 7_100_000, "name": "Bande 40 m", "mode": "LSB"},
    {"frequency": 14_200_000, "name": "Bande 20 m", "mode": "USB"},
    {"frequency": 14_074_000, "name": "20 m FT8", "mode": "USB"},
    {"frequency": 145_500_000, "name": "Appel 2 m FM", "mode": "NFM"},
    {"frequency": 433_500_000, "name": "Appel 70 cm FM", "mode": "NFM"},
]


def default_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "PyPhantom", "bookmarks.json")


def load(path: str | None = None) -> list[dict]:
    path = path or default_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return [dict(b) for b in DEFAULT_BOOKMARKS]


def save(bookmarks: list[dict], path: str | None = None) -> None:
    path = path or default_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bookmarks, f, indent=2, ensure_ascii=False)
