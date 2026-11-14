"""Mémoires en CSV, journal d'activité et enregistrements programmés."""
import csv
import datetime
import os

FIELDS = ["nom", "frequence_hz", "mode", "groupe"]
TIME_FORMAT = "%Y-%m-%d %H:%M"


def export_memories(path, memories):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:       # lisible directement par Excel
        w = csv.writer(f, delimiter=";")
        w.writerow(FIELDS)
        for m in memories:
            w.writerow([m["name"], int(m["hz"]), m["mode"], m.get("group", "")])


def import_memories(path, valid_modes):
    """Lit un CSV (séparateur ; ou ,) : nom, fréquence en Hz, mode, groupe facultatif."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    rows = list(csv.reader(text.splitlines(), delimiter=";" if text.count(";") >= text.count(",") else ","))
    out = []
    for row in rows:
        if len(row) < 2:
            continue
        try:
            hz = int(float(row[1].replace(" ", "").replace(" ", "")))
        except ValueError:
            continue                                                   # ligne d'en-tête ou ligne invalide
        mode = row[2].strip().upper() if len(row) > 2 else "USB"
        mode = {"FM": "NFM", "NBFM": "NFM", "WBFM": "WFM"}.get(mode, mode)
        out.append({"name": row[0].strip() or str(hz), "hz": hz, "mode": mode if mode in valid_modes else "USB",
                    "group": row[3].strip() if len(row) > 3 else ""})
    return out


def log_activity(folder, hz, mode, level_db, tone="", now=None):
    """Ajoute une ligne au journal du jour ; retourne le chemin du fichier."""
    now = now or datetime.datetime.now()
    path = os.path.join(folder, "RadioLab_journal_%s.csv" % now.strftime("%Y%m%d"))
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        if new:
            w.writerow(["heure", "frequence_hz", "mode", "niveau_db", "tonalite"])
        w.writerow([now.strftime("%H:%M:%S"), int(hz), mode, "%.0f" % level_db, tone or ""])
    return path


def parse_job(text, valid_modes):
    """« 2026-11-12 21:00;30;7100000;LSB;audio » -> enregistrement programmé."""
    parts = [p.strip() for p in text.split(";")]
    if len(parts) != 5:
        raise ValueError("format attendu : AAAA-MM-JJ HH:MM;minutes;fréquence en Hz;mode;audio ou iq")
    start = datetime.datetime.strptime(parts[0], TIME_FORMAT)
    minutes, hz, mode, kind = float(parts[1]), int(float(parts[2])), parts[3].upper(), parts[4].lower()
    if minutes <= 0 or mode not in valid_modes or kind not in ("audio", "iq"):
        raise ValueError("durée, mode ou type (audio / iq) invalide")
    return {"start": start.strftime(TIME_FORMAT), "minutes": minutes, "hz": hz, "mode": mode, "iq": kind == "iq"}


def job_label(job):
    return "%s — %g min — %d Hz %s — %s" % (job["start"], job["minutes"], job["hz"], job["mode"],
                                            "I/Q" if job["iq"] else "audio")


class Scheduler:
    """Décide, à chaque appel de `tick`, s'il faut lancer ou arrêter un enregistrement programmé."""

    def __init__(self, jobs):
        self.jobs = jobs                      # liste partagée avec la configuration
        self.current = None

    def tick(self, now=None):
        """Retourne une liste d'actions : ("start", job) ou ("stop", job)."""
        now = now or datetime.datetime.now()
        actions = []
        if self.current is not None:
            end = self._start(self.current) + datetime.timedelta(minutes=self.current["minutes"])
            if now >= end or self.current not in self.jobs:
                actions.append(("stop", self.current))
                if self.current in self.jobs:
                    self.jobs.remove(self.current)
                self.current = None
        if self.current is None:
            for job in list(self.jobs):
                start = self._start(job)
                end = start + datetime.timedelta(minutes=job["minutes"])
                if now >= end:
                    self.jobs.remove(job)                              # créneau déjà passé
                elif now >= start:
                    self.current = job
                    actions.append(("start", job))
                    break
        return actions

    @staticmethod
    def _start(job):
        return datetime.datetime.strptime(job["start"], TIME_FORMAT)
