"""Commande à distance par le réseau, au format texte de « rigctld » (Hamlib).

Permet à Gpredict (suivi Doppler des satellites) ou à tout programme parlant ce
protocole (WSJT-X, fldigi... via le modèle Hamlib « NET rigctl ») de lire et de
régler la fréquence et le mode de RadioLab.  RadioLab se présente comme un
récepteur seul : pas d'émission, pas de second VFO.

Les ordres reçus sont déposés dans une file ; c'est l'interface qui les exécute,
dans son propre fil d'exécution.
"""
import queue
import socketserver
import threading

TO_HAMLIB = {"NFM": "FM", "WFM": "WFM", "AM": "AM", "USB": "USB", "LSB": "LSB", "CW": "CW"}
FROM_HAMLIB = {"FM": "NFM", "FMN": "NFM", "WFM": "WFM", "AM": "AM", "USB": "USB", "LSB": "LSB", "CW": "CW",
               "CWR": "CW", "PKTUSB": "USB", "PKTLSB": "LSB"}
LONG = {"get_freq": "f", "set_freq": "F", "get_mode": "m", "set_mode": "M", "get_vfo": "v", "set_vfo": "V",
        "get_ptt": "t", "set_ptt": "T", "get_split_vfo": "s", "set_split_vfo": "S", "quit": "q",
        "chk_vfo": "chk_vfo", "dump_state": "dump_state", "get_powerstat": "powerstat", "get_level": "l",
        "get_func": "u"}

# Description du « poste » attendue par Hamlib à la connexion (format du protocole rigctld, version 0).
# Modes Hamlib : AM = 0x1, CW = 0x2, USB = 0x4, LSB = 0x8, FM = 0x20, WFM = 0x40.
DUMP_STATE = "\n".join([
    "0",                                              # version du protocole
    "2",                                              # modèle Hamlib (NET rigctl)
    "1",                                              # région UIT
    "0.000000 1800000000.000000 0x6f -1 -1 0x1 0x0",  # plage de réception, modes, pas d'émission, VFO A
    "0 0 0 0 0 0 0",                                  # fin des plages de réception
    "0 0 0 0 0 0 0",                                  # fin des plages d'émission (aucune)
    "0x6f 1", "0x6f 0", "0 0",                        # pas d'accord
    "0x02 500", "0x02 200", "0x02 2000",              # filtres CW
    "0x21 10000", "0x21 5000", "0x21 20000",          # AM, FM
    "0x0c 2700", "0x0c 1400", "0x0c 3900",            # BLU
    "0x40 160000", "0x40 120000", "0x40 200000",      # FM large
    "0 0",                                            # fin des filtres
    "0", "0", "0",                                    # RIT, XIT, décalage FI maximum
    "0",                                              # annonces
    "0",                                              # préamplificateurs
    "0",                                              # atténuateurs
    "0", "0",                                         # fonctions lisibles / réglables
    "0x40000000", "0",                                # niveaux lisibles (force du signal) / réglables
    "0", "0"]) + "\n"                                 # paramètres lisibles / réglables


def answer(line, get_state, commands):
    """Réponse à une ligne de commande (None = fermer la connexion)."""
    parts = line.strip().split()
    if not parts:
        return ""
    cmd = parts[0]
    if cmd.startswith("\\"):
        cmd = LONG.get(cmd[1:], "?")
    args = parts[1:]
    hz, mode, width, *rest = get_state()
    level = rest[0] if rest else -100.0
    try:
        if cmd == "f":
            return "%d\n" % round(hz)
        if cmd == "F":
            commands.put(("freq", float(args[0])))
            return "RPRT 0\n"
        if cmd == "m":
            return "%s\n%d\n" % (TO_HAMLIB.get(mode, "USB"), round(width))
        if cmd == "M":
            commands.put(("mode", FROM_HAMLIB[args[0].upper()]))
            return "RPRT 0\n"
        if cmd == "v":
            return "VFOA\n"
        if cmd == "t":
            return "0\n"
        if cmd == "s":
            return "0\nVFOA\n"
        if cmd == "chk_vfo":
            return "CHKVFO 0\n"
        if cmd == "dump_state":
            return DUMP_STATE
        if cmd == "powerstat":
            return "1\n"
        if cmd == "l":
            return "%d\n" % round(level) if args and args[0] == "STRENGTH" else "RPRT -11\n"
        if cmd == "u":
            return "0\n"
        if cmd in ("V", "T", "S"):
            return "RPRT 0\n"
        if cmd == "q":
            return None
    except (IndexError, ValueError, KeyError):
        return "RPRT -1\n"                    # paramètre invalide
    return "RPRT -11\n"                       # fonction non disponible


class RigCtlServer:
    def __init__(self, get_state, host="127.0.0.1", port=4532):
        self.commands = queue.Queue()
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                for raw in self.rfile:
                    reply = answer(raw.decode("ascii", "replace"), get_state, outer.commands)
                    if reply is None:
                        return
                    self.wfile.write(reply.encode("ascii"))

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = Server((host, port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
