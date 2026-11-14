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
import sys
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


# ------------------------------------------------------------------ CAT série
# Mêmes commandes texte rigctl, mais sur un port COM série (pour Gpredict en
# « rig série », OmniRig, ou tout client CAT texte).  Windows : ctypes pur
# (CreateFile sur COMn, 8N1, pas de dépendance) ; ailleurs : pyserial si présent.

_GENERIC_READ, _GENERIC_WRITE, _OPEN_EXISTING = 0x80000000, 0x40000000, 3
_INVALID_HANDLE = -1


def _open_com_windows(port, baud):
    """Ouvre un port COM (ex. 'COM3') en 8N1 via ctypes. Retourne le handle."""
    import ctypes

    k = ctypes.windll.kernel32
    h = k.CreateFileW("\\\\.\\" + port, _GENERIC_READ | _GENERIC_WRITE, 0, None, _OPEN_EXISTING, 0, None)
    if h in (None, _INVALID_HANDLE):
        raise OSError("ouverture de %s impossible (handle invalide)" % port)
    dcb = bytearray(28)                                   # DCB : 28 octets
    ctypes.windll.kernel32.GetCommState(h, dcb)
    struct_pack_into(dcb, 4, baud)                        # BaudRate
    struct_pack_into(dcb, 8, 1)                           # fBinary = 1 (le reste à 0)
    dcb[20] = 8                                           # ByteSize
    dcb[21] = 0                                           # Parity = NOPARITY
    dcb[22] = 0                                           # StopBits = ONESTOPBIT
    if not ctypes.windll.kernel32.SetCommState(h, dcb):
        k.CloseHandle(h)
        raise OSError("SetCommState a échoué sur %s" % port)
    timeouts = bytearray(20)                              # COMMTIMEOUTS
    struct_pack_into(timeouts, 0, 50)                     # ReadIntervalTimeout (max)
    struct_pack_into(timeouts, 4, 0)                      # ReadTotalTimeoutMultiplier
    struct_pack_into(timeouts, 8, 100)                    # ReadTotalTimeoutConstant
    ctypes.windll.kernel32.SetCommTimeouts(h, timeouts)
    ctypes.windll.kernel32.PurgeComm(h, 0x0C)             # TXABORT | RXABORT : on vide les files
    return h


def struct_pack_into(buf, offset, value):
    import struct

    struct.pack_into("<I", buf, offset, value)


def _com_read(h, n=256):
    import ctypes

    buf = ctypes.create_string_buffer(n)
    got = ctypes.c_ulong(0)
    if not ctypes.windll.kernel32.ReadFile(h, buf, n, ctypes.byref(got), None):
        return b""
    return buf.raw[:got.value]


def _com_write(h, data):
    import ctypes

    got = ctypes.c_ulong(0)
    ctypes.windll.kernel32.WriteFile(h, data, len(data), ctypes.byref(got), None)
    return got.value


class SerialCatServer:
    """Serveur CAT sur port série : mêmes commandes texte rigctl que RigCtlServer.

    `handle_line()` est séparé du transport : testable sans matériel.
    Les ordres reçus vont dans `self.commands` (file), comme RigCtlServer.
    """

    def __init__(self, get_state, port="COM3", baud=9600, autostart=True):
        self.get_state, self.port, self.baud = get_state, port, int(baud)
        self.commands = queue.Queue()
        self.error = None
        self._stop = threading.Event()
        self._tr = None
        self._thread = None
        if autostart:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def handle_line(self, line):
        """Réponse à une ligne de commande (None = fermer la connexion)."""
        return answer(line, self.get_state, self.commands)

    def _open(self):
        if sys.platform == "win32":
            return ("com", _open_com_windows(self.port, self.baud))
        import serial                                     # pyserial, hors Windows

        return ("py", serial.Serial(self.port, self.baud, bytesize=8, parity="N", stopbits=1, timeout=0.2))

    def _read(self, tr, n=256):
        if tr[0] == "com":
            return _com_read(tr[1], n)
        return tr[1].read(n) or b""

    def _write(self, tr, data):
        if tr[0] == "com":
            _com_write(tr[1], data)
        else:
            tr[1].write(data)

    def _run(self):
        try:
            self._tr = self._open()
        except Exception as e:
            self.error = "%s: %s" % (self.port, e)
            return
        buf = b""
        try:
            while not self._stop.is_set():
                chunk = self._read(self._tr)
                if chunk:
                    buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    reply = self.handle_line(line.decode("ascii", "replace"))
                    if reply is None:
                        return
                    self._write(self._tr, reply.encode("ascii"))
        except Exception as e:
            if not self._stop.is_set():
                self.error = "%s: %s" % (self.port, e)
        finally:
            self._close()

    def _close(self):
        tr, self._tr = self._tr, None
        if tr is None:
            return
        try:
            if tr[0] == "com":
                import ctypes

                ctypes.windll.kernel32.CloseHandle(tr[1])
            else:
                tr[1].close()
        except Exception:
            pass

    def close(self):
        self._stop.set()
        self._close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
