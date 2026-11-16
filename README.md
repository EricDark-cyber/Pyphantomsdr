# PyPhantom SDR

Récepteur SDR logiciel en **Python pur pour Windows**, inspiré de
[PhantomSDR-Plus](https://github.com/sv1btl/PhantomSDR-Plus) et basé sur le **port complet
de [RadioLab 0.15](https://github.com/EricDark-cyber/RadioLab)** (package `radiolab/` :
spectre + cascade cliquables, zoom, plan de bandes coloré, scanner, VFO B, TimeShift,
AM synchrone, anti-bruit NB/NR/ANF, rigctl Hamlib sur 127.0.0.1:4532 pour Gpredict/WSJT-X,
décodeurs : FT8/FT4, JS8Call, JT65, M17, D-STAR, NAVTEX, METAR, WSPR, PSK31/63, RTTY,
morse, packet AX.25/APRS, SSTV, fax météo, Hellschreiber, ADS-B/ModeS+CPR, AIS, ACARS,
POCSAG/DCS, CTCSS, RDS, TETRA, SIGMET, DCF77/Allouis…).

Le **Airspy R2 est piloté en ctypes directement sur `airspy.dll`** (fournie à la racine du
dépôt avec `libusb-1.0.dll` et `pthreadVC2.dll`) : **aucune installation de PothosSDR /
SoapySDR n'est nécessaire**. L'accès SoapySDR (autres récepteurs : RTL-SDR, HackRF…) reste
disponible en option via `pyphantom/device.py`.

Architecture : **PySide6** (GUI Qt native Windows) + **numpy** (DSP) + **pyqtgraph**
(spectre/waterfall) + **sounddevice** (audio).


```
pyphantom/
├── pyphantom/
│   ├── __main__.py        # point d'entrée : python -m pyphantom
│   ├── device.py          # SoapySDR (matériel) + IqFileSource (fichier .cf32)
│   ├── dsp.py             # démodulateurs AM/FM/SSB/CW, spectre, S-mètre, AGC, squelch
│   ├── filters.py         # filtres en flux : RIF, DDC polyphase, mixer, resampling
│   ├── audio.py           # sortie audio temps réel
│   ├── recorder.py        # enregistrements IQ (.cf32) et audio (.wav)
│   ├── bookmarks.py       # favoris (JSON : fréquence, nom, mode)
│   ├── config.py          # réglages persistants (%APPDATA%\PyPhantom)
│   ├── ft8/               # décodeur FT8/FT4 intégré (LDPC + CRC-14 + unpack WSJT-X)
│   ├── spectrum.py        # fenêtres d'analyse + spectre/zoom (utilisé aussi par filters.py)
│   ├── decoders/          # décodeurs : ADS-B/ModeS+CPR, AIS(GMSK), Olivia/Contestia, PSK31, RTTY, WSPR, packet AX.25, SSTV/fax/Hell (PNG), TETRA, POCSAG/DCS, CTCSS, RDS, SIGMET, DCF77/Allouis, FEC, EKF, AudioFX, réduction de bruit (noise blanker, Kalman, NLMS, NR spectrale), Receiver complet
│   └── gui/               # ANCIENNE interface (bibliothèque pyphantom conservée telle quelle,
│                          #  non utilisée au lancement : l'UI active est radiolab/gui.py)
├── tests/test_dsp.py      # tests hors-ligne du DSP (signaux synthétiques)
├── tests/test_filters.py  # tests des filtres en flux (équivalence morcelé / d'un coup)
├── tests/test_ft8.py      # tests du décodeur FT8 (aller-retour audio 8-FSK complet)
├── tests/test_decoders.py # tests des décodeurs (CRC, CPR, AIS, POCSAG/DCS, CTCSS, ADS-B...)
└── tools/make_demo_iq.py  # génère un fichier IQ de démonstration
```

## Installation sous Windows

1. **Python 3.9 ou plus récent, 64 bits** (python.org) — cochez « Add python.exe to PATH ».
   `python --version` doit afficher 3.9+ **64-bit** (3.9.12 convient ; `airspy.dll` est en 64 bits,
   un Python 32 bits ne pourrait pas la charger). Le code du dépôt est compatible 3.9
   (vérifié : aucun `match`, aucune union `X | Y` évaluée). pip choisira automatiquement les
   versions compatibles (numpy ≤2.2, PySide6 6.8.x, pyqtgraph 0.13.x sur un 3.9).
2. Dans un terminal (`Win + R`, `cmd`), à la racine du dépôt téléchargé :
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```
   (Gardez `.venv` en dehors du dossier téléchargé depuis GitHub : le ZIP de la branche
   est remplacé à chaque téléchargement.)
3. **Pilote du dongle** : avec [Zadig](https://zadig.akeo.ie/), remplacez le pilote USB
   de l'Airspy R2 par **WinUSB**. Fermez SDR# (un seul logiciel peut ouvrir l'Airspy).
   Aucun autre logiciel n'est requis : `airspy.dll` (64 bits), `libusb-1.0.dll` et
   `pthreadVC2.dll` sont **déjà fournis à la racine du dépôt** et chargés par ctypes
   (`radiolab/sources/airspy.py`) — pas besoin de PothosSDR/SoapySDR.
   *Optionnel* : pour d'autres récepteurs (RTL-SDR, HackRF…) via SoapySDR, installez
   PothosSDR et utilisez un venv **Python 3.9** (`.venv39`, bindings cp39 de PothosSDR
   2021) ; voir `pyphantom/device.py`.
4. Lancez :
   ```bat
   python -m pyphantom
   ```
   Cliquez sur **▶ Airspy**, cochez **SpyVerter** (+120 MHz) et **Bias-tee** si besoin.

## Airspy R2 + Spyverter (HF 0–60 MHz)

Branchez le **Spyverter** sur l'entrée antenne de l'Airspy R2 (IN du Spyverter côté
Airspy, antenne HF sur l'entrée du Spyverter). Dans PyPhantom :

1. Cochez **Spyverter (+120 MHz)** : la fréquence affichée reste la vraie fréquence HF
   (ex. 7,1 MHz) tandis que le LO de l'Airspy est réglé à fréquence + 120 MHz.
2. Cochez **Bias-tee (Spyverter)** pour alimenter le Spyverter en 4,5 V via le bias-tee
   de l'Airspy R2 (sinon alimentation externe).
3. **Ech. (S/s)** : l'Airspy R2 ne supporte que **2 500 000** et **10 000 000** S/s —
   la liste est automatiquement réduite quand le récepteur est détecté.
4. **Mode gain** : *Linéarité* (LNA/MIX/VGA, recommandé) ou *Sensibilité*. Le curseur
   « Gain RF » répartit le gain sur les étages exposés par le pilote SoapySDR.

Le récepteur est piloté directement par `airspy.dll` (ctypes, racine du dépôt) : `radiolab/sources/airspy.py`.

## Test sans récepteur (fichier IQ)

```bat
python tools/make_demo_iq.py demo.cf32 10 1000000
python -m pyphantom --iq demo.cf32 --rate 1000000
```

Le fichier `demo.cf32` contient des stations synthétiques : AM à −150 kHz, FM à
+200 kHz, USB à +60 kHz par rapport à la fréquence centrale (100 MHz par défaut).

Le port RadioLab lit aussi les fichiers **WAV I/Q stéréo** produits par SDR# :
menu « Ouvrir un fichier I/Q… », ou en ligne de commande :
```bat
python -m radiolab.cli enregistrement_IQ.wav --freq 7100000 --mode USB --nr -o audio.wav
python -m radiolab.cli SDRSharp_20260101_120000Z_7100000Hz.cf32 --rate 2500000 --freq 7100000 --mode LSB -o audio.wav
```
Les fichiers **`.cf32` de SDR#** sont lus directement (I/Q complexe 32 bits brut) : le débit
se donne avec `--rate` (un .cf32 n'a pas d'en-tête) ou est demandé à l'ouverture dans l'interface.

## Utilisation

- **▶ Airspy** : démarre le récepteur (ctypes sur `airspy.dll`). **Ouvrir un fichier I/Q…** : rejoue un
  enregistrement WAV I/Q ou **.cf32 de SDR#** (le débit est demandé à l'ouverture d'un .cf32).
- **Clic sur le spectre/cascade** : accorde. **Molette** : pas ; **Ctrl+molette** : zoom. Le plan de
  bandes colore le spectre (amateur/radio/aéro/divers).
- **Modes** : AM, NFM, WFM, USB, LSB, CW, DSB, RAW, TETRA ; AM synchrone + anti-fading, binaural, note CW.
- **Gain** (0–21, réparti LNA/MIX/VGA), AGC, volume, muet, squelch, S-mètre (calibration dBm), décimation.
- **Enregistrement** : « ● Audio » (WAV) et « ● I/Q » au format choisi à côté — **WAV 16 bits** ou
  **CF32** (format brut de SDR#, sans limite de 4 Go, rejouable dans SDR# ou PyPhantom). Programmable
  via « Programmer… » (section Outils).
- **Scanner** (section Mémoires) : parcourt les mémoires ou une plage, s'arrête sur squelch, journal CSV.
- **Commande à distance** (section Outils) : **Commande réseau** = rigctl texte sur 127.0.0.1:4532
  (Gpredict, WSJT-X « NET rigctl ») ; **CAT série** = mêmes commandes sur un port COM (Gpredict en
  série, OmniRig) : régler le port COM et le baud.
- **Anti-bruit** : NB (impulsions), NR spectrale (Wiener/MMSE-LSA/LMS), ANF (notch auto des porteuses).
- **Mémoires** : import/export CSV (nom ; fréquence Hz ; mode ; groupe).

### Raccourcis clavier (aussi affichés avec **F1**)

| Touche | Action |
|---|---|
| → / ← | fréquence : un pas vers le haut / le bas |
| Ctrl + → / Ctrl + ← | fréquence : dix pas |
| Page haut / Page bas | mode suivant / précédent |
| Espace | couper / rétablir le son |
| M | mémoriser la fréquence |
| R | enregistrer le son (marche / arrêt) |
| L | verrouiller la fréquence |
| P | fichier I/Q : pause / lecture |
| F1 | afficher cette liste |

### Dépannage

- **L'Airspy n'apparaît pas / « aucun périphérique »** : installez le pilote **WinUSB** avec
  [Zadig](https://zadig.akeo.ie/) (sélectionnez l'Airspy, WinUSB, Replace Driver), fermez
  SDR# ou tout autre logiciel qui l'occupe, puis relancez. Les DLL `airspy.dll`,
  `libusb-1.0.dll`, `pthreadVC2.dll` doivent être à côté du programme (elles sont à la racine
  du dépôt et embarquées dans l'exe fabriqué).
- **Le Spyverter n'est pas vu en HF** : cochez *Spyverter (+120 MHz)* et *Bias-tee* ; la
  fréquence affichée = fréquence réelle + 120 MHz.
- **Un autre logiciel (SDR#, RadioLab) ne marche plus** : le pilote WinUSB est pris par
  PyPhantom — quittez PyPhantom, ou réinstallez le pilote d'origine avec Zadig.
- **Audio haché** : montez la latence (Outils) ou décochez les réducteurs de bruit lourds
  activés en même temps (NR spectrale + LMS + Kalman + NeuralNR).
- **.cf32 illisible** : indiquez le débit à l'ouverture (`--rate` ou boîte de dialogue).
- **Fabriquer un .exe** : `fabriquer_exe.bat` (voir PyPhantom.spec) produit `dist\PyPhantom\`.

## Décodage FT8 / FT4 (intégré)

Le panneau **« FT8 / FT4 (décodeur intégré) »** (à droite) décode les messages FT8/FT4
directement depuis l'audio démodulé :

1. Accordez une fréquence FT8 en **USB** (ex. **7,074 MHz** avec le Spyverter : cochez
   *Spyverter (+120 MHz)* + *Bias-tee*), bande passante ~3 kHz.
2. Dans le panneau FT8 : cochez **Activer**, laissez **Offset (Hz) = 1500** (position du
   signal FT8 dans la bande audio ; ajustez entre 200 et 3000 Hz si besoin).
3. Les messages décodés (indicatif, grid, report SNR) s'affichent en haut de la liste
   avec l'heure, ex. `14:32:05  CQ F1ABC JN18  -12dB`.

Le décodeur (package `pyphantom/ft8/`) suit la norme WSJT-X : 79 symboles de 0,16 s,
8 tons espacés de 6,25 Hz, synchro Costas, décodage LDPC(174,91) par propagation de
croyance, CRC-14 puis dépaquetage du message de 77 bits.

## Décodeurs audio (panneau « Décodeurs »)

En NFM, cochez dans le panneau de droite : **CTCSS** (tonalité sous-audible affichée),
**DCS** (code numérique, avec correction Golay), **POCSAG** (radiomessagerie 512/1200/2400
bit/s, messages affichés avec RIC et texte), et **Notch 50 Hz** (filtre coupe-bande anti-
ronflette appliqué à l'audio, via `decoders/audiofx.py`). La bibliothèque complète
(`pyphantom/decoders/`) contient aussi ADS-B (1090 MHz, CPR), AIS (162 MHz), ACARS, C4FM,
DMR, RDS/stéréo FM, FEC (Hamming/RS/Viterbi), EKF, et les blocs de réduction de bruit
branchés dans `decoders/receiver.py` : **NoiseBlanker** (impulsions, avant le filtre étroit),
**LineEnhancer** NLMS (notch auto des porteuses ou "enhance"), **SpectralNR** (soustraction
spectrale Wiener / MMSE-LSA, estimation MCRA), **KalmanNR** et **NeuralNR** sur l'audio —
prêts à être branchés sur le flux IQ.

## Vérification (développement)

```bat
python tests/test_dsp.py
python tests/test_filters.py
python tests/test_ft8.py
python tests/test_decoders.py
python tests/test_radiolab_port.py
python tests/test_cf32_cat.py
```

Banc de tests pytest (nécessite `pip install pytest` ; aucun besoin de PySide6, un faux Qt
est injecté) — exécuté aussi en CI (`.github/workflows/tests.yml`) :
```bat
python -m pytest tests -q
python -m pytest tests_radiolab -q
```
(les lancer séparément : les deux dossiers contiennent des modules de même nom)

`test_dsp.py` teste hors-ligne les démodulateurs AM/NFM/USB/LSB/CW, le rééchantillonnage,
le spectre et le S-mètre sur des signaux synthétiques. `test_filters.py` vérifie les
filtres en flux (`filters.py`) : sortie identique morcelée ou d'un seul coup, DDC
polyphase, mixer à phase continue, resampling. `test_ft8.py` vérifie la chaîne
FT8 complète : message → CRC → LDPC → audio 8-FSK synthétisé → détection des tons →
bits → LDPC → CRC → message lisible. `test_decoders.py` valide les décodeurs : CRC-16/24
(trame Mode S officielle), CPR ADS-B (aller-retour), position AIS, BitSlicer, BCH/POCSAG,
Golay/DCS, CTCSS, notch AudioFX, trame ADS-B PPM synthétique (DF17 + ICAO), EKF, Hamming.
