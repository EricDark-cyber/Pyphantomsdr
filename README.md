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
│   └── gui/               # ancienne interface (bibliothèque pyphantom conservée)
├── tests/test_dsp.py      # tests hors-ligne du DSP (signaux synthétiques)
├── tests/test_filters.py  # tests des filtres en flux (équivalence morcelé / d'un coup)
├── tests/test_ft8.py      # tests du décodeur FT8 (aller-retour audio 8-FSK complet)
├── tests/test_decoders.py # tests des décodeurs (CRC, CPR, AIS, POCSAG/DCS, CTCSS, ADS-B...)
└── tools/make_demo_iq.py  # génère un fichier IQ de démonstration
```

## Installation sous Windows

1. **Python 3.10 ou plus récent, 64 bits** (python.org) — cochez « Add python.exe to PATH ».
   Vérifiez avec `python --version`.
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
```

## Utilisation

- **START** : ouvre la source sélectionnée (récepteur listé, ou fichier IQ via « Fichier IQ… »).
- **Clic sur le spectre/waterfall** : accorde la fréquence. **Molette** : zoom (curseur ×1…×64).
- **Mode / Bande passante** : AM, NFM, WFM, USB, LSB, CW (BFO 700 Hz).
- **Gain RF + AGC**, **volume**, **muet**, **squelch**, **calibration dBm** (pour le S-mètre).
- **● Rec IQ** : enregistre le flux brut en `.cf32` (+ métadonnées `.json`).
- **● Rec audio** : enregistre le signal démodulé en `.wav`.
- **Favoris** (panneau de droite) : double-clic pour accorder ; bouton « Ajouter la fréquence ».

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
```

Banc de tests pytest du port RadioLab 0.15 (nécessite `pip install pytest`) :
```bat
python -m pytest tests_radiolab -q
```

`test_dsp.py` teste hors-ligne les démodulateurs AM/NFM/USB/LSB/CW, le rééchantillonnage,
le spectre et le S-mètre sur des signaux synthétiques. `test_filters.py` vérifie les
filtres en flux (`filters.py`) : sortie identique morcelée ou d'un seul coup, DDC
polyphase, mixer à phase continue, resampling. `test_ft8.py` vérifie la chaîne
FT8 complète : message → CRC → LDPC → audio 8-FSK synthétisé → détection des tons →
bits → LDPC → CRC → message lisible. `test_decoders.py` valide les décodeurs : CRC-16/24
(trame Mode S officielle), CPR ADS-B (aller-retour), position AIS, BitSlicer, BCH/POCSAG,
Golay/DCS, CTCSS, notch AudioFX, trame ADS-B PPM synthétique (DF17 + ICAO), EKF, Hamming.
