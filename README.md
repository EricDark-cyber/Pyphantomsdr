# PyPhantom SDR

Récepteur SDR logiciel en **Python pur pour Windows**, inspiré de
[PhantomSDR-Plus](https://github.com/sv1btl/PhantomSDR-Plus) (spectre + waterfall
cliquable, zoom, modes AM/NFM/WFM/USB/LSB/CW, S-mètre en S-units/dBm, favoris de
fréquences, enregistrement IQ `.cf32` et audio `.wav`).

Architecture : **PySide6** (GUI Qt native Windows) + **SoapySDR** (accès au matériel :
RTL-SDR, HackRF, Airspy, etc.) + **numpy/scipy** (DSP) + **pyqtgraph** (spectre/waterfall)
+ **sounddevice** (audio).

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
│   ├── decoders/          # décodeurs numériques : ADS-B/ModeS+CPR, AIS(GMSK), Olivia/Contestia, ACARS, C4FM, DMR, POCSAG, CTCSS/DCS, RDS, FEC, EKF, AudioFX
│   └── gui/               # fenêtre principale, widget spectre/waterfall, thread d'acquisition
├── tests/test_dsp.py      # tests hors-ligne du DSP (signaux synthétiques)
├── tests/test_filters.py  # tests des filtres en flux (équivalence morcelé / d'un coup)
├── tests/test_ft8.py      # tests du décodeur FT8 (aller-retour audio 8-FSK complet)
├── tests/test_decoders.py # tests des décodeurs (CRC, CPR, AIS, POCSAG/DCS, CTCSS, ADS-B...)
└── tools/make_demo_iq.py  # génère un fichier IQ de démonstration
```

## Installation sous Windows

1. **Python 3.9** (requis par les bindings `_SoapySDR.pyd` de PothosSDR 2021, compilés
   cp39) : https://www.python.org/downloads/ — cochez « Add Python to PATH ».
   Vérifiez avec `py -3.9 --version` ; sinon installez Python 3.9.
2. **PothosSDR** (fournit `SoapySDR.dll` + les modules récepteurs) :
   téléchargez l'installeur sur https://downloads.pothosware.com/ et installez-le,
   puis ajoutez `C:\Program Files\PothosSDR\bin` à la variable d'environnement **PATH**
   (Paramètres système → Variables d'environnement).
3. **Pilote du dongle** : avec [Zadig](https://zadig.akeo.ie/), remplacez le pilote USB
   du RTL-SDR (etc.) par **WinUSB**.
4. Dans un terminal (`Win + R`, `cmd`), créez l'environnement **Python 3.9** et installez
   les dépendances :
   ```bat
   py -3.9 -m venv .venv39
   .venv39\Scripts\activate
   python -m pip install -r requirements.txt
   ```
   (Gardez `.venv39` en dehors du dossier téléchargé depuis GitHub : le ZIP de la branche
   est remplacé à chaque téléchargement.)
   **Ne faites PAS `pip install SoapySDR`** : le paquet PyPI n'a pas de roue Windows.
   Les bindings Python de SoapySDR viennent de PothosSDR. Deux possibilités :
   - Si l'installeur PothosSDR a enregistré ses bindings (c'est le cas des versions
     récentes), testez directement :
     ```bat
     python -c "import SoapySDR; print(SoapySDR.Device.enumerate())"
     ```
   - Sinon, pointez `PYTHONPATH` vers les bindings de PothosSDR (chez vous, vérifiez avec
     `dir "C:\Program Files\PothosSDR\lib" /b` : le dossier s'appelle `python3.9`, d'où
     l'obligation d'un venv Python 3.9) :
     ```bat
     set PATH=C:\Program Files\PothosSDR\bin;%PATH%
     set PYTHONPATH=C:\Program Files\PothosSDR\lib\python3.9\site-packages;%PYTHONPATH%
     python -c "import SoapySDR; print(SoapySDR.Device.enumerate())"
     ```
     L'application détecte aussi automatiquement ces chemins : lancez simplement
     `python -m pyphantom` depuis `.venv39`, sans variable d'environnement.
     L'Airspy doit apparaître avec `driver=airspy` (sinon refaire le pilote WinUSB avec Zadig).
5. Lancez :
   ```bat
   python -m pyphantom
   ```

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

Le pilote `airspy` est fourni par PothosSDR (étape 2). Le récepteur apparaît dans la
liste avec `driver=airspy`.

## Test sans récepteur (fichier IQ)

```bat
python tools/make_demo_iq.py demo.cf32 10 1000000
python -m pyphantom --iq demo.cf32 --rate 1000000
```

Le fichier `demo.cf32` contient des stations synthétiques : AM à −150 kHz, FM à
+200 kHz, USB à +60 kHz par rapport à la fréquence centrale (100 MHz par défaut).

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
DMR, RDS/stéréo FM, FEC (Hamming/RS/Viterbi), EKF — prête à être branchée sur le flux IQ.

## Vérification (développement)

```bat
python tests/test_dsp.py
python tests/test_filters.py
python tests/test_ft8.py
python tests/test_decoders.py
```

`test_dsp.py` teste hors-ligne les démodulateurs AM/NFM/USB/LSB/CW, le rééchantillonnage,
le spectre et le S-mètre sur des signaux synthétiques. `test_filters.py` vérifie les
filtres en flux (`filters.py`) : sortie identique morcelée ou d'un seul coup, DDC
polyphase, mixer à phase continue, resampling. `test_ft8.py` vérifie la chaîne
FT8 complète : message → CRC → LDPC → audio 8-FSK synthétisé → détection des tons →
bits → LDPC → CRC → message lisible. `test_decoders.py` valide les décodeurs : CRC-16/24
(trame Mode S officielle), CPR ADS-B (aller-retour), position AIS, BitSlicer, BCH/POCSAG,
Golay/DCS, CTCSS, notch AudioFX, trame ADS-B PPM synthétique (DF17 + ICAO), EKF, Hamming.
