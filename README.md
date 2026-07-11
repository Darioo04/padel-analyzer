# PADEL-ANALYZER

Sistema di *Match Analytics* per il padel. Traccia i 4 giocatori con YOLO, li
proietta su un campo 2D (Bird's-Eye View) tramite omografia e classifica la
dinamica tattica delle due squadre nel tempo con una **GRU multi-head** (una
testa indipendente per squadra).

Progetto finale del corso **AI-LAB** — ACSAI, Sapienza Università di Roma.

## Struttura

```
padel_analyzer/
├── data/                         # video delle partite (.mp4) + cache estratte
│   └── cache/                    # cache .npz (una per video), generate a runtime
├── calibration.json              # omografia (4 angoli del campo)
├── requirements.txt
├── main_demo.py                  # DEMO d'esame (video integrale + report finali)
├── src/
│   ├── court.py                  # geometria, omografia, etichettatura, mini-mappa
│   ├── calibrate.py              # calibrazione interattiva dell'omografia
│   ├── dataset.py                # estrazione YOLO + cache + Dataset temporale
│   ├── model.py                  # PadelMultiHeadGRU
│   ├── train.py                  # training multi-task (uno o più video) + grafici
│   ├── analytics.py              # logica tattica a 3 categorie + report
│   ├── analyze.py                # analisi oggettiva (stabilità temporale)
│   ├── ball_annotate.py          # (opzionale) annotazione pallina → dataset YOLO
│   ├── ball_train.py             # (opzionale) training del detector pallina
│   └── ball_track.py             # (opzionale) tracking pallina a video
└── AI_LAB_Report_Template_ACSAI_2025_2026.tex   # relazione (paper scientifico)
```

## Installazione

```bash
pip install -r requirements.txt
```

I pesi `yolov8m.pt` sono già inclusi; in caso di assenza vengono scaricati
automaticamente da Ultralytics alla prima esecuzione.

## Modello degli stati e situazioni tattiche

Per ogni squadra la posizione della coppia è codificata in **3 stati**:
`0 = FONDO`, `1 = RETE (attacco)`, `2 = DISALLINEATA` (i due giocatori sono
sfilacciati in profondità, oltre `SPLIT_DIFF = 4 m`).

Ogni stato mappa direttamente su **una** delle 3 situazioni tattiche, quindi i
due report (uno per squadra) sommano esattamente al 100%:

| Stato | Situazione tattica |
|-------|--------------------|
| 1 RETE | **Attacco** |
| 0 FONDO | **Difesa** |
| 2 DISALLINEATA | **Disallineamento** |

---

## Come testare tutto da zero

Tutti i comandi vanno eseguiti dalla cartella `padel_analyzer/`.

Prepara i video in `data/`. In questo progetto si usano **tre set**:

- `data/match_video.mp4`, `data/match_video_2.mp4` → usati per l'**addestramento**;
- `data/demo.mp4` → terzo set, tenuto da parte come **demo** (mai visto in training).

> I nomi dei file sono liberi: adatta i comandi qui sotto ai tuoi.

### 1. Calibrazione dell'omografia

L'omografia dipende dalla telecamera. Due casi:

- **Stessa telecamera** (es. set diversi della stessa partita): basta **una**
  `calibration.json`, valida per tutti i video.
- **Telecamere diverse** (partite diverse): calibra **ogni** video a parte,
  salvando file distinti.

Per (ri)calibrare, clicca i 4 angoli del campo nell'ordine indicato a schermo
(vicino-sx, vicino-dx, lontano-dx, lontano-sx):

```bash
python -m src.calibrate --video data/match_video.mp4 --time 30 --out calibration.json
# telecamere diverse:
python -m src.calibrate --video data/demo.mp4 --time 30 --out calibration_demo.json
```

### 2. Estrazione delle traiettorie (YOLO → cache)

YOLO gira una sola volta per video; il risultato è salvato in cache `.npz`. È lo
stadio più lento (usa la GPU se disponibile).

```bash
python -m src.dataset --video data/match_video.mp4   --calib calibration.json --out data/cache/set1.npz
python -m src.dataset --video data/match_video_2.mp4 --calib calibration.json --out data/cache/set2.npz
```

Suggerimento per un test rapido: `--frame-stride 3` processa 1 frame su 3;
`--max-frames 2000` limita il numero di frame.

### 3. Addestramento della GRU multi-head

Si addestra su **una o più** cache. Ogni partita è divisa temporalmente
(70/15/15) al proprio interno e i tre split sono poi concatenati, così ciascuna
partita contribuisce a train/val/test senza leakage tra finestre sovrapposte.

```bash
python -m src.train --caches data/cache/set1.npz data/cache/set2.npz --epochs 40
```

Produce `models/padel_gru.pt`, `loss_plot.png`, `confusion_matrix.png` e stampa
i `classification_report` (accuratezza e F1 per testa) sul test set.

### 4. Analisi oggettiva (per la relazione)

```bash
python -m src.analyze --cache data/cache/set2.npz --model models/padel_gru.pt
```

Stampa la **stabilità temporale**: la GRU riduce i cambi di stato spuri rispetto
all'euristica grezza (effetto di regolarizzazione temporale).

### 5. Demo d'esame (video integrale)

Elabora il **video demo per intero** (nessun `--start`/`--duration`), mostra
mini-mappa e banner tattico, e stampa i due report finali. Con `--save` salva
una copia annotata come artefatto d'esame:

```bash
# a schermo, interattivo (SPAZIO pausa | n avanti | q esci):
python main_demo.py --video data/demo.mp4 --calib calibration.json --model models/padel_gru.pt

# elaborazione integrale senza finestra, salvando il video annotato:
python main_demo.py --video data/demo.mp4 --calib calibration.json \
    --model models/padel_gru.pt --save output_demo.mp4 --no-show
```

Per un'anteprima veloce di un solo tratto:
`python main_demo.py --video data/demo.mp4 --start 120 --duration 60`.

### 6. Tracking della pallina a video

Un **detector dedicato** (addestrato sui frame annotati a mano) localizza la
pallina e un tracker ne segue la traiettoria disegnandola a video (cerchio
giallo). È **integrato nella demo**: se esiste `models/ball_yolo.pt` viene usato
automaticamente da `main_demo.py` (si disattiva con `--no-ball`).

```bash
# 6a. Annota la pallina (lente d'ingrandimento; ~200-300 frame + qualche negativo):
python -m src.ball_annotate --video data/match_video.mp4 --cache data/cache/set1.npz --n 300

# 6b. Addestra il detector (imgsz alto, indispensabile per un oggetto piccolo):
python -m src.ball_train --data data/ball_ds --imgsz 1280 --epochs 100
#    -> produce models/ball_yolo.pt

# 6c. Tester visivo su un tratto (solo traccia):
python -m src.ball_track --video data/demo.mp4 --start 120 --duration 30

# 6d. Nella demo è già attivo di default (basta che models/ball_yolo.pt esista):
python main_demo.py --video data/demo.mp4 --save output_demo.mp4 --no-show
```

> Il detector è **specifico per la telecamera/campo** su cui è stato annotato:
> per un video di un'altra partita conviene annotare e riaddestrare.
