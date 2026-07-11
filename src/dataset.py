"""Estrazione delle traiettorie (YOLO + omografia) e Dataset temporale.

Pipeline a due stadi:

  1. `extract_to_cache`  -- costoso, eseguito UNA sola volta.
     Scorre il video, rileva le persone con YOLO, proietta i punti-piede sul
     campo, scarta folla/arbitro/replay (tenendo solo i frame con esattamente
     4 giocatori dentro i confini, 2 per metà), applica l'auto-labeling
     geometrico e salva tutto in un file .npz.

  2. `PadelTacticsDataset` -- economico, usato dal training.
     Legge la cache e raggruppa i frame validi CONSECUTIVI in finestre mobili
     di 10 frame. Un salto (frame scartato) interrompe la continuità temporale.

"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.court import (build_sample, foot_point, load_calibration, project,
                       select_players)

WINDOW = 10          # lunghezza della finestra temporale (frame)
DEFAULT_STRIDE = 3   # passo tra finestre consecutive (training)


# --------------------------------------------------------------------------- #
#  Stadio 1: estrazione e cache                                                #
# --------------------------------------------------------------------------- #
def extract_to_cache(video_path, calib_path, out_path,
                     model_name="yolov8m.pt", conf=0.30,
                     frame_stride=1, max_frames=None, device=None):
    """Estrae le traiettorie dal video e salva la cache .npz.

    Ogni record corrisponde a un frame processato e contiene:
        frame_idx, valid, feats[8], labA, labB.
    I frame non validi (valid=False) restano nella cache come "buchi" che
    spezzano la continuità temporale delle finestre.
    """
    import cv2
    from ultralytics import YOLO
    from tqdm import tqdm

    cal = load_calibration(calib_path)
    H = cal["H"]
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire il video: {video_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx, feats, valids, labA, labB = [], [], [], [], []
    idx, processed = 0, 0
    pbar = tqdm(total=n_total, desc="Estrazione YOLO", unit="f")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        if idx % frame_stride != 0:
            idx += 1
            continue

        res = model.predict(frame, classes=[0], conf=conf, verbose=False,
                            device=device)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        feet = [foot_point(b) for b in boxes]
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        court = project(H, feet)                        # tutte le persone -> metri
        sample = build_sample(select_players(court, areas))  # scarta folla, tiene 2 vs 2

        frame_idx.append(idx)
        if sample is None:
            valids.append(False)
            feats.append(np.full(8, np.nan, dtype=np.float32))
            labA.append(-1)
            labB.append(-1)
        else:
            feat, la, lb, _, _ = sample
            valids.append(True)
            feats.append(feat)
            labA.append(la)
            labB.append(lb)

        idx += 1
        processed += 1
        if max_frames is not None and processed >= max_frames:
            break

    pbar.close()
    cap.release()

    frame_idx = np.array(frame_idx, dtype=np.int64)
    feats = np.stack(feats).astype(np.float32)
    valids = np.array(valids, dtype=bool)
    labA = np.array(labA, dtype=np.int64)
    labB = np.array(labB, dtype=np.int64)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, frame_idx=frame_idx, feats=feats,
                        valid=valids, labA=labA, labB=labB)

    n_valid = int(valids.sum())
    print(f"\nCache salvata in {out_path}")
    print(f"  Frame processati : {len(valids)}")
    print(f"  Frame validi     : {n_valid} ({100.0 * n_valid / max(1, len(valids)):.1f}%)")
    print(f"  Frame scartati   : {len(valids) - n_valid} (replay / <4 giocatori / folla)")
    return out_path


# --------------------------------------------------------------------------- #
#  Windowing                                                                   #
# --------------------------------------------------------------------------- #
def build_windows(feats, valid, labA, labB, window=WINDOW, stride=DEFAULT_STRIDE):
    """Raggruppa i frame validi CONSECUTIVI in finestre mobili.

    Ritorna (X[N,window,8], yA[N], yB[N], start[N]) dove `start` è l'indice del
    primo frame di ogni finestra (usato per lo split temporale train/val/test).
    L'etichetta della finestra è lo stato all'ULTIMO istante (predizione causale).
    """
    X, yA, yB, start = [], [], [], []
    n = len(valid)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        # estende il run di frame validi consecutivi
        j = i
        while j < n and valid[j]:
            j += 1
        run = list(range(i, j))               # [i, j)
        for s in range(0, len(run) - window + 1, stride):
            idxs = run[s:s + window]
            X.append(feats[idxs])
            yA.append(labA[idxs[-1]])
            yB.append(labB[idxs[-1]])
            start.append(idxs[0])
        i = j

    if not X:
        return (np.empty((0, window, 8), np.float32), np.empty(0, np.int64),
                np.empty(0, np.int64), np.empty(0, np.int64))
    return (np.stack(X).astype(np.float32), np.array(yA, np.int64),
            np.array(yB, np.int64), np.array(start, np.int64))


# --------------------------------------------------------------------------- #
#  Stadio 2: Dataset PyTorch                                                   #
# --------------------------------------------------------------------------- #
class PadelTacticsDataset(Dataset):
    """Dataset di finestre temporali tattiche costruito dalla cache .npz."""

    def __init__(self, cache_path=None, X=None, yA=None, yB=None,
                 window=WINDOW, stride=DEFAULT_STRIDE):
        if X is not None:
            self.X, self.yA, self.yB = X, yA, yB
            self.start = np.arange(len(X))
        else:
            data = np.load(cache_path)
            self.X, self.yA, self.yB, self.start = build_windows(
                data["feats"], data["valid"], data["labA"], data["labB"],
                window=window, stride=stride)
        self.window = window

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]),
                torch.tensor(self.yA[i]),
                torch.tensor(self.yB[i]))

    def class_counts(self):
        """Distribuzione delle classi per le due teste (utile per i pesi CE)."""
        ca = np.bincount(self.yA, minlength=3)
        cb = np.bincount(self.yB, minlength=3)
        return ca, cb


def _cli():
    p = argparse.ArgumentParser(description="Estrazione traiettorie PADEL-ANALYZER")
    p.add_argument("--video", default="data/match_video.mp4")
    p.add_argument("--calib", default="calibration.json")
    p.add_argument("--out", default="data/cache/frames.npz")
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--frame-stride", type=int, default=1,
                   help="processa 1 frame ogni N (1 = tutti)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="limita il numero di frame processati (debug)")
    p.add_argument("--device", default=None, help="'0' per GPU, 'cpu' per CPU")
    a = p.parse_args()
    extract_to_cache(a.video, a.calib, a.out, model_name=a.model, conf=a.conf,
                     frame_stride=a.frame_stride, max_frames=a.max_frames,
                     device=a.device)


if __name__ == "__main__":
    _cli()
