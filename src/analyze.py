"""Analisi oggettiva (senza ground-truth umana).

Stabilità temporale: quante volte lo stato cambia da un frame al successivo,
confrontando l'euristica grezza frame-per-frame con le predizioni della GRU.
Un numero più basso per la GRU dimostra quantitativamente l'effetto di
regolarizzazione temporale (de-noising) del modello ricorrente.
"""

import argparse

import numpy as np
import torch

from src.model import PadelMultiHeadGRU

FPS = 30.0
WINDOW = 10


# --------------------------------------------------------------------------- #
#  Run di frame validi consecutivi                                            #
# --------------------------------------------------------------------------- #
def _valid_runs(valid):
    runs, i, n = [], 0, len(valid)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


# --------------------------------------------------------------------------- #
#  1) Stabilità temporale                                                     #
# --------------------------------------------------------------------------- #
def temporal_stability(cache, model, device="cpu"):
    feats, valid = cache["feats"], cache["valid"]
    labA, labB = cache["labA"], cache["labB"]
    runs = _valid_runs(valid)

    def flips(seq):
        seq = np.asarray(seq)
        return int(np.sum(seq[1:] != seq[:-1])), len(seq)

    tr_h = tr_g = frames = 0
    for a, b in runs:
        L = b - a
        if L < WINDOW:
            continue
        # predizioni GRU per t = a+9 .. b-1 (finestra piena)
        wins = np.stack([feats[a + t - WINDOW + 1: a + t + 1]
                         for t in range(WINDOW - 1, L)]).astype(np.float32)
        with torch.no_grad():
            la, lb = model(torch.from_numpy(wins).to(device))
        gA = la.argmax(1).cpu().numpy(); gB = lb.argmax(1).cpu().numpy()
        # euristica sugli stessi frame
        hA = labA[a + WINDOW - 1: b]; hB = labB[a + WINDOW - 1: b]

        for seq in (hA, hB):
            f, _ = flips(seq); tr_h += f
        for seq in (gA, gB):
            f, _ = flips(seq); tr_g += f
        frames += 2 * (L - WINDOW + 1)   # A e B

    minutes = frames / FPS / 60.0
    rate_h = tr_h / minutes if minutes else 0.0
    rate_g = tr_g / minutes if minutes else 0.0
    print("\n=== Stabilita' temporale (cambi di stato al minuto) ===")
    print(f"  Euristica grezza : {rate_h:6.1f} flip/min")
    print(f"  GRU (T=10)       : {rate_g:6.1f} flip/min")
    if rate_h:
        print(f"  Riduzione        : {100 * (1 - rate_g / rate_h):4.1f}%")
    return rate_h, rate_g


def _cli():
    p = argparse.ArgumentParser(description="Analisi oggettiva PADEL-ANALYZER")
    p.add_argument("--cache", default="data/cache/frames.npz")
    p.add_argument("--model", default="models/padel_gru.pt")
    a = p.parse_args()

    cache = np.load(a.cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location=device)
    hp = ck["hparams"]
    model = PadelMultiHeadGRU(hp["input_size"], hp["hidden_size"],
                              hp["num_layers"], hp["num_classes"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])

    temporal_stability(cache, model, device)


if __name__ == "__main__":
    _cli()
