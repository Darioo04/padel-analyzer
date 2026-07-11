"""Addestramento Multi-Task della GRU multi-head.

La perdita totale è la somma delle cross-entropy delle due teste:

        loss_total = loss_A + loss_B

Ottimizzazione con Adam. Vengono prodotti:
  * models/padel_gru.pt          -- pesi del modello + iperparametri
  * loss_plot.png                -- curve di apprendimento (loss e accuracy)
  * confusion_matrix.png         -- matrici di confusione (test) per A e B

Si può addestrare su UNA o PIÙ partite: ogni video viene estratto in una cache
separata e passato con --caches. Le finestre di ogni partita vengono divise
temporalmente (70/15/15) al proprio interno e poi concatenate, così nessuna
partita finisce interamente in un solo split e non c'è leakage tra finestre
sovrapposte.

Uso:
    python -m src.train --caches data/cache/set1.npz data/cache/set2.npz --epochs 40
    python -m src.train --caches data/cache/frames.npz          # un solo video
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.court import STATE_NAMES
from src.dataset import DEFAULT_STRIDE, WINDOW, PadelTacticsDataset, build_windows
from src.model import PadelMultiHeadGRU

CLASS_LABELS = [STATE_NAMES[i] for i in range(3)]


# --------------------------------------------------------------------------- #
#  Split temporale (evita leakage tra finestre sovrapposte)                    #
# --------------------------------------------------------------------------- #
def _split_one(cache_path, window, stride, ratios=(0.70, 0.15, 0.15)):
    """Finestre di UNA partita, divise in ordine temporale (frame_idx)."""
    data = np.load(cache_path)
    X, yA, yB, start = build_windows(data["feats"], data["valid"],
                                     data["labA"], data["labB"],
                                     window=window, stride=stride)
    if len(X) == 0:
        raise RuntimeError(f"Nessuna finestra valida in {cache_path}: controlla la calibrazione e riesegui l'estrazione (src.dataset).")
    order = np.argsort(start)
    X, yA, yB = X[order], yA[order], yB[order]
    n = len(X)
    n_tr = int(ratios[0] * n)
    n_va = int((ratios[0] + ratios[1]) * n)
    return {
        "train": (X[:n_tr], yA[:n_tr], yB[:n_tr]),
        "val": (X[n_tr:n_va], yA[n_tr:n_va], yB[n_tr:n_va]),
        "test": (X[n_va:], yA[n_va:], yB[n_va:]),
    }


def temporal_split(cache_paths, window, stride, ratios=(0.70, 0.15, 0.15)):
    """Costruisce e divide le finestre di una o più partite.

    Ogni partita è divisa temporalmente al proprio interno; i tre split vengono
    poi concatenati fra le partite. In questo modo ciascuna partita contribuisce
    a train, val e test, e le finestre sovrapposte (che condividono frame)
    restano sempre nello stesso split.
    """
    if isinstance(cache_paths, (str, bytes)):
        cache_paths = [cache_paths]
    parts = [_split_one(p, window, stride, ratios) for p in cache_paths]

    sets = {}
    for k in ("train", "val", "test"):
        X = np.concatenate([p[k][0] for p in parts], axis=0)
        yA = np.concatenate([p[k][1] for p in parts], axis=0)
        yB = np.concatenate([p[k][2] for p in parts], axis=0)
        sets[k] = PadelTacticsDataset(X=X, yA=yA, yB=yB, window=window)
    return sets


def class_weights(y, n_classes=3):
    """Pesi per il bilanciamento della CE.

    Si usa la RADICE dell'inverso della frequenza (invece dell'inverso puro):
    attenua la spinta sulle classi molto rare -- come lo stato disallineato
    della squadra lontana -- evitando che vengano sovra-predette. I pesi sono
    normalizzati a media unitaria.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = np.sqrt(counts.sum() / (n_classes * counts))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  Loop di addestramento                                                       #
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, crit_a, crit_b, device, opt=None):
    train = opt is not None
    model.train(train)
    tot_loss = ca = cb = seen = 0
    for X, ya, yb in loader:
        X, ya, yb = X.to(device), ya.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            la, lb = model(X)
            loss = crit_a(la, ya) + crit_b(lb, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        bs = X.size(0)
        tot_loss += loss.item() * bs
        ca += (la.argmax(1) == ya).sum().item()
        cb += (lb.argmax(1) == yb).sum().item()
        seen += bs
    return tot_loss / seen, ca / seen, cb / seen


@torch.no_grad()
def collect_preds(model, loader, device):
    model.eval()
    ya_t, yb_t, ya_p, yb_p = [], [], [], []
    for X, ya, yb in loader:
        la, lb = model(X.to(device))
        ya_t.append(ya.numpy()); yb_t.append(yb.numpy())
        ya_p.append(la.argmax(1).cpu().numpy()); yb_p.append(lb.argmax(1).cpu().numpy())
    return (np.concatenate(ya_t), np.concatenate(ya_p),
            np.concatenate(yb_t), np.concatenate(yb_p))


def plot_curves(hist, path):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ep, hist["train_loss"], label="train")
    ax1.plot(ep, hist["val_loss"], label="val")
    ax1.set_title("Loss totale (A + B)"); ax1.set_xlabel("Epoca"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, hist["val_accA"], label="val acc A")
    ax2.plot(ep, hist["val_accB"], label="val acc B")
    ax2.set_title("Accuratezza di validazione")
    ax2.set_xlabel("Epoca"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  Curve di apprendimento -> {path}")


def plot_confusion(yat, yap, ybt, ybp, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, yt, yp, title in [(axes[0], yat, yap, "Squadra A"),
                              (axes[1], ybt, ybp, "Squadra B")]:
        cm = confusion_matrix(yt, yp, labels=[0, 1, 2])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Matrice di confusione -- {title}")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_LABELS, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_LABELS)
        ax.set_xlabel("Predetto"); ax.set_ylabel("Reale")
        thr = cm.max() / 2 if cm.max() else 0
        for r in range(3):
            for c in range(3):
                ax.text(c, r, int(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  Matrici di confusione  -> {path}")


def train(cache_paths, epochs=40, batch_size=128, lr=1e-3, hidden=64,
          layers=1, stride=DEFAULT_STRIDE, window=WINDOW, out_dir="models",
          seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if isinstance(cache_paths, (str, bytes)):
        cache_paths = [cache_paths]
    print(f"Partite (cache): {', '.join(cache_paths)}")

    sets = temporal_split(cache_paths, window, stride)
    print(f"Finestre  train/val/test: "
          f"{len(sets['train'])}/{len(sets['val'])}/{len(sets['test'])}")

    loaders = {k: DataLoader(v, batch_size=batch_size, shuffle=(k == "train"))
               for k, v in sets.items()}

    wa = class_weights(sets["train"].yA).to(device)
    wb = class_weights(sets["train"].yB).to(device)
    crit_a = nn.CrossEntropyLoss(weight=wa)
    crit_b = nn.CrossEntropyLoss(weight=wb)

    model = PadelMultiHeadGRU(input_size=8, hidden_size=hidden,
                              num_layers=layers, num_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {k: [] for k in
            ["train_loss", "val_loss", "val_accA", "val_accB"]}
    best_val, best_state = float("inf"), None

    for ep in range(1, epochs + 1):
        tl, _, _ = run_epoch(model, loaders["train"], crit_a, crit_b, device, opt)
        vl, va, vb = run_epoch(model, loaders["val"], crit_a, crit_b, device)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl)
        hist["val_accA"].append(va); hist["val_accB"].append(vb)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoca {ep:03d} | train_loss {tl:.4f} | val_loss {vl:.4f} "
              f"| val_acc A {va:.3f} B {vb:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Valutazione finale sul test ----
    yat, yap, ybt, ybp = collect_preds(model, loaders["test"], device)
    print("\n=== Squadra A (test) ===")
    print(classification_report(yat, yap, labels=[0, 1, 2],
                                target_names=CLASS_LABELS, zero_division=0))
    print("=== Squadra B (test) ===")
    print(classification_report(ybt, ybp, labels=[0, 1, 2],
                                target_names=CLASS_LABELS, zero_division=0))

    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "hparams": {"input_size": 8, "hidden_size": hidden,
                    "num_layers": layers, "num_classes": 3, "window": window},
    }
    torch.save(ckpt, os.path.join(out_dir, "padel_gru.pt"))
    print(f"\nModello salvato -> {os.path.join(out_dir, 'padel_gru.pt')}")

    plot_curves(hist, "loss_plot.png")
    plot_confusion(yat, yap, ybt, ybp, "confusion_matrix.png")
    return model


def _cli():
    p = argparse.ArgumentParser(description="Training PADEL-ANALYZER")
    p.add_argument("--caches", nargs="+", default=["data/cache/frames.npz"],
                   help="una o più cache .npz (un video ciascuna)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    a = p.parse_args()
    train(a.caches, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
          hidden=a.hidden, layers=a.layers, stride=a.stride)


if __name__ == "__main__":
    _cli()
