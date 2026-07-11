"""Addestra un detector della pallina (YOLOv8n) sui frame annotati.

Legge il dataset prodotto da `src.ball_annotate` (formato YOLO: images/ +
labels/), lo divide in train/val e riaddestra YOLOv8n. La pallina occupa pochi
pixel: si allena ad alta risoluzione (`--imgsz 1280`), fondamentale per gli
oggetti piccoli.

Uso:
    python -m src.ball_train --data data/ball_ds --imgsz 1280 --epochs 100

Produce models/ball_yolo.pt (pesi migliori).
"""

import argparse
import glob
import os
import shutil

import numpy as np


def _prepare_split(data_dir, val_frac, seed):
    img_dir = os.path.join(data_dir, "images")
    lbl_dir = os.path.join(data_dir, "labels")
    labels = glob.glob(os.path.join(lbl_dir, "*.txt"))
    items, npos = [], 0
    for lp in labels:
        stem = os.path.splitext(os.path.basename(lp))[0]
        ip = os.path.join(img_dir, stem + ".jpg")
        if not os.path.exists(ip):
            continue
        items.append(os.path.abspath(ip))
        if os.path.getsize(lp) > 0:
            npos += 1
    if not items:
        raise RuntimeError(f"Nessun frame annotato in {data_dir} (esegui src.ball_annotate).")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    nval = max(1, int(val_frac * len(items)))
    val_idx = set(order[:nval].tolist())
    train = [items[i] for i in range(len(items)) if i not in val_idx]
    val = [items[i] for i in range(len(items)) if i in val_idx]

    with open(os.path.join(data_dir, "train.txt"), "w") as f:
        f.write("\n".join(train))
    with open(os.path.join(data_dir, "val.txt"), "w") as f:
        f.write("\n".join(val))

    yaml_path = os.path.join(data_dir, "ball.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(data_dir)}\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("names:\n  0: ball\n")
    print(f"Dataset: {len(items)} frame ({npos} con pallina, {len(items)-npos} negativi) "
          f"-> train {len(train)} / val {len(val)}")
    return yaml_path


def train(data_dir, imgsz, epochs, batch, seed, val_frac=0.15,
          out="models/ball_yolo.pt"):
    from ultralytics import YOLO

    yaml_path = _prepare_split(data_dir, val_frac, seed)
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=yaml_path, imgsz=imgsz, epochs=epochs, batch=batch,
        patience=30, seed=seed, project="runs_ball", name="detector",
        exist_ok=True, pretrained=True,
    )
    best = os.path.join(results.save_dir, "weights", "best.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copyfile(best, out)
    print(f"\nDetector salvato -> {out}")
    print(f"(metriche e grafici in {results.save_dir})")
    return out


def _cli():
    p = argparse.ArgumentParser(description="Training detector pallina (YOLOv8n)")
    p.add_argument("--data", default="data/ball_ds")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8, help="riduci se la GPU va in OOM")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    train(a.data, a.imgsz, a.epochs, a.batch, a.seed)


if __name__ == "__main__":
    _cli()
