"""Annotazione della pallina per addestrare un detector dedicato.

Estrae un campione di frame dal video e chiede di cliccare il centro della
pallina; salva un dataset in formato YOLO (una classe: `ball`). Una lente
d'ingrandimento aiuta a individuare la pallina, che occupa pochi pixel.

Per avere piu' frame con la pallina in campo, si puo' campionare dai soli frame
"validi" (4 giocatori in campo) usando la cache dell'estrazione.

Uso:
    python -m src.ball_annotate --video data/match_video.mp4 --cache data/cache/set1.npz --n 300 --out data/ball_ds

Comandi (finestra):
    click sinistro = centro pallina      [ / ] = riduci / ingrandisci il box
    SPAZIO o n     = conferma e vai avanti (senza click = frame senza pallina)
    c              = cancella il punto           b = frame precedente
    s              = salta il frame (non salvarlo)
    q              = salva ed esci
"""

import argparse
import os

import cv2
import numpy as np

BOX_DEFAULT = 12          # semi-lato del box in pixel (box 24x24)
ZOOM = 4                  # fattore della lente
ZOOM_SRC = 40             # semi-lato (px) della regione ingrandita


def _sample_indices(video, n, cache_path, seed):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    rng = np.random.default_rng(seed)
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path)
        pool = d["frame_idx"][d["valid"]]
        pool = pool[pool < total]
    else:
        pool = np.arange(total)
    n = min(n, len(pool))
    idx = np.sort(rng.choice(pool, size=n, replace=False))
    return [int(i) for i in idx]


def _draw_overlay(frame, pt, half, mouse):
    disp = frame.copy()
    h, w = disp.shape[:2]
    # barra istruzioni
    cv2.rectangle(disp, (0, 0), (w, 30), (20, 20, 20), -1)
    cv2.putText(disp, "click=pallina  [ ]=box  SPAZIO=avanti  c=cancella  "
                "b=indietro  s=salta  q=esci", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    if pt is not None:
        cv2.rectangle(disp, (pt[0] - half, pt[1] - half),
                      (pt[0] + half, pt[1] + half), (0, 0, 255), 1)
        cv2.drawMarker(disp, pt, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
    # lente d'ingrandimento nell'angolo in alto a destra
    if mouse is not None:
        mx, my = mouse
        x0, y0 = max(0, mx - ZOOM_SRC), max(0, my - ZOOM_SRC)
        x1, y1 = min(w, mx + ZOOM_SRC), min(h, my + ZOOM_SRC)
        crop = disp[y0:y1, x0:x1]
        if crop.size:
            zoom = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM,
                              interpolation=cv2.INTER_NEAREST)
            zh, zw = zoom.shape[:2]
            px, py = w - zw - 10, 40
            cv2.rectangle(zoom, (0, 0), (zw - 1, zh - 1), (0, 255, 255), 1)
            cv2.drawMarker(zoom, ((mx - x0) * ZOOM, (my - y0) * ZOOM),
                           (0, 0, 255), cv2.MARKER_CROSS, 14, 1)
            disp[py:py + zh, px:px + zw] = zoom
    return disp


def annotate(video, out, n, cache_path, seed, redo):
    img_dir = os.path.join(out, "images")
    lbl_dir = os.path.join(out, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    idxs = _sample_indices(video, n, cache_path, seed)
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    win = "Annotazione pallina"
    cv2.namedWindow(win)
    state = {"pt": None, "half": BOX_DEFAULT, "mouse": None}

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_MOUSEMOVE:
            state["mouse"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)
    cv2.setMouseCallback(win, on_mouse)

    def name(i):
        return f"frame_{i:07d}"

    i = 0
    saved = 0
    while 0 <= i < len(idxs):
        fidx = idxs[i]
        lbl_path = os.path.join(lbl_dir, name(fidx) + ".txt")
        if os.path.exists(lbl_path) and not redo:
            i += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            i += 1
            continue
        state["pt"] = None
        state["half"] = BOX_DEFAULT

        while True:
            disp = _draw_overlay(frame, state["pt"], state["half"], state["mouse"])
            cv2.putText(disp, f"[{i + 1}/{len(idxs)}] frame {fidx}  salvati:{saved}",
                        (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, disp)
            k = cv2.waitKey(15) & 0xFF
            if k == ord("q"):
                cap.release(); cv2.destroyAllWindows()
                print(f"\nSalvati {saved} frame annotati in {out}")
                return
            if k == ord("["):
                state["half"] = max(4, state["half"] - 2)
            elif k == ord("]"):
                state["half"] = min(60, state["half"] + 2)
            elif k == ord("c"):
                state["pt"] = None
            elif k == ord("s"):                     # salta: non salva nulla
                i += 1
                break
            elif k == ord("b"):                     # indietro
                i = max(0, i - 1)
                break
            elif k in (ord(" "), ord("n"), 13, 10):  # conferma e salva
                cv2.imwrite(os.path.join(img_dir, name(fidx) + ".jpg"), frame)
                if state["pt"] is not None:
                    cx, cy = state["pt"]
                    hw = state["half"]
                    line = f"0 {cx / W:.6f} {cy / H:.6f} {2*hw / W:.6f} {2*hw / H:.6f}\n"
                    open(lbl_path, "w").write(line)
                else:                                # nessuna pallina = negativo
                    open(lbl_path, "w").write("")
                saved += 1
                i += 1
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCompletato: {saved} frame annotati in {out}")


def _cli():
    p = argparse.ArgumentParser(description="Annotazione pallina (dataset YOLO)")
    p.add_argument("--video", default="data/match_video.mp4")
    p.add_argument("--cache", default="data/cache/set1.npz",
                   help="campiona dai frame validi di questa cache (opzionale)")
    p.add_argument("--out", default="data/ball_ds")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--redo", action="store_true", help="riannota anche i frame gia' fatti")
    a = p.parse_args()
    annotate(a.video, a.out, a.n, a.cache, a.seed, a.redo)


if __name__ == "__main__":
    _cli()
