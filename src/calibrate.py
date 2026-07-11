"""Calibrazione interattiva dell'omografia.

La telecamera è fissa, quindi l'omografia va stimata UNA sola volta. Questo
strumento estrae un frame dal video e chiede all'utente di cliccare i 4 angoli
del campo da gioco, salvando le corrispondenze in calibration.json.

Ordine dei clic (IMPORTANTE, seguire esattamente):
    1. angolo VICINO-SINISTRA   -> (0, 0)  metri
    2. angolo VICINO-DESTRA     -> (10, 0)
    3. angolo LONTANO-DESTRA    -> (10, 20)
    4. angolo LONTANO-SINISTRA  -> (0, 20)

Uso:
    python -m src.calibrate --video data/match_video.mp4 --time 30
"""

import argparse
import json

import cv2
import numpy as np

from src.court import COURT_L, COURT_W

DST = [[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]]
LABELS = ["VICINO-SINISTRA (0,0)", "VICINO-DESTRA (10,0)",
          "LONTANO-DESTRA (10,20)", "LONTANO-SINISTRA (0,20)"]


def grab_frame(video_path, t_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire il video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Impossibile leggere il frame al tempo indicato.")
    return frame


def calibrate(video_path, t_sec, out_path):
    frame = grab_frame(video_path, t_sec)
    h, w = frame.shape[:2]
    pts = []
    win = "Calibrazione -- clicca i 4 angoli nell'ordine indicato"

    def draw():
        img = frame.copy()
        cv2.rectangle(img, (0, 0), (w, 30), (20, 20, 20), -1)
        msg = (f"Clicca: {LABELS[len(pts)]}" if len(pts) < 4
               else "Premi INVIO per salvare, 'r' per ricominciare")
        cv2.putText(img, msg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts)], len(pts) == 4, (0, 255, 0), 1)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])
            draw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    draw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            pts.clear(); draw()
        elif key in (13, 10) and len(pts) == 4:  # INVIO
            break
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            print("Calibrazione annullata.")
            return None
    cv2.destroyAllWindows()

    cal = {
        "image_size": [w, h],
        "src_points": pts,
        "dst_points": DST,
        "note": "near-left, near-right, far-right, far-left -> metri campo 10x20",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(f"Calibrazione salvata in {out_path}")
    print("Punti immagine:", pts)
    return cal


def _cli():
    p = argparse.ArgumentParser(description="Calibrazione omografia PADEL-ANALYZER")
    p.add_argument("--video", default="data/match_video.mp4")
    p.add_argument("--time", type=float, default=30.0, help="istante del frame (s)")
    p.add_argument("--out", default="calibration.json")
    a = p.parse_args()
    calibrate(a.video, a.time, a.out)


if __name__ == "__main__":
    _cli()
