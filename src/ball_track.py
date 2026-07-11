"""Tracking della pallina a video.

Usa il detector dedicato (models/ball_yolo.pt, addestrato con src.ball_train) per
localizzare la pallina in ogni frame. Un semplice tracker a nearest-neighbour
mantiene la continuità della traiettoria (sceglie il rilevamento più vicino alla
posizione prevista, non solo quello a confidenza più alta), così i falsi positivi
sparsi non fanno "saltare" la traccia.

"""

import argparse
from collections import deque

import cv2
import numpy as np


class BallTracker:
    """Associa i rilevamenti della pallina nel tempo per una traccia stabile.

    gate        raggio (px) di associazione al frame successivo (cresce coi buchi)
    reset_after dopo quanti frame senza match la traccia riparte da capo
    """

    def __init__(self, gate=140.0, reset_after=8):
        self.gate = gate
        self.reset_after = reset_after
        self.last = None          # (fidx, cx, cy) ultimo match accettato
        self.vel = (0.0, 0.0)     # velocità stimata (px/frame)
        self.misses = 0

    def _pick(self, dets, fidx):
        """Rilevamento da seguire: il più vicino alla posizione prevista se esiste
        una traccia, altrimenti quello a confidenza più alta."""
        if not dets:
            return None
        if self.last is None or self.misses > self.reset_after:
            return max(dets, key=lambda d: d[2])       # (cx, cy, conf)
        dt = max(1, fidx - self.last[0])
        px = self.last[1] + self.vel[0] * dt
        py = self.last[2] + self.vel[1] * dt
        gate = self.gate + 12.0 * self.misses
        best, bestd = None, gate
        for cx, cy, cf in dets:
            d = np.hypot(cx - px, cy - py)
            if d < bestd:
                best, bestd = (cx, cy, cf), d
        return best

    def update(self, dets, fidx):
        """Processa i rilevamenti (lista di (cx, cy, conf)); ritorna (cx, cy) o None."""
        pick = self._pick(dets, fidx)
        if pick is None:
            self.misses += 1
            if self.misses > self.reset_after:
                self.last = None
            return None
        cx, cy, _ = pick
        if self.last is not None:
            dt = max(1, fidx - self.last[0])
            self.vel = (0.6 * self.vel[0] + 0.4 * (cx - self.last[1]) / dt,
                        0.6 * self.vel[1] + 0.4 * (cy - self.last[2]) / dt)
        self.last = (fidx, cx, cy)
        self.misses = 0
        return (cx, cy)


def detect_ball(model, frame, conf, imgsz=1280):
    """Lista dei rilevamenti pallina [(cx, cy, conf), ...] su un frame."""
    res = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    b = res.boxes
    if b is None or len(b) == 0:
        return []
    xyxy = b.xyxy.cpu().numpy()
    cf = b.conf.cpu().numpy()
    return [((x1 + x2) / 2.0, (y1 + y2) / 2.0, float(c))
            for (x1, y1, x2, y2), c in zip(xyxy, cf)]


def _run_tester(video, model_path, start, duration, conf, save, show):
    from ultralytics import YOLO

    model = YOLO(model_path)
    tracker = BallTracker()
    cap = cv2.VideoCapture(video)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    end_msec = None if duration is None else (start + duration) * 1000.0

    writer = None
    if save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"Salvataggio -> {save}")

    trail = deque(maxlen=18)
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if end_msec is not None and cap.get(cv2.CAP_PROP_POS_MSEC) > end_msec:
            break
        pos = tracker.update(detect_ball(model, frame, conf), fidx)
        if pos is not None:
            trail.append(pos)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 7, (0, 255, 255), 2)
        for p in trail:
            cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)

        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("ball_track", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        fidx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def _cli():
    p = argparse.ArgumentParser(description="Tracking pallina (tester visivo)")
    p.add_argument("--video", default="data/demo.mp4")
    p.add_argument("--model", default="models/ball_yolo.pt")
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--save", default=None)
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()
    _run_tester(a.video, a.model, a.start, a.duration, a.conf, a.save,
                not a.no_show)


if __name__ == "__main__":
    _cli()
