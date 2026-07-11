"""PADEL-ANALYZER -- Demo d'esame.

Mostra il video con:
  * una mini-mappa 2D (Bird's-Eye View) nell'angolo, con 4 pallini colorati in
    movimento (Squadra A ciano, Squadra B arancione), orientata come il video;
  * il tracking della pallina a video (cerchio giallo) -- attivo per default se
    esiste models/ball_yolo.pt, disattivabile con --no-ball;
  * una scritta dinamica in alto che descrive l'azione corrente delle due squadre.

I frame di replay / cambio inquadratura (in cui non si rilevano esattamente 4
giocatori dentro il campo) vengono automaticamente scartati e segnalati.

Controlli durante la riproduzione:
    SPAZIO = pausa/riprendi     n = avanza di un frame (in pausa)     q = esci

Al termine (o premendo 'q') vengono stampati i DUE report statistici separati e
indipendenti per la Squadra A e la Squadra B, con i giudizi qualitativi.

Per la demo d'esame conviene elaborare il video INTEGRALE (nessun --start/
--duration) e salvarne una copia annotata con --save, così da avere un artefatto
riproducibile senza dover rieseguire tutto:

    python main_demo.py --video data/demo.mp4 --save output_demo.mp4 --no-show

Uso:
    python main_demo.py --start 60 --duration 60
    python main_demo.py --delay 100        # riproduzione piu lenta
"""

import argparse
import os
from collections import deque

import cv2
import numpy as np
import torch

from src.analytics import (ATTACCO, DIFESA, DISALLINEAMENTO, new_counter,
                           print_reports, situation)
from src.ball_track import BallTracker, detect_ball
from src.court import (NET_Y, STATE_NAMES, build_sample, draw_minimap,
                       foot_point, load_calibration, project,
                       select_player_indices)
from src.model import PadelMultiHeadGRU

WINDOW = 10
MAX_COAST = 10   # frame invalidi consecutivi tollerati prima di dichiarare un taglio

# Colori (BGR) delle situazioni
SIT_COLOR = {
    ATTACCO: (0, 220, 0),
    DIFESA: (0, 165, 255),
    DISALLINEAMENTO: (0, 0, 255),
}


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    hp = ckpt["hparams"]
    model = PadelMultiHeadGRU(input_size=hp["input_size"], hidden_size=hp["hidden_size"],
                              num_layers=hp["num_layers"], num_classes=hp["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def banner(frame, text, color):
    """Scritta dinamica in alto, su fascia semitrasparente."""
    w = frame.shape[1]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 46), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2, cv2.LINE_AA)


def compose_text(sit_a, sit_b):
    """Testo descrittivo dell'azione a partire dalle due situazioni."""
    if sit_a == sit_b == ATTACCO:
        return "ENTRAMBE A RETE", SIT_COLOR[ATTACCO]
    if sit_a == sit_b == DIFESA:
        return "ENTRAMBE A FONDO (scambio)", (200, 200, 200)
    if sit_a == sit_b == DISALLINEAMENTO:
        return "ENTRAMBE DISALLINEATE", SIT_COLOR[DISALLINEAMENTO]
    if sit_a == ATTACCO:
        return f"SQUADRA A ATTACCA  -  B: {sit_b.lower()}", SIT_COLOR[ATTACCO]
    if sit_b == ATTACCO:
        return f"SQUADRA B ATTACCA  -  A: {sit_a.lower()}", SIT_COLOR[DIFESA]
    # nessuna attacca: una difende, l'altra è disallineata
    who = "A" if sit_a == DISALLINEAMENTO else "B"
    return f"DISALLINEAMENTO  -  Squadra {who}", SIT_COLOR[DISALLINEAMENTO]


def _hint(frame):
    cv2.putText(frame, "SPAZIO pausa | n avanti | q esci",
                (16, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA)


def draw_boxes(frame, boxes, sel_idx, court, feet):
    """Disegna i rettangoli attorno alle persone: i 4 selezionati colorati per
    squadra (A ciano, B arancione), gli altri rilevamenti in grigio."""
    for b in boxes:
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (130, 130, 130), 1)
    for i in sel_idx:
        b = boxes[i]
        col = (255, 255, 0) if court[i][1] < NET_Y else (0, 165, 255)
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
        fx, fy = feet[i]
        cv2.circle(frame, (int(fx), int(fy)), 4, col, -1)


def draw_state(frame, sa, sb, sit_a, sit_b, near, far, tag=""):
    """Disegna mini-mappa, banner e righe di stato per un'analisi valida."""
    draw_minimap(frame, near, far)
    text, color = compose_text(sit_a, sit_b)
    banner(frame, text + tag, color)
    cv2.putText(frame, f"A: {STATE_NAMES[sa]} ({sit_a})", (16, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"B: {STATE_NAMES[sb]} ({sit_b})", (16, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)


def run(video_path, calib_path, model_path, start=0.0, duration=None,
        conf=0.30, yolo_model="yolov8m.pt", show=True, delay=60, save=None,
        ball=True, ball_model="models/ball_yolo.pt", ball_conf=0.20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cal = load_calibration(calib_path)
    H = cal["H"]
    model = load_model(model_path, device)

    from ultralytics import YOLO
    yolo = YOLO(yolo_model)

    ball_yolo = tracker = None
    ball_trail = deque(maxlen=18)   # scia della pallina (ultime posizioni)
    if ball:
        if os.path.exists(ball_model):
            ball_yolo = YOLO(ball_model)
            tracker = BallTracker()
            print(f"Tracking pallina attivo (detector: {ball_model})")
        else:
            print(f"[avviso] {ball_model} non trovato: tracking pallina disattivato. "
                  f"Allena il detector con 'python -m src.ball_train' oppure usa --no-ball.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossibile aprire il video: {video_path}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    end_msec = None if duration is None else (start + duration) * 1000.0

    writer = None
    if save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"Salvataggio video annotato -> {save}")

    buf = deque(maxlen=WINDOW)
    counter_a, counter_b = new_counter(), new_counter()
    n_valid = n_skip = n_pred = 0
    paused = False
    misses = 0          # frame invalidi consecutivi
    last = None         # ultima analisi valida (per il coasting)
    fidx = -1           # indice di frame (per il tracker della pallina)
    win = "PADEL-ANALYZER"

    while True:
        # --- gestione pausa: blocca finche' non arriva SPAZIO / n / q ---
        if show and paused:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = False
                continue
            if key != ord("n"):
                continue          # in pausa: ignora altri tasti
            # 'n': prosegue e processa un solo frame, restando in pausa

        ret, frame = cap.read()
        if not ret:
            break
        if end_msec is not None and cap.get(cv2.CAP_PROP_POS_MSEC) > end_msec:
            break
        fidx += 1

        # --- tracking pallina a video (opzionale): scia + cerchio ---
        if tracker is not None:
            pos = tracker.update(detect_ball(ball_yolo, frame, ball_conf), fidx)
            if pos is not None:
                ball_trail.append(pos)
            for p in ball_trail:
                cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)
            if pos is not None:
                cv2.circle(frame, (int(pos[0]), int(pos[1])), 7, (0, 255, 255), 2)

        res = yolo.predict(frame, classes=[0], conf=conf, verbose=False,
                           device=device)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        feet = [foot_point(b) for b in boxes]
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        court = project(H, feet)
        idx = select_player_indices(court, areas)
        draw_boxes(frame, boxes, idx, court, feet)
        sample = build_sample([court[i] for i in idx])

        if sample is not None:
            misses = 0
            feat, _, _, near, far = sample
            buf.append(feat)
            n_valid += 1

            if len(buf) == WINDOW:
                x = torch.from_numpy(np.stack(buf)).unsqueeze(0).to(device)
                with torch.no_grad():
                    la, lb = model(x)
                sa, sb = int(la.argmax()), int(lb.argmax())
                sit_a, sit_b = situation(sa), situation(sb)
                counter_a[sit_a] += 1
                counter_b[sit_b] += 1
                n_pred += 1
                last = (sa, sb, sit_a, sit_b, near, far)
                draw_state(frame, *last)
            else:
                draw_minimap(frame, near, far)
                banner(frame, f"Analisi in corso... ({len(buf)}/{WINDOW})",
                       (200, 200, 200))
        else:
            misses += 1
            if misses <= MAX_COAST and last is not None:
                # dropout transitorio: mantiene l'ultima analisi valida
                draw_state(frame, *last, tag="  [traccia mantenuta]")
            else:
                buf.clear()
                last = None
                n_skip += 1
                banner(frame, "FRAME SCARTATO  (replay / taglio inquadratura)",
                       (0, 0, 255))

        if paused:
            cv2.putText(frame, "[ PAUSA ]", (frame.shape[1] - 150, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)

        if show:
            _hint(frame)
            cv2.imshow(win, frame)
            key = cv2.waitKey(1 if paused else max(1, delay)) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("n"):
                paused = True

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"\nFrame analizzati: {n_pred} predetti | {n_valid} validi | "
          f"{n_skip} scartati")
    print_reports(counter_a, counter_b)


def _cli():
    p = argparse.ArgumentParser(description="PADEL-ANALYZER demo")
    p.add_argument("--video", default="data/match_video.mp4")
    p.add_argument("--calib", default="calibration.json")
    p.add_argument("--model", default="models/padel_gru.pt")
    p.add_argument("--yolo", default="yolov8m.pt")
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--start", type=float, default=0.0, help="inizio (secondi)")
    p.add_argument("--duration", type=float, default=None, help="durata (secondi)")
    p.add_argument("--delay", type=int, default=60,
                   help="ritardo per frame in ms (piu' alto = piu' lento)")
    p.add_argument("--no-show", action="store_true", help="non mostrare la finestra video")
    p.add_argument("--save", default=None,
                   help="salva il video annotato (es. output_demo.mp4)")
    p.add_argument("--no-ball", action="store_true",
                   help="disattiva il tracking pallina (attivo per default)")
    p.add_argument("--ball-model", default="models/ball_yolo.pt")
    p.add_argument("--ball-conf", type=float, default=0.20)
    a = p.parse_args()
    run(a.video, a.calib, a.model, start=a.start, duration=a.duration,
        conf=a.conf, yolo_model=a.yolo, show=not a.no_show, delay=a.delay,
        save=a.save, ball=not a.no_ball, ball_model=a.ball_model,
        ball_conf=a.ball_conf)


if __name__ == "__main__":
    _cli()
