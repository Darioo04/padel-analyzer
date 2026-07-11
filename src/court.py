"""Geometria del campo, omografia, etichettatura tattica e mini-mappa.

Sistema di riferimento del campo (metri), vista dall'alto (Bird's-Eye View):

    y = 20  +---------------------+   <- fondo campo Squadra B (lato lontano)
            |                     |
    y = 10  +======= RETE ========+   <- rete a metà campo
            |                     |
    y = 0   +---------------------+   <- fondo campo Squadra A (lato vicino)
           x=0                   x=10

  * Squadra A = metà vicina alla telecamera  (y < 10)
  * Squadra B = metà lontana                 (y > 10)

Ogni giocatore viene proiettato dal punto-piede (base del bounding box) sul
piano del campo. La posizione tattica di una coppia è codificata in 3 stati:

    0 = STATE_BACK  -> entrambi i giocatori a fondo campo
    1 = STATE_NET   -> entrambi i giocatori sotto rete (fase d'attacco)
    2 = STATE_SPLIT -> coppia "disallineata" (uno avanti, uno indietro)
"""

import json

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
#  Costanti del campo (metri)                                                  #
# --------------------------------------------------------------------------- #
COURT_W = 10.0        # larghezza del campo (asse x)
COURT_L = 20.0        # lunghezza del campo (asse y)
NET_Y = 10.0          # posizione della rete (metà campo)
NET_MIDLINE = 5.0     # soglia (m) sulla distanza media dalla rete per distinguere rete/fondo
SPLIT_DIFF = 4.0      # differenza (m) di profondità tra i 2 giocatori oltre cui la coppia è "disallineata"
BOUND_MARGIN = 1.5    # margine (m) oltre le linee tollerato per un rilevamento valido

# Codifica degli stati posizionali di una coppia
STATE_BACK = 0        # fondo campo
STATE_NET = 1         # sotto rete
STATE_SPLIT = 2       # coppia disallineata
STATE_NAMES = {0: "FONDO", 1: "RETE", 2: "DISALLINEATA"}


# --------------------------------------------------------------------------- #
#  Omografia                                                                   #
# --------------------------------------------------------------------------- #
def load_calibration(path):
    """Carica calibration.json e pre-calcola la matrice di omografia H.

    Il file contiene 4 corrispondenze immagine->campo nell'ordine:
    near-left, near-right, far-right, far-left.
    """
    with open(path, "r", encoding="utf-8") as f:
        cal = json.load(f)
    src = np.array(cal["src_points"], dtype=np.float32)
    dst = np.array(cal["dst_points"], dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("calibration.json deve contenere 4 punti src e 4 dst (x,y).")
    cal["H"] = cv2.getPerspectiveTransform(src, dst)
    return cal


def foot_point(box):
    """Punto-piede di un bounding box [x1, y1, x2, y2] = (centro-x, base-y)."""
    x1, _, x2, y2 = box
    return [(x1 + x2) / 2.0, float(y2)]


def project(H, pts):
    """Proietta punti immagine (Nx2, pixel) sul piano del campo (Nx2, metri)."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H)
    return out.reshape(-1, 2)


def in_court(pt, margin=BOUND_MARGIN):
    """True se il punto (metri) cade dentro i confini logici del campo."""
    x, y = pt
    return (-margin <= x <= COURT_W + margin) and (-margin <= y <= COURT_L + margin)


def select_player_indices(points, weights, margin=BOUND_MARGIN):
    """Indici dei 4 giocatori, scartando folla/arbitro/raccattapalle.

    Tiene solo le persone dentro il campo e, se in una metà ce ne sono più di
    due, mantiene le due con bounding box più grande (i giocatori sono vicini
    e grandi, gli estranei lontani e piccoli). Ritorna al più 4 indici, al più
    2 per metà; `build_sample` poi valida il vincolo 2 vs 2.
    """
    idx = [i for i, p in enumerate(points) if in_court(p, margin)]
    near = sorted([i for i in idx if points[i][1] < NET_Y],
                  key=lambda i: weights[i], reverse=True)[:2]
    far = sorted([i for i in idx if points[i][1] >= NET_Y],
                 key=lambda i: weights[i], reverse=True)[:2]
    return near + far


def select_players(points, weights, margin=BOUND_MARGIN):
    """Come select_player_indices ma ritorna direttamente i punti [x, y]."""
    return [list(map(float, points[i]))
            for i in select_player_indices(points, weights, margin)]


# --------------------------------------------------------------------------- #
#  Etichettatura tattica (auto-labeling geometrico)                            #
# --------------------------------------------------------------------------- #
def team_state(y_players, side):
    """Stato posizionale di una coppia a partire dalle y (metri) dei 2 giocatori.

    side = 'near' (Squadra A) oppure 'far' (Squadra B).

    Criterio (più aderente alla tattica del padel):
      * DISALLINEATA se i due giocatori differiscono in profondità (distanza
        dalla rete) di più di SPLIT_DIFF metri, cioè uno è avanti e l'altro
        indietro -- indipendentemente da dove si trovino sul campo;
      * altrimenti la coppia è compatta e viene classificata come SOTTO RETE
        o A FONDO in base alla distanza MEDIA dalla rete (soglia NET_MIDLINE).
    """
    if side == "near":
        dist_net = [NET_Y - y for y in y_players]
    else:
        dist_net = [y - NET_Y for y in y_players]

    if abs(dist_net[0] - dist_net[1]) > SPLIT_DIFF:
        return STATE_SPLIT

    mean_dist = 0.5 * (dist_net[0] + dist_net[1])
    return STATE_NET if mean_dist < NET_MIDLINE else STATE_BACK


def build_sample(court_pts):
    """Da una lista di punti-campo (metri) costruisce feature ed etichette.

    Ritorna (feat[8], labA, labB, near_pts, far_pts) oppure None se il frame
    non è valido (persone dentro il campo != 4 oppure ripartizione != 2 vs 2).

    Le feature sono le coordinate normalizzate dei 4 giocatori, ordinati per
    metà campo e, all'interno di ogni metà, da sinistra a destra:
        [Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2]  (in [0, 1])
    """
    pts = [list(map(float, p)) for p in court_pts if in_court(p)]
    if len(pts) != 4:
        return None

    near = sorted([p for p in pts if p[1] < NET_Y], key=lambda p: p[0])
    far = sorted([p for p in pts if p[1] >= NET_Y], key=lambda p: p[0])
    if len(near) != 2 or len(far) != 2:
        return None

    lab_a = team_state([p[1] for p in near], "near")
    lab_b = team_state([p[1] for p in far], "far")

    feat = []
    for p in near + far:
        feat.extend([p[0] / COURT_W, p[1] / COURT_L])
    return np.array(feat, dtype=np.float32), lab_a, lab_b, near, far


# --------------------------------------------------------------------------- #
#  Mini-mappa 2D (overlay per la demo)                                         #
# --------------------------------------------------------------------------- #
def draw_minimap(frame, near_pts, far_pts, corner="tr", width=150, pad=18):
    """Disegna la mappa Bird's-Eye-View con 4 pallini nell'angolo del frame.

    near_pts / far_pts: liste di [x, y] in metri (Squadra A / Squadra B).
    Squadra A in ciano, Squadra B in arancione.
    """
    h_img, w_img = frame.shape[:2]
    scale = width / COURT_W
    height = int(COURT_L * scale)
    width = int(width)

    if corner == "tr":
        x0, y0 = w_img - width - pad, pad
    elif corner == "tl":
        x0, y0 = pad, pad
    else:
        x0, y0 = w_img - width - pad, h_img - height - pad

    # Sfondo semitrasparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 6), (x0 + width + 6, y0 + height + 6),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def to_px(p):
        # Asse y ribaltato: la coppia vicina alla telecamera (y piccola, Squadra A)
        # va in BASSO nella mini-mappa, come appare nel video.
        return (x0 + int(np.clip(p[0], 0, COURT_W) * scale),
                y0 + int((COURT_L - np.clip(p[1], 0, COURT_L)) * scale))

    # Bordo del campo
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (200, 200, 200), 2)
    # Linee di servizio (a 3 m dai fondi: y=3 e y=17)
    for line_y in (3.0, 17.0):
        cv2.line(frame, to_px((0, line_y)), to_px((COURT_W, line_y)), (120, 120, 120), 1)
    # Linea centrale di servizio (x=5, fra le due linee di servizio)
    cv2.line(frame, to_px((COURT_W / 2, 3.0)), to_px((COURT_W / 2, 17.0)),
             (120, 120, 120), 1)
    # Rete (y=10) in rosso
    cv2.line(frame, to_px((0, NET_Y)), to_px((COURT_W, NET_Y)), (0, 0, 255), 2)

    for p in near_pts:
        cv2.circle(frame, to_px(p), 6, (255, 255, 0), -1)      # A: ciano
        cv2.circle(frame, to_px(p), 6, (0, 0, 0), 1)
    for p in far_pts:
        cv2.circle(frame, to_px(p), 6, (0, 165, 255), -1)      # B: arancione
        cv2.circle(frame, to_px(p), 6, (0, 0, 0), 1)

    cv2.putText(frame, "A", (x0 + 4, y0 + height - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "B", (x0 + width - 16, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
    return frame
