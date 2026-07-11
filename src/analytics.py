"""Logica tattica semplificata (3 categorie) e generazione dei report finali.

Ogni squadra, a ogni frame, ricade in UNA sola delle tre situazioni tattiche,
mappate direttamente dallo stato posizionale predetto dalla rispettiva testa
della rete. Le tre categorie partizionano gli stati, quindi la somma delle
percentuali di ogni report fa esattamente 100%.

    Attacco         : la coppia è sotto rete (stato 1).
    Difesa          : la coppia è a fondo campo (stato 0).
    Disallineamento : la coppia è disallineata in profondità (stato 2) -- un
                      giocatore avanzato e l'altro arretrato. Include anche le
                      fasi di servizio/risposta, in cui il disallineamento è
                      fisiologico e non costituisce un vero errore.
"""

from src.court import STATE_BACK, STATE_NET, STATE_SPLIT

# Ordine canonico delle 3 situazioni tattiche
ATTACCO = "Attacco"
DIFESA = "Difesa"
DISALLINEAMENTO = "Disallineamento"
SITUATIONS = [ATTACCO, DIFESA, DISALLINEAMENTO]

_STATE_TO_SITUATION = {
    STATE_NET: ATTACCO,
    STATE_BACK: DIFESA,
    STATE_SPLIT: DISALLINEAMENTO,
}


def situation(state):
    """Situazione tattica di una squadra dato il suo stato posizionale."""
    return _STATE_TO_SITUATION[state]


def new_counter():
    return {s: 0 for s in SITUATIONS}


def attack_judgment(pct):
    """Valutazione qualitativa del volume d'attacco."""
    if pct > 40:
        return "DOMINANTE"
    if pct >= 25:
        return "OFFENSIVO"
    return "PASSIVO"


def alignment_judgment(pct):
    """Valutazione qualitativa del disallineamento (quanto è compatta la coppia)."""
    if pct < 15:
        return "COMPATTA"
    if pct <= 30:
        return "MOBILE"
    return "SLEGATA"


def percentages(counter):
    """Converte un contatore in percentuali (dict) che sommano a 100."""
    total = sum(counter.values())
    if total == 0:
        return {s: 0.0 for s in SITUATIONS}, 0
    return {s: 100.0 * counter[s] / total for s in SITUATIONS}, total


def format_report(team_name, counter):
    """Restituisce una tabella testuale del report per una squadra."""
    pct, total = percentages(counter)
    atk = attack_judgment(pct[ATTACCO])
    ali = alignment_judgment(pct[DISALLINEAMENTO])

    width = 52
    lines = []
    lines.append("+" + "-" * width + "+")
    lines.append("| {:<{w}}|".format(f"REPORT TATTICO  --  {team_name}", w=width - 1))
    lines.append("+" + "-" * width + "+")
    lines.append("| {:<28}{:>10}{:>12} |".format("Situazione", "Frame", "Tempo %"))
    lines.append("+" + "-" * width + "+")
    for s in SITUATIONS:
        lines.append("| {:<28}{:>10d}{:>11.1f}% |".format(s, counter[s], pct[s]))
    lines.append("+" + "-" * width + "+")
    lines.append("| {:<28}{:>10d}{:>11.1f}% |".format("TOTALE", total, sum(pct.values())))
    lines.append("+" + "-" * width + "+")
    lines.append("| Volume d'Attacco : {:>5.1f}%   -> [{}]".format(pct[ATTACCO], atk)
                 .ljust(width + 1) + "|")
    lines.append("| Disallineamento  : {:>5.1f}%   -> [{}]".format(pct[DISALLINEAMENTO], ali)
                 .ljust(width + 1) + "|")
    lines.append("+" + "-" * width + "+")
    return "\n".join(lines)


def print_reports(counter_a, counter_b):
    """Stampa a terminale i due report distinti e indipendenti."""
    print()
    print("=" * 56)
    print("        PADEL-ANALYZER  --  REPORT FINALE DELLA PARTITA")
    print("=" * 56)
    print(format_report("SQUADRA A (lato vicino)", counter_a))
    print()
    print(format_report("SQUADRA B (lato lontano)", counter_b))
    print()
