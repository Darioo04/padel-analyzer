"""PADEL-ANALYZER -- pacchetto sorgente.

Sistema di Match Analytics per il Padel: tracciamento dei 4 giocatori,
proiezione Bird's-Eye-View tramite omografia, e classificazione tattica
temporale con una GRU multi-head (una testa per squadra).
"""

__all__ = ["court", "model", "dataset", "analytics"]
