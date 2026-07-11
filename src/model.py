"""Rete neurale ricorrente multi-head per l'analisi tattica del padel.

Un blocco GRU condiviso elabora la sequenza temporale delle posizioni dei 4
giocatori (finestre di 10 frame). Da esso si diramano DUE teste lineari
parallele e indipendenti, una per squadra, ciascuna con 3 uscite:

    head_A -> stato posizionale della Squadra A  (0 fondo / 1 rete / 2 disallineata)
    head_B -> stato posizionale della Squadra B  (0 fondo / 1 rete / 2 disallineata)

Le due teste sono indipendenti: risolvono la coesistenza degli stati (una
squadra può sbagliare mentre l'altra attacca).
"""

import torch
import torch.nn as nn


class PadelMultiHeadGRU(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=1,
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Blocco ricorrente condiviso
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)

        # Due rami di output paralleli e indipendenti
        self.head_A = nn.Linear(hidden_size, num_classes)
        self.head_B = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """x: [batch, seq_len=10, input_size=8] -> (logits_A, logits_B).

        Si usa lo stato dell'ultimo istante temporale come rappresentazione
        riassuntiva della finestra (classificazione causale della sequenza).
        """
        out, _ = self.gru(x)            # [B, T, H]
        feat = self.drop(out[:, -1, :])  # [B, H]  ultimo istante
        return self.head_A(feat), self.head_B(feat)

    @torch.no_grad()
    def predict_states(self, x):
        """Ritorna gli stati (argmax) predetti: (state_A, state_B) come interi."""
        self.eval()
        logits_a, logits_b = self.forward(x)
        return int(logits_a.argmax(dim=-1)), int(logits_b.argmax(dim=-1))
