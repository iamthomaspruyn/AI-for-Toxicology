"""Model definition matching the notebook architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEWithPredictor(nn.Module):
    """Sequence VAE + toxicity prediction head used in the notebooks."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        latent_dim: int,
        num_tasks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.conv_1 = nn.Conv1d(vocab_size, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy = torch.zeros(1, vocab_size, seq_len)
            dummy = self.relu(self.conv_1(dummy))
            dummy = self.relu(self.conv_2(dummy))
            dummy = self.relu(self.conv_3(dummy))
            self.flat_features = dummy.flatten(1).size(1)

        self.linear_0 = nn.Linear(self.flat_features, 435)
        self.fc_mu = nn.Linear(435, latent_dim)
        self.fc_logvar = nn.Linear(435, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 292)
        self.gru = nn.GRU(
            input_size=292, hidden_size=501, num_layers=3, batch_first=True
        )
        self.output = nn.Linear(501, vocab_size)

        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_tasks),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_onehot = (
            F.one_hot(x, num_classes=self.vocab_size)
            .float()
            .transpose(1, 2)
            .contiguous()
        )
        hidden = self.relu(self.conv_1(x_onehot))
        hidden = self.relu(self.conv_2(hidden))
        hidden = self.relu(self.conv_3(hidden))
        hidden = hidden.flatten(1)
        hidden = F.selu(self.linear_0(hidden))
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = 1e-2 * torch.randn_like(logvar)
        return torch.exp(0.5 * logvar) * eps + mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = F.selu(self.decoder_input(z))
        hidden = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        hidden, _ = self.gru(hidden)
        return self.output(hidden)

    def predict_logits(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        pred_logits = self.pred_head(mu)
        return pred_logits, mu, logvar

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon_logits = self.decode(z)
        pred_logits = self.pred_head(mu)
        return recon_logits, mu, logvar, pred_logits
