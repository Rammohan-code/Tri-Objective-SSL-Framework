from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__(); hid = max(128, in_dim)
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(inplace=True), nn.Linear(hid, proj_dim))
    def forward(self, y): return F.normalize(self.net(y), dim=-1)

class Decoder(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
    def forward(self, y): return self.net(y)

class TabSSL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, emb_dim: int = 256, proj_dim: int = 128, dropout: float = 0.1):
        super().__init__(); self.encoder = MLPEncoder(in_dim, hidden_dim, emb_dim, dropout); self.proj = ProjectionHead(emb_dim, proj_dim); self.decoder = Decoder(emb_dim, in_dim, hidden_dim, dropout)
    def forward(self, x):
        y = self.encoder(x); z = self.proj(y); x_hat = self.decoder(y); return y, z, x_hat
