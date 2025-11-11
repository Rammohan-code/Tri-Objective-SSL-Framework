from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class ViewConfig:
    mask_rate: float = 0.2
    noise_std: float = 0.01

def make_views(x: torch.Tensor, cfg: ViewConfig):
    noise1 = torch.randn_like(x) * cfg.noise_std
    noise2 = torch.randn_like(x) * cfg.noise_std
    m1 = (torch.rand_like(x) < cfg.mask_rate).float()
    m2 = (torch.rand_like(x) < cfg.mask_rate).float()
    v1 = x * (1.0 - m1) + noise1
    v2 = x * (1.0 - m2) + noise2
    return v1, v2, m1, m2
