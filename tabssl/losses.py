from __future__ import annotations
import torch, torch.nn.functional as F

def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    N, D = z1.shape
    z = torch.cat([z1, z2], dim=0)      # (2N, D)
    sim = (z @ z.t()) / tau
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    targets = torch.arange(N, 2*N, device=z.device)
    targets = torch.cat([targets, torch.arange(0, N, device=z.device)], dim=0)
    return F.cross_entropy(sim, targets)

def masked_reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (x_hat - x) ** 2
    masked = diff2 * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom

def alignment_mse(y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y1, y2)
