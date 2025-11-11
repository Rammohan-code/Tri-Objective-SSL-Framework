from __future__ import annotations
import random, numpy as np, torch
def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
