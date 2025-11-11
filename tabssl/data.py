from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

@dataclass
class TabularData:
    X: np.ndarray
    y: Optional[np.ndarray]
    scaler: StandardScaler

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32); self.y = y.astype(np.int64) if y is not None else None
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx: int):
        if self.y is None: return torch.from_numpy(self.X[idx])
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])

def load_sklearn_breast_cancer(test_size: float = 0.2, seed: int = 42) -> Tuple[TabularData, TabularData]:
    data = load_breast_cancer(); X, y = data.data, data.target
    n = X.shape[0]; rng = np.random.default_rng(seed); idx = rng.permutation(n)
    n_test = int(n * test_size); test_idx = idx[:n_test]; train_idx = idx[n_test:]
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx]); X_test = scaler.transform(X[test_idx])
    return TabularData(X_train, y[train_idx], scaler), TabularData(X_test, y[test_idx], scaler)

def make_dataloaders(train: TabularData, test: TabularData, batch_size: int = 256):
    ds_u = TabularDataset(train.X, None); ds_l_train = TabularDataset(train.X, train.y); ds_l_test = TabularDataset(test.X, test.y)
    return (
        DataLoader(ds_u, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(ds_l_train, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(ds_l_test, batch_size=batch_size, shuffle=False, drop_last=False),
    )
