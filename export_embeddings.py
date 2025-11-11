# export_embeddings.py
import argparse, os, json
import numpy as np, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabssl.model import TabSSL
from tabssl.utils import device_auto
from tabssl.data import load_csv_any  # from your earlier "any CSV" version

def get_embeddings(model, X, device, batch=1024):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch].astype(np.float32)).to(device)
            y, z, _ = model(xb)   # content y; projector z; recon xhat
            outs.append(y.cpu().numpy())
    return np.concatenate(outs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", type=str, required=True, help="Path to CSV (with a target column).")
    ap.add_argument("--target-col", type=str, required=True, help="Name of the label column.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to trained model .pt")
    ap.add_argument("--out", type=str, default="embeddings_proposed.csv")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    device = device_auto()
    print("[INFO] Device:", device)

    # Load CSV (auto categorical handling in your any-CSV loader)
    data = load_csv_any(args.csv_path, target_col=args.target_col, test_size=args.test_size, seed=42)
    X = data["X_all"].astype(np.float32)  # concatenated (train+test) if provided; else use X
    y = data["y_all"].astype(np.int32) if "y_all" in data else data["y"].astype(np.int32)

    # Standardize for a fair raw baseline
    scaler = StandardScaler().fit(X)
    X_std = scaler.transform(X)

    # Load model & weights
    in_dim = X.shape[1]
    model = TabSSL(in_dim=in_dim, hidden_dim=256, emb_dim=256, proj_dim=128, dropout=0.1).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    print("[INFO] Model loaded from:", args.ckpt)

    # Export embeddings
    E = get_embeddings(model, X_std, device)
    print("[INFO] Embeddings shape:", E.shape)

    # Save CSV with raw features reduced to 50D (optional) is done in plotting step; here we just save E + labels
    import pandas as pd
    df = pd.DataFrame(E)
    df["label"] = y
    df.to_csv(args.out, index=False)
    print("[SAVED]", args.out)

    # Also save RAW features (standardized) to a CSV for t-SNE comparison
    df_raw = pd.DataFrame(X_std)
    df_raw["label"] = y
    df_raw.to_csv("features_raw.csv", index=False)
    print("[SAVED] features_raw.csv")

if __name__ == "__main__":
    main()
