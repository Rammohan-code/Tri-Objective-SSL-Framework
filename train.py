from __future__ import annotations
import argparse, os, csv, json
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tabssl.data import load_sklearn_breast_cancer, make_dataloaders
from tabssl.model import TabSSL
from tabssl.losses import info_nce, masked_reconstruction_loss, alignment_mse
from tabssl.augment import ViewConfig, make_views
from tabssl.utils import set_seed, device_auto
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="Tabular SSL: Contrastive + Masked Reconstruction")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--emb-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--mask-rate", type=float, default=0.2)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--lambda-con", type=float, default=1.0)
    p.add_argument("--lambda-rec", type=float, default=1.0)
    p.add_argument("--lambda-align", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="outputs")
    return p.parse_args()

def save_loss_plots(hist_con, hist_rec, hist_align, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(hist_con, label="Contrastive")
    plt.plot(hist_rec, label="Reconstruction")
    plt.plot(hist_align, label="Alignment")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training Loss Curves")
    path = os.path.join(outdir, "loss_curves.png"); plt.savefig(path, bbox_inches="tight", dpi=150); plt.close()
    print(f"[Saved] {path}")

def save_epoch_csv(hist_con, hist_rec, hist_align, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "training_log.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["epoch", "Lcon", "Lrec", "Lalign"])
        for e, (a, b, c) in enumerate(zip(hist_con, hist_rec, hist_align), start=1):
            w.writerow([e, a, b, c])
    print(f"[Saved] {path}")

def save_metrics(acc, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f: json.dump({"linear_probe_accuracy": float(acc)}, f, indent=2)
    print(f"[Saved] {path}")

def pretrain(model: TabSSL, unlabeled_loader, args, device):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train(); view_cfg = ViewConfig(mask_rate=args.mask_rate, noise_std=args.noise_std)
    hist_con, hist_rec, hist_align = [], [], []
    for epoch in range(1, args.epochs+1):
        pbar = tqdm(unlabeled_loader, desc=f"Pretrain {epoch}/{args.epochs}", ncols=100)
        epoch_con = epoch_rec = epoch_align = 0.0
        for x in pbar:
            x = x.to(device)
            v1, v2, m1, m2 = make_views(x, view_cfg)
            v1, v2, m1, m2 = v1.to(device), v2.to(device), m1.to(device), m2.to(device)
            y1, z1, xh1 = model(v1); y2, z2, xh2 = model(v2)
            l_con = info_nce(z1, z2, tau=args.tau)
            l_rec = masked_reconstruction_loss(x, xh1, m1) + masked_reconstruction_loss(x, xh2, m2)
            l_align = alignment_mse(y1, y2)
            loss = args.lambda_con*l_con + args.lambda_rec*l_rec + args.lambda_align*l_align
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            epoch_con += l_con.item(); epoch_rec += l_rec.item(); epoch_align += l_align.item()
            pbar.set_postfix(con=f"{l_con.item():.3f}", rec=f"{l_rec.item():.3f}", align=f"{l_align.item():.3f}")
        n = len(unlabeled_loader)
        hist_con.append(epoch_con/n); hist_rec.append(epoch_rec/n); hist_align.append(epoch_align/n)
        print(f"Epoch {epoch}: Lcon={hist_con[-1]:.3f}  Lrec={hist_rec[-1]:.3f}  Lalign={hist_align[-1]:.3f}")
    return hist_con, hist_rec, hist_align

def linear_probe_sklearn(model: TabSSL, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, device):
    model.eval()
    with torch.no_grad():
        Z_tr = []
        for i in range(0, len(X_train), 1024):
            x = torch.from_numpy(X_train[i:i+1024].astype(np.float32)).to(device)
            y, z, _ = model(x); Z_tr.append(y.cpu().numpy())
        Z_tr = np.concatenate(Z_tr, axis=0)
        Z_te = []
        for i in range(0, len(X_test), 1024):
            x = torch.from_numpy(X_test[i:i+1024].astype(np.float32)).to(device)
            y, z, _ = model(x); Z_te.append(y.cpu().numpy())
        Z_te = np.concatenate(Z_te, axis=0)
    clf = LogisticRegression(max_iter=2000, solver="saga", n_jobs=-1)
    clf.fit(Z_tr, y_train); preds = clf.predict(Z_te)
    acc = accuracy_score(y_test, preds)
    return acc

def save_tsne_plot(model, X, y, device, outdir):
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        Y = []
        for i in range(0, len(X), 1024):
            xb = torch.from_numpy(X[i:i+1024].astype(np.float32)).to(device)
            yb, zb, _ = model(xb); Y.append(yb.cpu().numpy())
        Y = np.concatenate(Y, axis=0)
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=min(30, max(5, len(X)//10)))
    X2 = tsne.fit_transform(Y)
    plt.figure(); plt.scatter(X2[:,0], X2[:,1], c=y, alpha=0.7)
    plt.title("t-SNE of SSL Embeddings (content space)")
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    path = os.path.join(outdir, "tsne_embeddings.png"); plt.savefig(path, bbox_inches="tight", dpi=150); plt.close()
    print(f"[Saved] {path}")

def main():
    args = parse_args(); set_seed(args.seed); device = device_auto(); print("Using device:", device)
    train, test = load_sklearn_breast_cancer()
    unlabeled_loader, labeled_train_loader, labeled_test_loader = make_dataloaders(train, test, batch_size=args.batch_size)
    in_dim = train.X.shape[1]
    model = TabSSL(in_dim=in_dim, hidden_dim=args.hidden_dim, emb_dim=args.emb_dim, proj_dim=args.proj_dim, dropout=args.dropout).to(device)
    hist_con, hist_rec, hist_align = pretrain(model, unlabeled_loader, args, device)
    save_loss_plots(hist_con, hist_rec, hist_align, args.save_dir)
    save_epoch_csv(hist_con, hist_rec, hist_align, args.save_dir)
    acc = linear_probe_sklearn(model, train.X, train.y, test.X, test.y, device)
    print(f"[Linear Probe Accuracy] {acc:.4f}"); save_metrics(acc, args.save_dir)
    save_tsne_plot(model, test.X, test.y, device, args.save_dir)

if __name__ == "__main__":
    main()
