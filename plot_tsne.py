# plot_tsne.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_feats(path):
    df = pd.read_csv(path)
    y = df["label"].values
    X = df.drop(columns=["label"]).values
    return X, y

def reduce_for_tsne(X, target_dim=50):
    # Optional PCA down to 50D for stable t-SNE on high-D embeddings/features
    if X.shape[1] > target_dim:
        pca = PCA(n_components=target_dim, random_state=42)
        X = pca.fit_transform(X)
    return X

def tsne_2d(X, perplexity=30, seed=42):
    perplexity = max(5, min(perplexity, len(X)//10 if len(X)//10 >= 5 else 30))
    tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity)
    return tsne.fit_transform(X)

def scatter(ax, X2, y, title):
    sc = ax.scatter(X2[:,0], X2[:,1], c=y, s=10, alpha=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="features_raw.csv", help="CSV with raw standardized features (+label)")
    ap.add_argument("--proposed", type=str, default="embeddings_proposed.csv", help="CSV with Proposed embeddings (+label)")
    ap.add_argument("--contrastive_only", type=str, default="", help="Optional CSV with contrastive-only embeddings (+label)")
    ap.add_argument("--out", type=str, default="figure3_tsne.png")
    ap.add_argument("--perplexity", type=float, default=30)
    args = ap.parse_args()

    # Load sets
    X_raw, y_raw = load_feats(args.raw)
    X_prop, y_prop = load_feats(args.proposed)
    assert len(y_raw) == len(y_prop), "Mismatch in sample counts between RAW and Proposed."

    # Reduce and TSNE
    X_raw_r = reduce_for_tsne(X_raw, target_dim=50)
    X_prop_r = reduce_for_tsne(X_prop, target_dim=50)
    X2_raw = tsne_2d(X_raw_r, perplexity=args.perplexity)
    X2_prop = tsne_2d(X_prop_r, perplexity=args.perplexity)

    # Prepare figure
    if args.contrastive_only:
        X_con, y_con = load_feats(args.contrastive_only)
        assert len(y_con) == len(y_prop), "Mismatch with contrastive-only sample count."
        X_con_r = reduce_for_tsne(X_con, target_dim=50)
        X2_con = tsne_2d(X_con_r, perplexity=args.perplexity)

        # Three-panel figure
        plt.figure(figsize=(12,4))
        ax1 = plt.subplot(1,3,1); scatter(ax1, X2_raw, y_raw, "Raw Features")
        ax2 = plt.subplot(1,3,2); scatter(ax2, X2_con, y_con, "Contrastive-only")
        ax3 = plt.subplot(1,3,3); scatter(ax3, X2_prop, y_prop, "Proposed (Tri-Obj)")
    else:
        # Two-panel figure
        plt.figure(figsize=(8,4))
        ax1 = plt.subplot(1,2,1); scatter(ax1, X2_raw, y_raw, "Raw Features")
        ax2 = plt.subplot(1,2,2); scatter(ax2, X2_prop, y_prop, "Proposed (Tri-Obj)")

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.show()
    print("[SAVED]", args.out)

if __name__ == "__main__":
    main()
