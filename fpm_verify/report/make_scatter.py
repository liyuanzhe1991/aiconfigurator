# Compact PNG scatter (truth vs model, old/new side by side) for report embedding.
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--title", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

df = pd.read_csv(args.scores)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=110)
for ax, side, label in zip(axes, ("old", "new"),
                           ("old DB (random-content)", "new DB (this campaign)")):
    g = df[df.side == side]
    if not len(g):
        continue
    lo = min(g.truth.min(), g.model.min()) * 0.8
    hi = max(g.truth.max(), g.model.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color="#999", lw=1)
    ax.plot([lo, hi], [lo * 1.05, hi * 1.05], color="#ccc", lw=0.8, ls="--")
    ax.plot([lo, hi], [lo * 0.95, hi * 0.95], color="#ccc", lw=0.8, ls="--")
    ok = g[g.ape <= 0.05]
    bad = g[g.ape > 0.05]
    ax.scatter(ok.truth, ok.model, s=9, c="#2b8a3e", alpha=0.55, linewidths=0,
               label=f"APE<=5% (n={len(ok)})")
    ax.scatter(bad.truth, bad.model, s=12, c="#d9480f", alpha=0.7, linewidths=0,
               label=f"APE>5% (n={len(bad)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ground truth (ms)"); ax.set_ylabel("model (ms)")
    ax.set_title(f"{label}\nMAPE {g.ape.mean()*100:.2f}%  P95 {g.ape.quantile(.95)*100:.2f}%")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, which="both")
fig.suptitle(args.title, fontsize=11)
fig.tight_layout()
fig.savefig(args.out, bbox_inches="tight")
print(args.out)
