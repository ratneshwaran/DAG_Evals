"""
Experiment 3 — Supervised vs Unsupervised Flows

Compare FF1 scores for:
  - Supervised flow:   built from annotated intents (ground-truth)
  - Unsupervised flow: built from k-means clustering on utterance embeddings

We evaluate both on Bank and Hotel domains from STAR
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data_loader import load_star
from src.ff1 import ff1, ff1_breakdown
from src.graph import DialogueFlow
from experiments.exp2_hyperparam import discover_flow_kmeans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    domains=None,
    k_unsup=None,
    save_fig=True,
):
    if domains is None:
        domains = ["banking", "hotel"]

    print("=" * 60)
    print("Experiment 3: Supervised vs Unsupervised Flows (FF1)")
    print("=" * 60)

    data = load_star(domains=domains, max_dialogues_per_domain=200)

    results = {}

    for domain, info in data.items():
        dialogues = info["dialogues"][:100]
        sup_flow  = info["flow"]

        # Determine k for unsupervised: use number of nodes in supervised flow
        k = k_unsup if k_unsup is not None else sup_flow.num_nodes()
        k = max(2, k)

        print(f"\nDomain: {domain}")
        print(f"  Supervised flow:   {sup_flow}")
        print(f"  Unsupervised k={k}")

        unsup_flow = discover_flow_kmeans(
            dialogues, k=k, name=f"{domain}_unsup_k{k}"
        )

        sup_bd   = ff1_breakdown(dialogues, sup_flow)
        unsup_bd = ff1_breakdown(dialogues, unsup_flow)

        print(f"\n  Supervised:")
        _print_breakdown(sup_bd)
        print(f"\n  Unsupervised (k={k}):")
        _print_breakdown(unsup_bd)

        results[domain] = {
            "supervised":   sup_bd,
            "unsupervised": unsup_bd,
            "k_unsup":      k,
        }

    # -----------------------------------------------------------------------
    # Bar chart comparison
    # -----------------------------------------------------------------------
    _plot_comparison(results, save_fig)

    return results


def _print_breakdown(bd: dict):
    print(f"    FF1={bd['ff1']:.3f}  "
          f"faithfulness={bd['faithfulness']:.3f}  "
          f"compactness={bd['compactness']:.3f}  "
          f"avg_fudge={bd['avg_fudge']:.3f}  "
          f"nodes={bd['num_flow_nodes']}")


def _plot_comparison(results: dict, save_fig: bool):
    domains = list(results.keys())
    metrics = ["ff1", "faithfulness", "compactness", "avg_fudge"]
    metric_labels = ["FF1", "Faithfulness", "Compactness\n(controlled — same k)", "Avg FuDGE"]

    n_metrics = len(metrics)
    n_domains = len(domains)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    colors_sup   = "steelblue"
    colors_unsup = "tomato"

    for ax, metric, label in zip(axes, metrics, metric_labels):
        x = np.arange(n_domains)
        width = 0.35

        sup_vals   = [results[d]["supervised"][metric]   for d in domains]
        unsup_vals = [results[d]["unsupervised"][metric] for d in domains]

        bars1 = ax.bar(x - width/2, sup_vals,   width, color=colors_sup,   label="Supervised",   alpha=0.85)
        bars2 = ax.bar(x + width/2, unsup_vals, width, color=colors_unsup, label="Unsupervised", alpha=0.85)

        ax.set_title(label, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in domains], fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Value labels on bars
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        "Experiment 3: Supervised vs Unsupervised Flow Quality (FF1)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    if save_fig:
        out_path = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(out_path, exist_ok=True)
        fig_path = os.path.join(out_path, "exp3_sup_vs_unsup.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {fig_path}")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 3: Supervised vs Unsupervised")
    parser.add_argument("--domains", nargs="+", default=["banking", "hotel"])
    parser.add_argument("--k-unsup", type=int, default=None,
                        help="Fixed k for unsupervised (default: match supervised node count)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_experiment(
        domains=args.domains,
        k_unsup=args.k_unsup,
        save_fig=not args.no_save,
    )

    print("\n=== Summary ===")
    for domain, r in results.items():
        sup_ff1   = r["supervised"]["ff1"]
        unsup_ff1 = r["unsupervised"]["ff1"]
        better = "Supervised" if sup_ff1 >= unsup_ff1 else "Unsupervised"
        print(f"  {domain}: Supervised FF1={sup_ff1:.3f}, "
              f"Unsupervised FF1={unsup_ff1:.3f}  → {better} wins")
