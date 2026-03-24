"""
Experiment 1 — FuDGE discriminates in-task vs out-of-task dialogues

For each domain (Bank, Hotel):
  - Compute FuDGE(d, flow) for 50 in-task dialogues
  - Compute FuDGE(d, flow) for 50 out-of-task dialogues
  - Plot distributions; compute ROC-AUC and separation statistics
  - Verify that in-task FuDGE < out-of-task FuDGE (on average)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from src.data_loader import load_star
from src.fudge import fudge, _flow_max_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_scores(dialogues, flow, variant="min", desc=""):
    """Score all dialogues sharing precomputed caches for speed."""
    max_path = _flow_max_path(flow)
    centroid_cache = {}
    utt_embs_cache = {}
    scores = []
    for dial in tqdm(dialogues, desc=desc, leave=False):
        scores.append(fudge(
            dial, flow, variant=variant,
            _max_path=max_path,
            _centroid_cache=centroid_cache,
            _utt_embs_cache=utt_embs_cache,
        ))
    return np.array(scores)


def roc_auc(in_scores, out_scores):
    """
    Compute ROC-AUC treating lower FuDGE as "positive" (in-task).
    Label in-task=0, out-of-task=1 (we want in < out, so lower score → positive).
    """
    from sklearn.metrics import roc_auc_score
    scores = np.concatenate([in_scores, out_scores])
    labels = np.concatenate([
        np.zeros(len(in_scores)),
        np.ones(len(out_scores)),
    ])
    # Higher FuDGE → more likely out-of-task (label=1), no negation needed
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(domains=None, variant="min", save_fig=True):
    if domains is None:
        domains = ["banking", "hotel"]

    print("=" * 60)
    print("Experiment 1: In-task vs Out-of-task FuDGE Discrimination")
    print(f"Variant: {variant}")
    print("=" * 60)

    data = load_star(domains=domains, max_dialogues_per_domain=200)

    n_domains = len(data)
    fig, axes = plt.subplots(
        1, n_domains, figsize=(6 * n_domains, 5), sharey=False
    )
    if n_domains == 1:
        axes = [axes]

    results = {}

    for ax, (domain, info) in zip(axes, data.items()):
        flow = info["flow"]
        in_diags = info["split"]["in_task"][:50]
        out_diags = info["split"]["out_of_task"][:50]

        print(f"\nDomain: {domain}")
        print(f"  In-task dialogues:     {len(in_diags)}")
        print(f"  Out-of-task dialogues: {len(out_diags)}")
        print(f"  Flow: {flow}")

        in_scores  = compute_scores(in_diags,  flow, variant, f"  FuDGE in-task [{domain}]")
        out_scores = compute_scores(out_diags, flow, variant, f"  FuDGE out-of-task [{domain}]")

        # Stats
        auc = roc_auc(in_scores, out_scores)
        print(f"  In-task  FuDGE: mean={in_scores.mean():.3f} ± {in_scores.std():.3f}")
        print(f"  Out-task FuDGE: mean={out_scores.mean():.3f} ± {out_scores.std():.3f}")
        print(f"  ROC-AUC:        {auc:.3f}")
        print(f"  Separation OK:  {in_scores.mean() < out_scores.mean()}")

        results[domain] = {
            "in_scores": in_scores,
            "out_scores": out_scores,
            "auc": auc,
            "in_mean": in_scores.mean(),
            "out_mean": out_scores.mean(),
        }

        # Plot
        bins = np.linspace(0, 1, 25)
        ax.hist(in_scores,  bins=bins, alpha=0.6, color="steelblue",  label=f"In-task (n={len(in_scores)})")
        ax.hist(out_scores, bins=bins, alpha=0.6, color="tomato", label=f"Out-of-task (n={len(out_scores)})")
        ax.axvline(in_scores.mean(),  color="steelblue",  linestyle="--", linewidth=1.5, label=f"In mean={in_scores.mean():.3f}")
        ax.axvline(out_scores.mean(), color="tomato", linestyle="--", linewidth=1.5, label=f"Out mean={out_scores.mean():.3f}")
        ax.set_title(f"{domain.capitalize()} — ROC-AUC={auc:.3f}", fontsize=13)
        ax.set_xlabel("FuDGE score (lower = closer to flow)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    plt.suptitle(
        f"Experiment 1: FuDGE In-Task vs Out-of-Task ({variant} variant)",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    if save_fig:
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "results"
        )
        os.makedirs(out_path, exist_ok=True)
        fig_path = os.path.join(out_path, f"exp1_discrimination_{variant}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {fig_path}")

    plt.show()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1: FuDGE discrimination")
    parser.add_argument("--variant", choices=["min", "centroid"], default="min")
    parser.add_argument("--domains", nargs="+", default=["banking", "hotel"])
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    results = run_experiment(
        domains=args.domains,
        variant=args.variant,
        save_fig=not args.no_save,
    )

    print("\n=== Summary ===")
    all_ok = True
    for domain, r in results.items():
        ok = r["in_mean"] < r["out_mean"]
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {domain}: in={r['in_mean']:.3f} < out={r['out_mean']:.3f}, AUC={r['auc']:.3f}")

    if all_ok:
        print("\nExperiment 1 PASSED: FuDGE successfully discriminates in/out-of-task.")
    else:
        print("\nExperiment 1 FAILED: Some domains did not show clear separation.")
