"""
Experiment 2 — FF1 for hyperparameter selection (k-path optimisation)

We vary k (number of retained paths) in a simple greedy flow-discovery algorithm
and compute FF1 at each k. The flow with the best FF1 is the optimal k.

Flow discovery: cluster dialogue turns with k-means → k intent nodes →
build a linear chain. As k grows, faithfulness improves but compactness drops.
FF1 balances both and peaks at the "right" k.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans

from src.data_loader import load_star
from src.graph import DialogueFlow
from src.embeddings import encode
from src.ff1 import ff1, ff1_breakdown
from src.fudge import Dialogue


# ---------------------------------------------------------------------------
# Flow discovery: k-means clustering → intent nodes
# ---------------------------------------------------------------------------

def discover_flow_kmeans(
    dialogues: list,
    k: int,
    name: str = "discovered",
    random_state: int = 42,
) -> DialogueFlow:
    """
    Simple unsupervised flow discovery:
      1. Encode all utterances.
      2. K-means with k clusters → k intent nodes.
      3. Build a linear chain connecting clusters in order of first appearance.
    """
    # Collect all (actor, utterance) turns with their cluster indices
    all_actors = []
    all_utts = []
    for dial in dialogues:
        for actor, utt in dial:
            all_actors.append(actor)
            all_utts.append(utt)

    if not all_utts:
        flow = DialogueFlow(name=name)
        return flow

    embs = encode(all_utts)  # (N, D)
    k_actual = min(k, len(all_utts))

    km = KMeans(n_clusters=k_actual, random_state=random_state, n_init="auto")
    labels = km.fit_predict(embs)

    # For each cluster, collect example utterances and majority actor
    cluster_utts = {i: [] for i in range(k_actual)}
    cluster_actors = {i: [] for i in range(k_actual)}
    for i, (actor, utt, label) in enumerate(zip(all_actors, all_utts, labels)):
        cluster_utts[label].append(utt)
        cluster_actors[label].append(actor)

    def majority_actor(actors):
        n_user = sum(1 for a in actors if a == "user")
        n_agent = sum(1 for a in actors if a == "agent")
        return "user" if n_user >= n_agent else "agent"

    # Determine cluster ordering: mean position index across all dialogues
    cluster_positions = {i: [] for i in range(k_actual)}
    pos = 0
    for dial in dialogues:
        for j, (actor, utt) in enumerate(dial):
            cluster_positions[labels[pos]].append(j / max(len(dial) - 1, 1))
            pos += 1

    cluster_order = sorted(
        range(k_actual),
        key=lambda c: np.mean(cluster_positions[c]) if cluster_positions[c] else 0
    )

    # Build flow
    flow = DialogueFlow(name=name)
    for rank, cid in enumerate(cluster_order):
        node_id = f"cluster_{rank}"
        actor = majority_actor(cluster_actors[cid])
        utts = cluster_utts[cid][:20]  # cap examples
        flow.add_intent(node_id, actor=actor, utterances=utts, name=node_id)

    # Linear chain
    for rank in range(len(cluster_order) - 1):
        flow.add_transition(f"cluster_{rank}", f"cluster_{rank+1}")

    if cluster_order:
        flow.set_start("cluster_0")
        flow.set_end(f"cluster_{len(cluster_order)-1}")

    return flow


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    domains=None,
    k_values=None,
    save_fig=True,
):
    if domains is None:
        domains = ["banking", "hotel"]
    if k_values is None:
        k_values = list(range(2, 16))

    print("=" * 60)
    print("Experiment 2: FF1 for Hyperparameter (k) Selection")
    print(f"k range: {k_values[0]} .. {k_values[-1]}")
    print("=" * 60)

    data = load_star(domains=domains, max_dialogues_per_domain=200)

    n_domains = len(data)
    fig, axes = plt.subplots(1, n_domains, figsize=(6 * n_domains, 5))
    if n_domains == 1:
        axes = [axes]

    results = {}

    for ax, (domain, info) in zip(axes, data.items()):
        dialogues = info["dialogues"][:100]
        print(f"\nDomain: {domain} ({len(dialogues)} dialogues)")

        ff1_scores = []
        comp_scores = []
        faith_scores = []

        for k in tqdm(k_values, desc=f"  k-sweep [{domain}]"):
            flow_k = discover_flow_kmeans(dialogues, k=k, name=f"{domain}_k{k}")
            bd = ff1_breakdown(dialogues, flow_k)
            ff1_scores.append(bd["ff1"])
            comp_scores.append(bd["compactness"])
            faith_scores.append(bd["faithfulness"])

        ff1_arr  = np.array(ff1_scores)
        best_k   = k_values[int(np.argmax(ff1_arr))]
        best_ff1 = float(ff1_arr.max())

        print(f"  Best k: {best_k}  (FF1={best_ff1:.3f})")

        results[domain] = {
            "k_values":     k_values,
            "ff1_scores":   ff1_scores,
            "comp_scores":  comp_scores,
            "faith_scores": faith_scores,
            "best_k":       best_k,
            "best_ff1":     best_ff1,
        }

        # Plot
        ax.plot(k_values, ff1_scores,  "b-o", linewidth=2, markersize=5, label="FF1")
        ax.plot(k_values, comp_scores, "g--", linewidth=1.5, label="Compactness")
        ax.plot(k_values, faith_scores,"r--", linewidth=1.5, label="Faithfulness")
        ax.axvline(best_k, color="navy", linestyle=":", linewidth=1.5,
                   label=f"Optimal k={best_k}")
        ax.scatter([best_k], [best_ff1], color="navy", zorder=5, s=80)
        ax.set_title(f"{domain.capitalize()} — Optimal k={best_k}", fontsize=13)
        ax.set_xlabel("k (number of intent clusters)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Experiment 2: FF1 Peaks at Optimal k (Flow Complexity vs Faithfulness)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    if save_fig:
        out_path = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(out_path, exist_ok=True)
        fig_path = os.path.join(out_path, "exp2_hyperparam_k.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {fig_path}")

    plt.show()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 2: FF1 k-optimisation")
    parser.add_argument("--domains", nargs="+", default=["banking", "hotel"])
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    k_values = list(range(args.k_min, args.k_max + 1))

    results = run_experiment(
        domains=args.domains,
        k_values=k_values,
        save_fig=not args.no_save,
    )

    print("\n=== Summary ===")
    for domain, r in results.items():
        print(f"  {domain}: optimal k={r['best_k']}, FF1={r['best_ff1']:.3f}")
