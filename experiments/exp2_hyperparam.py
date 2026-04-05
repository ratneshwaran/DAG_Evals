"""
Experiment 2 — FF1 for hyperparameter selection (k = number of intent nodes)

We vary k (number of intent clusters / nodes) using k-means clustering.
This directly mirrors the paper's Figure 3 setup where k controls how many
intent nodes the discovered flow contains.

As k grows:
  - Faithfulness increases (finer-grained intents cover more dialogue patterns)
  - Compactness decreases (more nodes → higher complexity = nodes / utterances)
  - FF1 peaks at the "right" k that balances coverage vs complexity

This produces the inverted-U shape described in the paper (Figure 3).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tqdm import tqdm
from sklearn.cluster import KMeans

from src.data_loader import load_star

# Task-level defaults matching the paper's experimental setup.
# The paper used single-task corpora; using full domains inflates total_utterances
# and makes the complexity penalty negligible, suppressing the FF1 inverted-U shape.
DEFAULT_TASKS = {
    "banking": "bank_fraud_report",
    "hotel":   "hotel_book",
}

from src.graph import DialogueFlow
from src.embeddings import encode
from src.ff1 import ff1, ff1_breakdown
from src.fudge import Dialogue

# Number of clusters for building the full DAG (fixed; not the k being swept).
# Must be small enough that multiple dialogues share the same cluster-path,
# creating a skewed frequency distribution (a few common paths, many rare ones).
# bank_fraud_report / hotel_book each have ~8-12 distinct intents, so 10 works well.
N_BASE_CLUSTERS = 10


# ---------------------------------------------------------------------------
# Path-pruning flow discovery (ALG2-inspired)
# ---------------------------------------------------------------------------

def build_path_pruning_base(
    dialogues: list,
    n_clusters: int = N_BASE_CLUSTERS,
    random_state: int = 42,
) -> dict:
    """
    Pre-compute the full DAG and ranked path list for path-pruning.

    Steps:
      1. Cluster all utterances into n_clusters intent nodes (fixed).
      2. Map each dialogue to a sequence of cluster labels (deduplicated).
      3. Count how many dialogues follow each unique label sequence (path).
      4. Rank paths by frequency (most common first).

    Returns a dict with everything needed to build trimmed flows at any k.
    """
    all_actors: list = []
    all_utts:   list = []
    dial_slices: list = []    # (start, end) index into all_utts for each dialogue

    for dial in dialogues:
        start = len(all_utts)
        for actor, utt in dial:
            all_actors.append(actor)
            all_utts.append(utt)
        dial_slices.append((start, len(all_utts)))

    if not all_utts:
        return {
            "cluster_utts": {},
            "cluster_actors": {},
            "ranked_paths": [],
            "total_paths": 0,
        }

    n_clusters = max(2, min(n_clusters, len(all_utts)))

    embs   = encode(all_utts)
    km     = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(embs)

    # Per-cluster info
    cluster_utts:   dict = defaultdict(list)
    cluster_actors: dict = defaultdict(list)
    for i, (actor, utt) in enumerate(zip(all_actors, all_utts)):
        cluster_utts[labels[i]].append(utt)
        cluster_actors[labels[i]].append(actor)

    # Build deduplicated label sequence for each dialogue
    dial_paths: list = []
    for start, end in dial_slices:
        seq = list(labels[start:end])
        if not seq:
            continue
        deduped = [seq[0]]
        for lbl in seq[1:]:
            if lbl != deduped[-1]:
                deduped.append(lbl)
        dial_paths.append(tuple(deduped))

    # Rank paths by frequency
    path_counts  = Counter(dial_paths)
    ranked_paths = [path for path, _ in path_counts.most_common()]

    return {
        "cluster_utts":   dict(cluster_utts),
        "cluster_actors": dict(cluster_actors),
        "ranked_paths":   ranked_paths,
        "total_paths":    len(ranked_paths),
    }


def discover_flow_path_pruning(
    base_info: dict,
    k: int,
    name: str = "pruned",
) -> DialogueFlow:
    """
    Build a DialogueFlow by retaining only the top-k paths.

    Nodes = union of all cluster nodes appearing in any of the top-k paths.
    Edges = transitions observed in those paths.

    As k increases, both node count and edge count grow, reducing compactness.
    Faithfulness increases as more dialogue patterns are covered.
    """
    def majority_actor(actors):
        n_user  = sum(1 for a in actors if a == "user")
        n_agent = sum(1 for a in actors if a == "agent")
        return "user" if n_user >= n_agent else "agent"

    ranked_paths  = base_info["ranked_paths"]
    cluster_utts  = base_info["cluster_utts"]
    cluster_actors = base_info["cluster_actors"]

    k_actual  = max(1, min(k, len(ranked_paths)))
    top_paths = ranked_paths[:k_actual]

    used_nodes: set = set()
    used_edges: set = set()
    start_nodes: set = set()
    end_nodes:   set = set()

    for path in top_paths:
        if not path:
            continue
        for node in path:
            used_nodes.add(node)
        for i in range(len(path) - 1):
            used_edges.add((path[i], path[i + 1]))
        start_nodes.add(path[0])
        end_nodes.add(path[-1])

    flow = DialogueFlow(name=name)
    for cid in used_nodes:
        actor = majority_actor(cluster_actors.get(cid, []))
        utts  = cluster_utts.get(cid, [])[:20]
        flow.add_intent(str(cid), actor=actor, utterances=utts, name=f"cluster_{cid}")

    for src, dst in used_edges:
        flow.add_transition(str(src), str(dst))

    for n in start_nodes:
        flow.set_start(str(n))
    for n in end_nodes:
        flow.set_end(str(n))

    return flow


# ---------------------------------------------------------------------------
# k-means flow discovery (kept for exp3_sup_vs_unsup.py)
# ---------------------------------------------------------------------------

def discover_flow_kmeans(
    dialogues: list,
    k: int,
    name: str = "discovered",
    random_state: int = 42,
) -> DialogueFlow:
    """
    Simple unsupervised flow discovery via k-means clustering.
    Builds a linear chain of k cluster nodes.
    Retained for use by exp3_sup_vs_unsup.py.
    """
    all_actors = []
    all_utts   = []
    for dial in dialogues:
        for actor, utt in dial:
            all_actors.append(actor)
            all_utts.append(utt)

    if not all_utts:
        return DialogueFlow(name=name)

    embs     = encode(all_utts)
    k_actual = min(k, len(all_utts))

    km     = KMeans(n_clusters=k_actual, random_state=random_state, n_init="auto")
    labels = km.fit_predict(embs)

    cluster_utts   = {i: [] for i in range(k_actual)}
    cluster_actors = {i: [] for i in range(k_actual)}
    for actor, utt, label in zip(all_actors, all_utts, labels):
        cluster_utts[label].append(utt)
        cluster_actors[label].append(actor)

    def majority_actor(actors):
        n_user  = sum(1 for a in actors if a == "user")
        n_agent = sum(1 for a in actors if a == "agent")
        return "user" if n_user >= n_agent else "agent"

    cluster_positions = {i: [] for i in range(k_actual)}
    pos = 0
    for dial in dialogues:
        for j, (actor, utt) in enumerate(dial):
            cluster_positions[labels[pos]].append(j / max(len(dial) - 1, 1))
            pos += 1
    assert pos == len(all_utts)

    cluster_order = sorted(
        range(k_actual),
        key=lambda c: np.mean(cluster_positions[c]) if cluster_positions[c] else 0
    )

    flow = DialogueFlow(name=name)
    for rank, cid in enumerate(cluster_order):
        node_id = f"cluster_{rank}"
        actor   = majority_actor(cluster_actors[cid])
        utts    = cluster_utts[cid][:20]
        flow.add_intent(node_id, actor=actor, utterances=utts, name=node_id)

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
    tasks=None,
    save_fig=True,
):
    """
    Sweep k (number of intent nodes) via k-means clustering.

    tasks: dict mapping domain → STAR task name, e.g.
           {"banking": "bank_fraud_report", "hotel": "hotel_book"}.
           Defaults to DEFAULT_TASKS to match the paper's single-task setup.
           Pass tasks={} to disable task filtering and use the full domain.
    """
    if domains is None:
        domains = ["banking", "hotel"]
    if tasks is None:
        tasks = DEFAULT_TASKS

    print("=" * 60)
    print("Experiment 2: FF1 for Hyperparameter (k) Selection")
    print("Method: k-means (k = number of intent nodes)")
    if tasks:
        print(f"Task filter: {tasks}")
    print("=" * 60)

    # Cap at 40 dialogues to match the paper's utterance scale (~600 total):
    # STAR dialogues average ~16 turns, so 40 × 16 ≈ 640 utterances.
    DIAL_CAP = 40
    data = {}
    for domain in domains:
        task_name   = tasks.get(domain)
        task_filter = [task_name] if task_name else None
        domain_data = load_star(
            domains=[domain],
            max_dialogues_per_domain=DIAL_CAP,
            task_filter=task_filter,
        )
        data.update(domain_data)

    n_domains = len(data)
    fig, axes = plt.subplots(1, n_domains, figsize=(6 * n_domains, 5))
    if n_domains == 1:
        axes = [axes]

    results = {}

    for ax, (domain, info) in zip(axes, data.items()):
        task_label = tasks.get(domain, domain) if tasks else domain
        dialogues  = info["dialogues"]
        total_utts = sum(len(d) for d in dialogues)
        print(f"\nDomain: {domain} / task: {task_label}")
        print(f"  {len(dialogues)} dialogues, {total_utts} total utterances")

        # Sweep k from 2 up to a reasonable fraction of total utterances.
        # Paper sweeps to ~60 on ~600 utterances. We scale proportionally.
        k_max = max(10, min(80, total_utts // 8))
        if k_values is None:
            sweep = list(range(2, k_max + 1))
        else:
            sweep = [k for k in k_values if 2 <= k <= total_utts]
            if not sweep:
                sweep = list(range(2, k_max + 1))

        print(f"  Sweeping k from {sweep[0]} to {sweep[-1]}")

        ff1_scores   = []
        comp_scores  = []
        faith_scores = []

        for k in tqdm(sweep, desc=f"  k-sweep [{domain}]"):
            flow_k = discover_flow_kmeans(
                dialogues, k=k, name=f"{domain}_k{k}"
            )
            bd = ff1_breakdown(dialogues, flow_k)
            ff1_scores.append(bd["ff1"])
            comp_scores.append(bd["compactness"])
            faith_scores.append(bd["faithfulness"])

        ff1_arr  = np.array(ff1_scores)
        best_idx = int(np.argmax(ff1_arr))
        best_k   = sweep[best_idx]
        best_ff1 = float(ff1_arr.max())

        print(f"  Best k: {best_k}  (FF1={best_ff1:.3f})")

        results[domain] = {
            "k_values":     sweep,
            "ff1_scores":   ff1_scores,
            "comp_scores":  comp_scores,
            "faith_scores": faith_scores,
            "best_k":       best_k,
            "best_ff1":     best_ff1,
        }

        # Plot
        ax.plot(sweep, ff1_scores,  "b-o", linewidth=2, markersize=3, label="FF1")
        ax.plot(sweep, comp_scores, "g--", linewidth=1.5, label="Compactness")
        ax.plot(sweep, faith_scores,"r--", linewidth=1.5, label="Faithfulness")
        ax.axvline(best_k, color="navy", linestyle=":", linewidth=1.5,
                   label=f"Optimal k={best_k}")
        ax.scatter([best_k], [best_ff1], color="navy", zorder=5, s=80)
        ax.set_title(f"{task_label} — Optimal k={best_k} nodes", fontsize=13)
        ax.set_xlabel("k (number of intent nodes)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Experiment 2: FF1 Peaks at Optimal k (Compactness vs Faithfulness)",
        fontsize=12, y=1.02,
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

    parser = argparse.ArgumentParser(description="Experiment 2: FF1 k-optimisation (k-means)")
    parser.add_argument("--domains", nargs="+", default=["banking", "hotel"])
    parser.add_argument("--k-min", type=int, default=None,
                        help="Minimum k (default: 2)")
    parser.add_argument("--k-max", type=int, default=None,
                        help="Maximum k (default: auto from corpus size)")
    parser.add_argument("--no-tasks", action="store_true",
                        help="Disable task filtering (use full domain corpus)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    k_values = None
    if args.k_min is not None or args.k_max is not None:
        lo = args.k_min or 2
        hi = args.k_max or 80
        k_values = list(range(lo, hi + 1))

    tasks = {} if args.no_tasks else None  # None → use DEFAULT_TASKS

    results = run_experiment(
        domains=args.domains,
        k_values=k_values,
        tasks=tasks,
        save_fig=not args.no_save,
    )

    print("\n=== Summary ===")
    for domain, r in results.items():
        print(f"  {domain}: optimal k={r['best_k']} nodes, "
              f"FF1={r['best_ff1']:.3f}")
