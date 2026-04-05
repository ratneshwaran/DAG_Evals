"""
FuDGE — Fuzzy Dialogue-Graph Edit Distance

Implements the core metric from:
  "Automatic Evaluation of Task-Oriented Dialogue Flows"
  (arxiv 2411.10416)

A dialogue D = [(actor, utterance), ...] is compared against a flow DAG.
FuDGE is the minimum normalised edit distance between D and any path through
the DAG, where the substitution cost between an utterance and an intent node
is derived from cosine similarity of sentence embeddings.

Two substitution-cost variants are supported:
  - "min":      min cosine_dist(utterance, u) for u in intent.utterances
  - "centroid": cosine_dist(utterance, centroid_of_intent)

Algorithm: efficient DFS with memoised DP columns (Algorithm 2 in paper).
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional

import numpy as np
import networkx as nx

from .graph import DialogueFlow
from .embeddings import encode, cosine_dist, intent_centroid

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Dialogue = List[Tuple[str, str]]   # [(actor, utterance), ...]
DPCol   = np.ndarray               # shape (len(dialogue)+1,) float32

_INF = float("inf")


# ---------------------------------------------------------------------------
# Substitution cost
# ---------------------------------------------------------------------------

def _sub_cost(
    utt: str,
    utt_emb: np.ndarray,
    node_id: str,
    node_attr: dict,
    variant: str,
    actor: str,
    centroid_cache: Dict[str, np.ndarray],
    utt_embs_cache: Dict[str, np.ndarray],
    all_centroids: Optional[Dict[str, np.ndarray]] = None,
    all_actors: Optional[Dict[str, str]] = None,
    alpha: float = 0.5,
) -> float:
    """
    Cost of substituting utterance `utt` (actor `actor`) with intent `node_id`.

    Implements the paper's Equation 8:
        costsub(Br, u) = alpha * (d1(Br, u) + d2(Br, B*))
    where:
        d1 = intent-utterance distance (min or centroid variant)
        B* = nearest intent to u (same actor) in the full intent set
        d2 = cosine distance between centroids of Br and B*
        alpha = 0.5 (keeps cost in [0, 1])

    If all_centroids/all_actors are None, falls back to d1-only (no d2 term).
    Returns inf if actor mismatch.
    """
    node_actor = node_attr.get("actor")
    if node_actor is not None and node_actor != actor:
        return _INF

    node_utts = node_attr.get("utterances", [])
    if not node_utts:
        return 1.0

    # d1: intent-utterance distance
    if variant == "centroid":
        if node_id not in centroid_cache:
            centroid_cache[node_id] = intent_centroid(node_utts)
        cent = centroid_cache[node_id]
        d1 = cosine_dist(utt_emb, cent)
    else:  # "min"
        if node_id not in utt_embs_cache:
            utt_embs_cache[node_id] = encode(node_utts)
        node_embs = utt_embs_cache[node_id]
        sims = node_embs @ utt_emb  # (M,)
        sims = np.clip(sims, -1.0, 1.0)
        dists = (1.0 - sims) / 2.0
        d1 = float(dists.min())

    # d2: intent-intent distance between Br and B* (nearest intent to u)
    if all_centroids is None or all_actors is None:
        return d1

    # Ensure Br centroid is cached
    if node_id not in centroid_cache:
        centroid_cache[node_id] = intent_centroid(node_utts)
    br_cent = centroid_cache[node_id]

    # Find B* = argmin cosine_dist(centroid_Bs, utt_emb) where Bs.actor == actor
    best_d2 = 1.0
    for other_id, other_cent in all_centroids.items():
        if all_actors.get(other_id) != actor:
            continue
        # Distance from u to this intent centroid
        sim_u = float(np.dot(utt_emb, other_cent))
        sim_u = max(-1.0, min(1.0, sim_u))
        dist_u = (1.0 - sim_u) / 2.0
        if dist_u < best_d2:
            best_d2 = dist_u
            best_cent = other_cent

    # d2 = cosine_dist(Br centroid, B* centroid)
    d2 = cosine_dist(br_cent, best_cent) if best_d2 < 1.0 else 0.0

    return alpha * (d1 + d2)


# ---------------------------------------------------------------------------
# DP column extension
# ---------------------------------------------------------------------------

def _extend_column(
    prev_col: DPCol,
    dial: Dialogue,
    dial_embs: np.ndarray,
    node_id: str,
    node_attr: dict,
    variant: str,
    centroid_cache: Dict[str, np.ndarray],
    utt_embs_cache: Dict[str, np.ndarray],
    all_centroids: Optional[Dict[str, np.ndarray]] = None,
    all_actors: Optional[Dict[str, str]] = None,
    alpha: float = 0.5,
) -> DPCol:
    """
    Given the DP column `prev_col` (costs after consuming dialogue[0:i] and
    ending at the predecessor), produce the new column after visiting `node_id`.

    Standard Levenshtein DP extended along a new "column" (one flow node):
      col[0]   = prev_col[0] + 1  (delete this flow node, consume no dialogue)
      col[i]   = min(
                   col[i-1] + 1,           # delete dial[i-1] (ins in dialogue)
                   prev_col[i] + 1,         # skip this flow node (deletion)
                   prev_col[i-1] + sub,     # substitute
                 )
    where sub = _sub_cost(...).
    """
    n = len(dial)
    col = np.empty(n + 1, dtype=np.float64)
    col[0] = prev_col[0] + 1.0  # skip this intent (insert empty)

    for i in range(1, n + 1):
        actor_i, utt_i = dial[i - 1]
        sub = _sub_cost(
            utt_i,
            dial_embs[i - 1],
            node_id,
            node_attr,
            variant,
            actor_i,
            centroid_cache,
            utt_embs_cache,
            all_centroids,
            all_actors,
            alpha,
        )
        # Clamp ∞ substitution to a large finite penalty for DP stability
        sub_cost = sub if sub != _INF else n + 1.0

        col[i] = min(
            col[i - 1] + 1.0,        # skip dialogue turn (insertion)
            prev_col[i] + 1.0,       # skip flow node (deletion)
            prev_col[i - 1] + sub_cost,  # substitute
        )

    return col


# ---------------------------------------------------------------------------
# Efficient DFS algorithm (Algorithm 2)
# ---------------------------------------------------------------------------

def _fudge_efficient(
    dial: Dialogue,
    flow: DialogueFlow,
    variant: str = "min",
    centroid_cache: Optional[Dict[str, np.ndarray]] = None,
    utt_embs_cache: Optional[Dict[str, np.ndarray]] = None,
    alpha: float = 0.5,
) -> float:
    """
    Compute the unnormalised minimum edit distance between `dial` and the
    closest path in `flow` using iterative relaxation (Bellman-Ford style).

    Instead of DFS (exponential on dense/cyclic graphs), we propagate DP
    columns one hop at a time for n+2 rounds.  Each round is O(E * n),
    total O((n+2) * E * n) — tractable even for dense flows.

    memo[node] stores the element-wise best (minimum) DP column seen at
    that node across all paths of any length up to the current round.
    The best alignment score is min(col[n]) over all nodes.

    Returns raw edit distance (float).
    """
    graph = flow.graph
    n = len(dial)

    if n == 0:
        return 0.0

    # Pre-encode all dialogue utterances at once
    dial_embs = encode([utt for _, utt in dial])  # (n, D)

    # Caches keyed by node_id — shared across calls when passed in from avg_fudge
    if centroid_cache is None:
        centroid_cache = {}
    if utt_embs_cache is None:
        utt_embs_cache = {}

    # Precompute all intent centroids and actor map for the d2 term (Eq. 8)
    all_centroids: Dict[str, np.ndarray] = {}
    all_actors: Dict[str, str] = {}
    for nid in flow.intent_nodes():
        attr = graph.nodes[nid]
        all_actors[nid] = attr.get("actor", "agent")
        if nid not in centroid_cache:
            centroid_cache[nid] = intent_centroid(attr.get("utterances", []))
        all_centroids[nid] = centroid_cache[nid]

    # Initial column: cost of matching dialogue[0:i] with empty path
    # col[i] = i (delete all i dialogue turns)
    init_col = np.arange(n + 1, dtype=np.float64)

    # memo[node_id] = best DP column (element-wise min over all paths to node)
    memo: Dict[str, DPCol] = {}

    # Seed memo from source nodes
    for src in flow.source_nodes():
        attr = graph.nodes[src]
        col = _extend_column(
            init_col, dial, dial_embs, src, attr, variant,
            centroid_cache, utt_embs_cache,
            all_centroids, all_actors, alpha,
        )
        memo[src] = col.copy()

    # Iterative relaxation: propagate one hop per round for n+2 rounds.
    # After k rounds memo captures the best alignment over paths of length ≤ k+1.
    for _ in range(n + 2):
        # Snapshot incoming columns so this round is a single-hop step.
        snapshot = {nid: col.copy() for nid, col in memo.items()}
        changed = False

        for node_id, incoming_col in snapshot.items():
            for nxt in graph.successors(node_id):
                if nxt == DialogueFlow.END:
                    continue
                attr = graph.nodes[nxt]
                new_col = _extend_column(
                    incoming_col, dial, dial_embs, nxt, attr, variant,
                    centroid_cache, utt_embs_cache,
                    all_centroids, all_actors, alpha,
                )
                if nxt in memo:
                    if np.all(memo[nxt] <= new_col):
                        continue  # no improvement at any position
                    updated = np.minimum(memo[nxt], new_col)
                    if not np.array_equal(updated, memo[nxt]):
                        memo[nxt] = updated
                        changed = True
                else:
                    memo[nxt] = new_col.copy()
                    changed = True

        if not changed:
            break  # converged early

    if not memo:
        return float(n)
    return float(min(col[n] for col in memo.values()))


# ---------------------------------------------------------------------------
# Normalised FuDGE
# ---------------------------------------------------------------------------

def _flow_max_path(flow: DialogueFlow) -> int:
    """
    Return an upper bound on the longest path length through the flow.

    For sparse DAGs: enumerate paths (capped at 200) to get the true max.
    For dense or cyclic flows (where enumeration is expensive): fall back to
    the number of intent nodes as a proxy — this is O(1) and avoids the
    exponential all_paths() call.

    Density threshold: avg out-degree > 3 signals a dense/cyclic flow.
    """
    num_nodes = flow.num_nodes()
    if num_nodes == 0:
        return 1
    num_edges = flow.graph.number_of_edges()
    avg_out_degree = num_edges / max(num_nodes, 1)
    if avg_out_degree > 3.0:
        # Dense or cyclic flow — skip path enumeration
        return num_nodes
    # Sparse DAG: enumerate a sample of paths
    path_lens = [len(p) for p in flow.all_paths(max_depth=50, max_paths=200)]
    return max(path_lens) if path_lens else num_nodes


def fudge(
    dialogue: Dialogue,
    flow: DialogueFlow,
    variant: str = "min",
    alpha: float = 0.5,
    _max_path: Optional[int] = None,
    _centroid_cache: Optional[Dict[str, np.ndarray]] = None,
    _utt_embs_cache: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """
    Compute normalised FuDGE score for a single dialogue against a flow.

    FuDGE = raw_edit_distance / max(|D|, |best_path|)

    Returns a float in [0, 1].

    alpha: weight for the two-part substitution cost (paper Eq. 8, default 0.5).
    _max_path, _centroid_cache, _utt_embs_cache: pass from avg_fudge to share
    expensive precomputations (path length, node embeddings) across dialogues.
    """
    if not dialogue:
        return 0.0

    raw = _fudge_efficient(
        dialogue, flow, variant=variant,
        centroid_cache=_centroid_cache,
        utt_embs_cache=_utt_embs_cache,
        alpha=alpha,
    )

    if _max_path is None:
        _max_path = _flow_max_path(flow)

    denom = max(len(dialogue), _max_path)
    if denom == 0:
        return 0.0
    return float(raw) / denom


def avg_fudge(
    dialogues: List[Dialogue],
    flow: DialogueFlow,
    variant: str = "min",
    alpha: float = 0.5,
) -> float:
    """
    Average normalised FuDGE over a list of dialogues.

    Precomputes once per flow: max-path length and node embedding caches.
    Each node's utterances are encoded only once, not once per dialogue.
    """
    if not dialogues:
        return 0.0
    max_path = _flow_max_path(flow)
    centroid_cache: Dict[str, np.ndarray] = {}
    utt_embs_cache: Dict[str, np.ndarray] = {}
    scores = [
        fudge(
            d, flow, variant=variant, alpha=alpha,
            _max_path=max_path,
            _centroid_cache=centroid_cache,
            _utt_embs_cache=utt_embs_cache,
        )
        for d in dialogues
    ]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Naive O(paths × |D|) reference implementation (for small flows / testing)
# ---------------------------------------------------------------------------

def fudge_naive(
    dialogue: Dialogue,
    flow: DialogueFlow,
    variant: str = "min",
    alpha: float = 0.5,
) -> float:
    """
    Naively enumerate every path and compute Levenshtein distance.
    Only feasible for small DAGs. Useful for unit tests.
    """
    graph = flow.graph
    n = len(dialogue)

    if n == 0:
        return 0.0

    dial_embs = encode([utt for _, utt in dialogue])
    centroid_cache: Dict[str, np.ndarray] = {}
    utt_embs_cache: Dict[str, np.ndarray] = {}

    # Precompute all intent centroids for d2 term
    all_centroids: Dict[str, np.ndarray] = {}
    all_actors: Dict[str, str] = {}
    for nid in flow.intent_nodes():
        attr = graph.nodes[nid]
        all_actors[nid] = attr.get("actor", "agent")
        centroid_cache[nid] = intent_centroid(attr.get("utterances", []))
        all_centroids[nid] = centroid_cache[nid]

    best = _INF

    for path in flow.all_paths():
        m = len(path)
        # Standard edit distance DP
        dp = np.zeros((n + 1, m + 1), dtype=np.float64)
        dp[:, 0] = np.arange(n + 1)
        dp[0, :] = np.arange(m + 1)

        for i in range(1, n + 1):
            actor_i, utt_i = dialogue[i - 1]
            for j in range(1, m + 1):
                node_id = path[j - 1]
                attr = graph.nodes[node_id]
                sub = _sub_cost(
                    utt_i,
                    dial_embs[i - 1],
                    node_id,
                    attr,
                    variant,
                    actor_i,
                    centroid_cache,
                    utt_embs_cache,
                    all_centroids,
                    all_actors,
                    alpha,
                )
                sub_cost = sub if sub != _INF else n + 1.0
                dp[i, j] = min(
                    dp[i - 1, j] + 1.0,
                    dp[i, j - 1] + 1.0,
                    dp[i - 1, j - 1] + sub_cost,
                )

        dist = dp[n, m]
        denom = max(n, m)
        norm = dist / denom if denom > 0 else 0.0
        if norm < best:
            best = norm

    return best if best != _INF else 1.0
