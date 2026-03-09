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

import math
from typing import List, Tuple, Dict, Optional

import numpy as np
import networkx as nx

from .graph import DialogueFlow
from .embeddings import encode, cosine_dist, intent_centroid, pairwise_cosine_dist

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
) -> float:
    """
    Cost of substituting utterance `utt` (actor `actor`) with intent `node_id`.

    Returns ∞ if actor mismatch, otherwise a cosine distance in [0, 1].
    """
    node_actor = node_attr.get("actor")
    if node_actor is not None and node_actor != actor:
        return _INF

    node_utts = node_attr.get("utterances", [])
    if not node_utts:
        # No reference utterances → maximum cost (but not ∞, actor matched)
        return 1.0

    if variant == "centroid":
        if node_id not in centroid_cache:
            centroid_cache[node_id] = intent_centroid(node_utts)
        cent = centroid_cache[node_id]
        return cosine_dist(utt_emb, cent)

    else:  # "min"
        if node_id not in utt_embs_cache:
            utt_embs_cache[node_id] = encode(node_utts)
        node_embs = utt_embs_cache[node_id]
        # utt_emb shape (D,), node_embs shape (M, D)
        sims = node_embs @ utt_emb  # (M,)
        sims = np.clip(sims, -1.0, 1.0)
        dists = (1.0 - sims) / 2.0
        return float(dists.min())


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
) -> float:
    """
    Compute the unnormalised minimum edit distance between `dial` and the
    closest path in `flow` using a DFS with memoised DP columns.

    Returns raw edit distance (float).
    """
    graph = flow.graph
    n = len(dial)

    if n == 0:
        return 0.0

    # Pre-encode all dialogue utterances at once
    dial_embs = encode([utt for _, utt in dial])  # (n, D)

    # Caches keyed by node_id
    centroid_cache: Dict[str, np.ndarray] = {}
    utt_embs_cache: Dict[str, np.ndarray] = {}

    # memo[node_id] = best (element-wise min) DP column seen so far
    memo: Dict[str, DPCol] = {}

    # Initial column: cost of matching dialogue[0:i] with empty path
    # col[i] = i (delete all i dialogue turns)
    init_col = np.arange(n + 1, dtype=np.float64)

    best_distance = _INF

    # Source nodes (real, not START sentinel)
    sources = flow.source_nodes()

    def _dfs(node_id: str, incoming_col: DPCol, path_len: int):
        nonlocal best_distance

        attr = graph.nodes[node_id]
        col = _extend_column(
            incoming_col,
            dial,
            dial_embs,
            node_id,
            attr,
            variant,
            centroid_cache,
            utt_embs_cache,
        )

        # Memoisation: if we've visited this node with a dominating column,
        # prune. Otherwise update memo and continue.
        if node_id in memo:
            prev_best = memo[node_id]
            if np.all(prev_best <= col):
                # Previous visit dominated — prune this branch
                return
            # Element-wise min: keep the best from either visit
            memo[node_id] = np.minimum(prev_best, col)
        else:
            memo[node_id] = col.copy()

        # Candidate terminal: record best[n] (full dialogue consumed)
        candidate = col[n]
        if candidate < best_distance:
            best_distance = candidate

        # Recurse into successors (excluding END sentinel)
        for nxt in graph.successors(node_id):
            if nxt == DialogueFlow.END:
                continue
            _dfs(nxt, col, path_len + 1)

    for src in sources:
        _dfs(src, init_col, 1)

    return best_distance if best_distance != _INF else float(n)


# ---------------------------------------------------------------------------
# Normalised FuDGE
# ---------------------------------------------------------------------------

def _path_lengths(flow: DialogueFlow, max_depth: int = 100) -> List[int]:
    """Return list of lengths (node counts) of all paths in the flow."""
    return [len(p) for p in flow.all_paths(max_depth=max_depth)]


def fudge(
    dialogue: Dialogue,
    flow: DialogueFlow,
    variant: str = "min",
) -> float:
    """
    Compute normalised FuDGE score for a single dialogue against a flow.

    FuDGE = raw_edit_distance / max(|D|, |best_path|)

    Returns a float in [0, 1].
    """
    if not dialogue:
        return 0.0

    raw = _fudge_efficient(dialogue, flow, variant=variant)

    # Approximate best-path length using closest path in flow
    # For normalisation we use max(|D|, |flow_path|) over the best-matching path.
    # We use the maximum path length as a conservative upper bound.
    path_lens = _path_lengths(flow)
    if path_lens:
        max_path = max(path_lens)
    else:
        max_path = 1

    denom = max(len(dialogue), max_path)
    if denom == 0:
        return 0.0
    return float(raw) / denom


def avg_fudge(
    dialogues: List[Dialogue],
    flow: DialogueFlow,
    variant: str = "min",
) -> float:
    """
    Average normalised FuDGE over a list of dialogues.
    """
    if not dialogues:
        return 0.0
    scores = [fudge(d, flow, variant=variant) for d in dialogues]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Naive O(paths × |D|) reference implementation (for small flows / testing)
# ---------------------------------------------------------------------------

def fudge_naive(
    dialogue: Dialogue,
    flow: DialogueFlow,
    variant: str = "min",
) -> float:
    """
    Naively enumerate every path and compute Levenshtein distance.
    Only feasible for small DAGs. Useful for unit tests.
    """
    from .graph import DialogueFlow as _DF

    graph = flow.graph
    n = len(dialogue)

    if n == 0:
        return 0.0

    dial_embs = encode([utt for _, utt in dialogue])
    centroid_cache: Dict[str, np.ndarray] = {}
    utt_embs_cache: Dict[str, np.ndarray] = {}

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
