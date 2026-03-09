"""
FF1 — Flow-F1 score

Combines:
  - Faithfulness  f = 1 - avg_fudge(dialogues, flow)
  - Compactness   c = 1 - complexity(flow, total_utterances)

FF1 = harmonic mean of f and c.

Reference: "Automatic Evaluation of Task-Oriented Dialogue Flows" (arxiv 2411.10416)
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np

from .fudge import Dialogue, avg_fudge
from .graph import DialogueFlow


# ---------------------------------------------------------------------------
# Complexity penalty
# ---------------------------------------------------------------------------

def complexity(flow: DialogueFlow, total_utterances: int) -> float:
    """
    Relative complexity of the flow w.r.t. the corpus size.

    complexity = |nodes_in_flow| / total_utterances_in_corpus

    Clamped to [0, 1].
    """
    if total_utterances <= 0:
        return 0.0
    num_nodes = flow.num_nodes()
    return min(1.0, num_nodes / total_utterances)


# ---------------------------------------------------------------------------
# FF1
# ---------------------------------------------------------------------------

def ff1(
    dialogues: List[Dialogue],
    flow: DialogueFlow,
    fudge_variant: str = "min",
) -> float:
    """
    Compute FF1 (Flow-F1) score.

    Parameters
    ----------
    dialogues:     list of [(actor, utterance), ...] sequences
    flow:          DialogueFlow to evaluate
    fudge_variant: "min" or "centroid" for substitution cost

    Returns
    -------
    FF1 in [0, 1]  (higher is better)
    """
    total_utterances = sum(len(d) for d in dialogues)

    c = 1.0 - complexity(flow, total_utterances)
    f = 1.0 - avg_fudge(dialogues, flow, variant=fudge_variant)

    if c + f == 0.0:
        return 0.0
    return 2.0 * c * f / (c + f)


def ff1_breakdown(
    dialogues: List[Dialogue],
    flow: DialogueFlow,
    fudge_variant: str = "min",
) -> dict:
    """
    Return a dict with all intermediate values for inspection.
    """
    total_utterances = sum(len(d) for d in dialogues)
    compl = complexity(flow, total_utterances)
    avg_fd = avg_fudge(dialogues, flow, variant=fudge_variant)

    c = 1.0 - compl
    f = 1.0 - avg_fd
    score = 2.0 * c * f / (c + f) if (c + f) > 0 else 0.0

    return {
        "ff1": score,
        "compactness": c,
        "faithfulness": f,
        "complexity": compl,
        "avg_fudge": avg_fd,
        "num_flow_nodes": flow.num_nodes(),
        "total_utterances": total_utterances,
        "num_dialogues": len(dialogues),
    }
