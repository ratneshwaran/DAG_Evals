"""
Mermaid DAG Loader

Parses Mermaid graph syntax (as output by LLMs) into DialogueFlow objects.

Expected format (from gpt5derived.js etc.):

    graph TD
    B0["B – Warm greeting and invite user to share"]
    B0 --> U1a["U – Describes a specific concern"]
    U1a --> B2a["B – Validate concern and ask about emotions"]

Node labels starting with "B – " are agent nodes; "U – " are user nodes.
The text after the prefix becomes the node's utterance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .graph import DialogueFlow


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches node definitions in various Mermaid shapes:
#   B0["B – label"]          rectangle with quotes
#   B0([Bot: label])          stadium / pill
#   U1{{User: label}}         hexagon
#   B0(["B – label"])         stadium with quotes
#   B0["B – label"]:::class   with style suffix
_NODE_SHAPES = [
    r'([A-Za-z_]\w*)\["([^"]+)"\]',           # ["..."]
    r'([A-Za-z_]\w*)\(\["([^"]+)"\]\)',        # (["..."])
    r'([A-Za-z_]\w*)\(\[([^\]]+)\]\)',         # ([...])
    r'([A-Za-z_]\w*)\(([^)]+)\)',              # (...)
    r"([A-Za-z_]\w*)\{\{([^}]+)\}\}",          # {{...}}
    r'([A-Za-z_]\w*)\[([^\]]+)\]',             # [...]  (must be last — greedy)
]
_NODE_DEF_RES = [re.compile(p) for p in _NODE_SHAPES]

# Matches edges: A --> B  or  A --> B["label"]
_EDGE_RE = re.compile(
    r'([A-Za-z_]\w*)\s*-->\s*([A-Za-z_]\w*)'
)


def _parse_actor_and_text(label: str) -> Tuple[str, str]:
    """
    Extract actor and utterance text from a Mermaid node label.

    Conventions:
      "B – Do something"  → ("agent", "Do something")
      "U – Say something"  → ("user",  "Say something")

    Falls back to "agent" if prefix is unrecognised.
    Also handles variants: "B -", "Bot -", "U -", "User -" (case-insensitive).
    """
    # Try common prefixes: dash style ("B – ...") and colon style ("Bot: ...")
    m = re.match(
        r'^(B|Bot|A|Agent)\s*[\u2013\u2014\-:]+\s*(.+)$', label, re.IGNORECASE
    )
    if m:
        return "agent", m.group(2).strip()

    m = re.match(
        r'^(U|User)\s*[\u2013\u2014\-:]+\s*(.+)$', label, re.IGNORECASE
    )
    if m:
        return "user", m.group(2).strip()

    # No recognised prefix — guess from node ID if possible, else default agent
    return "agent", label.strip()


def parse_mermaid(text: str) -> Tuple[Dict[str, dict], List[Tuple[str, str]]]:
    """
    Parse Mermaid graph text into nodes and edges.

    Returns:
        nodes: {node_id: {"actor": str, "utterance": str}}
        edges: [(src_id, dst_id), ...]
    """
    nodes: Dict[str, dict] = {}
    edges: List[Tuple[str, str]] = []

    for line in text.splitlines():
        stripped = line.strip()

        # Skip blanks, comments, directives, style lines
        if not stripped or stripped.startswith("%%") or stripped.startswith("graph "):
            continue
        if stripped.startswith("classDef ") or stripped.startswith("class "):
            continue
        if stripped.startswith("subgraph") or stripped == "end":
            continue

        # Strip :::className suffixes before matching nodes
        clean = re.sub(r':::\w+', '', stripped)

        # Extract any node definitions on this line (could be inline on edge lines)
        for regex in _NODE_DEF_RES:
            for node_id, label in regex.findall(clean):
                if node_id not in nodes:
                    actor, utterance = _parse_actor_and_text(label)
                    nodes[node_id] = {"actor": actor, "utterance": utterance}

        # Extract edges
        for src, dst in _EDGE_RE.findall(stripped):
            edges.append((src, dst))

    return nodes, edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mermaid_flow(
    path: str,
    name: Optional[str] = None,
) -> DialogueFlow:
    """
    Load a Mermaid graph file and return a DialogueFlow.

    Each node gets a single utterance (the label text). For LLM-derived
    flows this is typically a description of the intent, which works as
    a semantic anchor for FuDGE embedding comparison.

    Parameters
    ----------
    path : str or Path
        Path to the .js / .md / .txt file containing Mermaid syntax.
    name : str, optional
        Name for the flow. Defaults to the filename stem.
    """
    p = Path(path)
    if name is None:
        name = p.stem

    text = p.read_text(encoding="utf-8")
    nodes, edges = parse_mermaid(text)

    if not nodes:
        raise ValueError(f"No nodes found in {path}")

    flow = DialogueFlow(name=name)

    # Add intent nodes
    for node_id, info in nodes.items():
        flow.add_intent(
            node_id,
            actor=info["actor"],
            utterances=[info["utterance"]],
            name=node_id,
        )

    # Add edges (skip any referencing undefined nodes)
    for src, dst in edges:
        if src in flow.graph and dst in flow.graph:
            flow.add_transition(src, dst)

    # Wire sentinels — sources get START, sinks get END
    for src in flow.source_nodes():
        flow.set_start(src)
    for snk in flow.sink_nodes():
        flow.set_end(snk)

    return flow


def load_mermaid_dir(
    dir_path: str,
    extensions: Tuple[str, ...] = (".js", ".md", ".txt", ".mmd"),
) -> Dict[str, DialogueFlow]:
    """
    Load all Mermaid files from a directory.

    Returns a dict keyed by filename stem.
    """
    d = Path(dir_path)
    flows = {}
    for f in sorted(d.iterdir()):
        if f.suffix.lower() in extensions and f.is_file():
            flows[f.stem] = load_mermaid_flow(str(f))
    return flows


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.mermaid_loader <path_to_mermaid_file>")
        sys.exit(1)

    flow = load_mermaid_flow(sys.argv[1])
    print(f"Flow: {flow}")
    print(f"  Nodes: {flow.num_nodes()}")
    print(f"  Edges: {flow.graph.number_of_edges()}")
    print(f"  Sources: {flow.source_nodes()}")
    print(f"  Sinks: {flow.sink_nodes()}")
    print("\nNodes:")
    for nid in flow.intent_nodes():
        attr = flow.node_attr(nid)
        print(f"  {nid}: [{attr['actor']}] {attr['utterances'][0][:80]}")
