"""
DAG/Flow representation for task-oriented dialogue flows.

Each node is an "intent" with actor (user/agent) and example utterances.
Edges represent valid transitions between intents.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterator
import networkx as nx


class DialogueFlow:
    """
    Wraps a networkx.DiGraph representing a task-oriented dialogue flow.

    Nodes carry attributes:
        - actor: "user" | "agent"
        - utterances: list of example utterance strings
        - name: human-readable intent label

    The graph must be a DAG (no cycles). A unique START sentinel node
    (id="__START__") and optional END sentinel ("__END__") help anchor paths.
    """

    START = "__START__"
    END = "__END__"

    def __init__(self, name: str = "flow"):
        self.name = name
        self.graph: nx.DiGraph = nx.DiGraph()
        # Add sentinel nodes
        self.graph.add_node(self.START, actor=None, utterances=[], name="START")
        self.graph.add_node(self.END, actor=None, utterances=[], name="END")

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_intent(
        self,
        node_id: str,
        actor: str,
        utterances: List[str],
        name: Optional[str] = None,
        **attrs,
    ) -> None:
        """Add (or update) an intent node."""
        if actor not in ("user", "agent"):
            raise ValueError(f"actor must be 'user' or 'agent', got {actor!r}")
        self.graph.add_node(
            node_id,
            actor=actor,
            utterances=list(utterances),
            name=name or node_id,
            **attrs,
        )

    def add_transition(self, src: str, dst: str, **attrs) -> None:
        """Add a directed edge (transition) from src to dst."""
        if src not in self.graph:
            raise KeyError(f"Source node {src!r} not in graph")
        if dst not in self.graph:
            raise KeyError(f"Destination node {dst!r} not in graph")
        self.graph.add_edge(src, dst, **attrs)

    def set_start(self, node_id: str) -> None:
        """Connect the START sentinel to the given node."""
        self.graph.add_edge(self.START, node_id)

    def set_end(self, node_id: str) -> None:
        """Connect the given node to the END sentinel."""
        self.graph.add_edge(node_id, self.END)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def node_attr(self, node_id: str) -> Dict[str, Any]:
        return self.graph.nodes[node_id]

    def intent_nodes(self) -> List[str]:
        """Return all real (non-sentinel) nodes."""
        return [
            n
            for n in self.graph.nodes
            if n not in (self.START, self.END)
        ]

    def source_nodes(self) -> List[str]:
        """Nodes with no real predecessors (first turns in conversations)."""
        return [
            n
            for n in self.intent_nodes()
            if all(p in (self.START,) for p in self.graph.predecessors(n))
            or self.graph.in_degree(n) == 0
        ]

    def sink_nodes(self) -> List[str]:
        """Nodes with no real successors (last turns)."""
        return [
            n
            for n in self.intent_nodes()
            if all(s in (self.END,) for s in self.graph.successors(n))
            or self.graph.out_degree(n) == 0
        ]

    # ------------------------------------------------------------------
    # Path enumeration
    # ------------------------------------------------------------------

    def all_paths(
        self, max_depth: int = 50
    ) -> Iterator[List[str]]:
        """
        DFS enumeration of all simple paths through the DAG
        (excluding START/END sentinels).

        Yields lists of real node IDs.
        """
        sources = self.source_nodes()
        sinks = set(self.sink_nodes())

        def _dfs(node: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            path = path + [node]
            successors = [
                s
                for s in self.graph.successors(node)
                if s != self.END
            ]
            if not successors or node in sinks:
                yield path
                return
            for nxt in successors:
                if nxt not in path:  # avoid cycles (safety)
                    yield from _dfs(nxt, path, depth + 1)

        for src in sources:
            yield from _dfs(src, [], 0)

    def num_nodes(self) -> int:
        return len(self.intent_nodes())

    def __repr__(self) -> str:
        return (
            f"DialogueFlow(name={self.name!r}, "
            f"nodes={self.num_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def flow_from_intent_sequence(
    name: str,
    intents: List[Dict[str, Any]],
) -> DialogueFlow:
    """
    Build a simple linear flow from an ordered list of intent dicts.

    Each dict: {"id": str, "actor": str, "utterances": [...]}
    """
    flow = DialogueFlow(name=name)
    for intent in intents:
        flow.add_intent(
            intent["id"],
            actor=intent["actor"],
            utterances=intent.get("utterances", []),
            name=intent.get("name", intent["id"]),
        )
    # Chain them linearly
    for i in range(len(intents) - 1):
        flow.add_transition(intents[i]["id"], intents[i + 1]["id"])
    if intents:
        flow.set_start(intents[0]["id"])
        flow.set_end(intents[-1]["id"])
    return flow


def flow_from_edges(
    name: str,
    nodes: List[Dict[str, Any]],
    edges: List[tuple],
) -> DialogueFlow:
    """
    Build a flow from explicit node list and edge list.

    nodes: [{"id", "actor", "utterances", ...}]
    edges: [(src_id, dst_id), ...]
    """
    flow = DialogueFlow(name=name)
    for nd in nodes:
        flow.add_intent(
            nd["id"],
            actor=nd["actor"],
            utterances=nd.get("utterances", []),
            name=nd.get("name", nd["id"]),
        )
    for src, dst in edges:
        flow.add_transition(src, dst)
    # Auto-detect sources/sinks for sentinels
    for src in flow.source_nodes():
        flow.set_start(src)
    for snk in flow.sink_nodes():
        flow.set_end(snk)
    return flow
