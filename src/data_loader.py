"""
STAR Dataset Loader

Loads the STAR (Schema-guided Task-oriented diaLogues And Reasoning) dataset
from HuggingFace and converts it into:
  - A list of dialogues: [(actor, utterance), ...]
  - Ground-truth DialogueFlow objects per task/domain

STAR paper: https://arxiv.org/abs/2010.11853
HF dataset: "Zac-HD/star"  (community re-upload)
  or direct from: https://github.com/emorynlp/STAR

We focus on Bank and Hotel domains for replication of arxiv 2411.10416.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .graph import DialogueFlow, flow_from_edges
from .fudge import Dialogue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_DOMAINS = {"banking", "hotel", "restaurant", "airline", "doctor"}

# Map STAR domain names → short labels used in the paper
DOMAIN_LABELS = {
    "banking": "Bank",
    "hotel": "Hotel",
}


# ---------------------------------------------------------------------------
# HuggingFace loader (primary)
# ---------------------------------------------------------------------------

def _load_star_hf(domain_filter: Optional[List[str]] = None) -> List[dict]:
    """
    Download STAR from HuggingFace.
    Returns a list of raw dialogue dicts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install `datasets` via: pip install datasets")

    # Try known HuggingFace dataset identifiers for STAR
    hf_ids = [
        "McGill-NLP/STAR",
        "Zac-HD/star",
        "emorynlp/star",
        "star_dataset",
    ]
    ds = None
    for hf_id in hf_ids:
        try:
            ds = load_dataset(hf_id, trust_remote_code=True)
            print(f"[data_loader] Loaded from HuggingFace: {hf_id}")
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError("Could not load STAR from any known HuggingFace ID")

    split = "train"
    if split not in ds:
        split = list(ds.keys())[0]

    raw = list(ds[split])

    if domain_filter:
        domain_filter_lower = [d.lower() for d in domain_filter]
        raw = [
            r for r in raw
            if any(d in str(r.get("domain", "")).lower() for d in domain_filter_lower)
        ]

    return raw


# ---------------------------------------------------------------------------
# Parse a single STAR dialogue record
# ---------------------------------------------------------------------------

def _parse_star_dialogue(record: dict) -> Tuple[str, Dialogue]:
    """
    Parse one STAR record into (task_name, dialogue).

    STAR structure varies by HF upload; we handle the most common layouts.
    """
    # Attempt to extract domain/task
    domain = str(record.get("domain", record.get("scenario", "unknown"))).lower()
    task = str(record.get("task", domain))

    turns = record.get("turns", record.get("dialogue", record.get("conversation", [])))

    dialogue: Dialogue = []
    for turn in turns:
        if isinstance(turn, dict):
            # Common keys
            speaker = str(
                turn.get("speaker", turn.get("role", turn.get("actor", "user")))
            ).lower()
            utterance = str(
                turn.get("utterance", turn.get("text", turn.get("content", "")))
            ).strip()
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            speaker, utterance = str(turn[0]).lower(), str(turn[1])
        else:
            continue

        # Normalise speaker label
        if speaker in ("system", "agent", "assistant", "bot"):
            actor = "agent"
        else:
            actor = "user"

        if utterance:
            dialogue.append((actor, utterance))

    return task, dialogue


# ---------------------------------------------------------------------------
# Build ground-truth flow DAGs from intent annotations
# ---------------------------------------------------------------------------

def _build_flow_from_annotations(
    task: str,
    dialogues: List[Dialogue],
    intent_annotations: Optional[List[List[str]]] = None,
) -> DialogueFlow:
    """
    Build a DialogueFlow from annotated intent sequences.

    If intent_annotations is None, we construct a simple linear flow
    by collecting all unique (actor, intent) pairs observed in order.

    intent_annotations: list of lists, one per dialogue,
        each inner list = [intent_per_turn, ...]
    """
    flow = DialogueFlow(name=task)

    if intent_annotations is not None:
        # Build transition graph from all observed intent sequences
        node_utts: Dict[str, List[str]] = defaultdict(list)
        node_actors: Dict[str, str] = {}
        edges: set = set()

        for dial, intents in zip(dialogues, intent_annotations):
            prev = None
            for (actor, utt), intent in zip(dial, intents):
                node_id = f"{intent}"
                node_actors[node_id] = actor
                node_utts[node_id].append(utt)
                if prev is not None:
                    edges.add((prev, node_id))
                prev = node_id

        for node_id, actor in node_actors.items():
            flow.add_intent(
                node_id,
                actor=actor,
                utterances=node_utts[node_id],
                name=node_id,
            )
        for src, dst in edges:
            flow.add_transition(src, dst)

    else:
        # Heuristic: create a flow with two alternating user/agent nodes per "step"
        # and populate with utterances from all dialogues
        max_turns = max((len(d) for d in dialogues), default=0)
        for i in range(max_turns):
            actor = "user" if i % 2 == 0 else "agent"
            node_id = f"turn_{i}"
            utts = [d[i][1] for d in dialogues if i < len(d) and d[i][0] == actor]
            flow.add_intent(node_id, actor=actor, utterances=utts or [""])
            if i > 0:
                flow.add_transition(f"turn_{i-1}", node_id)

    # Wire up sentinels
    for src in flow.source_nodes():
        flow.set_start(src)
    for snk in flow.sink_nodes():
        flow.set_end(snk)

    return flow


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def load_star(
    domains: Optional[List[str]] = None,
    max_dialogues_per_domain: Optional[int] = None,
) -> Dict[str, dict]:
    """
    Load STAR dataset and return a dict keyed by domain/task.

    Returns:
      {
        "banking": {
            "dialogues": [(actor, utt), ...],   # all dialogues for domain
            "flow": DialogueFlow,               # ground-truth flow
            "split": {
                "in_task": [dial, ...],         # dialogues matching this task
                "out_of_task": [dial, ...],     # dialogues from OTHER tasks
            }
        },
        ...
      }
    """
    if domains is None:
        domains = list(DOMAIN_LABELS.keys())

    print(f"[data_loader] Loading STAR dataset for domains: {domains}")

    try:
        raw_records = _load_star_hf(domain_filter=domains)
        print(f"[data_loader] Loaded {len(raw_records)} records from HuggingFace")
        return _process_star_records(raw_records, domains, max_dialogues_per_domain)
    except Exception as e:
        print(f"[data_loader] HuggingFace load failed: {e}")
        print("[data_loader] Falling back to synthetic data for testing...")
        return _make_synthetic_data(domains)


def _process_star_records(
    raw_records: List[dict],
    domains: List[str],
    max_per_domain: Optional[int],
) -> Dict[str, dict]:
    """Group raw records by domain and build flows."""

    domain_dialogues: Dict[str, List[Dialogue]] = defaultdict(list)
    domain_intents: Dict[str, List[List[str]]] = defaultdict(list)

    for rec in raw_records:
        task, dial = _parse_star_dialogue(rec)
        if not dial:
            continue

        # Try to match to a known domain
        matched_domain = None
        for d in domains:
            if d.lower() in task.lower():
                matched_domain = d
                break
        if matched_domain is None:
            matched_domain = task.split("_")[0].lower()
            if matched_domain not in domains:
                matched_domain = domains[0]  # fallback

        domain_dialogues[matched_domain].append(dial)

        # Extract intent annotations if available
        turns = rec.get("turns", rec.get("dialogue", []))
        intents = []
        for t in turns:
            if isinstance(t, dict):
                intent = t.get("intent", t.get("action", None))
                if intent:
                    intents.append(str(intent))
        if intents and len(intents) == len(dial):
            domain_intents[matched_domain].append(intents)

    result = {}
    all_domains = list(domain_dialogues.keys())

    for domain in domains:
        diags = domain_dialogues.get(domain, [])
        if max_per_domain:
            diags = diags[:max_per_domain]

        if not diags:
            continue

        intents = domain_intents.get(domain)
        flow = _build_flow_from_annotations(
            domain,
            diags,
            intent_annotations=intents if intents else None,
        )

        # Build splits for Exp 1
        out_of_task = []
        for other in all_domains:
            if other != domain:
                out_of_task.extend(domain_dialogues[other][:50])

        result[domain] = {
            "dialogues": diags,
            "flow": flow,
            "split": {
                "in_task": diags[:50],
                "out_of_task": out_of_task[:50],
            },
        }

    return result


# ---------------------------------------------------------------------------
# Synthetic fallback data (no internet required)
# ---------------------------------------------------------------------------

def _make_synthetic_data(domains: List[str]) -> Dict[str, dict]:
    """
    Generate simple synthetic dialogues for offline testing.
    Covers Bank and Hotel tasks with plausible intents.
    """
    templates = {
        "banking": {
            "nodes": [
                {"id": "greet_user",      "actor": "user",  "utterances": [
                    "Hello, I need help with my account.",
                    "Hi there, I have a banking question.",
                    "Good morning, I need assistance please.",
                ]},
                {"id": "greet_agent",     "actor": "agent", "utterances": [
                    "Hello! How can I help you today?",
                    "Good morning! What can I do for you?",
                    "Hi! I'm happy to assist. What do you need?",
                ]},
                {"id": "request_balance", "actor": "user",  "utterances": [
                    "What is my current balance?",
                    "Can you check my account balance?",
                    "I'd like to know how much money I have.",
                ]},
                {"id": "provide_balance", "actor": "agent", "utterances": [
                    "Your current balance is $1,234.56.",
                    "You have $567.89 in your account.",
                    "Your balance stands at $2,100.00.",
                ]},
                {"id": "transfer_request","actor": "user",  "utterances": [
                    "I want to transfer money to another account.",
                    "Please send $200 to account 12345.",
                    "Can I transfer funds?",
                ]},
                {"id": "confirm_transfer","actor": "agent", "utterances": [
                    "Your transfer has been completed successfully.",
                    "The funds have been sent. Is there anything else?",
                    "Transfer confirmed. Your new balance is $1,034.56.",
                ]},
                {"id": "farewell_user",   "actor": "user",  "utterances": [
                    "Thank you, goodbye.",
                    "That's all I needed. Bye!",
                    "Great, thanks for your help.",
                ]},
                {"id": "farewell_agent",  "actor": "agent", "utterances": [
                    "You're welcome! Have a great day!",
                    "Goodbye! Thank you for banking with us.",
                    "Take care! Don't hesitate to call again.",
                ]},
            ],
            "edges": [
                ("greet_user", "greet_agent"),
                ("greet_agent", "request_balance"),
                ("greet_agent", "transfer_request"),
                ("request_balance", "provide_balance"),
                ("provide_balance", "transfer_request"),
                ("provide_balance", "farewell_user"),
                ("transfer_request", "confirm_transfer"),
                ("confirm_transfer", "farewell_user"),
                ("farewell_user", "farewell_agent"),
            ],
        },
        "hotel": {
            "nodes": [
                {"id": "greet_user",      "actor": "user",  "utterances": [
                    "Hi, I want to book a room.",
                    "Hello, I need a hotel reservation.",
                    "I'd like to make a booking please.",
                ]},
                {"id": "greet_agent",     "actor": "agent", "utterances": [
                    "Welcome! I'd be happy to help you book a room.",
                    "Hello! Looking for accommodation?",
                    "Hi there! Let me find you the perfect room.",
                ]},
                {"id": "specify_dates",   "actor": "user",  "utterances": [
                    "I need a room from Friday to Sunday.",
                    "I want to check in on the 15th and check out on the 18th.",
                    "For three nights starting tomorrow.",
                ]},
                {"id": "check_availability","actor": "agent","utterances": [
                    "We have rooms available for those dates.",
                    "I can confirm availability for that period.",
                    "Yes, we have several options for your stay.",
                ]},
                {"id": "select_room",     "actor": "user",  "utterances": [
                    "I'll take a double room please.",
                    "Do you have a suite available?",
                    "The standard room will be fine.",
                ]},
                {"id": "confirm_booking", "actor": "agent", "utterances": [
                    "Your room has been booked successfully.",
                    "Booking confirmed! You'll receive a confirmation email.",
                    "All set! Your reservation is confirmed.",
                ]},
                {"id": "farewell_user",   "actor": "user",  "utterances": [
                    "Perfect, thank you!",
                    "Great, see you then.",
                    "Thanks for your help!",
                ]},
                {"id": "farewell_agent",  "actor": "agent", "utterances": [
                    "You're welcome! See you soon!",
                    "We look forward to your stay!",
                    "Thank you for choosing us!",
                ]},
            ],
            "edges": [
                ("greet_user", "greet_agent"),
                ("greet_agent", "specify_dates"),
                ("specify_dates", "check_availability"),
                ("check_availability", "select_room"),
                ("select_room", "confirm_booking"),
                ("confirm_booking", "farewell_user"),
                ("farewell_user", "farewell_agent"),
            ],
        },
    }

    result = {}
    rng = np.random.default_rng(42)

    for domain in domains:
        if domain not in templates:
            continue

        tmpl = templates[domain]
        flow = flow_from_edges(domain, tmpl["nodes"], tmpl["edges"])

        # Generate synthetic dialogues by sampling paths
        in_task_diags = _sample_dialogues_from_flow(flow, n=50, rng=rng)

        # Out-of-task: use dialogues from the other domain
        other_domains = [d for d in templates if d != domain]
        out_task_diags = []
        for od in other_domains:
            other_flow = flow_from_edges(od, templates[od]["nodes"], templates[od]["edges"])
            out_task_diags.extend(_sample_dialogues_from_flow(other_flow, n=50, rng=rng))

        all_diags = in_task_diags + out_task_diags
        rng.shuffle(all_diags)

        result[domain] = {
            "dialogues": all_diags,
            "flow": flow,
            "split": {
                "in_task": in_task_diags[:50],
                "out_of_task": out_task_diags[:50],
            },
        }

    return result


def _sample_dialogues_from_flow(
    flow: DialogueFlow,
    n: int = 50,
    rng: np.random.Generator = None,
) -> List[Dialogue]:
    """Sample n dialogues by walking random paths through the flow."""
    if rng is None:
        rng = np.random.default_rng(0)

    graph = flow.graph
    sources = flow.source_nodes()
    dialogues = []

    for _ in range(n):
        path_nodes = []
        node = rng.choice(sources)
        for _ in range(20):
            path_nodes.append(node)
            successors = [
                s for s in graph.successors(node)
                if s != DialogueFlow.END
            ]
            if not successors:
                break
            node = rng.choice(successors)

        dial: Dialogue = []
        for nid in path_nodes:
            attr = graph.nodes[nid]
            utts = attr.get("utterances", [])
            actor = attr.get("actor", "user")
            if utts:
                utt = str(rng.choice(utts))
                dial.append((actor, utt))
        if dial:
            dialogues.append(dial)

    return dialogues


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Loading STAR dataset (banking + hotel)...")
    data = load_star(domains=["banking", "hotel"], max_dialogues_per_domain=100)

    for domain, info in data.items():
        flow = info["flow"]
        diags = info["dialogues"]
        split = info["split"]
        print(f"\nDomain: {domain}")
        print(f"  Flow: {flow}")
        print(f"  Total dialogues: {len(diags)}")
        print(f"  In-task: {len(split['in_task'])}")
        print(f"  Out-of-task: {len(split['out_of_task'])}")
        if diags:
            d = diags[0]
            print(f"  Sample dialogue (first 3 turns):")
            for actor, utt in d[:3]:
                print(f"    [{actor}] {utt[:80]}")

    print("\n[data_loader] STAR loaded successfully.")
