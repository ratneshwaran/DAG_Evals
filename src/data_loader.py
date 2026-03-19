"""
STAR Dataset Loader

Loads the STAR (Schema-Guided Dialog Dataset) from RasaHQ/STAR and converts it into:
  - A list of dialogues: [(actor, utterance), ...]
  - Ground-truth DialogueFlow objects per task/domain

STAR source: https://github.com/RasaHQ/STAR
  5,820 dialogues, 13 domains, Wizard-of-Oz format.
  Each dialogue is a JSON file in the /dialogues/ directory.

Real STAR schema (per file):
  {
    "Events": [
      {
        "type": "utter" | "query" | "instruct" | ...,
        "actor": "User" | "Wizard" | "KnowledgeBase" | ...,
        "utterance": "...",      # only on utter events
        "intent": "...",         # semantic intent label
        "domain": "bank" | "hotel" | ...
      },
      ...
    ]
  }

We keep only type="utter" events from User/Wizard actors.
Domain mapping: "bank" → "banking", "hotel" → "hotel"

To use real data:
  git clone https://github.com/RasaHQ/STAR.git data/star
  # or download zip from https://github.com/RasaHQ/STAR/archive/refs/heads/master.zip

We focus on banking and hotel domains (as in arxiv 2411.10416).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .graph import DialogueFlow, flow_from_edges
from .fudge import Dialogue

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_DOMAINS = {"banking", "hotel", "restaurant", "ride", "weather", "trip"}

# Map raw STAR domain labels → canonical labels used in this project
DOMAIN_MAP = {
    "bank":       "banking",
    "banking":    "banking",
    "hotel":      "hotel",
    "restaurant": "restaurant",
    "ride":       "ride",
    "weather":    "weather",
    "trip":       "trip",
}

DEFAULT_DOMAINS = ["banking", "hotel"]


# ---------------------------------------------------------------------------
# Real STAR loader (local directory of JSON files)
# ---------------------------------------------------------------------------

def _load_star_local(data_dir: str, domain_filter: Optional[List[str]] = None) -> List[dict]:
    """
    Load STAR JSON files from a local directory.

    Expects files like: data/star/dialogues/1.json, 2.json, ...
    (cloned from https://github.com/RasaHQ/STAR)
    """
    dialogues_dir = os.path.join(data_dir, "dialogues")
    if not os.path.isdir(dialogues_dir):
        raise FileNotFoundError(
            f"STAR dialogues directory not found at {dialogues_dir}\n"
            f"Clone the dataset with:\n"
            f"  git clone https://github.com/RasaHQ/STAR {data_dir}"
        )

    files = sorted(
        f for f in os.listdir(dialogues_dir) if f.endswith(".json")
    )
    if not files:
        raise FileNotFoundError(f"No JSON files found in {dialogues_dir}")

    records = []
    for fname in files:
        path = os.path.join(dialogues_dir, fname)
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
            records.append(rec)
        except Exception:
            continue

    print(f"[data_loader] Loaded {len(records)} raw STAR files from {dialogues_dir}")

    if domain_filter:
        # Filter: keep records that have at least one utter event in the target domain
        canonical_filter = {DOMAIN_MAP.get(d.lower(), d.lower()) for d in domain_filter}
        filtered = []
        for rec in records:
            domains_in_rec = {
                DOMAIN_MAP.get(ev.get("domain", "").lower(), ev.get("domain", "").lower())
                for ev in rec.get("Events", [])
                if ev.get("type") == "utter"
            }
            if domains_in_rec & canonical_filter:
                filtered.append(rec)
        print(f"[data_loader] After domain filter {domain_filter}: {len(filtered)} records")
        records = filtered

    return records


# ---------------------------------------------------------------------------
# Parse a single real STAR record
# ---------------------------------------------------------------------------

def _parse_star_record(record: dict) -> List[Tuple[str, Dialogue, List[str]]]:
    """
    Parse one STAR dialogue file into a list of (domain, dialogue, intents).

    A single file may span multiple domains (multi-domain dialogues).
    We split by domain and return one entry per domain found.

    Returns: [(canonical_domain, dialogue, intent_list), ...]
    """
    events = record.get("Events", [])

    # Collect utter events only
    domain_turns: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    for ev in events:
        if ev.get("type") != "utter":
            continue

        raw_actor = str(ev.get("actor", "")).strip()
        if raw_actor not in ("User", "Wizard"):
            continue

        actor = "user" if raw_actor == "User" else "agent"
        utterance = str(ev.get("utterance", "")).strip()
        intent = str(ev.get("intent", "")).strip()
        raw_domain = str(ev.get("domain", "unknown")).lower().strip()
        domain = DOMAIN_MAP.get(raw_domain, raw_domain)

        if not utterance:
            continue

        domain_turns[domain].append((actor, utterance, intent))

    results = []
    for domain, turns in domain_turns.items():
        dialogue = [(actor, utt) for actor, utt, _ in turns]
        intents  = [intent for _, _, intent in turns]
        if dialogue:
            results.append((domain, dialogue, intents))

    return results


# ---------------------------------------------------------------------------
# Build ground-truth flows from intent annotations
# ---------------------------------------------------------------------------

def _build_flow_from_annotations(
    task: str,
    dialogues: List[Dialogue],
    intent_annotations: Optional[List[List[str]]] = None,
) -> DialogueFlow:
    """
    Build a DialogueFlow from observed intent sequences.

    If intent_annotations provided: nodes = unique intents, edges = observed transitions.
    Otherwise: heuristic linear flow with alternating user/agent nodes per turn position.
    """
    flow = DialogueFlow(name=task)

    if intent_annotations:
        node_utts:   Dict[str, List[str]] = defaultdict(list)
        node_actors: Dict[str, str] = {}
        edges: set = set()

        for dial, intents in zip(dialogues, intent_annotations):
            prev = None
            for (actor, utt), intent in zip(dial, intents):
                if not intent:
                    continue
                node_id = intent
                node_actors[node_id] = actor
                node_utts[node_id].append(utt)
                if prev is not None and prev != node_id:
                    edges.add((prev, node_id))
                prev = node_id

        for node_id, actor in node_actors.items():
            flow.add_intent(node_id, actor=actor, utterances=node_utts[node_id][:30])
        for src, dst in edges:
            if src in flow.graph and dst in flow.graph:
                flow.add_transition(src, dst)

    else:
        max_turns = max((len(d) for d in dialogues), default=0)
        for i in range(max_turns):
            actor = "user" if i % 2 == 0 else "agent"
            node_id = f"turn_{i}"
            utts = [d[i][1] for d in dialogues if i < len(d) and d[i][0] == actor]
            flow.add_intent(node_id, actor=actor, utterances=utts or [""])
            if i > 0:
                flow.add_transition(f"turn_{i-1}", node_id)

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
    data_dir: Optional[str] = None,
) -> Dict[str, dict]:
    """
    Load STAR dataset and return a dict keyed by domain.

    Priority:
      1. Local STAR clone at data_dir (or data/star/ by default)
      2. Synthetic fallback (for offline/testing use)

    Returns:
      {
        "banking": {
            "dialogues": [[(actor, utt), ...], ...],
            "flow": DialogueFlow,
            "split": {
                "in_task":     [dial, ...],
                "out_of_task": [dial, ...],
            }
        },
        ...
      }
    """
    if domains is None:
        domains = DEFAULT_DOMAINS

    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "star"
        )

    print(f"[data_loader] Loading STAR for domains: {domains}")

    # Try real STAR data first
    try:
        records = _load_star_local(data_dir, domain_filter=domains)
        return _process_real_records(records, domains, max_dialogues_per_domain)
    except FileNotFoundError as e:
        print(f"[data_loader] Real STAR not found: {e}")
        print("[data_loader] Falling back to synthetic data for testing...")
        print("[data_loader] To use real data: git clone https://github.com/RasaHQ/STAR data/star")
    except Exception as e:
        print(f"[data_loader] Error loading real STAR: {e}")
        print("[data_loader] Falling back to synthetic data...")

    return _make_synthetic_data(domains)


def _process_real_records(
    records: List[dict],
    domains: List[str],
    max_per_domain: Optional[int],
) -> Dict[str, dict]:
    """Parse real STAR records and group by domain."""

    domain_dialogues: Dict[str, List[Dialogue]]    = defaultdict(list)
    domain_intents:   Dict[str, List[List[str]]]   = defaultdict(list)

    for rec in records:
        for domain, dialogue, intents in _parse_star_record(rec):
            if domain not in domains:
                continue
            domain_dialogues[domain].append(dialogue)
            domain_intents[domain].append(intents)

    result = {}
    all_domains = list(domain_dialogues.keys())

    for domain in domains:
        diags = domain_dialogues.get(domain, [])
        if not diags:
            print(f"[data_loader] Warning: no dialogues found for domain '{domain}'")
            continue

        if max_per_domain:
            diags   = diags[:max_per_domain]
            intents = domain_intents[domain][:max_per_domain]
        else:
            intents = domain_intents[domain]

        flow = _build_flow_from_annotations(
            domain,
            diags,
            intent_annotations=intents if any(intents) else None,
        )

        # Out-of-task: dialogues from all other domains
        out_of_task: List[Dialogue] = []
        for other in all_domains:
            if other != domain:
                out_of_task.extend(domain_dialogues[other][:50])

        result[domain] = {
            "dialogues": diags,
            "flow": flow,
            "split": {
                "in_task":     diags[:50],
                "out_of_task": out_of_task[:50],
            },
        }

    return result


# ---------------------------------------------------------------------------
# Synthetic fallback data (no download required)
# ---------------------------------------------------------------------------

def _make_synthetic_data(domains: List[str]) -> Dict[str, dict]:
    """
    Generate synthetic dialogues based on realistic banking/hotel templates.
    Used when the real STAR dataset is not available locally.
    """
    templates = {
        "banking": {
            "nodes": [
                {"id": "greet_user",       "actor": "user",  "utterances": [
                    "Hello, I need help with my account.",
                    "Hi there, I have a banking question.",
                    "Good morning, I need assistance please.",
                ]},
                {"id": "greet_agent",      "actor": "agent", "utterances": [
                    "Hello! How can I help you today?",
                    "Good morning! What can I do for you?",
                    "Hi! I'm happy to assist. What do you need?",
                ]},
                {"id": "request_balance",  "actor": "user",  "utterances": [
                    "What is my current balance?",
                    "Can you check my account balance?",
                    "I'd like to know how much money I have.",
                ]},
                {"id": "provide_balance",  "actor": "agent", "utterances": [
                    "Your current balance is $1,234.56.",
                    "You have $567.89 in your account.",
                    "Your balance stands at $2,100.00.",
                ]},
                {"id": "transfer_request", "actor": "user",  "utterances": [
                    "I want to transfer money to another account.",
                    "Please send $200 to account 12345.",
                    "Can I transfer funds?",
                ]},
                {"id": "confirm_transfer", "actor": "agent", "utterances": [
                    "Your transfer has been completed successfully.",
                    "The funds have been sent. Is there anything else?",
                    "Transfer confirmed. Your new balance is $1,034.56.",
                ]},
                {"id": "farewell_user",    "actor": "user",  "utterances": [
                    "Thank you, goodbye.",
                    "That's all I needed. Bye!",
                    "Great, thanks for your help.",
                ]},
                {"id": "farewell_agent",   "actor": "agent", "utterances": [
                    "You're welcome! Have a great day!",
                    "Goodbye! Thank you for banking with us.",
                    "Take care! Don't hesitate to call again.",
                ]},
            ],
            "edges": [
                ("greet_user",      "greet_agent"),
                ("greet_agent",     "request_balance"),
                ("greet_agent",     "transfer_request"),
                ("request_balance", "provide_balance"),
                ("provide_balance", "transfer_request"),
                ("provide_balance", "farewell_user"),
                ("transfer_request","confirm_transfer"),
                ("confirm_transfer","farewell_user"),
                ("farewell_user",   "farewell_agent"),
            ],
        },
        "hotel": {
            "nodes": [
                {"id": "greet_user",        "actor": "user",  "utterances": [
                    "Hi, I want to book a room.",
                    "Hello, I need a hotel reservation.",
                    "I'd like to make a booking please.",
                ]},
                {"id": "greet_agent",       "actor": "agent", "utterances": [
                    "Welcome! I'd be happy to help you book a room.",
                    "Hello! Looking for accommodation?",
                    "Hi there! Let me find you the perfect room.",
                ]},
                {"id": "specify_dates",     "actor": "user",  "utterances": [
                    "I need a room from Friday to Sunday.",
                    "I want to check in on the 15th and check out on the 18th.",
                    "For three nights starting tomorrow.",
                ]},
                {"id": "check_availability","actor": "agent", "utterances": [
                    "We have rooms available for those dates.",
                    "I can confirm availability for that period.",
                    "Yes, we have several options for your stay.",
                ]},
                {"id": "select_room",       "actor": "user",  "utterances": [
                    "I'll take a double room please.",
                    "Do you have a suite available?",
                    "The standard room will be fine.",
                ]},
                {"id": "confirm_booking",   "actor": "agent", "utterances": [
                    "Your room has been booked successfully.",
                    "Booking confirmed! You'll receive a confirmation email.",
                    "All set! Your reservation is confirmed.",
                ]},
                {"id": "farewell_user",     "actor": "user",  "utterances": [
                    "Perfect, thank you!",
                    "Great, see you then.",
                    "Thanks for your help!",
                ]},
                {"id": "farewell_agent",    "actor": "agent", "utterances": [
                    "You're welcome! See you soon!",
                    "We look forward to your stay!",
                    "Thank you for choosing us!",
                ]},
            ],
            "edges": [
                ("greet_user",        "greet_agent"),
                ("greet_agent",       "specify_dates"),
                ("specify_dates",     "check_availability"),
                ("check_availability","select_room"),
                ("select_room",       "confirm_booking"),
                ("confirm_booking",   "farewell_user"),
                ("farewell_user",     "farewell_agent"),
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

        in_task_diags  = _sample_dialogues_from_flow(flow, n=50, rng=rng)

        other_domains  = [d for d in templates if d != domain]
        out_task_diags = []
        for od in other_domains:
            other_flow = flow_from_edges(od, templates[od]["nodes"], templates[od]["edges"])
            out_task_diags.extend(_sample_dialogues_from_flow(other_flow, n=50, rng=rng))

        all_diags = in_task_diags + out_task_diags
        rng.shuffle(all_diags)

        result[domain] = {
            "dialogues": all_diags,
            "flow":      flow,
            "split": {
                "in_task":     in_task_diags[:50],
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

    graph   = flow.graph
    sources = flow.source_nodes()
    dialogues = []

    for _ in range(n):
        path_nodes = []
        node = rng.choice(sources)
        for _ in range(20):
            path_nodes.append(node)
            successors = [s for s in graph.successors(node) if s != DialogueFlow.END]
            if not successors:
                break
            node = rng.choice(successors)

        dial: Dialogue = []
        for nid in path_nodes:
            attr  = graph.nodes[nid]
            utts  = attr.get("utterances", [])
            actor = attr.get("actor", "user")
            if utts:
                dial.append((actor, str(rng.choice(utts))))
        if dial:
            dialogues.append(dial)

    return dialogues


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading STAR dataset (banking + hotel)...")
    data = load_star(domains=["banking", "hotel"])

    for domain, info in data.items():
        flow  = info["flow"]
        diags = info["dialogues"]
        split = info["split"]
        print(f"\nDomain: {domain}")
        print(f"  Flow:            {flow}")
        print(f"  Total dialogues: {len(diags)}")
        print(f"  In-task:         {len(split['in_task'])}")
        print(f"  Out-of-task:     {len(split['out_of_task'])}")
        if diags:
            print(f"  Sample dialogue (first 3 turns):")
            for actor, utt in diags[0][:3]:
                print(f"    [{actor}] {utt[:80]}")

    print("\n[data_loader] Done.")
