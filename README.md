# FuDGE / FF1 — Replication of arxiv 2411.10416

Replication of **"Automatic Evaluation of Task-Oriented Dialogue Flows"**.

Implements:
- **FuDGE** (Fuzzy Dialogue-Graph Edit Distance) — minimum normalised edit distance between a dialogue and any path through a flow DAG
- **FF1** (Flow-F1) — harmonic mean of faithfulness and compactness

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
DAG_Evals/
├── requirements.txt
├── src/
│   ├── graph.py         # DialogueFlow DAG (networkx)
│   ├── embeddings.py    # Sentence-BERT (all-MiniLM-L6-v2)
│   ├── fudge.py         # FuDGE metric (efficient + naive)
│   ├── ff1.py           # FF1 score
│   └── data_loader.py   # STAR dataset loader
├── experiments/
│   ├── exp1_discrimination.py   # In-task vs out-of-task
│   ├── exp2_hyperparam.py       # FF1 for k-selection
│   └── exp3_sup_vs_unsup.py    # Supervised vs unsupervised flows
├── notebooks/
│   └── demo.ipynb       # End-to-end walkthrough
└── data/star/           # STAR dataset cache
```

---

## Quick Start

### Verify data loading
```bash
python src/data_loader.py
```

### Run experiments
```bash
# Exp 1: FuDGE discriminates in/out-of-task dialogues
python experiments/exp1_discrimination.py

# Exp 2: FF1 peaks at optimal k
python experiments/exp2_hyperparam.py --k-min 2 --k-max 15

# Exp 3: Supervised vs unsupervised flows
python experiments/exp3_sup_vs_unsup.py
```

### Notebook walkthrough
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Core API

```python
from src.graph import flow_from_intent_sequence
from src.fudge import fudge, avg_fudge
from src.ff1 import ff1, ff1_breakdown

# Build a flow
flow = flow_from_intent_sequence("banking", [
    {"id": "greet",   "actor": "user",  "utterances": ["Hello, I need help."]},
    {"id": "respond", "actor": "agent", "utterances": ["Sure, how can I help?"]},
    {"id": "balance", "actor": "user",  "utterances": ["What is my balance?"]},
    {"id": "answer",  "actor": "agent", "utterances": ["Your balance is $100."]},
])

# Score a dialogue
dialogue = [
    ("user",  "Hi, I need banking help please."),
    ("agent", "Happy to help! What do you need?"),
    ("user",  "Can you check my account balance?"),
    ("agent", "Your balance is $1,234.56."),
]

score = fudge(dialogue, flow, variant="min")       # lower = better match
print(f"FuDGE: {score:.3f}")

# FF1 over a corpus
breakdown = ff1_breakdown([dialogue] * 20, flow)
print(f"FF1={breakdown['ff1']:.3f}  "
      f"faithfulness={breakdown['faithfulness']:.3f}  "
      f"compactness={breakdown['compactness']:.3f}")
```

---

## Algorithm Notes

### FuDGE (efficient, Algorithm 2)
- DFS traversal of the DAG
- At each node, extends a DP edit-distance column
- Memoisation prunes dominated branches (element-wise column dominance)
- Normalised by `max(|dialogue|, |best_path|)`
- `variant="min"`: substitution cost = min cosine dist to any intent utterance
- `variant="centroid"`: substitution cost = cosine dist to intent centroid

### FF1
```
complexity(flow) = |flow_nodes| / total_utterances_in_corpus
compactness      = 1 - complexity
faithfulness     = 1 - avg_fudge
FF1              = 2 * compactness * faithfulness / (compactness + faithfulness)
```

---

## Dataset

Uses the **STAR** dataset (Schema-guided Task-Oriented Reasoning).
Downloaded automatically via HuggingFace `datasets`. Falls back to synthetic data if offline.

Domains used: **Banking**, **Hotel**

---

## Reference

```bibtex
@article{fudge2024,
  title   = {Automatic Evaluation of Task-Oriented Dialogue Flows},
  year    = {2024},
  journal = {arXiv},
  url     = {https://arxiv.org/abs/2411.10416}
}
```
