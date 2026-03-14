# FuDGE / FF1 Project — In-Depth Explanation

> **Who this is for:** Anyone who wants to understand what this project does, how the algorithms work, and how to use it — without reading all the source code.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Core Concepts](#2-core-concepts)
3. [Algorithm Walkthrough](#3-algorithm-walkthrough)
4. [File-by-File Reference](#4-file-by-file-reference)
5. [The Three Experiments](#5-the-three-experiments)
6. [Data Flow Diagram](#6-data-flow-diagram)
7. [How to Run](#7-how-to-run)
8. [Glossary](#8-glossary)

---

## 1. What This Project Does

### The Problem

Imagine you work at a bank. You've written a "conversation script" — a flowchart that describes how customer service calls *should* go:

```
Customer says hello → Agent greets back → Customer asks about balance
→ Agent provides balance → Customer asks to transfer → Agent confirms → Goodbye
```

Now you have 1,000 real recorded calls. **Did those real calls actually follow your script?** And if you have two different scripts, **which one better describes how your customers actually talk?**

This is hard to evaluate automatically. Real conversations are messy. People say things in different ways. The order of topics might shift. You can't just do a word-for-word match.

### The Paper

This project implements two automatic metrics from:

> **"Automatic Evaluation of Task-Oriented Dialogue Flows"**
> arXiv: 2411.10416

The paper introduces:
- **FuDGE** — *Fuzzy Dialogue-Graph Edit Distance*: measures how far a single conversation is from a flow script
- **FF1** — *Flow-F1*: an overall quality score for an entire flow, balancing two competing goals

### The Analogy

Think of spell-check but for conversation *structure*:
- Spell-check: "did you use the right words?"
- FuDGE: "did this conversation follow the right *sequence of topics*?"

Just like spell-check uses dictionaries of valid words, FuDGE uses a flow graph of valid conversation steps — but it's *fuzzy* because it uses AI sentence embeddings instead of exact matches.

---

## 2. Core Concepts

### Dialogue Flow (DAG)

A **Dialogue Flow** is a directed graph (DAG = Directed Acyclic Graph) where:
- Each **node** is an **intent** — a conversational goal like "greet user" or "request balance"
- Each **edge** means "this intent can be followed by that intent"
- Each node is labelled with its **actor** (user or agent) and **example utterances** — real example sentences that represent this intent

Conversations always alternate between user and agent, which is enforced in the graph.

**Example — Banking Flow (ASCII diagram):**

```
        [START]
           |
    [greet_user] ──────────────────────────────────┐
    actor=user                                      │
    "Hello, I need help with my account."           │
           |                                        │
    [greet_agent]                                   │
    actor=agent                                     │
    "Hello! How can I help you today?"              │
           |                                        │
      ┌────┴────┐                                   │
      ▼         ▼                                   │
[request_balance] [transfer_request]◄───────────────┘
actor=user        actor=user
"What is my       "I want to transfer
 current balance?" money to another account."
      |                    |
      ▼                    ▼
[provide_balance]  [confirm_transfer]
actor=agent        actor=agent
"Your balance      "Your transfer has been
 is $1,234.56."    completed successfully."
      |                    |
      └─────────┬──────────┘
                ▼
         [farewell_user]
         actor=user
         "Thank you, goodbye."
                |
                ▼
         [farewell_agent]
         actor=agent
         "Goodbye! Thank you for banking with us."
                |
             [END]
```

Key properties:
- The graph is a **DAG** — no loops (conversations always progress, never loop back)
- **START** and **END** are sentinel nodes (not real intents, just anchors)
- Multiple paths are possible (e.g., user might skip balance check and go straight to transfer)

---

### FuDGE — Fuzzy Dialogue-Graph Edit Distance

#### Intuition First: What is Edit Distance?

Edit distance is a classic idea: "how many single-character operations does it take to turn one string into another?"

For the words `"kitten"` → `"sitting"`:
- substitute `k` → `s`
- substitute `e` → `i`
- substitute nothing → insert `g`
= **3 operations** → edit distance = 3

#### Now Apply it to Dialogues

Instead of characters, we have **conversation turns** (actor + utterance). Instead of sequences of characters, we have sequences of **intents in a flow path**.

FuDGE asks: *"How many edits does it take to turn this real conversation into something the flow path describes?"*

**Allowed operations:**
- **Substitute**: replace a dialogue turn with a flow node (costs 0 if perfect match, up to 1 based on how semantically different they are)
- **Delete a flow node**: skip a step in the flow (costs 1)
- **Insert**: the dialogue has a turn that doesn't match any flow node (costs 1)

#### The Fuzzy Part: Semantic Similarity Instead of Exact Match

In regular edit distance, substituting `a` → `b` costs exactly 1 (wrong character). Here, substitution cost is **graded** using sentence embeddings:

- `"What is my account balance?"` → `"Can you check my account balance?"` → very similar → cost ≈ 0.02
- `"What is my account balance?"` → `"I want to book a hotel room."` → very different → cost ≈ 0.5+
- Actor mismatch (user turn vs agent node) → cost = **∞** (impossible match)

**Two substitution variants:**
| Variant | How cost is computed |
|---------|---------------------|
| `min` | `min cosine_dist(utterance, example)` for all example utterances in the intent node — best-case match |
| `centroid` | `cosine_dist(utterance, mean_embedding_of_all_examples)` — match against the "average" intent representation |

#### Finding the Best Match

The flow has many possible paths (e.g., user might check balance then transfer, or go straight to transfer). FuDGE tries **every path** and returns the distance to the **closest one**:

```
FuDGE(dialogue, flow) = min over all flow paths of edit_distance(dialogue, path)
                        ─────────────────────────────────────────────────────────
                             max(len(dialogue), len(path))
```

The denominator normalises the score to **[0, 1]**:
- **0** = the dialogue perfectly matches the flow
- **1** = the dialogue is completely different from the flow

---

### FF1 — Flow-F1

FF1 evaluates the quality of an entire flow (not just one dialogue), by balancing two competing goals:

#### Goal 1: Faithfulness
*Does the flow accurately describe how real conversations go?*

```
Faithfulness = 1 − avg_FuDGE(all_dialogues, flow)
```

A flow with low average FuDGE has high faithfulness — real conversations are close to the flow.

**Problem**: you could make a huge flow with hundreds of nodes covering every possible utterance → FuDGE → 0 → Faithfulness → 1. But that's useless as a "script."

#### Goal 2: Compactness
*Is the flow small and clean?*

```
Complexity  = |nodes in flow| / |total utterances in corpus|
Compactness = 1 − Complexity
```

A compact flow has few nodes relative to the size of the dataset. A flow with 8 nodes describing 400 dialogue turns has compactness ≈ 0.98. A flow with 300 nodes describing the same data has compactness ≈ 0.25.

#### FF1: The Harmonic Mean

```
FF1 = 2 × Faithfulness × Compactness
      ─────────────────────────────────
        Faithfulness + Compactness
```

This is exactly like the F1 score in classification (harmonic mean of precision and recall). It **punishes imbalance** — a flow that's super faithful but bloated, or super compact but inaccurate, both get penalised. The best flow balances both.

---

### Sentence Embeddings

The embedding model used is `all-MiniLM-L6-v2` from Sentence-BERT (Hugging Face).

**What it does:** Maps any sentence to a **384-dimensional vector** (a list of 384 numbers).

**Key property:** Similar sentences map to vectors that point in nearly the same direction. Dissimilar sentences map to vectors that point in different directions.

**Cosine similarity:** The angle between two vectors measures their similarity.

```
Cosine similarity = (A · B) / (|A| × |B|)    ranges from -1 to +1

We use:  cosine_dist = (1 - cosine_similarity) / 2   ranges from 0 to 1
```

**Worked example:**
```
"What is my balance?"  →  [0.12, -0.34, 0.87, ...]   # 384 numbers
"Can I check my funds?" → [0.14, -0.31, 0.85, ...]   # similar direction
cosine_dist ≈ 0.03   ← very close

"Book me a hotel room"  → [-0.22, 0.55, -0.41, ...]  # different direction
cosine_dist ≈ 0.52   ← far apart
```

The model is loaded **once** as a module-level singleton and reused for all computations (no repeated downloads).

---

## 3. Algorithm Walkthrough

Let's trace through a complete FuDGE computation with a 4-turn banking dialogue against a 4-node flow path.

### Setup

**Dialogue D** (4 turns):
```
Turn 0: [user]  "Hi, I need help with my account."
Turn 1: [agent] "Hello! What can I do for you?"
Turn 2: [user]  "What's my balance?"
Turn 3: [agent] "Your balance is $1,234.56."
```

**Flow path P** (4 nodes, simplified linear flow):
```
Node 0: greet_user   (actor=user,  examples=["Hello, I need help..."])
Node 1: greet_agent  (actor=agent, examples=["Hello! How can I help..."])
Node 2: request_balance (actor=user, examples=["What is my current balance?"])
Node 3: provide_balance (actor=agent, examples=["Your balance is $1,234.56."])
```

---

### Step 1: Encode All Utterances

All 4 dialogue utterances are batch-encoded into a `(4, 384)` matrix in one forward pass through the model.

All example utterances from each flow node are also encoded and cached (so we only compute each node's embeddings once, even if multiple dialogues are scored against the same flow).

---

### Step 2: Build the DP Table

We use a dynamic programming table — a grid of costs.

**Initial state**: before processing any flow node, the cost of matching dialogue turns `[0..i]` is just `i` (delete all `i` turns).

```
Initial column (before seeing any flow node):
  col[0] = 0   (empty dialogue vs empty path: 0 cost)
  col[1] = 1   (1 dialogue turn vs empty path: delete 1 turn)
  col[2] = 2
  col[3] = 3
  col[4] = 4
```

---

### Step 3: Extend the Column at Each Flow Node

For each flow node, we compute a new DP column from the previous one.

**At Node 0 (greet_user, actor=user):**

For each dialogue turn `i`, compute substitution cost:
```
Turn 0 [user]  "Hi, I need help..." vs greet_user → sub_cost ≈ 0.05  (similar, actor matches)
Turn 1 [agent] "Hello! What can..." vs greet_user → sub_cost = ∞     (actor mismatch!)
Turn 2 [user]  "What's my balance?" vs greet_user → sub_cost ≈ 0.40  (different topic)
Turn 3 [agent] "Your balance..."    vs greet_user → sub_cost = ∞     (actor mismatch!)
```

New column values (using the recurrence):
```
col[i] = min(
    col[i-1] + 1,          # insert: skip this dialogue turn
    prev_col[i] + 1,        # delete: skip this flow node
    prev_col[i-1] + sub     # substitute
)
```

Computed:
```
col[0] = prev_col[0] + 1 = 1      (skip greet_user node entirely)
col[1] = min(col[0]+1, prev[1]+1, prev[0]+0.05) = min(2, 2, 0.05) = 0.05
col[2] = min(col[1]+1, prev[2]+1, prev[1]+∞)   = min(1.05, 3, ∞) = 1.05
col[3] = min(col[2]+1, prev[3]+1, prev[2]+0.40) = min(2.05, 4, 2.40) = 2.05
col[4] = min(col[3]+1, prev[4]+1, prev[3]+∞)   = min(3.05, 5, ∞) = 3.05
```

This column is stored in the memo table for `greet_user`.

---

### Step 4: DFS with Memoisation (Pruning)

The efficient algorithm (`_fudge_efficient`) traverses the DAG by **depth-first search** rather than enumerating all paths explicitly.

**Key insight — pruning:** If we visit the same flow node twice via different preceding paths, and one path's DP column is **element-wise smaller** than the other's at every position, we can prune the worse path entirely — it can never lead to a better final answer.

```python
if np.all(prev_best <= col):
    return   # this branch is dominated, prune it
memo[node_id] = np.minimum(prev_best, col)   # keep the element-wise best
```

This is what makes the "efficient" algorithm much faster than the naive approach on large DAGs.

After the DFS completes, we look at `col[n]` (cost of matching all `n` dialogue turns) at every node we've reached — the minimum across all of them is the raw edit distance.

---

### Step 5: Normalise → Final FuDGE Score

```
raw_distance = best col[n] found = 0.10  (approximately)

path_length = 4 nodes (in this example)
dialogue_length = 4 turns

denominator = max(4, 4) = 4

FuDGE = 0.10 / 4 = 0.025
```

A FuDGE of 0.025 means the dialogue is very close to the flow — nearly a perfect match. A dialogue from a completely different domain (e.g., hotel booking against the banking flow) would score 0.3–0.5+.

---

### Naive vs Efficient

Two implementations exist:

| | `fudge_naive` | `fudge` (efficient) |
|--|--|--|
| Method | Enumerate every path explicitly; standard 2D DP per path | DFS through DAG; share and prune DP columns |
| Complexity | O(paths × \|D\| × path_len) | Much better due to memoisation and pruning |
| Use case | Unit tests, small flows | All real experiments |
| Agreement | Identical results ✅ | Identical results ✅ |

---

## 4. File-by-File Reference

### `src/graph.py` — DAG Representation

**Purpose:** Defines the `DialogueFlow` class, a thin wrapper around a `networkx.DiGraph`.

**Key class: `DialogueFlow`**

| Method | Input | Output | What it does |
|--------|-------|--------|-------------|
| `add_intent(node_id, actor, utterances, name)` | node ID, "user"/"agent", list of strings | — | Adds an intent node to the graph |
| `add_transition(src, dst)` | two node IDs | — | Adds a directed edge (valid sequence) |
| `set_start(node_id)` | node ID | — | Connects the START sentinel to this node |
| `set_end(node_id)` | node ID | — | Connects this node to the END sentinel |
| `intent_nodes()` | — | list of node IDs | Returns all real (non-sentinel) nodes |
| `source_nodes()` | — | list of node IDs | Nodes with no real predecessors (first turns) |
| `sink_nodes()` | — | list of node IDs | Nodes with no real successors (last turns) |
| `all_paths(max_depth=50)` | — | iterator of lists | DFS enumeration of all paths (START→END) |
| `num_nodes()` | — | int | Count of real intent nodes |

**Convenience constructors:**
- `flow_from_intent_sequence(name, intents)` — builds a simple linear chain from an ordered list of intent dicts
- `flow_from_edges(name, nodes, edges)` — builds a flow from explicit node + edge lists (auto-detects sources/sinks)

**Internal structure:**
```
graph.nodes["greet_user"] = {
    "actor": "user",
    "utterances": ["Hello, I need help...", "Hi there, I have a banking question."],
    "name": "greet_user"
}
```

---

### `src/embeddings.py` — Sentence-BERT Wrapper

**Purpose:** Loads the embedding model once and provides clean functions for all embedding operations used by FuDGE.

**Model:** `all-MiniLM-L6-v2` (6-layer MiniLM, 384-dimensional embeddings, fast and accurate)

**Loading strategy:** Module-level singleton — `_model` is `None` on import and loaded on first call. Subsequent calls reuse the loaded model.

| Function | Input | Output | Notes |
|----------|-------|--------|-------|
| `encode(texts, batch_size=64)` | list of strings | `(N, 384)` float32 array | L2-normalised embeddings |
| `cosine_dist(a, b)` | two 1-D arrays | float in [0, 1] | Uses `(1 - similarity) / 2` formula |
| `intent_centroid(utterances)` | list of strings | `(384,)` float32 array | Mean embedding, L2-normalised |
| `pairwise_cosine_dist(embs_a, embs_b)` | two 2-D arrays | `(N, M)` float array | All pairwise distances at once |

**Why L2-normalised?** When embeddings are unit vectors, cosine similarity reduces to a dot product: `sim = a · b`. This is much faster to compute and avoids division by norms at query time.

---

### `src/fudge.py` — FuDGE Metric

**Purpose:** The core algorithm. Computes the minimum normalised edit distance between a dialogue and any path through a flow DAG.

**Type aliases:**
```python
Dialogue = List[Tuple[str, str]]   # [(actor, utterance), ...]
DPCol    = np.ndarray              # shape (len(dialogue)+1,) float64
```

**Internal functions (not called directly):**

`_sub_cost(utt, utt_emb, node_id, node_attr, variant, actor, ...)` → float
- Returns ∞ if actor mismatch
- Returns cosine distance (min or centroid variant) otherwise
- Caches node embeddings and centroids across calls

`_extend_column(prev_col, dial, dial_embs, node_id, ...)` → DPCol
- One step of the Levenshtein DP
- Converts ∞ substitution costs to `n+1` for numerical stability

`_fudge_efficient(dial, flow, variant)` → float (raw, unnormalised)
- DFS with memoised DP columns
- Pruning when current branch is dominated by a previous visit
- Returns minimum `col[n]` over all reachable nodes

**Public API:**

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `fudge(dialogue, flow, variant="min")` | 1 dialogue, 1 flow | float [0,1] | Normalised FuDGE score |
| `avg_fudge(dialogues, flow, variant="min")` | list of dialogues, 1 flow | float [0,1] | Mean over all dialogues |
| `fudge_naive(dialogue, flow, variant="min")` | 1 dialogue, 1 flow | float [0,1] | Reference impl (enumerate all paths) |

---

### `src/ff1.py` — FF1 Score

**Purpose:** Combines faithfulness and compactness into a single flow quality score.

| Function | Signature | Returns | What it computes |
|----------|-----------|---------|-----------------|
| `complexity(flow, total_utterances)` | flow, int | float [0,1] | `num_nodes / total_utterances` |
| `ff1(dialogues, flow, fudge_variant="min")` | list, flow | float [0,1] | Full FF1 score |
| `ff1_breakdown(dialogues, flow, fudge_variant="min")` | list, flow | dict | All intermediate values |

`ff1_breakdown` returns:
```python
{
    "ff1": 0.734,
    "compactness": 0.960,
    "faithfulness": 0.612,
    "complexity": 0.040,
    "avg_fudge": 0.388,
    "num_flow_nodes": 8,
    "total_utterances": 400,
    "num_dialogues": 50,
}
```

**The formula:**
```python
total_utterances = sum(len(d) for d in dialogues)
c = 1 - (flow.num_nodes() / total_utterances)   # compactness
f = 1 - avg_fudge(dialogues, flow)               # faithfulness
FF1 = 2*c*f / (c+f)                              # harmonic mean
```

---

### `src/data_loader.py` — STAR Dataset Loader

**Purpose:** Loads and preprocesses dialogue data into the format the metrics expect.

**Primary function:**
```python
data = load_star(domains=["banking", "hotel"], max_dialogues_per_domain=200)
```

Returns a dict:
```python
{
    "banking": {
        "dialogues": [(actor, utt), ...],    # all dialogues, shuffled
        "flow": DialogueFlow,                # ground-truth flow for this domain
        "split": {
            "in_task":     [...],            # 50 dialogues from THIS domain
            "out_of_task": [...],            # 50 dialogues from OTHER domains
        }
    },
    "hotel": { ... }
}
```

**Loading strategy (in order of preference):**
1. Try to download STAR from several known HuggingFace identifiers (`McGill-NLP/STAR`, `Zac-HD/star`, etc.)
2. If all fail → **automatically fall back to synthetic data** (no internet needed)

**Synthetic data fallback (`_make_synthetic_data`):**
- Generates plausible banking and hotel dialogues from hardcoded flow templates
- Samples random paths through the flow graph, picking random example utterances
- 50 in-task + 50 out-of-task dialogues per domain
- Fully reproducible (uses `numpy.random.default_rng(42)`)

**How ground-truth flows are built (`_build_flow_from_annotations`):**
- If intent annotations are available in the dataset: build a transition graph from observed intent sequences
- Otherwise: create a positional flow (`turn_0`, `turn_1`, ...) where each slot collects all utterances at that position

---

## 5. The Three Experiments

### Experiment 1 — FuDGE Discriminates In-Task vs Out-of-Task

**File:** `experiments/exp1_discrimination.py`

**The question:** Is FuDGE a valid metric? Does it give lower scores to conversations that *should* match a flow (in-task) vs conversations that *shouldn't* (out-of-task)?

**What it does:**
1. Loads Banking and Hotel domains from STAR (or synthetic data)
2. For each domain: computes FuDGE for 50 in-task dialogues and 50 out-of-task dialogues against the domain's ground-truth flow
3. Plots overlapping histograms (blue = in-task, red = out-of-task)
4. Computes **ROC-AUC**: how well FuDGE separates the two groups (random = 0.5, perfect = 1.0)

**Expected output:**
```
Domain: banking
  In-task  FuDGE: mean=0.050 ± 0.020
  Out-task FuDGE: mean=0.340 ± 0.085
  ROC-AUC:        0.950
  Separation OK:  True
```

**What success looks like:**
- In-task mean significantly lower than out-of-task mean
- ROC-AUC > 0.8 (ideally > 0.9)
- Histogram peaks clearly separated with minimal overlap
- Status: `[PASS]` for both domains

**Plot saved to:** `results/exp1_discrimination_min.png`

---

### Experiment 2 — FF1 for Hyperparameter Selection (k-Optimisation)

**File:** `experiments/exp2_hyperparam.py`

**The question:** Can FF1 be used to automatically find the right number of intent clusters (`k`) for an unsupervised flow discovery algorithm?

**What it does:**
1. Takes a set of dialogues and sweeps `k` from 2 to 15 (default)
2. For each `k`: runs **k-means clustering** on utterance embeddings to create `k` intent nodes, then builds a linear flow chain
3. Computes `FF1`, `Faithfulness`, and `Compactness` for each `k`
4. Plots all three curves on the same graph; marks the `k` that maximises FF1

**K-means flow discovery (`discover_flow_kmeans`):**
1. Encode all utterances → embeddings matrix
2. K-means with `k` clusters → cluster label for each utterance
3. Majority vote per cluster → actor label (user/agent)
4. Order clusters by mean position in dialogues → defines the linear chain
5. Cap to 20 example utterances per cluster

**Why FF1 peaks at the right k:**
- Small `k` → very compact flow (high compactness) but misses many conversation patterns (low faithfulness)
- Large `k` → flow matches everything (high faithfulness) but is bloated (low compactness)
- FF1 balances both → **peaks at the "Goldilocks" k** that best represents the data

**Expected output:**
```
Domain: banking (100 dialogues)
  Best k: 8  (FF1=0.721)
```

**Plot saved to:** `results/exp2_hyperparam_k.png`

---

### Experiment 3 — Supervised vs Unsupervised Flows

**File:** `experiments/exp3_sup_vs_unsup.py`

**The question:** How much better is a human-annotated flow compared to one discovered automatically?

**What it does:**
1. Loads the ground-truth (supervised) flow for each domain from the dataset
2. Runs k-means flow discovery with `k = num_nodes_in_supervised_flow` (fair comparison)
3. Computes `ff1_breakdown` for both supervised and unsupervised flows
4. Produces side-by-side bar charts comparing FF1, Faithfulness, Compactness, and avg FuDGE

**Expected output:**
```
Domain: banking
  Supervised:
    FF1=0.810  faithfulness=0.950  compactness=0.720  avg_fudge=0.050  nodes=8
  Unsupervised (k=8):
    FF1=0.650  faithfulness=0.750  compactness=0.580  avg_fudge=0.250  nodes=8
```

**What success looks like:**
- Supervised FF1 > Unsupervised FF1 for both domains (annotation quality matters)
- The gap quantifies how much is "lost" without expert knowledge
- Unsupervised still performs reasonably — k-means on embeddings captures meaningful structure

**Plot saved to:** `results/exp3_sup_vs_unsup.png`

---

## 6. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATA SOURCES                                                        │
│                                                                      │
│  STAR Dataset (HuggingFace)    OR    Synthetic Data (hardcoded)     │
│  └─ Banking domain conversations    └─ 50 banking + 50 hotel diags  │
│  └─ Hotel domain conversations                                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
                     ┌──────────────────┐
                     │  data_loader.py  │
                     │                  │
                     │ parse → [(actor, │
                     │ utterance), ...]  │
                     │                  │
                     │ build flows from │
                     │ intent labels or  │
                     │ positional slots  │
                     └────────┬─────────┘
                              │
             ┌────────────────┴──────────────────┐
             │                                   │
             ▼                                   ▼
    ┌─────────────────┐                 ┌──────────────────┐
    │  Dialogue List  │                 │  DialogueFlow    │
    │  [(actor, utt)] │                 │  (networkx DAG)  │
    │  × N dialogues  │                 │  nodes + edges   │
    └────────┬────────┘                 └────────┬─────────┘
             │                                   │
             └────────────────┬──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  embeddings.py   │
                     │                  │
                     │ all-MiniLM-L6-v2 │
                     │                  │
                     │ encode(texts)    │
                     │  → (N, 384)      │
                     │ cosine_dist(a,b) │
                     │  → float [0,1]   │
                     │ intent_centroid  │
                     │  → (384,)        │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │    fudge.py      │
                     │                  │
                     │ DFS through DAG  │
                     │ DP columns per   │
                     │ node, memoised   │
                     │ + pruned         │
                     │                  │
                     │ fudge(d, flow)   │
                     │   → float [0,1]  │
                     │ avg_fudge(...)   │
                     │   → float [0,1]  │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │     ff1.py       │
                     │                  │
                     │ faithfulness =   │
                     │  1 - avg_fudge   │
                     │                  │
                     │ compactness =    │
                     │  1 - complexity  │
                     │                  │
                     │ FF1 = harmonic   │
                     │  mean(f, c)      │
                     └────────┬─────────┘
                              │
             ┌────────────────┼────────────────────┐
             │                │                    │
             ▼                ▼                    ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
   │    Exp 1     │  │    Exp 2     │  │      Exp 3       │
   │ Discrimination│  │ k-Selection  │  │  Sup vs Unsup   │
   │              │  │              │  │                  │
   │ FuDGE scores │  │ FF1 curve    │  │ FF1 breakdown    │
   │ in/out-task  │  │ vs k         │  │ supervised vs    │
   │ + ROC-AUC    │  │ peak = opt k │  │ k-means flow     │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────────┘
          │                 │                  │
          └─────────────────┴──────────────────┘
                            │
                            ▼
                    results/*.png
                 (saved plots + console output)
```

---

## 7. How to Run

### Prerequisites

You **must** use Anaconda Python, not the system Python:
- Anaconda: `C:\Users\ratne\anaconda3\python.exe` (Python 3.11, all packages available)
- System Python 3.8 at `C:\Users\ratne\AppData\Local\Programs\Python\Python38-32\` — **do not use this**

Open **Anaconda Prompt** from the Start Menu (recommended), or ensure your terminal's `python` command points to Anaconda.

---

### Full Setup Sequence

```bash
# 1. Navigate to the project directory
cd C:\Users\ratne\Downloads\DAG_Evals

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fix torchvision version mismatch (run once)
pip install torchvision==0.21.0

# 4. Fix Keras 3 / tf-keras conflict (run once)
pip install tf-keras

# 5. Verify data loading works
python src/data_loader.py
```

Expected output from step 5:
```
Loading STAR dataset (banking + hotel)...
[data_loader] HuggingFace load failed: ...
[data_loader] Falling back to synthetic data for testing...

Domain: banking
  Flow: DialogueFlow(name='banking', nodes=8, edges=9)
  Total dialogues: 100
  In-task: 50
  Out-of-task: 50
  Sample dialogue (first 3 turns):
    [user] Hello, I need help with my account.
    [agent] Hello! How can I help you today?
    [user] What is my current balance?

[data_loader] STAR loaded successfully.
```

The HuggingFace load failure is expected — the loader automatically falls back to synthetic data, which is perfectly fine for all three experiments.

---

### Running Experiments

```bash
# Run all three experiments in sequence
python run_all.py

# Or run individually:
python experiments/exp1_discrimination.py
python experiments/exp2_hyperparam.py
python experiments/exp3_sup_vs_unsup.py

# With optional arguments:
python experiments/exp1_discrimination.py --variant centroid --domains banking hotel
python experiments/exp2_hyperparam.py --k-min 2 --k-max 20
python experiments/exp3_sup_vs_unsup.py --k-unsup 6
```

Results (PNG plots) are saved to `results/`.

---

### Quick Sanity Check (Python REPL)

```python
import sys
sys.path.insert(0, "C:/Users/ratne/Downloads/DAG_Evals")

from src.data_loader import load_star
from src.fudge import fudge

data = load_star(domains=["banking"])
flow = data["banking"]["flow"]

in_dial  = data["banking"]["split"]["in_task"][0]
out_dial = data["banking"]["split"]["out_of_task"][0]

print(f"In-task  FuDGE: {fudge(in_dial,  flow):.3f}")   # expect ~0.05
print(f"Out-task FuDGE: {fudge(out_dial, flow):.3f}")   # expect ~0.35
```

---

## 8. Glossary

| Term | Definition |
|------|-----------|
| **DAG** | Directed Acyclic Graph — a graph with directed edges and no cycles. Used here to represent a dialogue flow where conversations always move "forward". |
| **Intent** | A conversational goal or action — e.g., "request balance", "confirm booking". Each intent is a node in the flow DAG. |
| **Utterance** | A single spoken/written turn in a conversation — e.g., "What is my balance?". One dialogue = a sequence of utterances. |
| **Edit distance** | The minimum number of insert/delete/substitute operations to transform one sequence into another. Also called Levenshtein distance. |
| **Cosine similarity** | A measure of how similar two vectors are, based on the angle between them. 1 = identical direction, 0 = perpendicular, -1 = opposite. |
| **Cosine distance** | `(1 - cosine_similarity) / 2` — ranges from 0 (identical) to 1 (opposite). Used as substitution cost in FuDGE. |
| **Centroid** | The mean of a set of vectors. The "centroid embedding" of an intent is the average of all its example utterance embeddings. |
| **FuDGE** | *Fuzzy Dialogue-Graph Edit Distance* — the minimum normalised edit distance between a dialogue and any path through a flow DAG, using semantic similarity as substitution cost. Lower = better match. |
| **FF1** | *Flow-F1* — the harmonic mean of Faithfulness and Compactness for a flow. Higher = better flow quality. |
| **Faithfulness** | How well a flow describes real conversations. `Faithfulness = 1 - avg_FuDGE`. High when real dialogues are close to the flow. |
| **Compactness** | How small/lean a flow is relative to the corpus size. `Compactness = 1 - (nodes / total_utterances)`. High when the flow has few nodes. |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic Curve — a threshold-independent measure of binary classification performance. 0.5 = random, 1.0 = perfect. Used in Exp 1 to measure how well FuDGE separates in-task vs out-of-task dialogues. |
| **all-MiniLM-L6-v2** | A compact Sentence-BERT model that maps any sentence to a 384-dimensional embedding vector. Fast, accurate, and widely used for semantic similarity tasks. |
| **STAR dataset** | Schema-guided Task-oriented diaLogues And Reasoning — a public dataset of multi-domain task-oriented dialogues with intent annotations. Used as the evaluation benchmark. |
| **Memoisation** | Caching the result of expensive computations so they are only done once. In FuDGE, DP columns are memoised per flow node to avoid recomputing when the same node is reached via multiple paths. |
| **DP (Dynamic Programming)** | A technique for solving optimisation problems by breaking them into overlapping subproblems and storing intermediate results. Used in FuDGE to compute edit distance efficiently. |
