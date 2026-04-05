# Applying FuDGE / FF1 to Mental Health Conversation DAGs

## Context

This project adapts the metrics from "Automatic Evaluation of Task-Oriented Dialogue Flows" (arXiv 2411.10416) to evaluate LLM-derived dialogue flow DAGs for mental health conversation skills. The original paper targets task-oriented domains (banking, hotel booking); here we apply the same framework to therapeutic conversation flows where an AI agent guides a user through grounding exercises, crisis assessment, and coping strategies.

Two LLM-derived DAGs are evaluated:

| DAG | Source Model | Nodes | Edges | Scope |
|-----|-------------|-------|-------|-------|
| `gpt5derived.js` | GPT-5 | 54 | 80 | Broad: crisis handling, grounding (flashback, nightmare, dissociation, generic), talk therapy, session wrap-up, opt-out |
| `kimik2derived.js` | Kimik2 Instruct | 23 | 29 | Focused: post-flashback/nightmare/body-memory/dissociation entry, shared grounding loop, optional next steps |

---

## How the Metrics Work Here

### FuDGE (Fuzzy Dialogue-Graph Edit Distance)

FuDGE measures how well a single dialogue aligns with the nearest path through a flow DAG. It computes a normalised edit distance where substitution costs are based on sentence-embedding similarity (all-MiniLM-L6-v2) rather than exact string matching.

For mental health DAGs this means: given a real conversation where a user describes a flashback and the agent walks them through grounding, FuDGE finds the path through the DAG that best matches that conversation's structure and measures how many insertions, deletions, or fuzzy substitutions are needed to align them.

- **Low FuDGE** (closer to 0): the dialogue closely follows a path in the DAG
- **High FuDGE** (closer to 1): the dialogue diverges significantly from any path

### FF1 (Flow-F1)

FF1 is the harmonic mean of two competing qualities:

- **Faithfulness** `f = 1 - avg_fudge`: how well the DAG's paths cover real conversation patterns. A DAG that captures the actual flow of dialogues scores high.
- **Compactness** `c = 1 - (nodes / total_utterances)`: how parsimonious the DAG is relative to the corpus. A DAG with fewer nodes relative to the data is more compact.

FF1 balances coverage against complexity. A DAG that perfectly covers every dialogue but has one node per utterance would score low on compactness. A DAG with 2 nodes would be compact but unfaithful.

---

## Baseline Results

Scoring both DAGs against 10 banking dialogues from STAR (as a cross-domain baseline):

| DAG | avg_fudge | Faithfulness | Compactness | FF1 |
|-----|-----------|-------------|-------------|-----|
| gpt5 | 0.508 | 0.492 | 0.669 | 0.567 |
| kimik2 | 0.524 | 0.476 | 0.859 | 0.612 |

### Interpretation

**Kimik2 scores higher on FF1 despite lower faithfulness.** This is because:

1. **Compactness advantage**: Kimik2 has 23 nodes vs GPT-5's 54. With 160 total utterances in the test set, GPT-5's complexity is 54/160 = 0.338 (compactness 0.662), while Kimik2's is 23/160 = 0.144 (compactness 0.856). The FF1 harmonic mean rewards this.

2. **Faithfulness is similar**: Both DAGs score poorly on faithfulness (~0.48-0.49) against banking dialogues. This is expected -- these are mental health conversation flows being scored against banking task dialogues. The semantic distance between "Invite present-moment grounding" and "I'd like to report a fraudulent transaction" is large.

3. **Cross-domain scoring is a sanity check, not the real evaluation.** The meaningful comparison requires mental health conversation data (see below).

### What These Numbers Mean in Practice

- **avg_fudge ~0.5** means roughly half the dialogue turns need to be inserted/deleted/substituted to align with the DAG. For cross-domain data, this is expected.
- **Compactness differences** directly reflect DAG design philosophy: GPT-5 produced a detailed, branching flow covering many scenarios; Kimik2 produced a more focused, linear flow.
- **FF1 is the decision metric**: if you must choose one DAG, FF1 tells you which one balances coverage and parsimony best for a given corpus.

---

## Key Differences from the Paper's Task-Oriented Setting

### 1. No Ground-Truth Flow Exists

In the paper's banking/hotel experiments, the STAR dataset provides schema-guided flows as ground truth. For mental health conversations, the LLM-derived DAGs *are* the flows being evaluated -- there is no authoritative reference flow. FuDGE/FF1 still work: they measure how well real conversations align with a proposed flow, regardless of whether that flow was hand-authored or LLM-generated.

### 2. Conversation Structure is Less Rigid

Banking dialogues follow fairly predictable patterns (greet -> state problem -> provide info -> confirm -> close). Mental health conversations are more fluid: a user might shift from describing a flashback to requesting crisis resources to trying a grounding exercise, all in one session. This means:

- DAGs need more branching to capture real conversational variety
- Faithfulness scores will generally be lower than in task-oriented domains
- The relative ranking between DAGs remains meaningful even if absolute scores are lower

### 3. Actor Roles Have Different Semantics

In task-oriented dialogues, "user" requests services and "agent" fulfils them. In mental health conversations, the roles are more collaborative: the agent guides therapeutic exercises while the user reports their emotional state. The FuDGE substitution cost handles this through actor matching -- a user utterance can only align with a user node, and vice versa -- which correctly enforces turn-taking structure.

### 4. Substitution Cost Captures Therapeutic Intent

The two-part substitution cost (paper Equation 8) computes:

```
cost_sub(Br, u) = alpha * (d1(Br, u) + d2(Br, B*))
```

Where d1 measures how close an utterance is to an intent node, and d2 measures how close that intent node is to the best-matching intent for this utterance. For mental health DAGs, this means:

- d1 captures whether "I can still feel the tightness in my chest" semantically matches a "body memory" node
- d2 penalises misalignment between the matched node and the globally best node, preventing spurious matches when multiple nodes have similar descriptions

---

## How to Use This for DAG Comparison

### Comparing Two LLM-Derived DAGs

```python
from src.mermaid_loader import load_mermaid_flow
from src.ff1 import ff1_breakdown

gpt5  = load_mermaid_flow("dags/gpt5derived.js")
kimik = load_mermaid_flow("dags/kimik2derived.js")

# dialogues: list of [(actor, utterance), ...] from your mental health corpus
bd_gpt5  = ff1_breakdown(dialogues, gpt5)
bd_kimik = ff1_breakdown(dialogues, kimik)

print(f"GPT-5:  FF1={bd_gpt5['ff1']:.3f}  faith={bd_gpt5['faithfulness']:.3f}  compact={bd_gpt5['compactness']:.3f}")
print(f"Kimik2: FF1={bd_kimik['ff1']:.3f}  faith={bd_kimik['faithfulness']:.3f}  compact={bd_kimik['compactness']:.3f}")
```

### Scoring Individual Dialogues

```python
from src.fudge import fudge

# Score a single conversation against a flow
score = fudge(dialogue, flow)
# score closer to 0 = good alignment, closer to 1 = poor alignment
```

### What to Look For

| Metric | What it tells you | When to prioritise |
|--------|------------------|--------------------|
| **FF1** | Overall flow quality (the main comparison metric) | Choosing between DAGs |
| **Faithfulness** | Does the DAG cover real conversation patterns? | DAG seems too simple |
| **Compactness** | Is the DAG parsimonious? | DAG seems over-specified |
| **avg_fudge** | Raw alignment distance | Diagnosing faithfulness issues |
| **Per-dialogue FuDGE** | Which conversations don't fit the DAG? | Finding coverage gaps |

### Corpus Requirements

For meaningful evaluation, the dialogue corpus should:

- **Match the DAG's domain**: score mental health DAGs against mental health conversations, not banking dialogues
- **Have sufficient size**: the paper uses 40-50 dialogues; fewer than 10 will produce noisy estimates
- **Include variety**: conversations covering different scenarios (crisis, grounding, talk therapy) exercise more of the DAG's structure
- **Use consistent formatting**: each dialogue as `[(actor, utterance), ...]` where actor is "user" or "agent"

---

## Limitations

1. **Embedding model scope**: all-MiniLM-L6-v2 is a general-purpose sentence encoder. It may not capture fine-grained distinctions in therapeutic language (e.g., difference between "grounding exercise" and "breathing exercise"). A domain-specific encoder could improve substitution cost accuracy.

2. **Single-utterance intent nodes**: the Mermaid loader creates each node with one utterance (the node label). In the paper, intent nodes have multiple example utterances from training data, giving richer embedding representations. Consider augmenting DAG nodes with example utterances from real conversations.

3. **Compactness scaling**: FF1's compactness term is `1 - nodes/total_utterances`. With small test corpora (e.g., 10 dialogues, 160 utterances), a 54-node DAG already has complexity ~0.34. With larger corpora the compactness difference between DAGs shrinks, making faithfulness the dominant factor.

4. **No path coverage analysis**: FuDGE tells you how well dialogues fit the DAG, but not which DAG paths are never used. Unused paths represent speculative conversation patterns the LLM generated that don't occur in practice.
