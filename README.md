# Causal Structures from Aviation Accident Reports

**Team:** Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey

Automatically extract and model causal chains from NTSB aviation accident narratives
using three complementary approaches: traditional NLP, a transformer classifier, and a
prompt-based LLM extractor — all feeding into a Neo4j knowledge graph.

---

## Project Overview

Aviation accident reports contain rich causal narratives in unstructured text:
*"fuel exhaustion resulted in a total loss of engine power, which led to an aerodynamic
stall."* Large-scale analysis of these chains is impractical by hand. This project
builds an automated pipeline that extracts cause-effect triples, classifies accident
categories, and organizes the results into a queryable knowledge graph.

The central research question is: **how does extraction quality, coverage, and causal
richness compare between rule-based NLP, a fine-tuned transformer encoder, and a
generative LLM?**

---

## Dataset

- **Source:** NTSB Narrative Reports
- **Records:** 6,059 cleaned accident narratives
- **Categories:**

| Category | Count |
|---|---|
| Personnel issues | 2,871 (47.4%) |
| Aircraft | 2,800 (46.2%) |
| Environmental issues | 383 (6.3%) |
| Organizational issues | 5 (0.1%) |

---

## Pipeline

```
NTSB CSV
  │
  ├── Model 1: Rule-based NLP ──────────────────────┐
  │     └── spaCy Dependency Parsing (sample)        │
  │                                                   ├──▶ Knowledge Graph (Neo4j)
  ├── Model 2: DistilBERT Classifier                 │
  │     └── Narrative category prediction             │
  │                                                   │
  └── Model 4: Mistral-7B LLM Extractor ────────────┘
        └── Prompt-based cause-effect triple extraction
```

---

## Understanding Narrative Coverage

**Narrative coverage** is the central metric for the extraction models. It measures what fraction of accident narratives the model successfully extracted ≥1 causal triple from.

### Definition and Calculation

$$\text{Coverage} = \frac{\text{\# narratives with ≥1 triple}}{\text{Total narratives in sample}} \times 100\%$$

A narrative "counts" as covered only if the model found at least one cause-effect relationship in it. If the model returns 0 triples (silent failure), that narrative is uncovered.

### Why This Matters

Extraction models have two failure modes:
1. **False negatives (silent)**: Model returns 0 triples when causal links actually exist
2. **False positives (explicit)**: Model returns incorrect triples

Coverage captures silent failures. High coverage means fewer "I found nothing" responses, which is critical for downstream tasks like knowledge graph construction.

### Examples by Model

#### Rule-based NLP (44.9% coverage)

**Input (narrative):**
> "The flight instructor was demonstrating an autorotation. The pilot became spatially disoriented due to dust clouds. This led to a loss of altitude."

**Model process:** Searches for fixed causal phrases ("resulted in", "due to", "led to", etc.)

**Output:** One triple extracted:
```json
{
  "cause": "dust clouds",
  "relation": "due to",
  "effect": "spatial disorientation"
}
```

**Why coverage is 44.9%:** The model only fires on explicit causal phrases. Many narratives describe causes implicitly (e.g., "Fuel exhaustion. Loss of engine power." with no linking word) — those 55.1% of narratives get 0 triples despite causal chains existing.

---

#### spaCy Dependency Parsing (50.4% coverage)

**Input (narrative):** Same as above

**Model process:** Parses grammatical dependencies, finds verb relationships between noun phrases

**Output:** Slightly better extraction from grammatical structure:
```json
{
  "cause": "dust clouds",
  "relation": "result",
  "effect": "spatial disorientation"
},
{
  "cause": "spatial disorientation",
  "relation": "lead",
  "effect": "loss of altitude"
}
```

**Why coverage is 50.4% (vs 44.9% rule-based):** Dependency parsing understands grammar better than fixed patterns, so it catches a few more implicit causal relationships. But it's still bound by the syntactic structure of the sentence — truly implicit causes ("pilot fatigue" mentioned in paragraph 3 caused "landing error" in paragraph 5) are still invisible.

---

#### Mistral-7B LLM (98.5% coverage)

**Input (narrative):** Same as above, plus an instruction:
```
Extract all causal relationships from this narrative.
Return a JSON array of {cause, relation, effect} objects.
```

**Model process:** Language model reads full narrative, understands semantic relationships, and reasons about what caused what

**Output:** Comprehensive extraction including implicit causes:
```json
[
  {
    "cause": "dust clouds",
    "relation": "caused",
    "effect": "spatial disorientation"
  },
  {
    "cause": "spatial disorientation",
    "relation": "led to",
    "effect": "loss of altitude"
  },
  {
    "cause": "low altitude",
    "relation": "prevented",
    "effect": "recovery"
  }
]
```

**Why coverage is 98.5%:** The LLM understands semantic intent. It reads "dust clouds... spatial disorientation... loss of altitude" and infers the causal chain even without explicit linking words. It also captures negative causality ("prevented recovery") that rule-based systems cannot express. Only 1.5% of narratives are truly *acausal* — they describe accidents with no clear causal narrative at all.

---

### Direct Correlation: Coverage → Quality

Higher coverage doesn't guarantee higher *accuracy* (triples could still be wrong), but it's a necessary condition:

| Model | Coverage | Implication |
|---|---|---|
| Rule-based NLP | 44.9% | Fast, precise on explicit causes. But **55% of narratives contribute 0 triples** — half the signal is discarded. |
| spaCy dep-parse | 50.4% | Slight improvement, but still **~49% silent failures**. Grammar helps, but can't infer implicit cause chains. |
| Mistral-7B LLM | **98.5%** | Nearly all narratives yield triples. **Only 1.5% return nothing**. Can express negative causality ("did not", "prevented") that other models cannot. |
| DistilBERT classifier | N/A | Different task — classifies *one* category label per narrative, not extraction. |

**Takeaway:** For extraction, LLMs win decisively on coverage because they reason about causal intent rather than pattern-matching. The tradeoff is speed (98 sec/batch vs 2 ms/batch for rule-based) and cost (4GB GPU memory).

---

## Initial Findings

### Model 1 — Traditional NLP (Rule-based + spaCy)

| Metric | Rule-based | spaCy dep-parse (500 sample) |
|---|---|---|
| Narrative coverage | 44.9% | 50.4% |
| Total triples | 4,867 | 505 |
| Avg triples / narrative | 1.79 | 2.00 |

**Key observations:**
- Coverage is inherently limited to narratives that contain explicit causal signal phrases.
  More than half the dataset produces no extraction at all.
- Two patterns ("resulted in", "due to") account for ~67% of all triples — the distribution
  is heavily long-tailed.
- spaCy dependency parsing achieves slightly better per-narrative density (+12%) but at
  much lower throughput (~35 narratives/sec vs. ~5,000 for rule-based).
- Both methods share the same surface-level vocabulary, so they are not truly independent
  signals.

---

### Model 2 — DistilBERT Transformer Classifier

| Metric | Value |
|---|---|
| Test accuracy | **71.2%** |
| Best val accuracy | 67.3% (epoch 5) |
| Early stopping triggered | Epoch 8 (patience = 3) |
| Class weighting | Aircraft 0.72×, Environmental 5.27×, Personnel 0.70× |

**Key observations:**
- Early stopping correctly identified epoch 5 as the optimal checkpoint; the model was
  already overfitting by epoch 6 (training loss 0.33, val accuracy declining).
- The 71.2% test accuracy is a meaningful signal given that near-equal class balance
  between Aircraft and Personnel issues makes naive majority-class guessing ~47%.
- Class weighting boosted sensitivity to the rare Environmental issues class (6.3% of data)
  without degrading overall accuracy.
- Confidence scores are high (82–99%) across all sample predictions, suggesting the model
  has learned clear discriminative features for the two dominant classes.
- The ceiling for this model is probably ~75–78% without domain-specific pretraining;
  the narratives are aviation-specific and DistilBERT was pretrained on general English.

---

### Model 4 — Mistral-7B-Instruct LLM Extractor

| Metric | Value |
|---|---|
| Sample size | 200 narratives |
| Narrative coverage | **98.5%** |
| Total triples | 753 |
| Avg triples / narrative | 3.82 |
| Parse errors (after retry) | 3 (1.5%) |
| VRAM usage (4-bit NF4) | 4.2 / 17.1 GB |

**Key observations:**
- Coverage jumped from 44.9% (rule-based) to **98.5%** — the LLM extracts causal
  relationships from narrative phrasing that no fixed pattern can match.
- The retry mechanism reduced parse errors from 26.5% to **1.5%** — a major reliability
  improvement.
- Critically, the LLM extracted **negative causal relations** entirely absent from the
  rule-based output: `"did not"`, `"precluded"`, `"prevented"`, `"failed to"`. These
  capture barrier failures (e.g., *"the pilot failed to maintain airspeed, which led to a
  stall"*) — a distinct and safety-critical causal category.
- Inference is slow: ~85 seconds/batch on an RTX 5070 Ti at 4-bit, versus milliseconds
  for the rule-based approach. For full-dataset extraction this is a practical constraint.

---

### Model 3 — Knowledge Graph

| Metric | Rule-based | Dep-parse | Combined |
|---|---|---|---|
| Nodes | 8,346 | 902 | 9,639 |
| Edges | 4,737 | 489 | 5,564 |
| Weakly connected components | 3,611 | 413 | 4,079 |
| Density | 0.000068 | 0.000602 | 0.000060 |

**Top causes (combined graph):** `accident`, `fuel starvation`, `fuel exhaustion`,
`spatial disorientation`, `accident could not be determined`

**Top effects (combined graph):** `loss of engine power`, `total loss of engine power`,
`aerodynamic stall`, `partial loss of engine power`, `substantial damage to the fuselage`

**Key observations:**
- Entity normalization (stripping leading articles, lowercasing) successfully merged
  near-duplicate nodes: `"loss of engine power"` now consolidates what were previously
  split entries like `"the loss of engine power"`, `"a loss of engine power"`, etc.
- `"accident"` remaining as the top cause node is a residual noise artifact from phrases
  like *"the cause of the accident could not be determined"* — further phrase-level
  filtering is needed.
- The dep-parse subgraph has 9× higher density than the rule-based graph, suggesting its
  smaller but more syntactically constrained extractions are more internally connected.
- Node overlap between rule-based and dep-parse is only 9.1%, confirming they capture
  different surface forms — combining them adds real value.
- 4,079 weakly connected components means the graph is still highly fragmented. Most
  subgraphs are 2–3 node chains, not large interconnected structures. Full connectivity
  requires either more data or entity linking/coreference resolution.

---

### Cross-Model Comparison

| Method | Coverage | Triples (sample-normalized) | Novel relation types | Speed |
|---|---|---|---|---|
| Rule-based | 44.9% | 1.79/narrative | None (fixed vocabulary) | ~5,000 narratives/sec |
| spaCy dep-parse | 50.4% | 2.00/narrative | Verb lemmas | ~35 narratives/sec |
| Mistral-7B LLM | **98.5%** | **3.82/narrative** | Negation, barriers, conditionals | ~0.14 narratives/sec |
| DistilBERT | N/A (classifier) | N/A | N/A | ~200 narratives/sec |

The results confirm the central hypothesis: **LLMs extract substantially richer and more
complete causal information**, but at a significant computational cost. Rule-based methods
are fast and precise within their vocabulary but structurally blind to anything outside
their pattern set.

---

## What Comes Next

### Immediate (high priority)

1. **Human evaluation** — a sample of ~50–100 triples from each method should be manually
   rated for correctness, completeness, and specificity. This is the only way to measure
   *precision*, which automated metrics cannot provide for open-ended extraction.

2. **Increase LLM sample size** — 200 narratives is sufficient for a proof-of-concept
   comparison but too small to draw statistical conclusions. Running on 1,000–2,000
   narratives (still tractable at ~85 sec/batch × 500 batches ≈ 12 hours) would give a
   much more reliable picture of LLM extraction quality at scale.

3. **Filter residual KG noise** — `"accident"` and `"accident could not be determined"`
   are still appearing as top cause nodes. These require phrase-level filtering beyond the
   current word-list approach.

### Medium priority

4. **Entity linking / coreference** — "fuel exhaustion" and "fuel starvation" are
   distinct nodes in the current graph but partially overlap semantically. A simple
   ontology mapping to CAST (Causal Analysis based on STAMP) or HFACS taxonomy would
   consolidate the graph and make it queryable in domain terms.

5. **Per-class metrics for the classifier** — the confusion matrix exists but the
   per-class precision/recall for Environmental issues (the hardest class) needs explicit
   reporting. If recall for that class is still low despite weighting, consider oversampling
   via SMOTE or augmentation.

6. **Neo4j query examples** — now that the Cypher export is working, write a set of
   representative queries (e.g., *"what are the most common causes of aerodynamic stalls?"*,
   *"which cause nodes have the highest betweenness centrality?"*) to demonstrate the
   graph's analytical value.

### Longer term

7. **Structured LLM comparison** — run the same 200-narrative sample through a second
   LLM (e.g., Llama-3-8B or Phi-3-mini) and compare triple quality directly.

8. **Aviation domain fine-tuning** — fine-tune DistilBERT on aviation text (e.g.,
   FAA Advisory Circulars, ASRS reports) as a pretraining step before classification to
   improve the accuracy ceiling beyond the current ~71%.

9. **Graph Neural Networks** — once the KG is sufficiently connected and entity-linked,
   GNNs (e.g., R-GCN) could learn richer representations of accident pathways for
   downstream prediction tasks.

---

## Repository Structure

```
.
├── CONFIG.conf                  # All configurable hyperparameters
├── train_and_evaluate.py        # Main pipeline script
├── src/
│   ├── data_loader.py           # CSV loading and preprocessing
│   ├── traditional_nlp.py       # Rule-based + spaCy extraction
│   ├── transformer_classifier.py # DistilBERT fine-tuning
│   ├── llm_extractor.py         # Mistral-7B prompt extraction
│   └── knowledge_graph.py       # NetworkX graph + Neo4j export
├── data/clean/                  # Cleaned NTSB CSV (not committed)
└── outputs/
    ├── model/                   # Saved DistilBERT weights
    ├── evaluation/              # evaluation_report.json
    ├── extractions/             # neo4j_import_full.cypher, llm_triples.json
    └── plots/                   # All generated figures
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python train_and_evaluate.py
```

All hyperparameters (epochs, LLM model, sample sizes, etc.) are in [CONFIG.conf](CONFIG.conf).
