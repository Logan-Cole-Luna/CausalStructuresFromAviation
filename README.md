# Causal Structures from Aviation Accident Reports

**Team:** Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey

Automatically extract and model causal chains from NTSB aviation accident narratives
using three extraction approaches — traditional NLP, a fine-tuned transformer classifier,
and a prompt-based LLM extractor — organized into a queryable knowledge graph built from
the extracted triples.

---

## Project Overview

Aviation accident reports contain rich causal narratives in unstructured text:
*"fuel exhaustion resulted in a total loss of engine power, which led to an aerodynamic
stall."* Large-scale analysis of these chains is impractical by hand. This project
builds an automated pipeline that extracts cause-effect triples, classifies accident
categories, and organizes the results into a queryable knowledge graph.

The central research question is: **how does extraction quality, coverage, and causal
richness compare between rule-based NLP, a fine-tuned transformer classifier, and a
generative LLM — and how well does each align with the NTSB's official causal findings
when evaluated on the same ground truth?**

---

## Results

Full dataset run — all 6,059 NTSB narratives.

### Extraction Models (Full Dataset)

| Metric | Rule-based NLP | spaCy Dep-parse | LLM (zero-shot) |
|---|---|---|---|
| Narrative coverage | 44.9% | 45.5% | **97.5%** |
| **Cause-confirmed coverage** | 43.3% | 43.9% | **99.2%** |
| Total triples | 4,867 | 5,880 | **22,189** |
| Avg triples / narrative | 1.79 | 2.13 | **3.76** |
| Category alignment (vs NTSB) | 50.0% | 50.0% | 48.6% |
| Finding keyword recall | 14.5% | 14.6% | **16.7%** |
| Parse errors | 0% | 0% | 0.86% |

### DistilBERT Classifier — Test Set (3-class)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Personnel issues | 0.674 | 0.680 | 0.677 | 431 |
| Aircraft | **0.725** | 0.664 | **0.693** | 420 |
| Environmental issues | 0.438 | **0.672** | 0.531 | 58 |
| **Weighted avg** | **0.682** | **0.672** | **0.675** | 909 |
| **Overall accuracy** | | | **67.2%** | |

### Unified Ground Truth — All Models on Same Test Set

All models are evaluated on the identical held-out 909 narratives from the DistilBERT
training split. Extraction models use cause/effect text; DistilBERT uses its predicted
category. This is the apples-to-apples comparison.

> Note: Cause-confirmed coverage denominates against all 5,321 NTSB C-finding accidents,
> so the test set's ~15% share of those (~800 accidents) sets the effective ceiling.

| Model | Narratives evaluated | Cause-confirmed coverage | Category alignment | Finding keyword recall |
|---|---|---|---|---|
| Rule-based NLP | 402 / 909 covered | 6.3% (338/5321) | 50.5% | 15.2% |
| spaCy Dep-parse | 409 / 909 covered | 6.5% (343/5321) | 49.4% | 15.1% |
| LLM (zero-shot) | 900 / 909 covered | 15.0% (799/5321) | 48.7% | 17.0% |
| LLM (few-shot) | 859 / 909 covered | 14.4% (765/5321) | 48.0% | **17.4%** |
| **DistilBERT** | **909 / 909** | **15.2% (807/5321)** | **67.2%** | N/A |

**Category alignment by NTSB finding type (unified test set):**

| NTSB Category | Rule-based | Dep-parse | LLM (zero-shot) | LLM (few-shot) | DistilBERT |
|---|---|---|---|---|---|
| Aircraft | 66.1% | 65.0% | 61.3% | 62.1% | **66.4%** |
| Personnel issues | 39.2% | 37.8% | 38.7% | 36.6% | **68.0%** |
| Environmental issues | 18.5% | 18.5% | 31.0% | **33.3%** | **67.2%** |

### Knowledge Graph (Output Artifact)

| Source | Nodes | Edges | WCC | Density |
|---|---|---|---|---|
| Rule-based | 8,346 | 4,737 | 3,611 | 6.8×10⁻⁵ |
| Dep-parse | 9,835 | 5,694 | 4,147 | 5.9×10⁻⁵ |
| LLM | 30,509 | 21,358 | 12,074 | 2.3×10⁻⁵ |
| **Combined** | **39,398** | **27,038** | **13,176** | **1.7×10⁻⁵** |

---

## Dataset

- **Source:** NTSB Narrative Reports
- **Records:** 6,059 cleaned accident narratives + official `finding_description` labels
- **Ground truth:** Each accident has one or more NTSB `finding_description` entries,
  structured as a hierarchy (e.g., `Personnel issues - Task performance - Use of checklist - Pilot - C`).
  The suffix `C` marks officially confirmed causes; `F` marks findings not designated as causes.

| Category | Count |
|---|---|
| Personnel issues | 2,871 (47.4%) |
| Aircraft | 2,800 (46.2%) |
| Environmental issues | 383 (6.3%) |
| Organizational issues | 5 (0.1%) |

---

## Evaluation Metrics Explained

### Narrative Coverage

$$\text{Coverage} = \frac{\text{\# narratives with ≥1 triple}}{\text{Total narratives}} \times 100\%$$

Measures what fraction of narratives the model successfully extracted at least one
causal triple from. A narrative is "uncovered" only when the model returns 0 triples —
a silent failure.

### Cause-Confirmed Coverage (ground truth)

$$\text{CC Coverage} = \frac{\text{\# C-finding accidents with ≥1 triple extracted}}{\text{\# accidents with ≥1 NTSB C finding}} \times 100\%$$

The NTSB labels each confirmed cause with `C`. This stricter metric uses only accidents
where a cause is officially confirmed as the denominator (5,321 of 5,959 accidents),
asking: *"for accidents we know have a cause, did the model find it?"*

### Category Alignment (ground truth)

For each accident where triples were extracted, the concatenated cause+effect text is
classified into one of the four NTSB top-level categories using keyword heuristics. The
result is compared to the official finding's top-level category. Score = % correct.

### Finding Keyword Recall (ground truth)

The NTSB finding hierarchy is tokenized into concept terms (e.g., `task performance`,
`fuel management`, `spatial disorientation`). Score = average % of those tokens that
appear anywhere in the extracted cause/effect text across all evaluated accidents.

---

## Model Results

### Model 1 — Traditional NLP (Rule-based + spaCy)

| Metric | Rule-based | spaCy dep-parse |
|---|---|---|
| Narrative coverage | 44.9% | 45.5% |
| Cause-confirmed coverage | 43.3% | 43.9% |
| Total triples | 4,867 | 5,880 |
| Avg triples / narrative | 1.79 | 2.13 |
| Category alignment | 50.0% | 50.0% |
| Finding keyword recall | 14.5% | 14.6% |
| Speed | ~5,000 narratives/sec | ~35 narratives/sec |

**Key observations:**

- Coverage plateaus at ~45% regardless of dataset size — confirmed by identical numbers
  on the 5,000-sample and 6,059-sample runs. This is a structural ceiling, not a data
  size effect. The 12 trigger patterns cover only explicitly-marked causal language.
- "Resulted in" (1,889) and "due to" (1,340) account for 67% of all rule-based triples —
  the distribution is heavily long-tailed.
- Dep-parse extracts 21% more triples per covered narrative (2.13 vs 1.79) because
  grammatical relations surface additional verb-argument pairs. Node overlap between the
  two graphs is 84.9%, confirming they capture the same vocabulary through different paths.
- 43.3% cause-confirmed coverage means rule-based extraction missed the officially
  confirmed cause in more than half of NTSB-confirmed-causal accidents.

---

### Model 2 — DistilBERT Transformer Classifier

DistilBERT is fine-tuned on the training split to classify each narrative into one of
three NTSB finding categories. The held-out test set is shared with the unified
ground-truth comparison so all models are evaluated on identical narratives.

| Metric | Value |
|---|---|
| Overall test accuracy | **67.2%** |
| Weighted avg F1 | 0.675 |
| Best val accuracy | 70.3% (epoch 3 of 6) |
| Early stopping triggered | Epoch 6 (patience = 3 from epoch 3) |
| Majority-class baseline | ~47% |
| Training data | 4,237 narratives |
| Test data | 909 narratives |
| Class weighting | Aircraft 0.72×, Environmental 5.27×, Personnel 0.70× |

**Per-class performance:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Personnel issues | 0.674 | 0.680 | 0.677 | 431 |
| Aircraft | **0.725** | 0.664 | **0.693** | 420 |
| Environmental issues | 0.438 | **0.672** | 0.531 | 58 |
| **Weighted avg** | **0.682** | **0.672** | **0.675** | 909 |

**Key observations:**

- **67.2%** against a 47% majority-class baseline = +20 points above chance. Dropping
  "Not determined" cleaned up the label space and produced a slight accuracy improvement
  over the prior 4-class run (66.3%).
- **Aircraft (high precision, lower recall):** The model is conservative — it predicts
  Aircraft confidently when mechanical vocabulary dominates, but pulls back on ambiguous
  narratives ("pilot failed to detect the mechanical failure") and misclassifies them as
  Personnel. Precision 0.725 > recall 0.664.
- **Personnel (lower precision, high recall):** The model over-predicts Personnel,
  absorbing the ambiguous Aircraft cases. Human-action words ("pilot", "crew", "decision")
  appear in both categories, biasing the model toward the more common class.
- **Environmental (recall=0.672, precision=0.438):** The high class weight (5.27×)
  successfully improved recall, but precision is low — the model correctly identifies most
  actual environmental accidents but also mislabels some Personnel/Aircraft narratives as
  Environmental. Only 58 test samples make this class inherently noisy.
- **Category alignment on test set: 67.2%** — this is the true held-out performance.
  The full-dataset figure (74.5%) is inflated because it includes training narratives
  the classifier has already seen.
- Early stopping at epoch 6 (best at epoch 3) indicates faster convergence on the cleaner
  3-class problem compared to the 4-class run (which trained all 10 epochs).

---

### Model 3 — Mistral-7B-Instruct LLM Extractor

| Metric | Value |
|---|---|
| Dataset size | 6,059 narratives |
| Narrative coverage | **97.5%** (5,908 / 6,059) |
| **Cause-confirmed coverage** | **99.2%** (5,277 / 5,321) |
| Total triples | **22,189** |
| Avg triples / narrative | **3.76** |
| Parse errors (after retry) | 52 (0.86%) |
| VRAM usage (4-bit NF4) | 4.2 / 17.1 GB |
| Inference time | ~4.6 hours for 3,284 uncached narratives |
| LLM response cache | 5,959 entries |

**Top relation phrases:**

| Relation | Count | Type |
|---|---|---|
| resulted in | 7,952 | Positive causal |
| led to | 6,198 | Positive causal |
| caused | 2,044 | Positive causal |
| contributed to | 933 | Positive causal |
| precluded | 257 | Barrier failure |
| prevented | 211 | Barrier failure |
| did not prevent | 181 | Barrier failure |

**Key observations:**

- **99.2% cause-confirmed coverage** means the LLM found at least one cause-effect triple
  in 5,277 of the 5,321 accidents NTSB officially confirmed had causes. The 44 misses are
  almost entirely the 52 parse failures (invalid JSON even after retry).
- The LLM extracts **barrier failures** ("precluded", "prevented", "did not prevent") —
  a structurally distinct causal category that rule-based methods cannot express. These
  capture cases where a safety mechanism *failed to activate*, not where something actively
  caused the outcome.
- **Category alignment (48.6%)** is slightly *lower* than rule-based (50%), not because
  the LLM extracts worse causes, but because it uses richer and more varied technical
  language ("manifold pressure drop", "cyclic trim misalignment") that the keyword
  heuristic can't classify as cleanly as rule-based extractions do.
- **Environmental alignment (26.5% LLM vs 17% rule):** The LLM's strongest improvement
  over rule-based is on weather/environment causes. It correctly identifies "density
  altitude reduced climb performance" as an Environmental cause even without explicit
  causal phrases; rule-based misses this entirely.
- **Response caching** made the full-dataset run practical: 2,775 of 6,059 narratives
  were already cached from earlier runs, reducing GPU time by ~46%.

#### LLM Few-Shot Variant (test set only)

The few-shot variant runs on the same 909 held-out test narratives but uses 3 demonstrations
(one per NTSB category, drawn from the training set) plus a required-vocabulary block
listing approved relation phrases and NTSB category terminology in the prompt.

| Metric | Zero-shot (test set) | Few-shot (test set) |
|---|---|---|
| Narratives covered | 900 / 909 (99.0%) | 859 / 909 (94.5%) |
| Total triples | 3,312 | 3,451 |
| Avg triples / covered narrative | 3.68 | 4.02 |
| Parse errors | ~1% | **5.4%** |
| Category alignment | 48.7% | 48.0% |
| Finding keyword recall | 17.0% | **17.4%** |
| Cause-confirmed coverage | 15.0% | 14.4% |

**Key observations:**

- **Parse error rate jumped 6×** (0.86% → 5.4%) because the larger prompt with examples
  leaves less token budget for the narrative, causing more truncation and malformed JSON
  responses. The 929 uncached test narratives required ~74 minutes of GPU inference.
- **Coverage decreased** from 99.0% to 94.5% — consistent with the parse error increase
  and tighter token budget leaving some narratives partially truncated.
- **Keyword recall improved marginally** (+0.4 pp to 17.4%) — the terminology guidance
  nudged the LLM toward more NTSB-native vocabulary in its cause/effect spans. Environmental
  alignment also improved (31.0% → 33.3%), the category most helped by explicit vocabulary
  cues.
- **Category alignment was flat** (48.7% → 48.0%) — the terminology block helps with
  span vocabulary but the keyword-heuristic classifier is the bottleneck, not the LLM.
- **Overall assessment:** Few-shot + terminology prompting is a wash at the current scale.
  The vocabulary alignment benefit is real but small; the parse error increase is a
  significant drawback. A larger token budget (`max_length > 1536`) or a smarter example
  selection strategy would be needed to make few-shot consistently better.

---

## Knowledge Graph (Output Artifact)

The knowledge graph is not a fourth model — it is a queryable output artifact assembled
from the triples produced by Models 1 and 3. Each triple becomes a directed edge; nodes
are normalized entity strings.

| Source | Nodes | Edges | WCC | Density |
|---|---|---|---|---|
| Rule-based | 8,346 | 4,737 | 3,611 | 6.8×10⁻⁵ |
| Dep-parse | 9,835 | 5,694 | 4,147 | 5.9×10⁻⁵ |
| LLM | 30,509 | 21,358 | 12,074 | 2.3×10⁻⁵ |
| **Combined** | **39,398** | **27,038** | **13,176** | **1.7×10⁻⁵** |

**Top causes (combined):** `wind gust` (107), `loss of engine power` (82), `hard landing` (74),
`fuel exhaustion` (39), `spatial disorientation` (22)

**Top effects (combined):** `loss of engine power` (275), `total loss of engine power` (169),
`airplane nosed over` (160), `aerodynamic stall` (38)

**Key observations:**

- The LLM contributes 30,509 nodes — 3.6× more than rule-based alone — reflecting its
  richer extraction vocabulary. The flip side is more fragmentation: "fuel exhaustion",
  "fuel depletion", and "fuel starvation" remain as separate nodes without entity linking.
- 13,176 weakly connected components on 39,398 nodes means average component size is ~3
  nodes. Most accidents form isolated causal chains — entity disambiguation is needed before
  the graph becomes traversable at scale.
- `loss of engine power` appears as both a top cause (82) and top effect (275), making it
  the central hub concept in aviation accident causation — simultaneously the outcome of
  mechanical failures and the trigger for aerodynamic stalls and crashes.
- Node overlap between rule-based and dep-parse graphs is 84.9%, confirming they extract
  the same vocabulary through different mechanisms and combining them adds limited
  incremental value over using LLM triples alone.

---

## Ground Truth Alignment Summary

All models are evaluated against the NTSB `finding_description` ground truth.
Two evaluation contexts are reported:

**Unified test set (909 held-out narratives)** — the fairest comparison. All models run on
exactly the same narratives. DistilBERT's category alignment here (67.2%) is its true
held-out accuracy. The full-dataset number (74.5%) is inflated because it includes
training narratives the classifier has already seen.

| Model | Narratives | Cause-confirmed coverage | Category alignment | KW recall |
|---|---|---|---|---|
| Rule-based NLP | 402 covered | 6.3% | 50.5% | 15.2% |
| spaCy Dep-parse | 409 covered | 6.5% | 49.4% | 15.1% |
| LLM (zero-shot) | 900 covered | 15.0% | 48.7% | 17.0% |
| LLM (few-shot) | 859 covered | 14.4% | 48.0% | **17.4%** |
| **DistilBERT** | **909 (all)** | **15.2%** | **67.2%** | N/A |

**Full-dataset results (6,059 narratives, different coverage denominators per model):**

| Model | Cause-confirmed coverage | Category alignment | KW recall |
|---|---|---|---|
| Rule-based NLP | 43.3% | 50.0% | 14.5% |
| spaCy Dep-parse | 43.9% | 50.0% | 14.6% |
| LLM (zero-shot) | **99.2%** | 48.6% | 16.7% |
| DistilBERT | **99.9%** | 74.5%* | N/A |

*\* Inflated — includes training narratives. Use 67.2% from test set for true held-out performance.*

---

## Understanding Narrative Coverage

### Examples by Model

#### Rule-based NLP (44.9% coverage)

**Input:**
> "The pilot became spatially disoriented due to dust clouds. This led to a loss of altitude."

**Output:**
```json
[
  { "cause": "dust clouds", "relation": "due to", "effect": "spatial disorientation" },
  { "cause": "spatial disorientation", "relation": "led to", "effect": "loss of altitude" }
]
```

**Why 44.9%:** Fires only on explicit trigger phrases. Narratives with no matching phrase
(e.g., "Fuel exhaustion. Loss of engine power.") return 0 triples — 55.1% silent failures.

---

#### spaCy Dependency Parsing (45.5% coverage)

**Input:** Same narrative.

**Output:** Identical structure — grammatical relations surface the same cause-effect pairs
when explicit causal verbs are present. Marginal gain (+0.6 pp) comes from catching a
few verb-argument pairs the regex patterns miss.

---

#### Mistral-7B LLM (97.5% coverage)

**Input:** Same narrative, with extraction prompt.

**Output:**
```json
[
  { "cause": "dust clouds", "relation": "caused", "effect": "spatial disorientation" },
  { "cause": "spatial disorientation", "relation": "led to", "effect": "loss of altitude" },
  { "cause": "low altitude at onset of disorientation", "relation": "prevented", "effect": "recovery" }
]
```

**Why 97.5%:** Understands semantic intent across the full narrative. Captures implicit
causation and negative causality ("prevented recovery") invisible to pattern-based methods.
Only 0.86% parse failures remained after retry.

---

## What Comes Next

### High priority

1. **Entity linking** — "fuel exhaustion", "fuel starvation", "fuel depletion" are
   separate KG nodes. Mapping them to a shared concept (e.g., via WordNet or a
   domain ontology like CAST/HFACS) would consolidate the graph from 13,176 WCC into
   a meaningfully smaller, more traversable structure.

2. **Category-guided LLM prompting** — use DistilBERT's classification as a prefix to
   the LLM prompt ("This is a Personnel issues accident — find the specific pilot
   decision or action..."). The current few-shot vocabulary guidance gave +0.4 pp keyword
   recall but no category alignment gain; injecting DistilBERT's predicted label as a
   hard constraint is a stronger signal.

3. **Human evaluation** — automated metrics cannot measure extraction *precision*.
   A sample of ~50–100 triples per method, manually rated for correctness and
   specificity, is needed to complete the quality picture.

### Medium priority

4. **Expand rule patterns** — "stemmed from", "triggered by", "aggravated by",
   "following", "after" would raise the rule-based ceiling from ~45% toward ~55%
   with minimal implementation cost.

5. **KG noise filter expansion** — `"accident"` and `"pilot"` still appear as generic
   top-cause nodes. Phrase-level pattern matching (rather than exact token matching)
   would clean up these artifacts.

6. **Neo4j query examples** — the Cypher export is complete (27,171 statements).
   Writing representative queries (*"what are the most common causes of aerodynamic
   stalls?"*, *"which nodes have the highest betweenness centrality?"*) would
   demonstrate the graph's analytical value.

### Longer term

7. **Aviation domain fine-tuning** — pretraining DistilBERT on FAA Advisory Circulars
   or ASRS reports before fine-tuning on NTSB categories should push the 67.2% accuracy
   ceiling higher. The current model's Environmental F1 (0.531) has the most room to gain.

8. **Graph Neural Networks** — once entity linking reduces fragmentation, GNNs
   (e.g., R-GCN) could learn richer accident-pathway representations for downstream
   prediction tasks.

---

## Repository Structure

```
.
├── CONFIG.conf                   # All configurable hyperparameters (sample_n=0 for full dataset)
├── main.py                       # Pipeline entry point (runs train + eval)
├── generate_plots.py             # Regenerate all plots from saved artifacts (no retraining)
├── src/
│   ├── train.py                  # Extraction + DistilBERT training
│   ├── eval.py                   # Evaluation + plot generation
│   ├── plotting.py               # All matplotlib/seaborn plot functions
│   ├── finding_evaluator.py      # Ground-truth alignment vs NTSB finding_description
│   ├── data_loader.py            # CSV loading and preprocessing
│   ├── traditional_nlp.py        # Rule-based + spaCy extraction
│   ├── transformer_classifier.py # DistilBERT fine-tuning and inference
│   ├── llm_extractor.py          # Mistral-7B prompt extraction + response cache
│   └── knowledge_graph.py        # NetworkX graph + Neo4j Cypher export
├── data/clean/                   # Cleaned NTSB CSV (not committed)
└── outputs/
    ├── model/                    # Saved DistilBERT weights + label map
    ├── evaluation/               # evaluation_report.json (all metrics)
    ├── extractions/              # llm_triples.json, llm_response_cache.json,
    │                             # neo4j_import_full.cypher, graph_stats_updated.json
    ├── training/                 # rule_triples.json, dep_triples.json, train_history.json
    └── plots/                    # All generated figures (16 PNG files)
```

---

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Full pipeline (train + eval, uses entire dataset by default)
python main.py

# Regenerate all plots from existing artifacts (no GPU needed for extraction models)
python generate_plots.py

# Run independently
python -m src.train     # extraction + DistilBERT training only
python -m src.eval      # evaluation + plots only
```

All hyperparameters are in [CONFIG.conf](CONFIG.conf). Set `sample_n = 0` to use the
entire dataset (default); set to any positive integer to subsample.
