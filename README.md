# Causal Structures from Aviation Accident Reports

**Team:** Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey

Automatically extract and model causal chains from NTSB aviation accident narratives
using three extraction approaches — traditional NLP, a fine-tuned BERT causal extractor,
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
richness compare between rule-based NLP, a fine-tuned BERT causal extractor, and a
generative LLM — and how well does each align with the NTSB's official causal findings
when evaluated on the same ground truth?**

---

## Results

All models evaluated on the same **909 held-out test narratives** (15% stratified split,
fixed seed). Every model produces `{cause, relation, effect}` triples — identical output
format — enabling direct comparison on all metrics.

> Ground-truth metrics use the NTSB `finding_description` labels.
> Cause-confirmed coverage denominates against 5,321 accidents with official C-findings;
> the test set's ~15% share (~800 accidents) sets the effective ceiling.

### Extraction Summary — 909 Test Narratives

| Metric | Rule-based NLP | spaCy Dep-parse | BERT Extractor | LLM (zero-shot) | LLM (few-shot) |
|---|---|---|---|---|---|
| **Coverage** (narratives with ≥1 triple) | 44.2% | 45.0% | 43.1% | **99.0%** | 94.5% |
| Total triples | 721 | 878 | 624 | 3,390 | 3,375 |
| Avg triples / covered narrative | 1.79 | 2.13 | 1.59 | 3.77 | **3.93** |
| **Cause-confirmed coverage** | 6.3% | 6.5% | 6.2% | **15.0%** | 14.4% |
| **Category alignment** (vs NTSB) | **50.5%** | 49.4% | 50.0% | 48.7% | 48.0% |
| **Keyword recall** | 15.2% | 15.1% | 14.6% | 17.0% | **17.4%** |

**Category alignment by NTSB finding type:**

| NTSB Category | Rule-based | Dep-parse | BERT Extractor | LLM (zero-shot) | LLM (few-shot) |
|---|---|---|---|---|---|
| Aircraft | **66.1%** | 65.0% | 64.5% | 61.3% | 62.1% |
| Environmental issues | 18.5% | 18.5% | 19.2% | 31.0% | **33.3%** |
| Personnel issues | 39.2% | 37.8% | **39.9%** | 38.7% | 36.6% |

### Knowledge Graph (Output Artifact)

| Source | Nodes | Edges | WCC | Density |
|---|---|---|---|---|
| Rule-based | 8,346 | 4,737 | 3,611 | 6.8×10⁻⁵ |
| Dep-parse | 9,835 | 5,694 | 4,147 | 5.9×10⁻⁵ |
| LLM | 30,509 | 21,358 | 12,074 | 2.3×10⁻⁵ |
| **Combined** | **39,398** | **27,038** | **13,176** | **1.7×10⁻⁵** |

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

---

### Model 2 — BERT Causal Extractor

DistilBERT is fine-tuned for **causal span extraction** — the same task as Models 1 and 3.
Rather than classifying narratives into categories, it predicts BIO (Begin-Inside-Outside)
token labels to identify cause and effect spans directly from sentence text.

**Approach: BIO Token Classification**

For each sentence in a narrative, the model assigns one of five token-level labels:

| Label | Meaning |
|---|---|
| `O` | Not part of a causal span |
| `B-CAUSE` | Beginning of a cause span |
| `I-CAUSE` | Continuation of a cause span |
| `B-EFFECT` | Beginning of an effect span |
| `I-EFFECT` | Continuation of an effect span |

Decoded cause/effect spans are assembled into `{cause, relation, effect}` triples
using the same relation vocabulary as rule-based extraction.

**Training data:** Rule-based triples from training narratives are used as pseudo-labels.
Each triple's `sentence`, `cause`, and `effect` fields are aligned to token positions via
character-offset mapping. Negative examples (sentences without causal patterns) are sampled
at 2× the positive count to balance training.

| Training parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Positive BIO examples | 3,416 (rule-based triples from training narratives) |
| Negative BIO examples | 6,832 (2× positive, sentences with no causal pattern) |
| Train / val split | 8,711 / 1,537 examples |
| Loss function | Cross-entropy, O tokens down-weighted (0.2×) |
| Optimizer | AdamW, lr=2e-5, weight_decay=0.01 |
| Epochs | 5 (no early stopping triggered) |
| Best val span-F1 | **0.9578** (prec=0.924, rec=0.994) at epoch 5 |
| Saved to | `outputs/model_bert_extractor/` |

**Training curve:**

| Epoch | Loss | Val span-F1 | Prec | Rec |
|---|---|---|---|---|
| 1 | 0.608 | 0.935 | 0.886 | 0.991 |
| 2 | 0.186 | 0.947 | 0.902 | 0.997 |
| 3 | 0.145 | 0.958 | 0.928 | 0.989 |
| 4 | 0.129 | 0.955 | 0.918 | 0.995 |
| 5 | 0.118 | **0.958** | **0.924** | **0.994** |

**Test-set extraction results (909 narratives):**

| Metric | Value |
|---|---|
| Narratives covered | 392 / 909 (**43.1%**) |
| Total triples | 624 |
| Avg triples / covered narrative | 1.59 |

**Top relation phrases extracted:**

| Relation | Count |
|---|---|
| resulted in | 247 |
| due to | 178 |
| contributed to | 76 |
| led to | 40 |
| caused | 39 |
| as a result of | 17 |

**Finding-alignment on test set:**

| Metric | Value |
|---|---|
| Cause-confirmed coverage | 6.2% (329 / 5,321) |
| Category alignment | 50.0% |
| Keyword recall | 14.6% |
| Aircraft alignment | 64.5% (118/183) |
| Environmental alignment | 19.2% (5/26) |
| Personnel alignment | 39.9% (73/183) |

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

#### LLM Few-Shot

The few-shot variant runs on the same 909 held-out test narratives but uses 3 demonstrations
(one per NTSB category, drawn from the training set) plus a required-vocabulary block
listing approved relation phrases and NTSB category terminology in the prompt.

| Metric | Zero-shot (test set) | Few-shot (test set) |
|---|---|---|
| Narratives covered | 900 / 909 (99.0%) | 859 / 909 (94.5%) |
| Total triples | 3,312 | 3,375 |
| Avg triples / covered narrative | 3.68 | 3.93 |
| Parse errors | ~1% | **5.4%** |
| Category alignment | 48.7% | 48.0% |
| Finding keyword recall | 17.0% | **17.4%** |
| Cause-confirmed coverage | 15.0% | 14.4% |

---

## Knowledge Graph

The knowledge graph is a queryable output artifact assembled
from the triples produced by all three extraction models. Each triple becomes a directed
edge; nodes are normalized entity strings.

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
