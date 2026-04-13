# Causal Structures from Aviation Accident Reports

**Team:** Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey

Automatically extract and model causal chains from NTSB aviation accident narratives
using **six extraction approaches** — traditional NLP (rule-based + dependency parsing),
two fine-tuned **transformer extractors** (BERT & T5 seq2seq), and **LLM-based extraction**
(zero-shot and few-shot) — organized into a queryable knowledge graph.

---

## Project Overview

Aviation accident reports contain rich causal narratives in unstructured text:
*"fuel exhaustion resulted in a total loss of engine power, which led to an aerodynamic
stall."* Large-scale analysis of these chains is impractical by hand. This project
builds an automated pipeline that extracts cause-effect triples and organizes them
into a queryable knowledge graph.

The central research question is: **how does extraction quality, coverage, and causal
richness compare across rule-based NLP, transformer-based extractors (BERT vs. T5),
and generative LLMs — and how well does each align with the NTSB's official causal
findings on held-out test data?**

---

## Results

### Evaluation Setup

**Cross-validation split:** 6,059 narratives split into **60% train (3,575), 20% val (1,192), 20% test (1,192)**.

- **Training set (3,575):** Used to train BERT and T5 extractors with hyperparameter tuning
- **Validation set (1,192):** Used for hyperparameter optimization (Bayesian search via Optuna)
- **Test set (1,192):** Held-out evaluation set — all final metrics reported here

All models produce identical output format: `{cause, relation, effect}` triples, enabling direct
comparison across all six approaches on identical metrics.

> **Ground-truth evaluation:** Metrics use NTSB `finding_description` labels.
> Cause-confirmed coverage denominates against 5,321 accidents with official C-findings;
> the test set's share (~800 accidents) sets the effective ceiling for ground-truth metrics.

### Final Results — 1,192 Test Narratives

| Metric | Rule-based | Dep-parse | BERT | T5 | LLM (0-shot) | LLM (few-shot) |
|---|---|---|---|---|---|---|
| **Coverage** | 45.0% | 45.5% | 42.3% | **47.2%** | **99.5%** | 72.1% |
| Total triples | 944 | 1,124 | 753 | 910 | 4,358 | 3,375 |
| Avg per narrative | 1.76 | 2.07 | 1.49 | 1.62 | 3.67 | 3.93 |
| **Cause-confirmed coverage** | 8.4% | 8.5% | 8.0% | **8.8%** | **19.9%** | 14.4% |
| **Category alignment** | 50.4% | 49.8% | 49.4% | **52.2%** | 51.3% | 48.0% |
| **Keyword recall** | 14.1% | 14.1% | 13.4% | **14.6%** | 16.7% | **17.4%** |

**Insights:**
- **T5 seq2seq** outperforms BERT token classification on coverage (47.2% vs 42.3%) and has best cause-confirmed coverage among transformers (8.8%)
- **LLM zero-shot** dominates coverage (99.5%) but lower cause-confirmation (19.9%), suggesting broad but less precise extraction
- **Rule-based NLP** remains competitive (45% coverage), with strong keyword recall (14.1%)
- **Category alignment** across all methods clustering ~50% indicates systematic challenge in NTSB category prediction from raw text

### Category Alignment Breakdown (Test Set)

| NTSB Category | Rule-based | Dep-parse | BERT | T5 | LLM (0-shot) | LLM (few-shot) |
|---|---|---|---|---|---|---|
| Aircraft | 66.9% | 66.8% | 65.8% | **69.6%** | 65.3% | 62.1% |
| Environmental | 8.8% | 8.8% | 9.4% | 8.3% | 20.8% | **33.3%** |
| Personnel | 35.8% | 34.7% | 35.4% | **37.2%** | 40.3% | 36.6% |

**Key finding:** Aircraft-related causes are easiest to extract (~65-70%); environmental factors much harder (~8-33%), likely due to their descriptive rather than causal nature in narratives.

### Knowledge Graph (Combined Output)

| Source | Nodes | Edges | Weakly Connected Components | Density |
|---|---|---|---|---|---|
| Rule-based | 8,346 | 4,737 | 3,611 | 6.8×10⁻⁵ |
| Dep-parse | 9,835 | 5,694 | 4,147 | 5.9×10⁻⁵ |
| LLM combined | 30,509 | 21,358 | 12,074 | 2.3×10⁻⁵ |
| **All sources** | **39,398** | **27,038** | **13,176** | **1.7×10⁻⁵** |

---

## Training & Evaluation Pipeline

### Architecture Overview

```
Data (6,059 narratives)
    ↓
[1] Train/Val/Test Split (60/20/20)
    ├─→ Train set (3,575): Pseudo-label triples from rule-based extraction
    ├─→ Val set (1,192): Hyperparameter tuning (Optuna Bayesian search)
    └─→ Test set (1,192): Final evaluation (held-out)
    
[2] Hyperparameter Tuning (src/train.py)
    ├─→ BERT tuning: 15 Optuna trials on validation set
    │   └─→ Search space: lr [5e-6, 5e-5], batch_size [8, 32], epochs [3, 8]
    │   └─→ Objective: Maximize validation F1 on BIO token classification
    ├─→ T5 tuning: 15 Optuna trials on validation set  
    │   └─→ Search space: same as BERT
    │   └─→ Objective: Minimize validation loss on seq2seq generation
    └─→ Best params saved for final training
    
[3] Final Training (src/train.py, continued)
    ├─→ BERT: Train on full training set with best params (early stopping patience=3)
    │   └─→ Model: DistilBERT with 5-class BIO token classification head
    │   └─→ Loss: Class-weighted CrossEntropyLoss (weight O=0.2 to handle class imbalance)
    ├─→ T5: Train on full training set with best params (early stopping patience=3)
    │   └─→ Model: T5-base with seq2seq generation head
    │   └─→ Output format: "cause: [text] | effect: [text]"
    └─→ Trained models saved to outputs/model_{bert,t5}_extractor_tuned/
    
[4] Evaluation (src/eval.py)
    ├─→ Load trained BERT & T5 models
    ├─→ Inference on test set (1,192 narratives)
    ├─→ Baseline extraction:
    │   ├─→ Rule-based: Pattern matching with 12 causal connectives
    │   ├─→ Dep-parse: Dependency parsing with heuristics
    │   └─→ LLM: Mistral-7B zero-shot + few-shot prompting
    ├─→ Ground-truth alignment: Compare to NTSB findings
    └─→ Report: Metrics, plots, knowledge graph cypher statements
```

### Step-by-Step Execution

**1. Train BERT & T5 (tuning + final training):**
```bash
python -m src.train --bert-trials 15 --t5-trials 15
```
- Creates CV split if not exists
- Runs Optuna tuning for each model on validation set
- Trains final models with best hyperparameters on training set
- Saves: `outputs/model_{bert,t5}_extractor_tuned/`

**2. Evaluate all six models:**
```bash
python -m src.eval
```
- Loads tuned BERT & T5 models
- Runs inference on test set (1,192 narratives)
- Extracts rule-based and LLM baselines
- Computes ground-truth alignment metrics
- Generates plots and evaluation report
- Saves: `outputs/evaluation/evaluation_report.json`, `outputs/plots/*.png`

### Training Data & Pseudo-Labeling

BERT and T5 trained on **pseudo-labeled** rule-based extractions from training set:
- **Positive examples:** 2,810 sentences from rule-based triples (training set only)
  - Each triple provides: sentence, cause span, effect span
  - Aligned to token/subword positions for BERT (BIO labels) or included in seq2seq target for T5
- **Negative examples:** 5,620 sentences with no causal pattern
  - Sampled from training narratives without any matching causal connective
  - Ratio 2:1 (negative to positive) to balance dataset

Total dataset: **8,430 examples** (2,810 pos + 5,620 neg) split 85/15 train/val for model training.

### Bias-Variance Analysis

BERT training (5 epochs):
| Epoch | Train Loss | Val Loss | Train F1 | Val F1 | Regime |
|---|---|---|---|---|---|
| 1 | 0.5432 | 0.3847 | 0.7853 | 0.8941 | high_bias |
| 2 | 0.2104 | 0.1876 | 0.9215 | 0.9532 | balanced |
| 3 | 0.1673 | 0.1823 | 0.9384 | 0.9534 | balanced |
| 4 | 0.1521 | 0.1851 | 0.9421 | 0.9562 | balanced |
| 5 | 0.1487 | 0.1902 | 0.9432 | 0.9475 | high_variance |

**Interpretation:** Model learns quickly (epoch 1 high-bias), reaches balanced regime by epoch 2, maintains through epoch 4, shows early signs of overfitting at epoch 5. Best weights restored from epoch 4.

---

## Architecture & Design Decisions

### Why T5 over GPT2 for Token-Level Extraction

**GPT2 (Autoregressive):**
- Strength: Natural language generation
- **Weakness for BIO tagging:** Left-to-right only — misses 50% of context needed to classify token roles
- Result: 1.6% coverage (failure case: architectural misalignment)

**BERT (Bidirectional Encoder):**
- Strength: Full sentence context via masked language modeling
- Weakness for variable-length extraction: Constrained to token-level BIO labels → alignment errors
- Result: 42.3% coverage

**T5 Seq2Seq (Bidirectional Encoder + Autoregressive Decoder):**
- Strength: Full context (encoder) + flexible text generation (decoder)
- Perfect for: "Given sentence, generate cause and effect as natural text"
- Result: **47.2% coverage** — best among transformers
- Output format: `"cause: [cause text] | effect: [effect text]"` → robust parsing

**Conclusion:** T5's seq2seq architecture matches the task (structured text generation) better than token classification. No character-position alignment needed, handles variable-length spans naturally.

### Model Comparison Summary

| Model | Type | Context | Output | Coverage | Cause-Confirmed | Why Best/Worst |
|---|---|---|---|---|---|---|
| **Rule-based** | Pattern matching | Local | Pattern + span | 45.0% | 8.4% | Simple, interpretable; limited to known patterns |
| **Dep-parse** | Syntactic structure | Local | Syntactic edges | 45.5% | 8.5% | Captures grammatical relations; misses semantic causality |
| **BERT** | Token classification | Sentence-wide | BIO tags | 42.3% | 8.0% | Bidirectional, strong; BIO alignment errors |
| **T5** | Seq2seq generation | Sentence-wide | Natural text | **47.2%** | **8.8%** | Best transformer; flexible output; semantic understanding |
| **LLM 0-shot** | Prompt-based reasoning | Full prompt | Natural text | **99.5%** | 19.9% | Highest coverage; lower precision; no training |
| **LLM few-shot** | Prompt + examples | Full prompt | Natural text | 72.1% | 14.4% | Balanced coverage/precision; requires exemplars |

---

## Evaluation Metrics Explained

### Narrative Coverage
$$\text{Coverage} = \frac{\text{# narratives with ≥1 triple}}{\text{Total test narratives}} \times 100\%$$
Measures fraction of test narratives where model extracts ≥1 triple. Range: [0, 100%].

### Cause-Confirmed Coverage (Ground Truth)
$$\text{CC Coverage} = \frac{\text{# test accidents with C-finding + ≥1 extracted triple}}{\text{# test accidents with any C-finding}} \times 100\%$$
Stricter metric using only NTSB-confirmed cause ("C" label) as denominator. Answers: "For accidents officially confirmed to have a cause, did the model find it?"

### Category Alignment (Ground Truth)
For each extracted triple, concatenate cause+effect text and classify into NTSB top-level category (Aircraft, Personnel, Environmental) using keyword matching. Compare to official finding category. Score = % of extracted accidents correctly classified.

### Finding Keyword Recall (Ground Truth)
Extract keywords from NTSB `finding_description` (task performance, fuel management, etc.). Score = average fraction of keywords appearing in extracted cause/effect text across all evaluated accidents.

---

## Code Organization

```
src/
├── train.py                 # Hyperparameter tuning + final model training
├── eval.py                  # Evaluation pipeline (all 6 models)
├── bert_extractor.py        # BERT BIO token classification extractor
├── t5_extractor.py          # T5 seq2seq generation extractor
├── rule_based.py            # Rule-based pattern matching + dep-parse extraction
├── knowledge_graph.py       # Knowledge graph construction & statistics
├── finding_evaluator.py     # Ground-truth alignment metrics
├── plotting.py              # Visualization (cross-model comparison, knowledge graph)
├── data_loader.py           # Data loading & preprocessing
├── cross_validation.py      # 60/20/20 train/val/test split
└── hyperparameter_tuning.py # Optuna + bias-variance utilities

outputs/
├── model_bert_extractor_tuned/     # Tuned BERT model weights
├── model_t5_extractor_tuned/       # Tuned T5 model weights
├── extractions/
│   ├── bert_triples.json           # BERT-extracted triples (test set)
│   ├── t5_triples.json             # T5-extracted triples (test set)
│   ├── causal_triples_rules.json   # Rule-based triples (all sets)
│   └── llm_triples.json            # LLM-extracted triples (all sets)
├── plots/
│   ├── eval_cross_model_comparison.png
│   ├── eval_knowledge_graph_full.png
│   └── [other analysis plots]
└── evaluation/
    └── evaluation_report.json      # Final metrics & statistics
```

---

## Key Findings & Insights

1. **Transformer extraction is viable:**
   - BERT (42.3%) and T5 (47.2%) compete with rule-based NLP (45.0%)
   - Bidirectional models beat unidirectional: T5 >> GPT2 (47.2% vs 1.6%)
   - Seq2seq better than token classification for variable-length spans

2. **LLMs dominate coverage but lose precision:**
   - Zero-shot LLM: 99.5% coverage but only 19.9% cause-confirmed
   - Few-shot LLM: 72.1% coverage, 14.4% cause-confirmed
   - Suggests LLMs extract *plausible* relations, not necessarily *confirmed* causality

3. **Category alignment is hard:**
   - All methods plateau at ~50% category alignment
   - Aircraft causes much easier (65-70%) than environmental (8-33%)
   - Environmental factors often descriptive rather than causal in raw text

4. **Hyperparameter tuning matters:**
   - BERT best performance at epoch 4; naive training → overfitting by epoch 5
   - Early stopping (patience=3) crucial to prevent degradation
   - Bayesian optimization efficiently explored 15 trials vs exhaustive grid

5. **Pseudo-labeling limits:**
   - Transformer models limited by quality of rule-based pseudo-labels
   - T5 slightly higher coverage despite same training data → better output format
   - Gap vs LLM suggests rule-based lacks semantic causality markers

---

## Usage

### Full Pipeline
```bash
# Train BERT & T5 with hyperparameter tuning
python -m src.train --bert-trials 15 --t5-trials 15

# Evaluate all models on test set
python -m src.eval

# View results
cat outputs/evaluation/evaluation_report.json
open outputs/plots/eval_cross_model_comparison.png
```

### Inference Only (Trained Models)
```python
from src.bert_extractor import BERTCausalExtractor
from src.t5_extractor import T5CausalExtractor
import pandas as pd

df = pd.read_csv('data/clean/narratives.csv')

bert = BERTCausalExtractor(model_name='distilbert-base-uncased')
bert.load('outputs/model_bert_extractor_tuned')
bert_triples = bert.extract(df)  # Returns list of {ev_id, cause, relation, effect, ...}

t5 = T5CausalExtractor(model_name='t5-base')
t5.load('outputs/model_t5_extractor_tuned')
t5_triples = t5.extract(df)
```

---

## References

- **BERT:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- **T5:** Raffel et al., "Exploring the Limits of Transfer Learning with Unified Text-to-Text Transformer" (2020)
- **Optuna:** Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework" (2019)
- **NTSB Accident Database:** https://data.ntsb.gov/

