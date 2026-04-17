# Causal Structures from Aviation Accident Reports

**Team:** Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey

Automatically extract and model causal chains from NTSB aviation accident narratives
using **six extraction approaches** - traditional NLP (rule-based + dependency parsing),
two fine-tuned **transformer extractors** (BERT & T5 seq2seq), and **LLM-based extraction**
(zero-shot and few-shot) - organized into a queryable knowledge graph.

---

## Project Overview

Aviation accident reports contain rich causal narratives in unstructured text:
*"fuel exhaustion resulted in a total loss of engine power, which led to an aerodynamic
stall."* Large-scale analysis of these chains is impractical by hand. This project
builds an automated pipeline that extracts cause-effect triples and organizes them
into a queryable knowledge graph.

The central research question is: **how does extraction quality, coverage, and causal
richness compare across rule-based NLP, transformer-based extractors (BERT vs. T5),
and generative LLMs - and how well does each align with the NTSB's official causal
findings on held-out test data?**

---

## Results

### Evaluation Setup

**Cross-validation split:** 6,059 narratives split into **60% train (3,575), 20% val (1,192), 20% test (1,192)**.

- **Training set (3,575):** Used to train BERT and T5 extractors with hyperparameter tuning
- **Validation set (1,192):** Used for hyperparameter optimization (Bayesian search via Optuna)
- **Test set (1,192):** Held-out evaluation set - all final metrics reported here

All models produce identical output format: `{cause, relation, effect}` triples, enabling direct
comparison across all six approaches on identical metrics.

> **Ground-truth evaluation:** Metrics use NTSB `finding_description` labels.
> Cause-confirmed coverage denominates against 5,321 accidents with official C-findings;
> the test set's share (~800 accidents) sets the effective ceiling for ground-truth metrics.

### Final Results - 1,192 Test Narratives

| Metric | Rule-based | Dep-parse | BERT | T5 | LLM (0-shot) | LLM (few-shot) |
|---|---|---|---|---|---|---|
| **Coverage** | 45.0% (536) | 45.5% (542) | 35.2% (420) | **47.1% (561)** | **99.5% (1,186)** | 72.1% (859) |
| Total triples | 944 | 1,124 | 596 | 901 | 4,358 | 3,375 |
| Avg per narrative | 1.76 | 2.07 | 1.42 | 1.61 | 3.67 | 3.93 |
| **Cause-confirmed coverage** | 8.4% (448/5,321) | 8.5% (454/5,321) | 6.6% (352/5,321) | **8.8% (469/5,321)** | **19.9% (1,056/5,321)** | 14.4% (765/5,321) |
| **Category alignment** | 50.4% | 49.8% | 46.9% | **52.9%** | 51.3% | 48.0% |
| **Keyword recall** | 19.4% | 19.3% | 18.2% | **20.0%** | 25.1% | 25.6% |

### Category Alignment Breakdown (Test Set)

| NTSB Category | Rule-based | Dep-parse | BERT | T5 | LLM (0-shot) | LLM (few-shot) |
|---|---|---|---|---|---|---|
| Aircraft | 66.9% (188/281) | 66.8% (189/283) | 65.0% (139/214) | **69.7% (202/290)** | 65.3% (377/577) | 62.1% (242/390) |
| Environmental | 8.8% (3/34) | 8.8% (3/34) | 0.0% (0/31) | 8.8% (3/34) | 20.8% (15/72) | **33.3% (19/57)** |
| Personnel | 35.8% (79/221) | 34.7% (78/225) | 33.1% (58/175) | **38.8% (92/237)** | 40.3% (216/536) | 36.6% (151/412) |

**Key finding:** Aircraft-related causes are easiest to extract (~65-70%); environmental factors much harder (~8-33%), likely due to their descriptive rather than causal nature in narratives.

### Binary Cause-Detection Metrics (Test Set)

Full-narrative binary classification: does the model extract **any** cause-effect relation?

| Metric | Rule-based | Dep-parse | BERT | T5 | LLM (0-shot) | LLM (few-shot) |
|---|---|---|---|---|---|---|
| **Accuracy** | 33.4% | 33.6% | 27.9% | 34.6% | **80.2%** | 21.9% |
| **Precision** | 83.0% | 83.1% | 83.2% | 82.8% | **89.4%** | 90.6% |
| **Recall** | 31.7% | 31.9% | 23.8% | **33.6%** | **88.2%** | 13.7% |
| **F1 Score** | 45.8% | 46.2% | 37.1% | **47.8%** | **88.8%** | 23.8% |
| **AUC-ROC** | 0.3786 | 0.3798 | 0.4075 | 0.3651 | **0.4347** | 0.5110 |
| **Composite Score** | 27.0% | 27.2% | 21.6% | **28.3%** | **55.7%** | 13.6% |
| Token coverage | 8.2% | 8.3% | 6.0% | 8.8% | 22.6% | 3.4% |

**Interpretation:**
- LLM zero-shot dominates on F1 (88.8%) and accuracy (80.2%), leveraging high recall (88.2%) from aggressive extraction
- T5 best among extractors trained on pseudo-labels: 47.8% F1, 33.6% recall, 34.6% accuracy
- BERT underperforms (37.1% F1) due to BIO token classification limitations
- Traditional methods (rule-based, dep-parse) achieve F1 ~46%, matching T5—viable baselines
- LLM few-shot paradox: 90.6% precision but only 13.7% recall (too conservative)

### Top Extracted Relation Phrases

**BERT extraction on test set (596 triples):**
- 'resulted in': 261, 'due to': 169, 'led to': 44, 'caused': 39, 'contributed to': 36, 'because of': 15, ...

**T5 seq2seq on test set (901 triples):**
- 'resulted in': 364, 'due to': 224, 'caused': 99, 'contributed to': 81, 'led to': 55, 'because of': 22, ...

**Key observation:** T5 extracts ~50% more relations than BERT with higher diversity (more balanced phrase distribution), indicating better span selection.

### Knowledge Graph (Combined Output)

| Source | Nodes | Edges | Weakly Connected Components | Density |
|---|---|---|---|---|
| Rule-based | 8,346 | 4,737 | 3,611 | 6.8×10⁻⁵ |
| Dep-parse | 9,835 | 5,694 | 4,147 | 5.9×10⁻⁵ |
| **All sources combined** | **39,398** | **27,038** | **13,176** | **1.7×10⁻⁵** |

### Bias-Variance Analysis

**BERT Token Classification** (8 epochs, best params: lr=2.55e-05, batch_size=8):

| Epoch | Train Loss | Val Loss | Loss Gap | Train F1 | Val F1 | Regime |
|---|---|---|---|---|---|---|
| 1 | 0.6148 | 0.2540 | -0.3608 | 0.7734 | 0.9151 | **high_bias** |
| 2 | 0.1981 | 0.2465 | 0.0483 | 0.9511 | 0.9556 | balanced |
| 3 | 0.1522 | 0.2405 | 0.0884 | 0.9596 | 0.9548 | balanced |
| 4 | 0.1306 | 0.2247 | 0.0941 | 0.9643 | 0.9563 | balanced |
| 5 | 0.1176 | 0.2372 | 0.1196 | 0.9674 | 0.9524 | balanced |
| 6 | 0.1027 | 0.2311 | 0.1285 | 0.9699 | 0.9567 | balanced |
| 7 | 0.0943 | 0.2829 | 0.1886 | 0.9719 | 0.9576 | balanced |
| 8 | 0.0867 | 0.3363 | 0.2496 | 0.9741 | 0.9568 | balanced |
| **Best** | — | — | — | — | **0.9576** (epoch 7) | — |

**T5 Seq2Seq** (8 epochs, best params: lr=4.82e-05, batch_size=8):

| Epoch | Train Loss | Val Loss | Loss Gap | Train F1 | Val F1 | Regime |
|---|---|---|---|---|---|---|
| 1 | 2.4796 | 0.0308 | -2.4488 | 0.7196 | 0.9782 | **high_bias** |
| 2 | 0.0281 | 0.0221 | -0.0060 | 0.9784 | 0.9942 | balanced |
| 3 | 0.0196 | 0.0193 | -0.0003 | 0.9889 | 0.9965 | balanced |
| 4 | 0.0163 | 0.0182 | 0.0019 | 0.9918 | 0.9965 | balanced |
| 5 | 0.0125 | 0.0184 | 0.0059 | 0.9941 | 0.9977 | balanced |
| 6 | 0.0108 | 0.0183 | 0.0075 | 0.9962 | 0.9977 | balanced |
| 7 | 0.0095 | 0.0195 | 0.0100 | 0.9971 | 0.9977 | balanced |
| **Best** | — | — | — | — | **0.9965** (epoch 4) | — |

**Interpretation:**
- **BERT**: Single epoch of high-bias (epoch 1: gap=-0.3608 due to strong initial improvement), transitions to balanced by epoch 2. Maintains stable performance through epoch 7 (val F1=0.9576), but early stopping restored best weights from epoch 7. Overall 7/8 epochs in balanced regime indicates good convergence without overfitting.
- **T5**: Rapid high-bias → balanced transition (epoch 1→2). Achieves state-of-the-art validation F1=0.9965 by epoch 4, with minimal loss gap. Restores best weights from epoch 4 (val F1=0.9965). T5's superior sequence-to-sequence architecture enables higher F1 scores and tighter validation performance.
- **Summary:** Both models show stable convergence with early stopping preventing overfitting. T5 significantly outperforms BERT (0.9965 vs 0.9576 val F1), validating the architectural choice for flexible span generation.

---

## Key Findings & Insights

1. **T5 seq2seq outperforms BERT token classification:**
  - T5 achieves 47.1% coverage vs BERT's 35.2% - validates architectural advantage of seq2seq over token classification
  - T5 has best cause-confirmed coverage among transformers (8.8% vs BERT 6.6%)
  - T5 achieves highest category alignment among transformers (52.9% vs BERT 46.9%)
  - Flexible seq2seq output format better suited for variable-length spans than BIO tag alignment; less constrained by token boundaries

2. **T5 extraction is competitive with traditional NLP, BERT lags:**
  - T5 (47.1%) exceeds rule-based (45.0%) and matches dep-parse (45.5%)
  - BERT (35.2%) significantly underperforms traditional baselines - token classification architecture appears poorly suited to task
  - Demonstrates viability of seq2seq neural extraction; token classification fundamentally limited for causal span extraction
  - Gap suggests: BIO tag alignment loses semantic boundaries; T5's free generation preserves span semantics

3. **LLMs achieve highest coverage but lower precision:**
  - Zero-shot LLM: 99.5% coverage but only 19.9% cause-confirmed (broad extraction, high hallucination)
  - Few-shot LLM: 72.1% coverage, 14.4% cause-confirmed (more selective than zero-shot but still unreliable)
  - Suggests LLMs extract plausible causal relations that don't always map to NTSB-confirmed causes
  - LLM coverage comes at cost: zero-shot precision is ~4× lower than T5 despite comparable coverage

4. **Category alignment plateaus at ~50% across most methods:**
  - Aircraft-related causes are easiest to predict (~65-70% alignment across all models)
  - Environmental factors much harder (~8-33%), typically descriptive rather than causal
  - T5 achieves highest category alignment among transformers (52.9%), exceeding rule-based (50.4%)
  - LLM few-shot is only method with strong environmental alignment (33.3% vs 8.8% for rule-based)
  - Indicates systematic challenge: raw narrative causality markers don't map 1:1 to NTSB's official categorization

5. **Training and hyperparameter choices matter:**
  - Early stopping (patience=3) prevents overfitting: BERT maintains stable val F1 0.91-0.96 across epochs
  - T5 converges to exceptional val F1=0.9965 (epoch 4), with minimal loss gap, validating hyperparameter selection
  - Bayesian hyperparameter optimization (Optuna) efficiently selected best params in 15 trials
  - Pseudo-labeling quality matters: T5 achieves higher coverage with same training data due to better architecture and output format
  - Restored best weights (not final): BERT uses epoch 7, T5 uses epoch 4

6. **Knowledge graph aggregation provides comprehensive causal network:**
  - Combined extraction yields 39,398 nodes and 27,038 edges across all approaches
  - Rule-based (4,737 edges) + Dep-parse (5,694 edges) + Neural+LLM (rest) creates dense, multi-sourced causality network
  - Node overlap: rule-based and dep-parse share 84.9% of nodes, enabling high-confidence core graph
  - Enables multi-method cross-validation and broader causal discovery; consensus triples = high-confidence causal facts

---

## Models

This project implements six extraction approaches, spanning traditional NLP, deep learning, and large language models.

### Rule-Based Pattern Matching

**Approach:** Regex-based extraction using 12 hand-crafted causal connectives (e.g., "resulted in", "due to", "caused by", "led to", etc.).

**Strengths:**
- Fully interpretable - patterns are human-readable and auditable
- No training required; applies immediately to any narrative
- Fast inference; minimal computational overhead
- Strong performance on explicit causal connectives

**Weaknesses:**
- Limited to known patterns - misses semantic causality not expressed via target connectives
- Brittleness to paraphrasing (e.g., "caused" vs "brought about")
- Requires manual pattern engineering to cover new causal expressions

### Dependency Parsing

**Approach:** Syntactic structure analysis using spaCy's dependency parser to extract grammatical relations (nsubj, nmod, etc.) and heuristics to identify cause-effect edges.

**Strengths:**
- Captures grammatical structure without training
- Flexible application across domains with standard syntactic dependencies
- Interpretable through dependency graphs

**Weaknesses:**
- Misses semantic causality - grammatical relations ≠ causal relations
- Limited to sentence boundaries; struggles with multi-sentence causality chains
- Requires post-hoc heuristics to convert syntactic edges to causal triples

### BERT Token Classification

**Approach:** DistilBERT fine-tuned for BIO (Begin-Inside-Outside) token classification on pseudo-labeled training data. Each token is labeled O (outside), B-CAUSE (cause beginning), I-CAUSE (cause inside), B-EFFECT (effect beginning), or I-EFFECT (effect inside).

**Strengths:**
- Bidirectional context from masked language modeling
- Leverages pre-trained knowledge from BERT
- Learns from data; adapts to domain-specific causality patterns

**Weaknesses:**
- BIO tag decoding introduces alignment errors for variable-length spans
- Token-level predictions may not align cleanly to semantic boundaries
- Requires extensive training data (2,810+ positive examples) to be competitive
- Limited to sentence-level extraction

### T5 Seq2Seq Generation

**Approach:** T5-base model fine-tuned for sequence-to-sequence generation. Input: narrative sentence. Output: natural text in format "cause: [text] | effect: [text]".

**Strengths:**
- Flexible output format - generates natural language without span alignment constraints
- Full bidirectional encoder context + autoregressive decoder for natural text generation
- Better suited for variable-length, semantically-meaningful spans
- Achieves highest coverage among transformer models

**Weaknesses:**
- Requires training data and hyperparameter tuning
- More computationally expensive than rule-based or BERT
- Generated text may require parsing to extract cause/effect boundaries

### LLM Zero-Shot Prompting

**Approach:** Mistral-7B model prompted to extract causal triples without in-context examples. Single prompt instructs model to identify all cause-effect pairs and return as JSON list.

**Strengths:**
- Highest coverage (99.5%) - extracts all plausible causal relations
- No fine-tuning required; works out-of-the-box
- Leverages large-scale pre-training and emergent reasoning
- Flexible to prompt variation and reasoning instructions

**Weaknesses:**
- Lower precision (19.9% cause-confirmed) - extracts plausible but unconfirmed relations
- Hallucination risk - may infer causality not explicitly stated
- Less interpretable - difficult to audit reasoning
- Computationally expensive inference

### LLM Few-Shot Prompting

**Approach:** Mistral-7B with 3 in-context examples (cause-effect triples from training set) to steer extraction toward more conservative, confirmed causal relations.

**Strengths:**
- Balanced coverage (72.1%) and precision (14.4% cause-confirmed)
- Few-shot examples provide grounding without full fine-tuning
- Better precision than zero-shot - examples guide model to confirmed causality
- Still interpretable through provided examples

**Weaknesses:**
- Requires selection of representative examples
- Coverage lower than zero-shot - more selective extraction
- Still computationally expensive
- Example quality affects output quality

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


### Training Data & Pseudo-Labeling

BERT and T5 trained on **pseudo-labeled** rule-based extractions from training set:
- **Positive examples:** 2,810 sentences from rule-based triples (training set only)
 - Each triple provides: sentence, cause span, effect span
 - Aligned to token/subword positions for BERT (BIO labels) or included in seq2seq target for T5
- **Negative examples:** 5,620 sentences with no causal pattern
 - Sampled from training narratives without any matching causal connective
 - Ratio 2:1 (negative to positive) to balance dataset

---

## Evaluation Metrics

### Narrative Coverage
Measures fraction of test narratives where model extracts ≥1 triple. Range: [0, 100%].

### Cause-Confirmed Coverage (Ground Truth)
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

---

## Limitations

1. **Pseudo-labeling quality bottleneck:** BERT and T5 are trained on rule-based extractions, which limits their potential beyond rule-based coverage (45%). Superior coverage by LLM zero-shot (99.5%) suggests rule-based labels miss significant semantic causality.

2. **Single-domain evaluation:** All models trained and evaluated exclusively on NTSB aviation narratives. Generalization to other accident domains (rail, maritime, industrial) unknown.

3. **Sentence-level extraction:** Current architectures extract causality within sentence boundaries. Multi-sentence causal chains (e.g., "Fuel leak led to engine failure, which caused cabin depressurization") are split and may lose context.

4. **Category alignment plateau at ~50%:** Even best-performing models (T5, zero-shot LLM) achieve only ~51% category alignment. Gap suggests raw narrative causality markers don't directly correspond to NTSB's official categorization logic.

5. **LLM hallucination risk:** Zero-shot LLM extraction (99.5% coverage) has only 19.9% cause-confirmation rate, indicating significant hallucination. No confidence scoring or uncertainty quantification provided.

6. **No temporal or causal direction modeling:** Current output format (cause, effect, relation) is symmetric. Actual causality is directed and temporal -model doesn't explicitly capture "A caused B at time T" semantics.

7. **Environmental factors severely underperformance:** All methods achieve only 8-33% alignment for environmental causes. This appears to reflect structural mismatch: environmental text is often descriptive (e.g., "icing conditions") rather than explicitly causal.

8. **Training data imbalance:** Negative examples (5,620) outnumber positive (2,810) by 2:1. Class-weighted loss partially mitigates, but ratio may not reflect real distribution in full narratives.

---

## Future Work

1. **Human-annotated ground truth:** Current evaluation uses NTSB finding descriptions as proxy labels. Human annotation of causal triples on held-out test set would provide absolute performance ceiling and fine-grained error analysis.

2. **Multi-sentence and discourse-level extraction:** Extend models to capture causal chains across sentence boundaries using discourse parsing or hierarchical architectures (e.g., document-level BERT with span extraction).

3. **Temporal and causal direction modeling:** 
  - Augment output format to capture direction ("A → B" vs "B → A") and temporal sequence
  - Experiment with temporal relation extraction (TempEval) frameworks
  - Jointly predict causality and temporal relations

4. **Confidence scoring and uncertainty quantification:**
  - Add confidence head to transformer models
  - Implement Bayesian variants (e.g., MC Dropout) to quantify model uncertainty
  - Use uncertainty to filter LLM hallucinations

5. **Cross-domain generalization:**
  - Evaluate trained models on rail, maritime, and industrial accident datasets
  - Domain adaptation techniques (e.g., adversarial training) to improve transfer

6. **Environmental and organizational factor specialization:**
  - Collect domain-specific training data for environmental/org factors
  - Task-specific classifiers for non-accident categories (e.g., "weather description" → "environmental context")
  - Separate extraction pipeline for descriptive vs. causal environmental text

7. **Ensemble and voting strategies:**
  - Combine all six methods with confidence-weighted voting
  - Implement cascade: fast rule-based first, BERT/T5 for uncertain cases, LLM for coverage
  - Cross-model consistency scoring to identify high-confidence extractions

8. **Knowledge graph refinement:**
  - Implement link prediction to infer missing causal edges
  - Community detection to identify causal clusters and sub-graphs
  - Integration with external knowledge bases (e.g., aircraft type properties, weather terminology)

9. **Interactive annotation and active learning:**
  - User interface for refinement of extractions post-hoc
  - Active learning to select informative examples for re-training
  - Iterative improvement loop with domain expert feedback

10. **Explainability and interpretability:**
   - Implement attention visualization for transformer models
   - Generate natural language explanations for extracted triples
   - Comparative analysis of which model components drive decisions

---

## References

- **BERT:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- **T5:** Raffel et al., "Exploring the Limits of Transfer Learning with Unified Text-to-Text Transformer" (2020)
- **Optuna:** Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework" (2019)
- **NTSB Accident Database:** https://data.ntsb.gov/

