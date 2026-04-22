# Causal Structures from Aviation Accident Reports

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

## References

- **BERT:** Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
- **T5:** Raffel et al., "Exploring the Limits of Transfer Learning with Unified Text-to-Text Transformer" (2020)
- **Optuna:** Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework" (2019)
- **NTSB Accident Database:** https://data.ntsb.gov/
