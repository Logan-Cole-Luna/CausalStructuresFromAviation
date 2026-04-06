"""
train.py — Extract triples and train the DistilBERT classifier.

Run standalone:
    python train.py
    python train.py --sample 2000
    python train.py --sample 2000 --no-llm
    python train.py --sample 2000 --no-distilbert

Artifacts saved to outputs/training/ and outputs/extractions/ for eval.py to consume.
"""
import argparse
import configparser
import json
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_data, preprocess_data
from src.traditional_nlp import batch_extract, load_nlp
from src.transformer_classifier import NTSBClassifier, LABEL_COLS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_cfg(path: str = 'CONFIG.conf') -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    cfg.read(path)
    return cfg


def section(title: str):
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    def _default(o):
        if hasattr(o, 'item'):
            return o.item()
        if isinstance(o, (np.integer, np.floating)):
            return float(o)
        return str(o)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=_default)


# ---------------------------------------------------------------------------
# Model 1 — Traditional NLP extraction
# ---------------------------------------------------------------------------

def run_nlp_extraction(df: pd.DataFrame, sample_n: int, training_dir: Path):
    section('MODEL 1: Traditional NLP — Causal Extraction')

    sample_df = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)

    # Rule-based
    print(f'\n[1a] Rule-based extraction (n={len(sample_df)})...')
    rule_triples = batch_extract(sample_df, nlp=None, sample_n=None)
    ev_with = len({t['ev_id'] for t in rule_triples})
    pattern_counts = Counter(t['relation'] for t in rule_triples)
    per_ev = Counter(t['ev_id'] for t in rule_triples)
    densities = list(per_ev.values())
    print(f'  Narratives with ≥1 triple: {ev_with} ({ev_with/len(sample_df):.1%})')
    print(f'  Total triples:             {len(rule_triples)}')
    print(f'  Avg triples/narrative:     {np.mean(densities):.2f}')
    for pat, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f'    \'{pat}\': {cnt}')

    _save_json(rule_triples, training_dir / 'rule_triples.json')
    print(f'  Saved → {training_dir / "rule_triples.json"}')

    # spaCy dep-parse
    print(f'\n[1b] spaCy dep-parse (n={len(sample_df)})...')
    nlp = load_nlp()
    dep_triples = []
    if nlp is not None:
        dep_triples = batch_extract(sample_df, nlp=nlp, sample_n=None)
        dep_ev = {t['ev_id'] for t in dep_triples}
        dep_per_ev = Counter(t['ev_id'] for t in dep_triples)
        dep_densities = list(dep_per_ev.values())
        print(f'  Narratives with ≥1 triple: {len(dep_ev)} ({len(dep_ev)/len(sample_df):.1%})')
        print(f'  Total triples:             {len(dep_triples)}')
        if dep_densities:
            print(f'  Avg triples/narrative:     {np.mean(dep_densities):.2f}')
    else:
        print('  spaCy not available — run: python -m spacy download en_core_web_sm')

    _save_json(dep_triples, training_dir / 'dep_triples.json')
    print(f'  Saved → {training_dir / "dep_triples.json"}')

    return rule_triples, dep_triples


# ---------------------------------------------------------------------------
# Model 2 — DistilBERT classifier
# ---------------------------------------------------------------------------

def run_distilbert_training(df: pd.DataFrame, sample_n: int, cfg, output_dir: Path, training_dir: Path):
    section('MODEL 2: DistilBERT Transformer Classifier')

    try:
        import torch
    except ImportError:
        print('  torch not installed — skipping.')
        return

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f'  CUDA: {name} ({mem:.1f} GB)')
    else:
        print('  CUDA not detected — training on CPU.')

    t = cfg['transformer'] if 'transformer' in cfg else {}
    model_name           = t.get('model_name',            'distilbert-base-uncased')
    epochs               = int(t.get('epochs',            10))
    batch_size           = int(t.get('batch_size',        16))
    lr                   = float(t.get('lr',              2e-5))
    use_amp              = t.get('use_amp',                'true').lower() == 'true'
    amp_dtype            = t.get('amp_dtype',              'auto')
    compile_model        = t.get('compile_model',          'false').lower() == 'true'
    use_class_weights    = t.get('use_class_weights',      'true').lower() == 'true'
    patience             = int(t.get('early_stopping_patience', 3))
    test_size            = float(t.get('test_size',        0.15))
    val_size             = float(t.get('val_size',         0.15))

    sample_df = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)
    print(f'\n  Training on n={len(sample_df)} records ({len(LABEL_COLS)} categories)...')

    clf = NTSBClassifier(
        num_labels=len(LABEL_COLS),
        model_name=model_name,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        compile_model=compile_model,
    )

    train_ds, val_ds, test_ds, label_map = clf.prepare_data(
        sample_df,
        text_col='narr_clean',
        label_col='top_category',
        test_size=test_size,
        val_size=val_size,
        max_samples=None,
    )
    print(f'  Label map: {label_map}')

    train_history = clf.train(
        train_ds, val_ds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        save_path=str(output_dir / 'model'),
        use_class_weights=use_class_weights,
        patience=patience,
    )

    # Save training artifacts for eval.py
    _save_json(train_history, training_dir / 'train_history.json')

    # Save test split so eval.py can reproduce the exact held-out set
    inv_map = {v: k for k, v in label_map.items()}
    test_split = {
        'texts':     [test_ds.encodings['input_ids'][i].tolist() for i in range(len(test_ds))],
        'labels':    [test_ds.labels[i] for i in range(len(test_ds))],
        'label_map': label_map,
    }
    _save_json(test_split, training_dir / 'test_split.json')
    print(f'  Training artifacts saved → {training_dir}')


# ---------------------------------------------------------------------------
# Model 4 — LLM extraction
# ---------------------------------------------------------------------------

def run_llm_extraction(df: pd.DataFrame, sample_n: int, cfg, extractions_dir: Path):
    section('MODEL 4: LLM Prompt-Based Causal Extraction (Mistral-7B-Instruct-v0.3)')

    try:
        from src.llm_extractor import LLMCausalExtractor
    except ImportError as e:
        print(f'  LLM extractor unavailable: {e}')
        return []

    lcfg           = cfg['llm_extractor'] if 'llm_extractor' in cfg else {}
    model_name     = lcfg.get('model_name',     'mistralai/Mistral-7B-Instruct-v0.3')
    max_new_tokens = int(lcfg.get('max_new_tokens', 300))
    load_in_4bit   = lcfg.get('load_in_4bit',   'true').lower() == 'true'
    temperature    = float(lcfg.get('temperature', 0.0))
    batch_size     = int(lcfg.get('batch_size',   4))
    seed           = int(lcfg.get('seed',         42))
    max_retries    = int(lcfg.get('max_retries',  1))
    cache_path_str = lcfg.get('cache_path', 'outputs/extractions/llm_response_cache.json').strip()
    cache_path     = Path(cache_path_str) if cache_path_str else None

    print(f'\n  Model:   {model_name}')
    print(f'  Sample:  {sample_n}  |  4-bit: {load_in_4bit}  |  max_new_tokens: {max_new_tokens}')

    try:
        extractor = LLMCausalExtractor(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as e:
        print(f'  Failed to load LLM: {e}')
        return []

    llm_triples = extractor.extract_batch(
        df,
        text_col='narr_clean',
        id_col='ev_id',
        sample_n=sample_n,
        batch_size=batch_size,
        seed=seed,
        max_retries=max_retries,
        cache_path=cache_path,
    )

    if not llm_triples:
        print('  No triples extracted.')
        return []

    ev_with        = len({t['ev_id'] for t in llm_triples})
    pattern_counts = Counter(t['relation'] for t in llm_triples)
    print(f'\n  Coverage:              {ev_with}/{sample_n} ({ev_with/sample_n:.1%})')
    print(f'  Total triples:         {len(llm_triples)}')
    print(f'  Avg per narrative:     {len(llm_triples)/max(1, ev_with):.2f}')
    print(f'  Top relation phrases:')
    for rel, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]:
        print(f'    \'{rel}\': {cnt}')

    out_path = extractions_dir / 'llm_triples.json'
    _save_json(llm_triples, out_path)
    print(f'\n  Saved → {out_path}')
    return llm_triples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='NTSB — Training & Extraction')
    parser.add_argument('--sample',         type=int,  default=None,
                        help='Narratives per model (overrides CONFIG.conf global.sample_n)')
    parser.add_argument('--config',         type=str,  default='CONFIG.conf')
    parser.add_argument('--no-llm',         action='store_true', help='Skip LLM extraction')
    parser.add_argument('--no-distilbert',  action='store_true', help='Skip DistilBERT training')
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    output_dir     = Path(cfg.get('paths', 'output_dir', fallback='outputs'))
    data_path      = cfg.get('paths', 'data_path', fallback='data/clean/cleaned_narritives_and_findings.csv')
    training_dir   = output_dir / 'training'
    extractions_dir = output_dir / 'extractions'
    training_dir.mkdir(parents=True, exist_ok=True)
    extractions_dir.mkdir(parents=True, exist_ok=True)

    # Resolve sample_n: CLI > config global > default
    if args.sample is not None:
        sample_n = args.sample
    else:
        sample_n = int(cfg.get('global', 'sample_n', fallback=2000))

    print('=' * 70)
    print('  NTSB Causal Chain Extraction — Training')
    print('  Team: Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey')
    print('=' * 70)
    print(f'\n  Sample size (all models): {sample_n}')

    print('\nLoading and preprocessing data...')
    df = load_data(data_path)
    df = preprocess_data(df)
    print(f'  Records: {len(df)}  |  Categories: {df["top_category"].value_counts().to_dict()}')

    # Save run config so eval.py can read it
    _save_json({'sample_n': sample_n, 'data_path': data_path}, training_dir / 'run_config.json')

    # Model 1: NLP extraction
    run_nlp_extraction(df, sample_n, training_dir)

    # Model 2: DistilBERT
    if not args.no_distilbert:
        run_distilbert_training(df, sample_n, cfg, output_dir, training_dir)
    else:
        print('\n  [Skipping DistilBERT — --no-distilbert flag set]')

    # Model 4: LLM extraction
    if not args.no_llm:
        run_llm_extraction(df, sample_n, cfg, extractions_dir)
    else:
        print('\n  [Skipping LLM — --no-llm flag set]')

    print('\n' + '=' * 70)
    print(f'  Training complete. Artifacts in {output_dir}/')
    print('  Run  python eval.py  to generate evaluation plots.')
    print('=' * 70)


if __name__ == '__main__':
    main()
