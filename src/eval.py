"""
eval.py -- Evaluate all models and generate plots.

Reads artifacts saved by train.py. Run standalone after training:
    python eval.py
    python eval.py --sample 2000
    python eval.py --no-llm   (skip LLM plots if llm_triples.json absent)
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
from src.knowledge_graph import build_graph, graph_stats, to_neo4j_cypher, visualize_subgraph
from src.bert_extractor import BERTCausalExtractor
from src.t5_extractor import T5CausalExtractor
from src.cross_validation import create_cv_split, save_cv_split, load_cv_split, print_cv_split
from src.plotting import (
    plot_traditional_nlp,
    plot_llm_analysis,
    plot_kg_stats,
    plot_cross_model_comparison,
    plot_top_relation_phrases,
)

try:
    from src.finding_evaluator import load_findings, evaluate_finding_alignment, print_finding_report
    FINDING_EVAL_AVAILABLE = True
except ImportError:
    FINDING_EVAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config / helpers
# ---------------------------------------------------------------------------

def _load_cfg(path: str = 'CONFIG.conf') -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    cfg.read(path)
    return cfg


def section(title: str):
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


def _load_json(path: Path) -> list | dict:
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)


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
# Model 1 -- Traditional NLP evaluation
# ---------------------------------------------------------------------------

def eval_traditional_nlp(training_dir: Path, sample_n: int, plots_dir: Path,
                          test_ev_ids=None) -> dict:
    section('MODEL 1: Traditional NLP -- Causal Extraction (Evaluation)')

    rule_triples = _load_json(training_dir / 'rule_triples.json')
    dep_triples  = _load_json(training_dir / 'dep_triples.json')

    if not rule_triples:
        print('  rule_triples.json not found -- run train.py first.')
        return {'all_rule_triples': [], 'all_dep_triples': [], 'rule_test': [], 'dep_test': []}

    # Restrict reporting to test set when ev_ids are available
    if test_ev_ids:
        rule_test = [t for t in rule_triples if str(t['ev_id']) in test_ev_ids]
        dep_test  = [t for t in dep_triples  if str(t['ev_id']) in test_ev_ids]
        n_report  = len(test_ev_ids)
        print(f'\n  Evaluating on test set ({n_report} held-out narratives)')
    else:
        rule_test = rule_triples
        dep_test  = dep_triples
        n_report  = sample_n

    ev_with_rule   = len({t['ev_id'] for t in rule_test})
    ev_with_dep    = len({t['ev_id'] for t in dep_test})
    pattern_counts = Counter(t['relation'] for t in rule_test)
    direction_counts = Counter(t['direction'] for t in rule_test)
    per_ev         = Counter(t['ev_id'] for t in rule_test)
    densities      = list(per_ev.values()) if per_ev else [0]

    print(f'  Rule-based  -- narratives with >=1 triple: {ev_with_rule}/{n_report} ({ev_with_rule/n_report:.1%})')
    print(f'  Rule-based  -- total triples: {len(rule_test)}  avg: {np.mean(densities):.2f}')
    print(f'  Dep-parse   -- narratives with >=1 triple: {ev_with_dep}/{n_report} ({ev_with_dep/n_report:.1%})')
    print(f'  Dep-parse   -- total triples: {len(dep_test)}')
    print(f'  Direction breakdown: {dict(direction_counts)}')
    print(f'  Pattern hit counts:')
    for pat, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f'    \'{pat}\': {cnt}')

    plot_traditional_nlp(rule_test, dep_test, n_report, plots_dir)

    return {
        'rule_based': {
            'total_triples': len(rule_test),
            'coverage': round(ev_with_rule / n_report, 4),
            'avg_density': round(float(np.mean(densities)), 2),
            'pattern_counts': dict(pattern_counts),
            'direction_counts': dict(direction_counts),
        },
        'dep_parsing': {
            'sample_size': n_report,
            'total_triples': len(dep_test),
            'coverage': round(ev_with_dep / n_report, 4),
        },
        'all_rule_triples': rule_triples,   # full dataset — used for KG
        'all_dep_triples':  dep_triples,
        'rule_test': rule_test,             # test-set slice — used for metrics/plot
        'dep_test':  dep_test,
    }


# ---------------------------------------------------------------------------
# Model 2 -- BERT Causal Extractor evaluation
# ---------------------------------------------------------------------------

def eval_bert_extractor(
    training_dir: Path,
    output_dir: Path,
    df: pd.DataFrame,
    sample_n: int,
    cfg,
) -> list:
    """
    Train (or load) BERTCausalExtractor on training-split rule-based triples,
    then run extraction on the test-split narratives.

    Returns a list of extracted triples in the same format as the rule-based
    and LLM extractors: [{ev_id, cause, relation, effect, direction, method}].
    """
    section('MODEL 2: BERT Causal Extractor (Evaluation)')

    bert_dir   = output_dir / 'model_bert_extractor'
    triples_path = output_dir / 'extractions' / 'bert_triples.json'
    rule_triples = _load_json(training_dir / 'rule_triples.json')

    if not rule_triples:
        print('  rule_triples.json not found -- run train.py first.')
        return []

    try:
        import torch
    except ImportError:
        print('  torch not installed -- skipping BERT extractor.')
        return []

    # Load CV split (should already be created in main())
    cv_split = _load_json(training_dir / 'cv_split.json')
    if not isinstance(cv_split, dict) or 'test_ev_ids' not in cv_split:
        print('  cv_split.json not found -- please ensure CV split is created first.')
        return []

    test_ev_ids  = cv_split['test_ev_ids']
    train_ev_ids = cv_split.get('train_ev_ids', [])

    print(f'  Test narratives: {len(test_ev_ids)}  '
          f'Training pool for BERT: {len(train_ev_ids)}')

    # Load or train BERT extractor
    # Try tuned model first, fall back to default
    bert_dir_tuned = output_dir / 'model_bert_extractor_tuned'
    extractor = BERTCausalExtractor(model_name='distilbert-base-uncased')

    if bert_dir_tuned.exists() and (bert_dir_tuned / 'extractor_meta.json').exists():
        print(f'  Loading tuned BERT model from {bert_dir_tuned}...')
        extractor.load(str(bert_dir_tuned))
    elif bert_dir.exists() and (bert_dir / 'extractor_meta.json').exists():
        extractor.load(str(bert_dir))
    else:
        print('  No saved model found -- training BERT extractor...')
        train_ds, val_ds = extractor.prepare_data(
            df=df,
            rule_triples=rule_triples,
            train_ev_ids=train_ev_ids,
        )
        bert_cfg = cfg['bert_extractor'] if 'bert_extractor' in cfg else {}
        epochs     = int(bert_cfg.get('epochs',     5))
        batch_size = int(bert_cfg.get('batch_size', 16))
        lr         = float(bert_cfg.get('lr',       2e-5))
        history = extractor.train(
            train_ds, val_ds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=str(bert_dir),
        )
        # Log bias-variance analysis
        if 'bias_variance_logs' in history:
            from src.hyperparameter_tuning import print_bias_variance_analysis
            print_bias_variance_analysis(history['bias_variance_logs'])

    # Run extraction on test-set narratives only
    print(f'\n  Running BERT extraction on {len(test_ev_ids)} test narratives...')
    test_df = df[df['ev_id'].astype(str).isin(set(str(e) for e in test_ev_ids))]
    bert_triples = extractor.extract(
        df=test_df,
        text_col='narr_clean',
        id_col='ev_id',
    )

    # Persist
    triples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(triples_path, 'w', encoding='utf-8') as f:
        json.dump(bert_triples, f, indent=2, ensure_ascii=False)

    # Report same metrics as other extraction models
    ev_with = len({t['ev_id'] for t in bert_triples})
    pattern_counts = Counter(t['relation'] for t in bert_triples)
    n_test = len(test_ev_ids)

    print(f'\n  Coverage:          {ev_with}/{n_test} ({ev_with/max(1,n_test):.1%})')
    print(f'  Total triples:     {len(bert_triples)}')
    print(f'  Avg per narrative: {len(bert_triples)/max(1, ev_with):.2f}')
    print(f'  Top relation phrases:')
    for rel, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    '{rel}': {cnt}")
    print(f'  Saved {len(bert_triples)} triples -> {triples_path}')

    return bert_triples


# ---------------------------------------------------------------------------
# Model 4 -- T5 Causal Extractor evaluation
# ---------------------------------------------------------------------------

def eval_t5_extractor(
    training_dir: Path,
    output_dir: Path,
    df: pd.DataFrame,
    sample_n: int,
    cfg,
) -> list:
    """
    Train (or load) T5CausalExtractor on training-split rule-based triples,
    then run extraction on the test-split narratives.

    Returns a list of extracted triples.
    """
    section('MODEL 4: T5 Seq2Seq Causal Extractor (Evaluation)')

    t5_dir   = output_dir / 'model_t5_extractor'
    triples_path = output_dir / 'extractions' / 't5_triples.json'
    rule_triples = _load_json(training_dir / 'rule_triples.json')

    if not rule_triples:
        print('  rule_triples.json not found -- run train.py first.')
        return []

    try:
        import torch
    except ImportError:
        print('  torch not installed -- skipping T5 extractor.')
        return []

    # Load CV split
    cv_split = _load_json(training_dir / 'cv_split.json')
    if not isinstance(cv_split, dict) or 'test_ev_ids' not in cv_split:
        print('  cv_split.json not found -- please run BERT evaluator first.')
        return []

    test_ev_ids = cv_split['test_ev_ids']
    train_ev_ids = cv_split.get('train_ev_ids', [])

    print(f'  Test narratives: {len(test_ev_ids)}  '
          f'Training pool for T5: {len(train_ev_ids)}')

    # Load or train T5 extractor
    # Try tuned model first, fall back to default
    t5_dir_tuned = output_dir / 'model_t5_extractor_tuned'
    extractor = T5CausalExtractor(model_name='t5-base')

    if t5_dir_tuned.exists() and (t5_dir_tuned / 'extractor_meta.json').exists():
        print(f'  Loading tuned T5 model from {t5_dir_tuned}...')
        extractor.load(str(t5_dir_tuned))
    elif t5_dir.exists() and (t5_dir / 'extractor_meta.json').exists():
        extractor.load(str(t5_dir))
    else:
        print('  No saved model found -- training T5 extractor...')
        train_ds, val_ds = extractor.prepare_data(
            df=df,
            rule_triples=rule_triples,
            train_ev_ids=train_ev_ids,
        )
        t5_cfg = cfg['t5_extractor'] if 't5_extractor' in cfg else {}
        epochs     = int(t5_cfg.get('epochs',     5))
        batch_size = int(t5_cfg.get('batch_size', 16))
        lr         = float(t5_cfg.get('lr',       1e-4))
        history = extractor.train(
            train_ds, val_ds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=str(t5_dir),
        )

    # Run extraction on test-set narratives only
    print(f'\n  Running T5 extraction on {len(test_ev_ids)} test narratives...')
    test_df = df[df['ev_id'].astype(str).isin(set(str(e) for e in test_ev_ids))]
    t5_triples = extractor.extract(
        df=test_df,
        text_col='narr_clean',
        id_col='ev_id',
    )

    # Persist
    triples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(triples_path, 'w', encoding='utf-8') as f:
        json.dump(t5_triples, f, indent=2, ensure_ascii=False)

    # Report same metrics as other extraction models
    ev_with = len({t['ev_id'] for t in t5_triples})
    pattern_counts = Counter(t['relation'] for t in t5_triples)
    n_test = len(test_ev_ids)

    print(f'\n  Coverage:          {ev_with}/{n_test} ({ev_with/max(1,n_test):.1%})')
    print(f'  Total triples:     {len(t5_triples)}')
    print(f'  Avg per narrative: {len(t5_triples)/max(1, ev_with):.2f}')
    print(f'  Top relation phrases:')
    for rel, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    '{rel}': {cnt}")
    print(f'  Saved {len(t5_triples)} triples -> {triples_path}')

    return t5_triples


# ---------------------------------------------------------------------------
# Model 5 -- LLM evaluation
# ---------------------------------------------------------------------------

def eval_llm(extractions_dir: Path, sample_n: int, plots_dir: Path,
             test_ev_ids=None) -> tuple:
    """Returns (llm_all_triples, llm_test_triples)."""
    section('MODEL 3: LLM Prompt-Based Causal Extraction (Evaluation)')

    llm_all = _load_json(extractions_dir / 'llm_triples.json')
    if not llm_all:
        print('  llm_triples.json not found or empty -- run train.py first.')
        return [], []

    # Restrict to test set for metrics reporting
    if test_ev_ids:
        llm_test  = [t for t in llm_all if str(t['ev_id']) in test_ev_ids]
        n_report  = len(test_ev_ids)
        print(f'\n  Evaluating on test set ({n_report} held-out narratives)')
    else:
        llm_test = llm_all
        n_report  = sample_n

    ev_with        = len({t['ev_id'] for t in llm_test})
    pattern_counts = Counter(t['relation'] for t in llm_test)
    print(f'  Coverage:          {ev_with}/{n_report} ({ev_with/n_report:.1%})')
    print(f'  Total triples:     {len(llm_test)}')
    print(f'  Avg per narrative: {len(llm_test)/max(1, ev_with):.2f}')
    print(f'  Top relation phrases:')
    for rel, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]:
        print(f'    \'{rel}\': {cnt}')

    plot_llm_analysis(llm_test, n_report, plots_dir)
    return llm_all, llm_test


# ---------------------------------------------------------------------------
# Model 3 -- Knowledge graph evaluation
# ---------------------------------------------------------------------------

def eval_knowledge_graph(rule_triples: list, dep_triples: list, llm_triples: list,
                          cfg, output_dir: Path, plots_dir: Path) -> dict:
    section('MODEL 3: Knowledge Graph -- Structural Evaluation')

    kg_cfg             = cfg['knowledge_graph'] if 'knowledge_graph' in cfg else {}
    noise_filter       = kg_cfg.get('noise_filter',       'true').lower() == 'true'
    normalize_entities = kg_cfg.get('normalize_entities', 'true').lower() == 'true'
    top_n              = int(kg_cfg.get('visualize_top_n', 40))

    llm_triples  = llm_triples or []
    all_triples  = rule_triples + dep_triples + llm_triples

    G_rules = build_graph(rule_triples, noise_filter=noise_filter, normalize=normalize_entities)
    G_deps  = build_graph(dep_triples,  noise_filter=noise_filter, normalize=normalize_entities)
    G_all   = build_graph(all_triples,  noise_filter=noise_filter, normalize=normalize_entities)

    stats_rules = graph_stats(G_rules)
    stats_deps  = graph_stats(G_deps)
    stats_all   = graph_stats(G_all)

    def _fmt(stats, name):
        print(f'\n  [{name}]')
        print(f'    Nodes: {stats["num_nodes"]}  Edges: {stats["num_edges"]}  '
              f'Density: {stats["density"]:.6f}  WCC: {stats["weakly_connected_components"]}')
        print(f'    Top-5 causes:  ' +
              ', '.join(f'({d}) {n[:30]}' for n, d in stats['top_causes'][:5]))
        print(f'    Top-5 effects: ' +
              ', '.join(f'({d}) {n[:30]}' for n, d in stats['top_effects'][:5]))

    _fmt(stats_rules, 'Rule-based graph')
    _fmt(stats_deps,  'Dep-parse graph')
    _fmt(stats_all,   'Combined graph')

    if G_rules.number_of_nodes() > 0 and G_deps.number_of_nodes() > 0:
        rule_nodes = set(G_rules.nodes())
        dep_nodes  = set(G_deps.nodes())
        overlap    = len(rule_nodes & dep_nodes)
        total_u    = len(rule_nodes | dep_nodes)
        print(f'\n  Node overlap (rule intersect dep): {overlap}/{total_u} ({overlap/max(1,total_u):.1%})')

    cypher_path = output_dir / 'extractions' / 'neo4j_import_full.cypher'
    to_neo4j_cypher(all_triples, path=str(cypher_path),
                    noise_filter=noise_filter, normalize=normalize_entities)

    kg_viz_path = str(plots_dir / 'eval_knowledge_graph_full.png')
    visualize_subgraph(G_all, top_n=top_n, save_path=kg_viz_path)
    print(f'  KG visualization saved -> {kg_viz_path}')

    plot_kg_stats(stats_rules, stats_deps, stats_all, plots_dir)

    return {
        'rule_graph':     {k: v for k, v in stats_rules.items()
                           if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
        'dep_graph':      {k: v for k, v in stats_deps.items()
                           if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
        'combined_graph': {k: v for k, v in stats_all.items()
                           if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
        '_stats':         {'rules': stats_rules, 'deps': stats_deps, 'all': stats_all},
    }


# ---------------------------------------------------------------------------
# LLM few-shot extraction on test set
# ---------------------------------------------------------------------------

def eval_llm_fewshot_testset(
    df: pd.DataFrame,
    training_dir: Path,
    extractions_dir: Path,
    cfg,
) -> list:
    """
    Run LLM extraction on the held-out test set using few-shot examples built
    from the training split.  Uses a separate cache so results are independent
    of the zero-shot full-dataset run.

    Returns a list of triples (same format as extract_batch).
    """
    section('LLM Few-Shot Extraction on Test Set')

    # Load test split to get ev_id splits
    test_split_path = training_dir / 'test_split.json'
    if not test_split_path.exists():
        print('  test_split.json not found — run train.py first.')
        return []
    import json as _json
    with open(test_split_path) as f:
        test_split = _json.load(f)
    test_ev_ids  = test_split.get('test_ev_ids', [])
    train_ev_ids = test_split.get('train_ev_ids', [])
    if not test_ev_ids:
        print('  test_ev_ids missing from test_split.json — run train.py first.')
        return []
    print(f'  Test set: {len(test_ev_ids)} narratives  |  '
          f'Training pool for examples: {len(train_ev_ids)} narratives')

    # Load the zero-shot cache to source few-shot examples from training responses
    zero_shot_cache_path = Path(cfg.get('llm_extractor', 'cache_path',
                                        fallback='outputs/extractions/llm_response_cache.json'))
    from src.llm_extractor import (
        LLMCausalExtractor, build_few_shot_examples,
        _load_cache,
    )
    zero_shot_cache = _load_cache(zero_shot_cache_path) if zero_shot_cache_path.exists() else {}

    # Try to load findings for category-stratified example selection
    data_path = cfg.get('paths', 'data_path',
                        fallback='data/clean/cleaned_narritives_and_findings.csv')
    findings_df = None
    if Path(data_path).exists():
        try:
            from src.finding_evaluator import load_findings
            findings_df = load_findings(data_path)
        except Exception:
            pass

    # Build few-shot example block from training set
    print('  Building few-shot examples from training set...')
    few_shot_block = build_few_shot_examples(
        train_ev_ids=train_ev_ids,
        df=df,
        cache=zero_shot_cache,
        findings_df=findings_df,
        n_per_category=1,
    )
    n_examples = few_shot_block.count('Example ')
    print(f'  Built {n_examples} few-shot examples.')
    if n_examples == 0:
        print('  WARNING: no examples built — falling back to zero-shot for test set.')

    # Few-shot cache is separate from zero-shot cache
    fewshot_cache_path = extractions_dir / 'llm_response_cache_fewshot.json'
    fewshot_triples_path = extractions_dir / 'llm_triples_fewshot.json'

    # Try to load any already-computed few-shot triples without loading the GPU model
    if fewshot_triples_path.exists() and _load_cache(fewshot_cache_path):
        fewshot_cache = _load_cache(fewshot_cache_path)
        already_done = set(fewshot_cache.keys()) & set(str(e) for e in test_ev_ids)
        if len(already_done) == len(test_ev_ids):
            print(f'  All {len(test_ev_ids)} test narratives already in few-shot cache.')
            from src.llm_extractor import _parse_triples
            triples = []
            for eid in test_ev_ids:
                raw = fewshot_cache.get(str(eid), '')
                triples.extend(_parse_triples(raw, str(eid)))
            ev_with = len({t['ev_id'] for t in triples})
            print(f'  Few-shot triples: {len(triples)} from {ev_with} narratives')
            return triples

    # Load GPU model for inference
    llm_cfg = cfg['llm_extractor'] if 'llm_extractor' in cfg else {}
    model_name    = llm_cfg.get('model_name',     'mistralai/Mistral-7B-Instruct-v0.3')
    load_in_4bit  = llm_cfg.get('load_in_4bit',   'true').lower() == 'true'
    max_new_tokens = int(llm_cfg.get('max_new_tokens', 350))
    batch_size     = int(llm_cfg.get('batch_size',     4))

    extractor = LLMCausalExtractor(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        max_new_tokens=max_new_tokens,
    )

    triples = extractor.extract_batch(
        df=df,
        text_col='narr_clean',
        id_col='ev_id',
        sample_n=None,
        batch_size=batch_size,
        restrict_ev_ids=test_ev_ids,
        few_shot_block=few_shot_block,
        cache_path=fewshot_cache_path,
    )

    # Persist triples
    fewshot_triples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fewshot_triples_path, 'w', encoding='utf-8') as f:
        _json.dump(triples, f, indent=2, ensure_ascii=False)
    ev_with = len({t['ev_id'] for t in triples})
    print(f'  Saved {len(triples)} few-shot triples ({ev_with} narratives) -> {fewshot_triples_path}')
    return triples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='NTSB -- Evaluation & Plotting')
    parser.add_argument('--sample', type=int,  default=None,
                        help='Sample size (must match what was used in train.py)')
    parser.add_argument('--config', type=str,  default='CONFIG.conf')
    parser.add_argument('--no-llm',     action='store_true', help='Skip LLM evaluation')
    parser.add_argument('--no-fewshot', action='store_true', help='Skip few-shot LLM test-set extraction')
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    output_dir      = Path(cfg.get('paths', 'output_dir', fallback='outputs'))
    data_path       = cfg.get('paths', 'data_path', fallback='data/clean/cleaned_narritives_and_findings.csv')
    training_dir    = output_dir / 'training'
    extractions_dir = output_dir / 'extractions'
    eval_dir        = output_dir / 'evaluation'
    plots_dir       = output_dir / 'plots'

    for d in (plots_dir, eval_dir, extractions_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Resolve sample_n: CLI > saved run config > config global > default
    run_cfg  = _load_json(training_dir / 'run_config.json')
    if args.sample is not None:
        sample_n = args.sample
    elif isinstance(run_cfg, dict) and 'sample_n' in run_cfg:
        sample_n = run_cfg['sample_n']
    else:
        sample_n = int(cfg.get('global', 'sample_n', fallback=2000))

    print('=' * 70)
    print('  NTSB Causal Chain Extraction -- Evaluation')
    print('  Team: Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey')
    print('=' * 70)
    print(f'\n  Sample size (all models): {sample_n}')

    print('\nLoading and preprocessing data...')
    df = load_data(data_path)
    df = preprocess_data(df)
    print(f'  Records: {len(df)}')

    # Resolve sentinel: 0 -> full dataset
    if sample_n == 0:
        sample_n = len(df)
        print(f'  sample_n=0 resolved to full dataset: {sample_n}')

    # Load findings for ground-truth evaluation
    findings_df = None
    if FINDING_EVAL_AVAILABLE and Path(data_path).exists():
        try:
            findings_df = load_findings(data_path)
        except Exception as e:
            print(f'  [warn] Could not load findings: {e}')

    # Create or load cross-validation split (60/20/20)
    cv_split_path = training_dir / 'cv_split.json'
    cv_split = load_cv_split(cv_split_path)

    if not cv_split:
        section('Creating Cross-Validation Split (60/20/20)')
        cv_split = create_cv_split(df, id_col='ev_id', train_frac=0.6, val_frac=0.2, test_frac=0.2)
        save_cv_split(cv_split, cv_split_path)
        print_cv_split(cv_split)
    else:
        section('Loading Existing Cross-Validation Split')
        print_cv_split(cv_split)

    test_ev_ids = cv_split.get('test_ev_ids', [])
    test_ev_set = set(str(e) for e in test_ev_ids)
    n_test = len(test_ev_ids)

    # Model 2: BERT Causal Extractor (runs first so cv_split always exists)
    bert_triples = eval_bert_extractor(training_dir, output_dir, df, sample_n, cfg)

    # Model 3: T5 Seq2Seq Causal Extractor
    t5_triples = eval_t5_extractor(training_dir, output_dir, df, sample_n, cfg)

    # Model 1 — filter to test set for metrics; keep all for KG
    trad_results = eval_traditional_nlp(training_dir, sample_n, plots_dir, test_ev_set)

    # Model 3: LLM — filter to test set for metrics; keep all for KG
    if args.no_llm:
        llm_all, llm_test = [], []
    else:
        llm_all, llm_test = eval_llm(extractions_dir, sample_n, plots_dir, test_ev_set)

    # LLM few-shot on test set
    fewshot_triples = []
    if not args.no_fewshot and not args.no_llm:
        fewshot_triples = eval_llm_fewshot_testset(df, training_dir, extractions_dir, cfg)

    # Convenience aliases for test-set triples per model
    rule_test    = trad_results.get('rule_test', [])
    dep_test     = trad_results.get('dep_test',  [])

    # -----------------------------------------------------------------------
    # Ground-truth finding alignment — ALL models on the same test set
    # -----------------------------------------------------------------------
    alignment_results = []
    if findings_df is not None:
        section('Finding-Alignment Evaluation (Ground Truth) -- Test Set')
        print(f'  Evaluating all models on {n_test} held-out test narratives\n')
        for label, triples in [
            ('Rule-based',      rule_test),
            ('Dep-parse',       dep_test),
            ('BERT Extractor',  bert_triples),
            ('T5 Extractor',    t5_triples),
            ('LLM (zero-shot)', llm_test),
            ('LLM (few-shot)',  fewshot_triples),
        ]:
            if not triples:
                continue
            res = evaluate_finding_alignment(triples, findings_df, label=label)
            alignment_results.append(res)
            pca = res['per_category_alignment']
            print(f'  [{label}]')
            print(f'    Coverage:               {res["ev_ids_extracted"]}/{n_test} '
                  f'({res["ev_ids_extracted"]/max(1,n_test):.1%})')
            print(f'    Cause-confirmed cov.:   {res["cause_confirmed_coverage"]:.1%}  '
                  f'({res["cause_confirmed_n"]}/{res["cause_confirmed_denom"]})')
            print(f'    Category alignment:     {res["category_alignment_score"]:.1%}  '
                  f'(n={res["category_alignment_n"]})')
            print(f'    Keyword recall:         {res["finding_keyword_recall"]:.1%}  '
                  f'(n={res["keyword_recall_n"]})')
            for cat, v in sorted(pca.items()):
                print(f'      {cat:<25} {v["score"]:.1%}  ({v["correct"]}/{v["total"]})')
            print()
        if alignment_results:
            print_finding_report(alignment_results)

    # Knowledge Graph (output artifact — uses full-dataset triples for richness)
    kg_results = eval_knowledge_graph(
        trad_results.get('all_rule_triples', []),
        trad_results.get('all_dep_triples', []),
        llm_all,
        cfg, output_dir, plots_dir,
    )

    # Cross-model comparison — ALL on test set
    plot_cross_model_comparison(
        rule_test, dep_test, llm_test, n_test, bert_triples, t5_triples, plots_dir,
    )

    # Top relation phrases — three-panel figure
    plot_top_relation_phrases(rule_test, bert_triples, llm_test, plots_dir)

    # Save evaluation report
    alignment_map = {r['label']: {k: v for k, v in r.items() if k != 'label'}
                     for r in alignment_results}
    report = {
        'test_set_n':   n_test,
        'sample_n':     sample_n,
        'traditional_nlp': {
            'rule_based':  trad_results.get('rule_based',  {}),
            'dep_parsing': trad_results.get('dep_parsing', {}),
        },
        'bert_extractor': {
            'total_triples':          len(bert_triples),
            'narratives_with_triple': len({t['ev_id'] for t in bert_triples}),
        },
        'llm_extractor': {
            'total_triples':          len(llm_test),
            'narratives_with_triple': len({t['ev_id'] for t in llm_test}),
        },
        'llm_fewshot_testset': {
            'total_triples':          len(fewshot_triples),
            'narratives_with_triple': len({t['ev_id'] for t in fewshot_triples}),
        },
        'finding_alignment': alignment_map,
        'knowledge_graph': {k: v for k, v in kg_results.items() if k != '_stats'},
    }
    report_path = eval_dir / 'evaluation_report.json'
    _save_json(report, report_path)

    section('Evaluation Complete -- Output Files')
    print(f'  Evaluation report:          {report_path}')
    print(f'  Traditional NLP plot:       {plots_dir / "eval_traditional_nlp.png"}')
    print(f'  BERT extractor triples:     {output_dir / "extractions" / "bert_triples.json"}')
    print(f'  LLM extraction plot:        {plots_dir / "eval_llm_extraction.png"}')
    print(f'  Top relation phrases:       {plots_dir / "eval_top_relation_phrases.png"}')
    print(f'  KG visualization:           {plots_dir / "eval_knowledge_graph_full.png"}')
    print(f'  KG stats plot:              {plots_dir / "eval_kg_stats.png"}')
    print(f'  Cross-model comparison:     {plots_dir / "eval_cross_model_comparison.png"}')
    print(f'  Neo4j Cypher:               {output_dir / "extractions" / "neo4j_import_full.cypher"}')
    print('=' * 70)


if __name__ == '__main__':
    main()
