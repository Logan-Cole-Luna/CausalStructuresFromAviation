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
from src.transformer_classifier import NTSBClassifier, LABEL_COLS
from src.plotting import (
    plot_traditional_nlp,
    plot_confusion_matrix,
    plot_training_curves,
    plot_per_class_metrics,
    plot_llm_analysis,
    plot_kg_stats,
    plot_cross_model_comparison,
)


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

def eval_traditional_nlp(training_dir: Path, sample_n: int, plots_dir: Path) -> dict:
    section('MODEL 1: Traditional NLP -- Causal Extraction (Evaluation)')

    rule_triples = _load_json(training_dir / 'rule_triples.json')
    dep_triples  = _load_json(training_dir / 'dep_triples.json')

    if not rule_triples:
        print('  rule_triples.json not found -- run train.py first.')
        return {'all_rule_triples': [], 'all_dep_triples': []}

    ev_with_rule   = len({t['ev_id'] for t in rule_triples})
    ev_with_dep    = len({t['ev_id'] for t in dep_triples})
    pattern_counts = Counter(t['relation'] for t in rule_triples)
    direction_counts = Counter(t['direction'] for t in rule_triples)
    per_ev         = Counter(t['ev_id'] for t in rule_triples)
    densities      = list(per_ev.values())

    print(f'\n  Rule-based  -- narratives with >=1 triple: {ev_with_rule}/{sample_n} ({ev_with_rule/sample_n:.1%})')
    print(f'  Rule-based  -- total triples: {len(rule_triples)}  avg: {np.mean(densities):.2f}')
    print(f'  Dep-parse   -- narratives with >=1 triple: {ev_with_dep}/{sample_n} ({ev_with_dep/sample_n:.1%})')
    print(f'  Dep-parse   -- total triples: {len(dep_triples)}')
    print(f'  Direction breakdown: {dict(direction_counts)}')
    print(f'  Pattern hit counts:')
    for pat, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f'    \'{pat}\': {cnt}')

    plot_traditional_nlp(rule_triples, dep_triples, sample_n, plots_dir)

    return {
        'rule_based': {
            'total_triples': len(rule_triples),
            'coverage': round(ev_with_rule / sample_n, 4),
            'avg_density': round(float(np.mean(densities)), 2),
            'pattern_counts': dict(pattern_counts),
            'direction_counts': dict(direction_counts),
        },
        'dep_parsing': {
            'sample_size': sample_n,
            'total_triples': len(dep_triples),
            'coverage': round(ev_with_dep / sample_n, 4),
        },
        'all_rule_triples': rule_triples,
        'all_dep_triples': dep_triples,
    }


# ---------------------------------------------------------------------------
# Model 2 -- DistilBERT evaluation
# ---------------------------------------------------------------------------

def eval_distilbert(training_dir: Path, output_dir: Path, plots_dir: Path, df: pd.DataFrame,
                    sample_n: int, cfg) -> dict:
    section('MODEL 2: DistilBERT Transformer Classifier (Evaluation)')

    model_dir     = output_dir / 'model'
    history_path  = training_dir / 'train_history.json'

    if not model_dir.exists():
        print('  Model not found -- run train.py first.')
        return {}

    try:
        import torch
        from src.transformer_classifier import NarrativeDataset
    except ImportError:
        print('  torch not installed -- skipping.')
        return {}

    # Load saved label map
    label_map_path = model_dir / 'label_map.json'
    if not label_map_path.exists():
        print('  label_map.json not found.')
        return {}
    with open(label_map_path) as f:
        label_map = json.load(f)
    inv_map = {v: k for k, v in label_map.items()}

    # Rebuild test set from the same sample + same split params
    t = cfg['transformer'] if 'transformer' in cfg else {}
    test_size  = float(t.get('test_size', 0.15))
    val_size   = float(t.get('val_size',  0.15))

    clf = NTSBClassifier(num_labels=len(label_map), model_name='distilbert-base-uncased')
    clf.label_map     = label_map
    clf.inv_label_map = inv_map

    sample_df = df.sample(n=min(sample_n, len(df)), random_state=42).reset_index(drop=True)
    _, _, test_ds, _ = clf.prepare_data(
        sample_df,
        text_col='narr_clean',
        label_col='top_category',
        id_col='ev_id',
        test_size=test_size,
        val_size=val_size,
        max_samples=None,
    )
    # clf.test_ev_ids is now populated; update test_split.json with ev_ids
    try:
        existing_split = _load_json(training_dir / 'test_split.json')
        if isinstance(existing_split, dict):
            existing_split['test_ev_ids']  = clf.test_ev_ids
            existing_split['train_ev_ids'] = clf.train_ev_ids
            _save_json(existing_split, training_dir / 'test_split.json')
    except Exception:
        pass

    # Load best weights
    clf.load(str(model_dir))
    print(f'  Loaded model from {model_dir}')

    # Evaluate
    results = clf.evaluate(test_ds)
    acc     = results['accuracy']
    print(f'\n  Test Accuracy: {acc:.4f} ({acc*100:.1f}%)')

    if 'classification_report' in results:
        report = results['classification_report']
        print(f'\n  {"Class":<25} {"Precision":>10} {"Recall":>8} {"F1":>8} {"Support":>9}')
        print('  ' + '-' * 65)
        for label in LABEL_COLS:
            if label in report:
                m = report[label]
                print(f'  {label:<25} {m["precision"]:>10.3f} {m["recall"]:>8.3f} '
                      f'{m["f1-score"]:>8.3f} {int(m["support"]):>9}')
        for avg in ('macro avg', 'weighted avg'):
            if avg in report:
                m = report[avg]
                print(f'  {avg:<25} {m["precision"]:>10.3f} {m["recall"]:>8.3f} {m["f1-score"]:>8.3f}')

    # Load training history for curves
    history = _load_json(history_path) if history_path.exists() else {}
    results['train_history'] = history

    # Plots
    all_preds, all_labels = clf.get_predictions(test_ds)
    class_names = [inv_map[i] for i in sorted(inv_map)]
    plot_confusion_matrix(all_preds, all_labels, class_names, plots_dir)
    plot_training_curves(history, plots_dir)
    plot_per_class_metrics(results.get('classification_report', {}), label_map, plots_dir)

    # Sample predictions
    print('\n  Sample predictions:')
    sample_texts = sample_df[sample_df['top_category'].isin(LABEL_COLS)]['narr_clean'].sample(
        5, random_state=99).tolist()
    for i, text in enumerate(sample_texts):
        label, conf = clf.predict(text)
        print(f'  [{i+1}] {label} ({conf:.1%}) | {text[:80].replace(chr(10), " ")}')

    # Run inference on ALL narratives with a finding so we can compare against
    # the extraction models on the same ground-truth denominator.
    full_df = df[df['top_category'].isin(LABEL_COLS) & df['ev_id'].notna()].copy()
    print(f'\n  Running full-dataset inference for finding alignment ({len(full_df)} narratives)...')
    predictions: dict = {}
    loader_full = torch.utils.data.DataLoader(
        NarrativeDataset(
            full_df['narr_clean'].astype(str).tolist(),
            [0] * len(full_df),   # dummy labels
            clf.tokenizer,
        ),
        batch_size=64,
        shuffle=False,
        **clf.loader_kwargs,
    )
    clf.model.eval()
    all_pred_ids = []
    with torch.no_grad():
        for batch in loader_full:
            batch = clf._to_device(batch)
            with clf._autocast_context():
                outputs = clf.model(**{k: v for k, v in batch.items() if k != 'labels'})
            all_pred_ids.extend(outputs.logits.argmax(dim=-1).cpu().tolist())

    ev_ids = full_df['ev_id'].astype(str).tolist()
    for ev_id, pred_idx in zip(ev_ids, all_pred_ids):
        predictions[ev_id] = inv_map.get(pred_idx, str(pred_idx))

    pred_path = output_dir / 'evaluation' / 'distilbert_predictions.json'
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
    print(f'  Saved {len(predictions)} predictions -> {pred_path}')

    # Also save test-set-only predictions for unified cross-model evaluation.
    test_set = set(clf.test_ev_ids)
    test_predictions = {eid: cat for eid, cat in predictions.items() if eid in test_set}
    test_pred_path = output_dir / 'evaluation' / 'distilbert_test_predictions.json'
    with open(test_pred_path, 'w') as f:
        json.dump(test_predictions, f)
    print(f'  Saved {len(test_predictions)} test-set predictions -> {test_pred_path}')

    results['full_predictions'] = predictions
    return results


# ---------------------------------------------------------------------------
# Model 4 -- LLM evaluation
# ---------------------------------------------------------------------------

def eval_llm(extractions_dir: Path, sample_n: int, plots_dir: Path) -> list:
    section('MODEL 4: LLM Prompt-Based Causal Extraction (Evaluation)')

    llm_triples = _load_json(extractions_dir / 'llm_triples.json')
    if not llm_triples:
        print('  llm_triples.json not found or empty -- run train.py first.')
        return []

    ev_with        = len({t['ev_id'] for t in llm_triples})
    pattern_counts = Counter(t['relation'] for t in llm_triples)
    print(f'\n  Coverage:          {ev_with}/{sample_n} ({ev_with/sample_n:.1%})')
    print(f'  Total triples:     {len(llm_triples)}')
    print(f'  Avg per narrative: {len(llm_triples)/max(1, ev_with):.2f}')
    print(f'  Top relation phrases:')
    for rel, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1])[:10]:
        print(f'    \'{rel}\': {cnt}')

    plot_llm_analysis(llm_triples, sample_n, plots_dir)
    return llm_triples


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

    # Model 1
    trad_results = eval_traditional_nlp(training_dir, sample_n, plots_dir)

    # Model 2
    transformer_results = eval_distilbert(training_dir, output_dir, plots_dir, df, sample_n, cfg)

    # Model 4
    llm_triples = [] if args.no_llm else eval_llm(extractions_dir, sample_n, plots_dir)

    # LLM few-shot on test set (independent of full-dataset zero-shot run)
    fewshot_triples = []
    if not args.no_fewshot and not args.no_llm:
        fewshot_triples = eval_llm_fewshot_testset(df, training_dir, extractions_dir, cfg)

    # Model 3 (KG)
    kg_results = eval_knowledge_graph(
        trad_results.get('all_rule_triples', []),
        trad_results.get('all_dep_triples', []),
        llm_triples,
        cfg, output_dir, plots_dir,
    )

    # Cross-model comparison
    plot_cross_model_comparison(
        trad_results.get('all_rule_triples', []),
        trad_results.get('all_dep_triples', []),
        llm_triples,
        sample_n,
        transformer_results,
        plots_dir,
    )

    # Save evaluation report
    report = {
        'sample_n': sample_n,
        'traditional_nlp': {
            'rule_based':  trad_results.get('rule_based',  {}),
            'dep_parsing': trad_results.get('dep_parsing', {}),
        },
        'transformer': {k: v for k, v in transformer_results.items()
                        if k not in ('train_history')},
        'llm_extractor': {
            'total_triples':          len(llm_triples),
            'narratives_with_triple': len({t['ev_id'] for t in llm_triples}),
        },
        'llm_fewshot_testset': {
            'total_triples':          len(fewshot_triples),
            'narratives_with_triple': len({t['ev_id'] for t in fewshot_triples}),
        },
        'knowledge_graph': {k: v for k, v in kg_results.items() if k != '_stats'},
    }
    report_path = eval_dir / 'evaluation_report.json'
    _save_json(report, report_path)

    section('Evaluation Complete -- Output Files')
    print(f'  Evaluation report:          {report_path}')
    print(f'  Traditional NLP plot:       {plots_dir / "eval_traditional_nlp.png"}')
    print(f'  Transformer confusion:      {plots_dir / "eval_transformer_confusion.png"}')
    print(f'  Transformer training curves:{plots_dir / "eval_transformer_training_curves.png"}')
    print(f'  Transformer per-class:      {plots_dir / "eval_transformer_per_class.png"}')
    print(f'  LLM extraction plot:        {plots_dir / "eval_llm_extraction.png"}')
    print(f'  KG visualization:           {plots_dir / "eval_knowledge_graph_full.png"}')
    print(f'  KG stats plot:              {plots_dir / "eval_kg_stats.png"}')
    print(f'  Cross-model comparison:     {plots_dir / "eval_cross_model_comparison.png"}')
    print(f'  Neo4j Cypher:               {output_dir / "extractions" / "neo4j_import_full.cypher"}')
    print('=' * 70)


if __name__ == '__main__':
    main()
