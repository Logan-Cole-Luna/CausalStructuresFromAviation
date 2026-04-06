"""
eval.py — Evaluate all models and generate plots.

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
# Model 1 — Traditional NLP evaluation
# ---------------------------------------------------------------------------

def eval_traditional_nlp(training_dir: Path, sample_n: int, plots_dir: Path) -> dict:
    section('MODEL 1: Traditional NLP — Causal Extraction (Evaluation)')

    rule_triples = _load_json(training_dir / 'rule_triples.json')
    dep_triples  = _load_json(training_dir / 'dep_triples.json')

    if not rule_triples:
        print('  rule_triples.json not found — run train.py first.')
        return {'all_rule_triples': [], 'all_dep_triples': []}

    ev_with_rule   = len({t['ev_id'] for t in rule_triples})
    ev_with_dep    = len({t['ev_id'] for t in dep_triples})
    pattern_counts = Counter(t['relation'] for t in rule_triples)
    direction_counts = Counter(t['direction'] for t in rule_triples)
    per_ev         = Counter(t['ev_id'] for t in rule_triples)
    densities      = list(per_ev.values())

    print(f'\n  Rule-based  — narratives with ≥1 triple: {ev_with_rule}/{sample_n} ({ev_with_rule/sample_n:.1%})')
    print(f'  Rule-based  — total triples: {len(rule_triples)}  avg: {np.mean(densities):.2f}')
    print(f'  Dep-parse   — narratives with ≥1 triple: {ev_with_dep}/{sample_n} ({ev_with_dep/sample_n:.1%})')
    print(f'  Dep-parse   — total triples: {len(dep_triples)}')
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
# Model 2 — DistilBERT evaluation
# ---------------------------------------------------------------------------

def eval_distilbert(training_dir: Path, output_dir: Path, plots_dir: Path, df: pd.DataFrame,
                    sample_n: int, cfg) -> dict:
    section('MODEL 2: DistilBERT Transformer Classifier (Evaluation)')

    model_dir     = output_dir / 'model'
    history_path  = training_dir / 'train_history.json'

    if not model_dir.exists():
        print('  Model not found — run train.py first.')
        return {}

    try:
        import torch
        from src.transformer_classifier import NarrativeDataset
    except ImportError:
        print('  torch not installed — skipping.')
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
        test_size=test_size,
        val_size=val_size,
        max_samples=None,
    )

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
        print(f'  [{i+1}] {label} ({conf:.1%}) | {text[:80].replace(chr(10), " ")}…')

    return results


# ---------------------------------------------------------------------------
# Model 4 — LLM evaluation
# ---------------------------------------------------------------------------

def eval_llm(extractions_dir: Path, sample_n: int, plots_dir: Path) -> list:
    section('MODEL 4: LLM Prompt-Based Causal Extraction (Evaluation)')

    llm_triples = _load_json(extractions_dir / 'llm_triples.json')
    if not llm_triples:
        print('  llm_triples.json not found or empty — run train.py first.')
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
# Model 3 — Knowledge graph evaluation
# ---------------------------------------------------------------------------

def eval_knowledge_graph(rule_triples: list, dep_triples: list, llm_triples: list,
                          cfg, output_dir: Path, plots_dir: Path) -> dict:
    section('MODEL 3: Knowledge Graph — Structural Evaluation')

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
        print(f'\n  Node overlap (rule ∩ dep): {overlap}/{total_u} ({overlap/max(1,total_u):.1%})')

    cypher_path = output_dir / 'extractions' / 'neo4j_import_full.cypher'
    to_neo4j_cypher(all_triples, path=str(cypher_path),
                    noise_filter=noise_filter, normalize=normalize_entities)

    kg_viz_path = str(plots_dir / 'eval_knowledge_graph_full.png')
    visualize_subgraph(G_all, top_n=top_n, save_path=kg_viz_path)
    print(f'  KG visualization saved → {kg_viz_path}')

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
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='NTSB — Evaluation & Plotting')
    parser.add_argument('--sample', type=int,  default=None,
                        help='Sample size (must match what was used in train.py)')
    parser.add_argument('--config', type=str,  default='CONFIG.conf')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM evaluation')
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
    print('  NTSB Causal Chain Extraction — Evaluation')
    print('  Team: Madeline Gorman, Katherine Hoffsetz, Logan Luna, Stephanie Ramsey')
    print('=' * 70)
    print(f'\n  Sample size (all models): {sample_n}')

    print('\nLoading and preprocessing data...')
    df = load_data(data_path)
    df = preprocess_data(df)
    print(f'  Records: {len(df)}')

    # Model 1
    trad_results = eval_traditional_nlp(training_dir, sample_n, plots_dir)

    # Model 2
    transformer_results = eval_distilbert(training_dir, output_dir, plots_dir, df, sample_n, cfg)

    # Model 4
    llm_triples = [] if args.no_llm else eval_llm(extractions_dir, sample_n, plots_dir)

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
                        if k not in ('classification_report', 'train_history')},
        'llm_extractor': {
            'total_triples':          len(llm_triples),
            'narratives_with_triple': len({t['ev_id'] for t in llm_triples}),
        },
        'knowledge_graph': {k: v for k, v in kg_results.items() if k != '_stats'},
    }
    report_path = eval_dir / 'evaluation_report.json'
    _save_json(report, report_path)

    section('Evaluation Complete — Output Files')
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
