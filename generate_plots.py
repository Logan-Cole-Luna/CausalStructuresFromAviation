"""
generate_plots.py — Regenerate all evaluation plots from saved artifacts.

Does NOT reload or retrain any model. Reads only JSON files produced by
train.py / eval.py and the LLM response cache.

Usage:
    python generate_plots.py
    python generate_plots.py --config CONFIG.conf
    python generate_plots.py --skip-distilbert   # skip plots that need eval_report metrics
"""
import argparse
import configparser
import json
import sys
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cfg(path: str = 'CONFIG.conf') -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    cfg.read(path)
    return cfg


def _load_json(path: Path):
    if not path.exists():
        print(f'  [warn] {path} not found — skipping.')
        return {} if path.suffix == '.json' else []
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f'  Saved -> {path}')


def section(title: str):
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


# ---------------------------------------------------------------------------
# Import all plot functions
# ---------------------------------------------------------------------------

try:
    from src.plotting import (
        plot_traditional_nlp,
        plot_confusion_matrix,
        plot_training_curves,
        plot_per_class_metrics,
        plot_llm_analysis,
        plot_kg_stats,
        plot_cross_model_comparison,
        plot_radar_extraction,
        plot_radar_classifier,
        plot_kg_per_source,
        plot_llm_cache_growth,
        plot_finding_alignment,
    )
except ImportError as exc:
    sys.exit(f'[error] Could not import plotting functions: {exc}')

try:
    from src.knowledge_graph import build_graph, graph_stats, visualize_subgraph
    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False
    print('[warn] networkx not available — KG visualizations will be skipped.')

try:
    from src.finding_evaluator import (
        load_findings, evaluate_finding_alignment,
        evaluate_classifier_alignment, print_finding_report,
    )
    FINDING_EVAL_AVAILABLE = True
except ImportError as exc:
    FINDING_EVAL_AVAILABLE = False
    print(f'[warn] finding_evaluator import failed: {exc}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='NTSB — Plot generation (no retraining)')
    parser.add_argument('--config',          default='CONFIG.conf')
    parser.add_argument('--skip-distilbert', action='store_true',
                        help='Skip confusion matrix / per-class plots (need predictions)')
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    output_dir      = Path(cfg.get('paths', 'output_dir',  fallback='outputs'))
    training_dir    = output_dir / 'training'
    extractions_dir = output_dir / 'extractions'
    eval_dir        = output_dir / 'evaluation'
    plots_dir       = output_dir / 'plots'
    model_dir       = output_dir / 'model'

    kg_cfg             = cfg['knowledge_graph'] if 'knowledge_graph' in cfg else {}
    noise_filter       = kg_cfg.get('noise_filter',       'true').lower() == 'true'
    normalize_entities = kg_cfg.get('normalize_entities', 'true').lower() == 'true'
    top_n              = int(kg_cfg.get('visualize_top_n', 40))

    for d in (plots_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load run config (get sample_n used during training)
    # -----------------------------------------------------------------------
    run_cfg  = _load_json(training_dir / 'run_config.json')
    sample_n = run_cfg.get('sample_n') if isinstance(run_cfg, dict) else None
    if sample_n is None:
        sample_n = int(cfg.get('global', 'sample_n', fallback=2000))
    # 0 is the sentinel for "full dataset" — generate_plots reads it from the saved
    # run_config, so it's already the resolved integer if train.py ran correctly.
    # Guard in case generate_plots is run before a full training pass.
    if sample_n == 0:
        sample_n = 6059  # conservative upper bound; will be capped by actual data

    print('=' * 70)
    print('  NTSB Causal Chain Extraction — Plot Generation (no retraining)')
    print('=' * 70)
    print(f'\n  Sample n (from training run): {sample_n}')
    print(f'  Output dir: {output_dir}')

    # -----------------------------------------------------------------------
    # Load extraction artifacts
    # -----------------------------------------------------------------------
    section('Loading Artifacts')
    rule_triples    = _load_json(training_dir    / 'rule_triples.json')
    dep_triples     = _load_json(training_dir    / 'dep_triples.json')
    llm_triples     = _load_json(extractions_dir / 'llm_triples.json')
    fewshot_triples = _load_json(extractions_dir / 'llm_triples_fewshot.json')
    train_history   = _load_json(training_dir    / 'train_history.json')
    eval_report     = _load_json(eval_dir        / 'evaluation_report.json')
    cache_path      = Path(cfg.get('llm_extractor', 'cache_path',
                                   fallback='outputs/extractions/llm_response_cache.json'))
    llm_cache = _load_json(cache_path) if cache_path.exists() else {}

    # Load DistilBERT label map
    label_map_path = model_dir / 'label_map.json'
    label_map = _load_json(label_map_path) if label_map_path.exists() else {}
    inv_map   = {v: k for k, v in label_map.items()} if label_map else {}

    print(f'  Rule triples loaded:        {len(rule_triples):,}')
    print(f'  Dep triples loaded:         {len(dep_triples):,}')
    print(f'  LLM triples loaded:         {len(llm_triples):,}')
    print(f'  LLM few-shot triples:       {len(fewshot_triples):,}')
    print(f'  LLM cache entries:          {len(llm_cache):,}')
    print(f'  Train history epochs:       {len(train_history.get("train_loss", []))}')

    # -----------------------------------------------------------------------
    # Parse full LLM cache → extended triple set
    # -----------------------------------------------------------------------
    section('Parsing Full LLM Cache')
    full_llm_triples = llm_triples  # fallback if cache parsing fails

    if llm_cache:
        from src.plotting import plot_llm_cache_growth
        result = plot_llm_cache_growth(llm_cache, llm_triples, sample_n, plots_dir)
        if result:
            full_llm_triples = result
            print(f'  Full cache parsed: {len(full_llm_triples):,} triples from '
                  f'{len({t["ev_id"] for t in full_llm_triples}):,} narratives')
            # Persist extended triples so eval.py can use them later
            extended_path = extractions_dir / 'llm_triples_full.json'
            _save_json(full_llm_triples, extended_path)

    # -----------------------------------------------------------------------
    # Model 1 — Traditional NLP plots
    # -----------------------------------------------------------------------
    section('Traditional NLP Plots')
    if rule_triples:
        plot_traditional_nlp(rule_triples, dep_triples, sample_n, plots_dir)
    else:
        print('  No rule triples — skipping.')

    # -----------------------------------------------------------------------
    # Model 2 — DistilBERT training curves (no inference needed)
    # -----------------------------------------------------------------------
    section('DistilBERT Training Curves')
    if train_history:
        plot_training_curves(train_history, plots_dir)
        best = train_history.get('best_val_acc', 0)
        best_ep = train_history.get('best_epoch', '?')
        print(f'  Best val acc: {best:.4f} at epoch {best_ep}')

    # Per-class metrics — now persisted in evaluation_report.json by eval.py
    class_report = eval_report.get('transformer', {}).get('classification_report', {})
    if class_report and label_map and not args.skip_distilbert:
        plot_per_class_metrics(class_report, label_map, plots_dir)
        plot_radar_classifier(class_report, label_map, plots_dir)
    else:
        print('  Per-class metrics not in saved report -- run src/eval.py once to generate them.')

    # -----------------------------------------------------------------------
    # Model 4 — LLM extraction plots
    # -----------------------------------------------------------------------
    section('LLM Extraction Plots')
    if full_llm_triples:
        # Use full cache sample size if available, otherwise training sample_n
        llm_sample = len(llm_cache) if llm_cache else sample_n
        plot_llm_analysis(full_llm_triples, llm_sample, plots_dir)
    else:
        print('  No LLM triples — skipping.')

    # -----------------------------------------------------------------------
    # Model 3 — Knowledge graph plots
    # -----------------------------------------------------------------------
    section('Knowledge Graph Plots')
    if KG_AVAILABLE and (rule_triples or dep_triples or full_llm_triples):
        G_rules = build_graph(rule_triples,       noise_filter=noise_filter, normalize=normalize_entities)
        G_deps  = build_graph(dep_triples,        noise_filter=noise_filter, normalize=normalize_entities)
        G_llm   = build_graph(full_llm_triples,   noise_filter=noise_filter, normalize=normalize_entities)
        G_all   = build_graph(
            rule_triples + dep_triples + full_llm_triples,
            noise_filter=noise_filter, normalize=normalize_entities,
        )

        stats_rules = graph_stats(G_rules)
        stats_deps  = graph_stats(G_deps)
        stats_llm   = graph_stats(G_llm)
        stats_all   = graph_stats(G_all)

        print(f'  KG sizes — Rule: {G_rules.number_of_nodes()}n/{G_rules.number_of_edges()}e  '
              f'| Dep: {G_deps.number_of_nodes()}n/{G_deps.number_of_edges()}e  '
              f'| LLM: {G_llm.number_of_nodes()}n/{G_llm.number_of_edges()}e  '
              f'| Combined: {G_all.number_of_nodes()}n/{G_all.number_of_edges()}e')

        # Stats bar chart (rule / dep / combined)
        plot_kg_stats(stats_rules, stats_deps, stats_all, plots_dir)

        # Per-source KG network visualizations
        plot_kg_per_source(
            rule_triples, dep_triples, full_llm_triples,
            noise_filter=noise_filter, normalize=normalize_entities,
            top_n=top_n, plots_dir=plots_dir,
        )

        # Full combined graph visualization
        viz_path = str(plots_dir / 'eval_knowledge_graph_full.png')
        visualize_subgraph(G_all, top_n=top_n, save_path=viz_path)

        # Save updated graph stats
        _save_json({
            'rule':     {k: v for k, v in stats_rules.items()
                         if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
            'dep':      {k: v for k, v in stats_deps.items()
                         if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
            'llm':      {k: v for k, v in stats_llm.items()
                         if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
            'combined': {k: v for k, v in stats_all.items()
                         if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
        }, extractions_dir / 'graph_stats_updated.json')
    else:
        print('  networkx not available or no triples — skipping KG plots.')
        stats_rules = stats_deps = stats_llm = {}

    # -----------------------------------------------------------------------
    # Radar — extraction models
    # -----------------------------------------------------------------------
    section('Radar Charts')
    if rule_triples or dep_triples or full_llm_triples:
        # Use the unified sample_n for the radar (all models were run on full dataset)
        plot_radar_extraction(
            rule_triples, dep_triples, full_llm_triples,
            stats_rules, stats_deps, stats_llm,
            sample_n=sample_n,
            plots_dir=plots_dir,
        )

    # Radar for DistilBERT per-class (requires classification_report in eval_report)
    class_report = eval_report.get('transformer', {}).get('classification_report', {})
    if class_report and label_map:
        plot_radar_classifier(class_report, label_map, plots_dir)

    # -----------------------------------------------------------------------
    # Cross-model comparison
    # -----------------------------------------------------------------------
    section('Cross-Model Comparison')
    transformer_results = {
        'accuracy':      eval_report.get('transformer', {}).get('accuracy', None),
        'train_history': train_history,
        'classification_report': eval_report.get('transformer', {}).get('classification_report', {}),
    }
    plot_cross_model_comparison(
        rule_triples, dep_triples, full_llm_triples,
        sample_n, transformer_results, plots_dir,
    )

    # -----------------------------------------------------------------------
    # Finding-alignment evaluation (ground truth)
    # -----------------------------------------------------------------------
    section('Finding-Alignment Evaluation (Ground Truth)')
    data_path = cfg.get('paths', 'data_path',
                        fallback='data/clean/cleaned_narritives_and_findings.csv')

    if FINDING_EVAL_AVAILABLE and Path(data_path).exists():
        findings_df = load_findings(data_path)
        print(f'  Findings loaded: {len(findings_df):,} rows, '
              f'{findings_df["ev_id"].nunique():,} unique accidents')
        print(f'  Cause-coded (C): {findings_df["is_cause"].sum():,}')

        # Load test_split ev_ids if available (produced by train.py / eval.py after a
        # training run).  When present we run a UNIFIED evaluation — all models evaluated
        # on the same held-out narratives so results are directly comparable.
        test_split = _load_json(training_dir / 'test_split.json')
        test_ev_ids = (
            set(test_split.get('test_ev_ids', []))
            if isinstance(test_split, dict) else set()
        )
        unified_mode = bool(test_ev_ids)
        if unified_mode:
            print(f'\n  Unified test-set mode: {len(test_ev_ids)} held-out narratives')
        else:
            print('\n  Full-dataset mode (no test_ev_ids in test_split.json)')

        def _filter_triples(triples, ev_ids_set):
            """Filter to test ev_ids when in unified mode, else return all."""
            if not ev_ids_set:
                return triples
            return [t for t in triples if str(t.get('ev_id', '')) in ev_ids_set]

        alignment_results = []

        # --- Full-dataset evaluation (always run) ---
        section('Finding-Alignment Evaluation -- Full Dataset')
        for label, triples in [
            ('Rule-based', rule_triples),
            ('Dep-parse',  dep_triples),
            ('LLM',        full_llm_triples),
        ]:
            if not triples:
                continue
            res = evaluate_finding_alignment(triples, findings_df, label=label)
            alignment_results.append(res)
            print(f'\n  [{label}]')
            print(f'    Category alignment:       {res["category_alignment_score"]:.1%}  '
                  f'(n={res["category_alignment_n"]})')
            print(f'    Cause-confirmed coverage: {res["cause_confirmed_coverage"]:.1%}  '
                  f'({res["cause_confirmed_n"]}/{res["cause_confirmed_denom"]})')
            print(f'    Finding keyword recall:   {res["finding_keyword_recall"]:.1%}  '
                  f'(n={res["keyword_recall_n"]})')

        pred_path = eval_dir / 'distilbert_predictions.json'
        if pred_path.exists():
            distilbert_preds = _load_json(pred_path)
            if distilbert_preds:
                db_res = evaluate_classifier_alignment(
                    distilbert_preds, findings_df, label='DistilBERT (full)'
                )
                alignment_results.append(db_res)
                print(f'\n  [DistilBERT (full)]')
                print(f'    Category alignment:       {db_res["category_alignment_score"]:.1%}  '
                      f'(n={db_res["category_alignment_n"]})')
                print(f'    Cause-confirmed coverage: {db_res["cause_confirmed_coverage"]:.1%}  '
                      f'({db_res["cause_confirmed_n"]}/{db_res["cause_confirmed_denom"]})')

        if alignment_results:
            print_finding_report(alignment_results)
            plot_finding_alignment(alignment_results, plots_dir)

        # --- Unified test-set evaluation (only when test_ev_ids are available) ---
        if unified_mode:
            section('Finding-Alignment Evaluation -- Unified Test Set')
            unified_results = []

            # LLM few-shot triples are already restricted to the test set
            for label, triples in [
                ('Rule-based',     rule_triples),
                ('Dep-parse',      dep_triples),
                ('LLM (zero-shot)', full_llm_triples),
                ('LLM (few-shot)', fewshot_triples),
            ]:
                if not triples:
                    continue
                filtered = _filter_triples(triples, test_ev_ids)
                res = evaluate_finding_alignment(filtered, findings_df, label=label)
                unified_results.append(res)
                print(f'\n  [{label}]  (test set: {len(filtered)} triples from '
                      f'{res["ev_ids_extracted"]} narratives)')
                print(f'    Category alignment:       {res["category_alignment_score"]:.1%}')
                print(f'    Cause-confirmed coverage: {res["cause_confirmed_coverage"]:.1%}  '
                      f'({res["cause_confirmed_n"]}/{res["cause_confirmed_denom"]})')
                print(f'    Finding keyword recall:   {res["finding_keyword_recall"]:.1%}')

            # Prefer test-set predictions; fall back to filtering full predictions
            test_pred_path = eval_dir / 'distilbert_test_predictions.json'
            if test_pred_path.exists():
                test_preds = _load_json(test_pred_path)
            elif pred_path.exists():
                all_preds = _load_json(pred_path)
                test_preds = {eid: cat for eid, cat in all_preds.items()
                              if eid in test_ev_ids}
            else:
                test_preds = {}

            if test_preds:
                db_test_res = evaluate_classifier_alignment(
                    test_preds, findings_df, label='DistilBERT'
                )
                unified_results.append(db_test_res)
                print(f'\n  [DistilBERT]  (test set: {len(test_preds)} predictions)')
                print(f'    Category alignment:       {db_test_res["category_alignment_score"]:.1%}')
                print(f'    Cause-confirmed coverage: {db_test_res["cause_confirmed_coverage"]:.1%}  '
                      f'({db_test_res["cause_confirmed_n"]}/{db_test_res["cause_confirmed_denom"]})')

            if unified_results:
                print_finding_report(unified_results)
                from src.plotting import plot_finding_alignment as _pfa
                _pfa(unified_results, plots_dir, suffix='_unified')

                existing = _load_json(eval_dir / 'evaluation_report.json')
                if isinstance(existing, dict):
                    existing['finding_alignment_unified'] = {
                        r['label']: {k: v for k, v in r.items() if k != 'label'}
                        for r in unified_results
                    }

        # Persist full-dataset alignment into evaluation_report.json
        existing = _load_json(eval_dir / 'evaluation_report.json')
        if isinstance(existing, dict):
            existing['finding_alignment'] = {
                r['label']: {k: v for k, v in r.items() if k != 'label'}
                for r in alignment_results
            }
            _save_json(existing, eval_dir / 'evaluation_report.json')
    else:
        print('  finding_evaluator not available or data_path missing -- skipping.')

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    section('All Plots Generated')
    plot_files = sorted(plots_dir.glob('*.png'))
    for p in plot_files:
        size_kb = p.stat().st_size / 1024
        print(f'  {p.name:<50}  {size_kb:>6.0f} KB')
    print(f'\n  Total: {len(plot_files)} plots in {plots_dir}')


if __name__ == '__main__':
    main()
