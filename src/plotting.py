"""
Plotting utilities for NTSB Causal Chain Extraction.
Standardized functions for all model comparison plots.

No torch / transformers imports - purely matplotlib + seaborn + numpy.
"""
from __future__ import annotations

import math
import json as _json
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.extractor_utils import _JUNK_NODES, _JSON_RE


# =========================================================================
# Helpers
# =========================================================================

def _save(fig, plots_dir: Path, filename: str):
    """Save figure and close."""
    path = plots_dir / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved -> {path}")


def _model_colors():
    """Return standard color palette for 6 models."""
    return {
        'Rule-based': '#2196F3',      # Blue
        'Dep-parse': '#4CAF50',        # Green
        'BERT': '#FF9800',             # Orange
        'T5': '#FFC107',               # Amber
        'LLM (0-shot)': '#9C27B0',     # Purple
        'LLM (few-shot)': '#e91e63',   # Pink
    }


def _get_color_list(models: List[str]) -> List[str]:
    """Get colors for a list of model names."""
    palette = _model_colors()
    return [palette.get(m, '#666666') for m in models]


# =========================================================================
# Standardized Extraction Metrics
# =========================================================================

def _compute_extraction_stats(triples: List[dict]) -> Tuple[int, int, float]:
    """
    Compute coverage, total triples, and avg density for a triple list.
    Returns: (num_narratives, total_triples, avg_per_narrative)
    """
    if not triples:
        return 0, 0, 0.0
    narratives = len({t['ev_id'] for t in triples})
    total = len(triples)
    avg = total / max(1, narratives)
    return narratives, total, avg


def plot_cross_model_comparison_all_six(
    models_dict: Dict[str, List[dict]],
    sample_n: int,
    plots_dir: Path,
):
    """
    Coverage, density, and total yield bars for all 6 models.
    models_dict: {'Rule-based': [...], 'BERT': [...], ...}
    """
    model_names = list(models_dict.keys())
    stats = [_compute_extraction_stats(models_dict[name]) for name in model_names]

    narratives_list = [s[0] for s in stats]
    triples_list = [s[1] for s in stats]
    avg_density = [s[2] for s in stats]
    coverage_pct = [n / sample_n * 100 for n in narratives_list]

    colors = _get_color_list(model_names)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Coverage %
    bars = axes[0].bar(model_names, coverage_pct, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, coverage_pct):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[0].set_ylabel('Narratives with ≥1 triple (%)')
    axes[0].set_title('Extraction Coverage')
    axes[0].set_ylim(0, max(coverage_pct) * 1.15)
    axes[0].grid(True, axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # Average density
    bars = axes[1].bar(model_names, avg_density, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, avg_density):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)
    axes[1].set_ylabel('Avg triples per narrative')
    axes[1].set_title('Extraction Density')
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    # Total triples
    bars = axes[2].bar(model_names, triples_list, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, triples_list):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 50, f'{val:,}',
                     ha='center', va='bottom', fontweight='bold', fontsize=9)
    axes[2].set_ylabel('Total triples')
    axes[2].set_title('Triple Yield')
    axes[2].grid(True, axis='y', alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)

    plt.suptitle('Cross-Model Comparison - All Six Extraction Methods (Test Set)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_cross_model_comparison_all_six.png')


# =========================================================================
# Relation Phrases
# =========================================================================

def plot_top_relation_phrases(
    models_dict: Dict[str, List[dict]],
    plots_dir: Path,
    top_n: int = 10,
):
    """
    Six-panel horizontal bar chart: top relation phrases for all six models.
    models_dict: {'Rule-based': [...], 'BERT': [...], ...}
    """
    model_names = list(models_dict.keys())
    colors = _get_color_list(model_names)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes_flat = axes.flatten()

    for ax, name, color in zip(axes_flat, model_names, colors):
        triples = models_dict[name]
        if not triples:
            ax.set_title(f'{name}\n(no triples)')
            ax.axis('off')
            continue

        counts = Counter(t['relation'] for t in triples)
        top = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
        labels = [p for p, _ in top]
        vals = [v for _, v in top]
        total = sum(counts.values())

        bars = ax.barh(labels, vals, color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            pct = val / max(1, total) * 100
            ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:,} ({pct:.1f}%)', va='center', fontsize=8)

        ax.invert_yaxis()
        ax.set_xlabel('Triple count')
        n_ev = len({t['ev_id'] for t in triples})
        ax.set_title(
            f'{name}\n{len(triples):,} triples - {n_ev} narratives\n'
            f'top {top_n} of {len(counts)} unique relations',
            fontsize=10, fontweight='bold',
        )
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlim(right=max(vals) * 1.25 if vals else 1)

    plt.suptitle('Top Relation Phrases - Test Set (1,192 narratives) - All Six Models',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_top_relation_phrases.png')


# =========================================================================
# Finding Alignment (Ground Truth)
# =========================================================================

def plot_finding_alignment(
    alignment_results: List[dict],
    plots_dir: Path,
    suffix: str = '',
):
    """
    3-panel figure: category alignment, cause-confirmed coverage, keyword recall.
    alignment_results: list of dicts from finding_evaluator
    """
    if not alignment_results:
        return

    labels = [r['label'] for r in alignment_results]
    colors = _get_color_list(labels)

    cat_aln = [r['category_alignment_score'] * 100 for r in alignment_results]
    cc_cov = [r['cause_confirmed_coverage'] * 100 for r in alignment_results]
    kw_rec = [r['finding_keyword_recall'] * 100 if r['finding_keyword_recall'] is not None else None
              for r in alignment_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _bar_with_na(ax, vals, model_labels, model_colors, title, ylabel):
        x = np.arange(len(model_labels))
        w = 0.65
        for i, (val, _, col) in enumerate(zip(vals, model_labels, model_colors)):
            if val is None:
                ax.bar(i, 0, w, color='#e0e0e0', alpha=0.5, edgecolor='white')
                ax.text(i, 3, 'N/A', ha='center', va='bottom', fontsize=9, color='#999', fontweight='bold')
            else:
                ax.bar(i, val, w, color=col, alpha=0.85, edgecolor='white')
                ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom',
                        fontweight='bold', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=10, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 115)
        ax.grid(True, axis='y', alpha=0.3)

    _bar_with_na(axes[0], cat_aln, labels, colors,
                 'Category Alignment\n(predicted cat == NTSB finding cat)', 'Accuracy (%)')
    _bar_with_na(axes[1], cc_cov, labels, colors,
                 'Cause-Confirmed Coverage\n(C-findings only)', '% of C-finding accidents covered')
    _bar_with_na(axes[2], kw_rec, labels, colors,
                 'Finding Keyword Recall\n(% finding tokens in text)', 'Avg recall (%)')

    for ax, n_items in zip(axes, [
        [f'n={r["category_alignment_n"]}' for r in alignment_results],
        [f'{r["cause_confirmed_n"]}/{r["cause_confirmed_denom"]}' for r in alignment_results],
        [f'n={r["keyword_recall_n"]}' if r['finding_keyword_recall'] is not None else 'N/A'
         for r in alignment_results],
    ]):
        for i, label in enumerate(n_items):
            ax.text(i, ax.get_ylim()[0] + 2, label, ha='center', va='bottom', fontsize=7, color='grey')

    title_note = ' (Unified Test Set)' if suffix else ''
    plt.suptitle(f'NTSB Finding-Alignment Evaluation — Ground Truth Comparison{title_note}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, f'eval_finding_alignment{suffix}.png')

    # Per-category alignment breakdown
    all_cats = sorted({cat for r in alignment_results
                       for cat in r.get('per_category_alignment', {})})
    if not all_cats:
        return

    x = np.arange(len(all_cats))
    width = 0.8 / max(1, len(labels))
    fig2, ax2 = plt.subplots(figsize=(13, 6))
    for i, (r, color) in enumerate(zip(alignment_results, colors)):
        pa = r.get('per_category_alignment', {})
        vals = [pa.get(cat, {}).get('score', 0) * 100 for cat in all_cats]
        ns = [pa.get(cat, {}).get('total', 0) for cat in all_cats]
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax2.bar(x + offset, vals, width * 0.9,
                       label=r['label'], color=color, alpha=0.85, edgecolor='white')
        for bar, val, n in zip(bars, vals, ns):
            if n > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                         f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

    short_cats = [c.replace(' issues', '') for c in all_cats]
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_cats, fontsize=11)
    ax2.set_ylabel('Category alignment accuracy (%)')
    ax2.set_ylim(0, 115)
    ax2.set_title(f'Category Alignment Score — Breakdown by NTSB Finding Category{title_note}',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig2, plots_dir, f'eval_finding_alignment_by_category{suffix}.png')


# =========================================================================
# Training Curves (Load from Tuning Results)
# =========================================================================

def plot_training_loss_curves(tuning_results: dict, plots_dir: Path):
    """
    Plot training loss curves for BERT and T5 from tuning results.
    tuning_results: dict with 'bert' and 't5' keys
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BERT
    if 'bert' in tuning_results and 'training_history' in tuning_results['bert']:
        bert_hist = tuning_results['bert']['training_history']
        train_loss = bert_hist.get('train_loss', [])
        # Extract val_loss from bias_variance_logs
        val_loss = [log['val_loss'] for log in bert_hist.get('bias_variance_logs', [])]
        if train_loss and val_loss:
            epochs = list(range(1, len(train_loss) + 1))
            axes[0].plot(epochs, train_loss, 'o-', color='#FF9800', label='Train Loss', linewidth=2)
            axes[0].plot(epochs, val_loss, 's-', color='#FF6F00', label='Val Loss', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('BERT - Training Loss Curve')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

    # T5
    if 't5' in tuning_results and 'training_history' in tuning_results['t5']:
        t5_hist = tuning_results['t5']['training_history']
        train_loss = t5_hist.get('train_loss', [])
        val_loss = t5_hist.get('val_loss', [])
        if train_loss and val_loss:
            epochs = list(range(1, len(train_loss) + 1))
            axes[1].plot(epochs, train_loss, 'o-', color='#FFC107', label='Train Loss', linewidth=2)
            axes[1].plot(epochs, val_loss, 's-', color='#FF6F00', label='Val Loss', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('T5 - Training Loss Curve')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Loss Curves - Best Hyperparameter Trials', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_training_loss_curves.png')


def plot_training_metrics(tuning_results: dict, plots_dir: Path):
    """
    Plot training accuracy (F1 for BERT, metric for T5) from tuning results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # BERT F1
    if 'bert' in tuning_results and 'training_history' in tuning_results['bert']:
        bert_hist = tuning_results['bert']['training_history']
        train_f1 = bert_hist.get('train_f1', [])
        val_f1 = bert_hist.get('val_f1', [])
        if train_f1 and val_f1:
            epochs = list(range(1, len(train_f1) + 1))
            axes[0].plot(epochs, train_f1, 'o-', color='#FF9800', label='Train F1', linewidth=2)
            axes[0].plot(epochs, val_f1, 's-', color='#FF6F00', label='Val F1', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('F1 Score')
            axes[0].set_title('BERT - Token Classification F1')
            axes[0].set_ylim([0.7, 1.0])
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

    # T5 Metric - compute from val_loss
    if 't5' in tuning_results and 'training_history' in tuning_results['t5']:
        t5_hist = tuning_results['t5']['training_history']
        val_loss = t5_hist.get('val_loss', [])
        if val_loss:
            epochs = list(range(1, len(val_loss) + 1))
            val_metric = [1 / (1 + loss) for loss in val_loss]
            axes[1].plot(epochs, val_metric, 's-', color='#FFC107', label='Val Metric (1/(1+loss))', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Metric Value')
            axes[1].set_title('T5 - Generation Quality Metric')
            axes[1].set_ylim([0, 1.0])
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Metrics - Best Hyperparameter Trials', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_training_metrics.png')


def plot_bias_variance_tradeoff(tuning_results: dict, plots_dir: Path):
    """
    Plot bias-variance tradeoff (train vs val loss gap) for BERT and T5.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # BERT
    if 'bert' in tuning_results and 'training_history' in tuning_results['bert']:
        bert_hist = tuning_results['bert']['training_history']
        train_loss = bert_hist.get('train_loss', [])
        bv_logs = bert_hist.get('bias_variance_logs', [])
        val_loss = [log['val_loss'] for log in bv_logs]
        if train_loss and val_loss:
            epochs = list(range(1, len(train_loss) + 1))
            gap = np.array(val_loss) - np.array(train_loss)

            colors_bert = []
            for g in gap:
                if g < -0.05:
                    colors_bert.append('#e74c3c')  # Red - underfitting
                elif g > 0.05:
                    colors_bert.append('#f39c12')  # Orange - overfitting
                else:
                    colors_bert.append('#27ae60')  # Green - balanced

            axes[0].bar(epochs, gap, color=colors_bert, alpha=0.85, edgecolor='white')
            axes[0].axhline(0, color='black', linestyle='-', linewidth=1)
            axes[0].axhline(0.05, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Overfitting threshold')
            axes[0].axhline(-0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Underfitting threshold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss Gap (Val - Train)')
            axes[0].set_title('BERT - Bias-Variance Tradeoff')
            axes[0].legend()
            axes[0].grid(True, axis='y', alpha=0.3)

    # T5
    if 't5' in tuning_results and 'training_history' in tuning_results['t5']:
        t5_hist = tuning_results['t5']['training_history']
        train_loss = t5_hist.get('train_loss', [])
        val_loss = t5_hist.get('val_loss', [])
        if train_loss and val_loss:
            epochs = list(range(1, len(train_loss) + 1))
            gap = np.array(val_loss) - np.array(train_loss)

            colors_t5 = []
            for g in gap:
                if g < -0.01:
                    colors_t5.append('#e74c3c')  # Red - underfitting
                elif g > 0.01:
                    colors_t5.append('#f39c12')  # Orange - overfitting
                else:
                    colors_t5.append('#27ae60')  # Green - balanced

            axes[1].bar(epochs, gap, color=colors_t5, alpha=0.85, edgecolor='white')
            axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
            axes[1].axhline(0.01, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Overfitting threshold')
            axes[1].axhline(-0.01, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Underfitting threshold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss Gap (Val - Train)')
            axes[1].set_title('T5 - Bias-Variance Tradeoff')
            axes[1].legend()
            axes[1].grid(True, axis='y', alpha=0.3)

    plt.suptitle('Bias-Variance Tradeoff - Training Dynamics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_bias_variance_tradeoff.png')


# =========================================================================
# Traditional NLP and LLM (single-model plots)
# =========================================================================

def plot_traditional_nlp(
    rule_triples: List[dict],
    dep_triples: List[dict],
    sample_n: int,
    plots_dir: Path,
):
    """Overview of rule-based and dep-parse extraction."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Rule pattern counts
    pattern_counts = Counter(t['relation'] for t in rule_triples)
    sorted_pats = sorted(pattern_counts.items(), key=lambda x: -x[1])
    pats = [p for p, _ in sorted_pats]
    vals = [v for _, v in sorted_pats]
    bars = axes[0, 0].barh(pats, vals, color='#2196F3', alpha=0.85)
    for bar, val in zip(bars, vals):
        axes[0, 0].text(val + 5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=8)
    axes[0, 0].set_xlabel('Count')
    axes[0, 0].set_title('Rule-based: Top Relation Patterns')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, axis='x', alpha=0.3)

    # Density distribution
    per_ev_rule = Counter(t['ev_id'] for t in rule_triples)
    densities = list(per_ev_rule.values())
    axes[0, 1].hist(densities, bins=20, color='#4CAF50', edgecolor='white', alpha=0.85)
    axes[0, 1].axvline(np.mean(densities), color='red', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(densities):.1f}')
    axes[0, 1].set_xlabel('Triples per narrative')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Rule-based: Density Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Coverage comparison
    ev_rule = len(per_ev_rule)
    ev_dep = len({t['ev_id'] for t in dep_triples})
    methods = ['Rule-based', 'Dep-parse']
    covered = [ev_rule, ev_dep]
    not_covered = [sample_n - c for c in covered]
    x = np.arange(len(methods))
    axes[1, 0].bar(x, [c/sample_n*100 for c in covered], color=['#2196F3', '#4CAF50'], alpha=0.85, label='Covered')
    axes[1, 0].bar(x, [nc/sample_n*100 for nc in not_covered], bottom=[c/sample_n*100 for c in covered],
                   color='#e0e0e0', alpha=0.85, label='Not covered')
    for i, c in enumerate(covered):
        axes[1, 0].text(i, c/sample_n*100/2, f'{c/sample_n:.0%}', ha='center', va='center',
                       fontweight='bold', color='white')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods)
    axes[1, 0].set_ylabel('% of narratives')
    axes[1, 0].set_ylim(0, 115)
    axes[1, 0].set_title('Coverage Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis='y', alpha=0.3)

    # Direction breakdown (rule-based only)
    direction_counts = Counter(t['direction'] for t in rule_triples)
    dir_labels = list(direction_counts.keys())
    dir_vals = list(direction_counts.values())
    colors_dir = ['#2196F3' if d == 'forward' else '#FF9800' for d in dir_labels]
    axes[1, 1].pie(dir_vals, labels=dir_labels, colors=colors_dir, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Rule-based: Causal Direction')

    plt.suptitle('Traditional NLP - Rule-based & Dep-parse Overview', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_traditional_nlp.png')


def plot_llm_analysis(
    llm_triples: List[dict],
    sample_n: int,
    plots_dir: Path,
):
    """Overview of LLM extraction."""
    if not llm_triples:
        return

    pattern_counts = Counter(t['relation'] for t in llm_triples)
    per_ev = Counter(t['ev_id'] for t in llm_triples)
    densities = list(per_ev.values())
    ev_with = len(per_ev)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Top relations
    top_rels = sorted(pattern_counts.items(), key=lambda x: -x[1])[:15]
    rel_labels = [r for r, _ in top_rels]
    rel_vals = [v for _, v in top_rels]
    neg_kw = ('not', 'fail', 'preclu', 'prevent')
    colors_rel = ['#e74c3c' if any(kw in r.lower() for kw in neg_kw) else '#9C27B0' for r in rel_labels]
    bars = axes[0].barh(rel_labels, rel_vals, color=colors_rel, alpha=0.85)
    for bar, val in zip(bars, rel_vals):
        axes[0].text(val + 5, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Count')
    axes[0].set_title('LLM: Top Relation Phrases')
    axes[0].grid(True, axis='x', alpha=0.3)

    # Density
    axes[1].hist(densities, bins=min(20, max(densities)), color='#9C27B0', edgecolor='white', alpha=0.85)
    axes[1].axvline(np.mean(densities), color='red', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(densities):.1f}')
    axes[1].set_xlabel('Triples per narrative')
    axes[1].set_ylabel('Count')
    axes[1].set_title('LLM: Density Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Coverage donut
    not_covered = max(0, sample_n - ev_with)
    axes[2].pie([ev_with, not_covered], labels=['Covered', 'Not covered'], colors=['#9C27B0', '#e0e0e0'],
               autopct='%1.1f%%', startangle=90)
    axes[2].set_title(f'LLM: Coverage\n{ev_with}/{sample_n}')

    plt.suptitle('LLM (Mistral-7B) - Causal Extraction Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_llm_analysis.png')


# =========================================================================
# Knowledge Graph
# =========================================================================

def plot_kg_stats(stats_rules: dict, stats_deps: dict, stats_all: dict, plots_dir: Path):
    """Plot knowledge graph statistics."""
    sources = ['Rule-based', 'Dep-parse', 'All Combined']
    stats = [stats_rules, stats_deps, stats_all]
    colors = ['#2196F3', '#4CAF50', '#9C27B0']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Nodes
    nodes = [s.get('num_nodes', 0) for s in stats]
    axes[0, 0].bar(sources, nodes, color=colors, alpha=0.85, edgecolor='white')
    for i, val in enumerate(nodes):
        axes[0, 0].text(i, val + 500, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Nodes')
    axes[0, 0].set_title('Knowledge Graph - Nodes')
    axes[0, 0].grid(True, axis='y', alpha=0.3)

    # Edges
    edges = [s.get('num_edges', 0) for s in stats]
    axes[0, 1].bar(sources, edges, color=colors, alpha=0.85, edgecolor='white')
    for i, val in enumerate(edges):
        axes[0, 1].text(i, val + 200, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    axes[0, 1].set_ylabel('Number of Edges')
    axes[0, 1].set_title('Knowledge Graph - Edges')
    axes[0, 1].grid(True, axis='y', alpha=0.3)

    # Weakly Connected Components
    wccs = [s.get('weakly_connected_components', 0) for s in stats]
    axes[1, 0].bar(sources, wccs, color=colors, alpha=0.85, edgecolor='white')
    for i, val in enumerate(wccs):
        axes[1, 0].text(i, val + 100, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Knowledge Graph - Weakly Connected Components')
    axes[1, 0].grid(True, axis='y', alpha=0.3)

    # Density
    densities = [s.get('density', 0) for s in stats]
    axes[1, 1].bar(sources, densities, color=colors, alpha=0.85, edgecolor='white')
    for i, val in enumerate(densities):
        axes[1, 1].text(i, val + val*0.1, f'{val:.2e}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Knowledge Graph - Density')
    axes[1, 1].grid(True, axis='y', alpha=0.3)

    plt.suptitle('Knowledge Graph Statistics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_kg_stats.png')


# =========================================================================
# Cross-model comparison
# =========================================================================

def plot_cross_model_comparison(
    rule_triples: List[dict],
    dep_triples: List[dict],
    llm_triples: List[dict],
    sample_n: int,
    bert_triples: Optional[List[dict]] = None,
    t5_triples: Optional[List[dict]] = None,
    plots_dir: Path = Path('outputs/plots'),
):
    """Coverage, density, and total-yield bars across all extraction methods."""
    bert_triples = bert_triples or []
    t5_triples   = t5_triples   or []

    ev_rule = len({t['ev_id'] for t in rule_triples})
    ev_dep  = len({t['ev_id'] for t in dep_triples})
    ev_bert = len({t['ev_id'] for t in bert_triples}) if bert_triples else 0
    ev_t5   = len({t['ev_id'] for t in t5_triples})   if t5_triples  else 0
    ev_llm  = len({t['ev_id'] for t in llm_triples})  if llm_triples else 0

    rule_avg = len(rule_triples) / max(1, ev_rule)
    dep_avg  = len(dep_triples)  / max(1, ev_dep)
    bert_avg = len(bert_triples) / max(1, ev_bert) if bert_triples else 0
    t5_avg   = len(t5_triples)   / max(1, ev_t5)   if t5_triples  else 0
    llm_avg  = len(llm_triples)  / max(1, ev_llm)  if llm_triples else 0

    all_methods = [
        (f'Rule-based\n(n={sample_n})',     ev_rule, rule_avg, len(rule_triples),             '#2196F3'),
        (f'spaCy dep\n(n={sample_n})',      ev_dep,  dep_avg,  len(dep_triples),              '#4CAF50'),
        (f'BERT Extractor\n(test set)',     ev_bert, bert_avg, len(bert_triples),             '#FF9800'),
        (f'T5 Seq2Seq\n(test set)',         ev_t5,   t5_avg,   len(t5_triples),               '#FFC107'),
        (f'LLM Mistral-7B\n(n={sample_n})', ev_llm, llm_avg,  len(llm_triples) if llm_triples else 0, '#9C27B0'),
    ]
    rows = [(m, ev, avg, tot, c) for m, ev, avg, tot, c in all_methods if ev > 0 or tot > 0]
    if not rows:
        return

    methods       = [r[0] for r in rows]
    coverage_pct  = [r[1] / sample_n * 100 for r in rows]
    avg_density   = [r[2] for r in rows]
    total_triples = [r[3] for r in rows]
    colors        = [r[4] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, vals, ylabel, title, fmt in [
        (axes[0], coverage_pct,  'Narratives with ≥1 triple (%)', 'Extraction Coverage', '{:.1f}%'),
        (axes[1], avg_density,   'Avg triples / narrative (≥1)',  'Extraction Density',  '{:.2f}'),
        (axes[2], total_triples, 'Total triples extracted',        'Total Triple Yield',  '{:,}'),
    ]:
        bars = ax.bar(methods, vals, color=colors, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02 + 0.5,
                    fmt.format(val), ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=20)
        if title == 'Extraction Coverage':
            ax.set_ylim(0, 120)

    plt.suptitle(f'Cross-Model Comparison — Causal Triple Extraction (sample n={sample_n})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_cross_model_comparison.png')


# =========================================================================
# Radar (spider) chart — extraction models
# =========================================================================

def plot_radar_extraction(
    rule_triples: List[dict],
    dep_triples: List[dict],
    llm_triples: List[dict],
    rule_kg_stats: dict,
    dep_kg_stats: dict,
    llm_kg_stats: dict,
    sample_n: int,
    plots_dir: Path,
):
    """Spider chart comparing Rule-based, Dep-parse and LLM on five extraction metrics."""

    def _metrics(triples, kg_stats):
        ev_with   = len({t['ev_id'] for t in triples}) if triples else 0
        coverage  = ev_with / max(1, sample_n) * 100
        density   = len(triples) / max(1, ev_with)
        n_rels    = len({t['relation'].lower().strip() for t in triples})
        diversity = n_rels / max(1, len(triples)) * 100
        nodes     = kg_stats.get('num_nodes', 0)
        edges     = kg_stats.get('num_edges', 0)
        return coverage, density, diversity, nodes, edges

    r_cov, r_den, r_div, r_nod, r_edg = _metrics(rule_triples, rule_kg_stats)
    d_cov, d_den, d_div, d_nod, d_edg = _metrics(dep_triples,  dep_kg_stats)
    l_cov, l_den, l_div, l_nod, l_edg = _metrics(llm_triples,  llm_kg_stats)

    raw = np.array([
        [r_cov, r_den, r_div, r_nod, r_edg],
        [d_cov, d_den, d_div, d_nod, d_edg],
        [l_cov, l_den, l_div, l_nod, l_edg],
    ], dtype=float)

    col_max = raw.max(axis=0)
    col_max[col_max == 0] = 1.0
    norm = raw / col_max

    categories = ['Coverage\n(%)', 'Density\n(triples/narr)', 'Relation\nDiversity (%)', 'KG Nodes', 'KG Edges']
    N      = len(categories)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    labels  = ['Rule-based', 'Dep-parse', 'LLM (Mistral-7B)']
    colors  = ['#2196F3', '#4CAF50', '#9C27B0']
    raw_fmt = [
        [f'{r_cov:.1f}%', f'{r_den:.2f}', f'{r_div:.1f}%', f'{r_nod:,}', f'{r_edg:,}'],
        [f'{d_cov:.1f}%', f'{d_den:.2f}', f'{d_div:.1f}%', f'{d_nod:,}', f'{d_edg:,}'],
        [f'{l_cov:.1f}%', f'{l_den:.2f}', f'{l_div:.1f}%', f'{l_nod:,}', f'{l_edg:,}'],
    ]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))

    for i, (row, label, color, fmt) in enumerate(zip(norm, labels, colors, raw_fmt)):
        vals = row.tolist() + row[:1].tolist()
        ax.plot(angles, vals, 'o-', linewidth=2, color=color,
                label=f'{label}  [{", ".join(fmt)}]')
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=7, color='grey')
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title('Extraction Models — Radar Comparison\n(axes normalised to best-in-class)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=8,
              title='[coverage, density, diversity, KG nodes, KG edges]',
              title_fontsize=7)
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_radar_extraction.png')


# =========================================================================
# Per-source knowledge graph visualizations
# =========================================================================

def plot_kg_per_source(
    rule_triples: List[dict],
    dep_triples: List[dict],
    llm_triples: List[dict],
    noise_filter: bool = True,
    normalize: bool = True,
    top_n: int = 30,
    plots_dir: Path = Path('outputs/plots'),
):
    """Three side-by-side network graphs — one per extraction source."""
    import networkx as nx
    from src.knowledge_graph import build_graph

    sources = [
        ('Rule-based', rule_triples, '#2196F3'),
        ('Dep-parse',  dep_triples,  '#4CAF50'),
        ('LLM',        llm_triples,  '#9C27B0'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    for ax, (name, triples, color) in zip(axes, sources):
        if not triples:
            ax.set_title(f'{name} — no triples')
            ax.axis('off')
            continue

        G = build_graph(triples, noise_filter=noise_filter, normalize=normalize)

        if G.number_of_nodes() == 0:
            ax.set_title(f'{name} — graph empty after filtering')
            ax.axis('off')
            continue

        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:top_n]
        sub = G.subgraph(top_nodes)
        pos = nx.spring_layout(sub, seed=42, k=1.8)

        node_sizes  = [200 + 80 * sub.degree(n) for n in sub.nodes()]
        node_colors = [
            color if sub.nodes[n].get('type') == 'cause_node' else '#ecf0f1'
            for n in sub.nodes()
        ]

        nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=node_sizes,
                               node_color=node_colors, alpha=0.88, linewidths=0.5, edgecolors='#555')
        nx.draw_networkx_edges(sub, pos, ax=ax, edge_color='#aaa',
                               arrows=True, arrowsize=12, width=1.0, alpha=0.5)

        top15 = set(sorted(sub.nodes(), key=lambda n: sub.degree(n), reverse=True)[:15])
        labels_map = {n: (n[:22] + '…' if len(n) > 22 else n) for n in top15}
        nx.draw_networkx_labels(sub, pos, labels=labels_map, ax=ax, font_size=6)

        stats_txt = (
            f'Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}\n'
            f'Triples: {len(triples):,}  WCC: {nx.number_weakly_connected_components(G)}'
        )
        ax.set_title(f'{name} Knowledge Graph\n(top {top_n} nodes shown)\n{stats_txt}',
                     fontsize=10, fontweight='bold')
        ax.axis('off')
        ax.legend(handles=[
            mpatches.Patch(facecolor=color,     label='Cause node'),
            mpatches.Patch(facecolor='#ecf0f1', label='Effect node', edgecolor='#555'),
        ], fontsize=7, loc='lower left')

    plt.suptitle('Knowledge Graphs by Extraction Source', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_kg_per_source.png')


def plot_kg_rule_bert_llm(
    rule_triples: List[dict],
    bert_triples: List[dict],
    llm_triples: List[dict],
    noise_filter: bool = True,
    normalize: bool = True,
    top_n: int = 30,
    plots_dir: Path = Path('outputs/plots'),
):
    """Three side-by-side KG panels: Rule-based | BERT Extractor | LLM."""
    import networkx as nx
    from src.knowledge_graph import build_graph

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    def _draw_kg(ax, triples, color, title):
        if not triples:
            ax.set_title(title + '\n(no triples)')
            ax.axis('off')
            return
        G = build_graph(triples, noise_filter=noise_filter, normalize=normalize)
        if G.number_of_nodes() == 0:
            ax.set_title(title + '\n(empty after filtering)')
            ax.axis('off')
            return
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:top_n]
        sub = G.subgraph(top_nodes)
        pos = nx.spring_layout(sub, seed=42, k=1.8)
        node_sizes = [200 + 80 * sub.degree(n) for n in sub.nodes()]
        node_colors = [
            color if sub.nodes[n].get('type') == 'cause_node' else '#ecf0f1'
            for n in sub.nodes()
        ]
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=node_sizes,
                               node_color=node_colors, alpha=0.88, linewidths=0.5, edgecolors='#555')
        nx.draw_networkx_edges(sub, pos, ax=ax, edge_color='#aaa',
                               arrows=True, arrowsize=12, width=1.0, alpha=0.5)
        top15 = set(sorted(sub.nodes(), key=lambda n: sub.degree(n), reverse=True)[:15])
        labels_map = {n: (n[:22] + '...' if len(n) > 22 else n) for n in top15}
        nx.draw_networkx_labels(sub, pos, labels=labels_map, ax=ax, font_size=6)
        ev_with = len({t['ev_id'] for t in triples})
        stats = (
            f'Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}\n'
            f'Triples: {len(triples):,}  Narratives: {ev_with:,}  '
            f'WCC: {nx.number_weakly_connected_components(G)}'
        )
        ax.set_title(f'{title}\n(top {top_n} nodes shown)\n{stats}', fontsize=10, fontweight='bold')
        ax.axis('off')
        ax.legend(handles=[
            mpatches.Patch(facecolor=color,     label='Cause node'),
            mpatches.Patch(facecolor='#ecf0f1', label='Effect node', edgecolor='#555'),
        ], fontsize=7, loc='lower left')

    _draw_kg(axes[0], rule_triples, '#2196F3', 'Rule-based Knowledge Graph')
    _draw_kg(axes[1], bert_triples, '#FF9800', 'BERT Extractor Knowledge Graph')
    _draw_kg(axes[2], llm_triples,  '#9C27B0', 'LLM Knowledge Graph')

    plt.suptitle('Knowledge Graphs: Rule-based  |  BERT Extractor  |  LLM',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_kg_rule_bert_llm.png')


# =========================================================================
# LLM cache growth / utilisation
# =========================================================================

def plot_llm_cache_growth(
    cache: dict,
    llm_triples: List[dict],
    sample_n: int,
    plots_dir: Path,
):
    """Bar chart showing cache size, triples parsed, coverage, and avg density."""
    if not cache:
        return

    total_cached = len(cache)
    full_triples = []
    parse_errors = 0
    for ev_id, raw in cache.items():
        m = _JSON_RE.search(raw)
        if not m:
            parse_errors += 1
            continue
        try:
            items = _json.loads(m.group())
        except Exception:
            parse_errors += 1
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            c = str(item.get('cause', '')).strip()
            e = str(item.get('effect', '')).strip()
            r = str(item.get('relation', 'caused')).strip()
            if c and e and c.lower() not in _JUNK_NODES and e.lower() not in _JUNK_NODES:
                full_triples.append({'ev_id': ev_id, 'cause': c, 'relation': r, 'effect': e})

    ev_with_full  = len({t['ev_id'] for t in full_triples})
    coverage_full = ev_with_full / max(1, total_cached) * 100
    density_full  = len(full_triples) / max(1, ev_with_full)

    ev_with_orig  = len({t['ev_id'] for t in llm_triples})
    coverage_orig = ev_with_orig / max(1, sample_n) * 100
    density_orig  = len(llm_triples) / max(1, ev_with_orig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels  = [f'Training sample\n(n={sample_n})', f'Full cache\n(n={total_cached})']
    covered = [ev_with_orig, ev_with_full]
    colors  = ['#9C27B0', '#4CAF50']
    bars = axes[0].bar(labels, covered, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, covered):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 10,
                     f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[0].set_ylabel('Narratives with ≥1 triple')
    axes[0].set_title('LLM: Narratives Covered')
    axes[0].grid(True, axis='y', alpha=0.3)

    triple_counts = [len(llm_triples), len(full_triples)]
    bars2 = axes[1].bar(labels, triple_counts, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars2, triple_counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 20,
                     f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('Total triples extracted')
    axes[1].set_title('LLM: Total Triple Yield')
    axes[1].grid(True, axis='y', alpha=0.3)

    x = np.arange(2)
    w = 0.35
    ax2  = axes[2]
    ax2b = ax2.twinx()
    b1 = ax2.bar(x - w/2, [coverage_orig, coverage_full], w,
                 color=['#9C27B0', '#4CAF50'], alpha=0.85, label='Coverage %')
    b2 = ax2b.bar(x + w/2, [density_orig, density_full], w,
                  color=['#FF9800', '#e74c3c'], alpha=0.75, label='Avg density')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Training\nsample', 'Full cache'])
    ax2.set_ylabel('Coverage (%)', color='#555')
    ax2b.set_ylabel('Avg triples / narrative', color='#555')
    ax2.set_title('Coverage & Density\n(sample vs full cache)')
    for bar, val in zip(b1, [coverage_orig, coverage_full]):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(b2, [density_orig, density_full]):
        ax2b.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                  f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3)

    fig.text(0.5, -0.04,
             f'Parse errors: {parse_errors}/{total_cached} cached entries '
             f'({parse_errors/max(1,total_cached):.1%})  |  '
             f'Full cache coverage: {coverage_full:.1f}%  |  '
             f'Full cache triples: {len(full_triples):,}',
             ha='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f5e9', alpha=0.8))

    plt.suptitle('LLM Cache Utilisation — Training Sample vs Full Overnight Cache',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_llm_cache_growth.png')

    return full_triples


# =========================================================================
# Entry point — generate all plots from saved artifacts
# =========================================================================

def _load_cfg(path: str = 'CONFIG.conf'):
    import configparser
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    cfg.read(path)
    return cfg


def _load_json_file(path: Path):
    if not path.exists():
        print(f'  [warn] {path} not found — skipping.')
        return {} if path.suffix == '.json' else []
    with open(path, encoding='utf-8') as f:
        return _json.load(f)


def _save_json_file(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        _json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f'  Saved -> {path}')


def _section(title: str):
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


def main():
    import argparse
    import warnings
    warnings.filterwarnings('ignore')

    from src.knowledge_graph import build_graph, graph_stats, visualize_subgraph
    from src.finding_evaluator import load_findings, evaluate_finding_alignment, print_finding_report

    parser = argparse.ArgumentParser(description='NTSB — Plot generation (no retraining)')
    parser.add_argument('--config', default='CONFIG.conf')
    args = parser.parse_args()

    cfg = _load_cfg(args.config)

    output_dir      = Path(cfg.get('paths', 'output_dir', fallback='outputs'))
    training_dir    = output_dir / 'training'
    extractions_dir = output_dir / 'extractions'
    eval_dir        = output_dir / 'evaluation'
    plots_dir       = output_dir / 'plots'

    kg_cfg             = cfg['knowledge_graph'] if 'knowledge_graph' in cfg else {}
    noise_filter       = kg_cfg.get('noise_filter',       'true').lower() == 'true'
    normalize_entities = kg_cfg.get('normalize_entities', 'true').lower() == 'true'
    top_n              = int(kg_cfg.get('visualize_top_n', 40))

    for d in (plots_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_cfg  = _load_json_file(training_dir / 'run_config.json')
    sample_n = run_cfg.get('sample_n') if isinstance(run_cfg, dict) else None
    if sample_n is None:
        sample_n = int(cfg.get('global', 'sample_n', fallback=2000))
    if sample_n == 0:
        sample_n = 6059

    print('=' * 70)
    print('  NTSB Causal Chain Extraction — Plot Generation (no retraining)')
    print('=' * 70)
    print(f'\n  Sample n : {sample_n}')
    print(f'  Output   : {output_dir}')

    # ------------------------------------------------------------------
    # Training plots (BERT / T5 tuning results)
    # ------------------------------------------------------------------
    _section('Training Plots')
    tuning_results_path = output_dir / 'tuning_results.json'
    tuning_results = _load_json_file(tuning_results_path) if tuning_results_path.exists() else {}
    if tuning_results:
        plot_training_loss_curves(tuning_results, plots_dir)
        plot_training_metrics(tuning_results, plots_dir)
        plot_bias_variance_tradeoff(tuning_results, plots_dir)
    else:
        print('  No tuning_results.json — skipping training plots.')

    eval_report = _load_json_file(eval_dir / 'evaluation_report.json')
    det_raw = eval_report.get('detection_metrics', {}) if isinstance(eval_report, dict) else {}
    if det_raw:
        detection_results = [{'label': k, **v} for k, v in det_raw.items()]
        plot_detection_metrics(detection_results, plots_dir)

    # ------------------------------------------------------------------
    # Load extraction artifacts
    # ------------------------------------------------------------------
    _section('Loading Extraction Artifacts')

    def _find(primary: Path, fallback: Path):
        return primary if primary.exists() else fallback

    rule_triples    = _load_json_file(_find(training_dir    / 'rule_triples.json',
                                             output_dir / 'causal_triples_rules.json'))
    dep_triples     = _load_json_file(_find(training_dir    / 'dep_triples.json',
                                             output_dir / 'causal_triples_deps.json'))
    llm_triples     = _load_json_file(_find(extractions_dir / 'llm_triples.json',
                                             output_dir / 'llm_triples.json'))
    fewshot_triples = _load_json_file(_find(extractions_dir / 'llm_triples_fewshot.json',
                                             output_dir / 'llm_triples_fewshot.json'))
    bert_triples    = _load_json_file(_find(extractions_dir / 'bert_triples.json',
                                             output_dir / 'bert_triples.json'))
    t5_triples      = _load_json_file(_find(extractions_dir / 't5_triples.json',
                                             output_dir / 't5_triples.json'))

    cache_path = Path(cfg.get('llm_extractor', 'cache_path',
                              fallback='outputs/extractions/llm_response_cache.json'))
    if not cache_path.exists():
        cache_path = output_dir / 'llm_response_cache.json'
    llm_cache = _load_json_file(cache_path) if cache_path.exists() else {}

    print(f'  Rule triples:      {len(rule_triples):,}')
    print(f'  Dep triples:       {len(dep_triples):,}')
    print(f'  BERT triples:      {len(bert_triples):,}')
    print(f'  T5 triples:        {len(t5_triples):,}')
    print(f'  LLM triples:       {len(llm_triples):,}')
    print(f'  LLM few-shot:      {len(fewshot_triples):,}')
    print(f'  LLM cache entries: {len(llm_cache):,}')

    # ------------------------------------------------------------------
    # Parse full LLM cache
    # ------------------------------------------------------------------
    _section('Parsing Full LLM Cache')
    full_llm_triples = llm_triples

    if llm_cache:
        result = plot_llm_cache_growth(llm_cache, llm_triples, sample_n, plots_dir)
        if result:
            full_llm_triples = result
            print(f'  Full cache parsed: {len(full_llm_triples):,} triples from '
                  f'{len({t["ev_id"] for t in full_llm_triples}):,} narratives')
            out_path = (extractions_dir if extractions_dir.exists() else output_dir) / 'llm_triples_full.json'
            _save_json_file(full_llm_triples, out_path)

    # ------------------------------------------------------------------
    # Traditional NLP plots
    # ------------------------------------------------------------------
    _section('Traditional NLP Plots')
    if rule_triples:
        plot_traditional_nlp(rule_triples, dep_triples, sample_n, plots_dir)

    # ------------------------------------------------------------------
    # LLM extraction plots
    # ------------------------------------------------------------------
    _section('LLM Extraction Plots')
    if full_llm_triples:
        plot_llm_analysis(full_llm_triples, len(llm_cache) if llm_cache else sample_n, plots_dir)

    # ------------------------------------------------------------------
    # Knowledge graph plots
    # ------------------------------------------------------------------
    _section('Knowledge Graph Plots')
    stats_rules = stats_deps = stats_llm = {}

    if rule_triples or dep_triples or full_llm_triples:
        G_rules = build_graph(rule_triples,     noise_filter=noise_filter, normalize=normalize_entities)
        G_deps  = build_graph(dep_triples,      noise_filter=noise_filter, normalize=normalize_entities)
        G_llm   = build_graph(full_llm_triples, noise_filter=noise_filter, normalize=normalize_entities)
        G_bert  = build_graph(bert_triples,     noise_filter=noise_filter, normalize=normalize_entities) \
                  if bert_triples else None
        G_all   = build_graph(rule_triples + dep_triples + full_llm_triples,
                              noise_filter=noise_filter, normalize=normalize_entities)

        stats_rules = graph_stats(G_rules)
        stats_deps  = graph_stats(G_deps)
        stats_llm   = graph_stats(G_llm)
        stats_all   = graph_stats(G_all)
        stats_bert  = graph_stats(G_bert) if G_bert else {}

        print(f'  KG sizes — Rule: {G_rules.number_of_nodes()}n/{G_rules.number_of_edges()}e  '
              f'| Dep: {G_deps.number_of_nodes()}n/{G_deps.number_of_edges()}e  '
              f'| LLM: {G_llm.number_of_nodes()}n/{G_llm.number_of_edges()}e  '
              f'| Combined: {G_all.number_of_nodes()}n/{G_all.number_of_edges()}e')

        plot_kg_stats(stats_rules, stats_deps, stats_all, plots_dir)
        plot_kg_per_source(rule_triples, dep_triples, full_llm_triples,
                           noise_filter=noise_filter, normalize=normalize_entities,
                           top_n=top_n, plots_dir=plots_dir)
        plot_kg_rule_bert_llm(rule_triples, bert_triples, full_llm_triples,
                              noise_filter=noise_filter, normalize=normalize_entities,
                              top_n=top_n, plots_dir=plots_dir)
        visualize_subgraph(G_all, top_n=top_n, save_path=str(plots_dir / 'eval_knowledge_graph_full.png'))

        _save_json_file({
            'rule':     {k: v for k, v in stats_rules.items()
                         if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
            'dep':      {k: v for k, v in stats_deps.items()
                         if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
            'bert':     {k: v for k, v in stats_bert.items()
                         if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
            'llm':      {k: v for k, v in stats_llm.items()
                         if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
            'combined': {k: v for k, v in stats_all.items()
                         if k not in ('top_causes', 'top_effects', 'top_nodes_by_betweenness')},
        }, (extractions_dir if extractions_dir.exists() else output_dir) / 'graph_stats_updated.json')

    # ------------------------------------------------------------------
    # Radar charts
    # ------------------------------------------------------------------
    _section('Radar Charts')
    if rule_triples or dep_triples or full_llm_triples:
        plot_radar_extraction(rule_triples, dep_triples, full_llm_triples,
                              stats_rules, stats_deps, stats_llm,
                              sample_n=sample_n, plots_dir=plots_dir)

    # ------------------------------------------------------------------
    # Cross-model comparison
    # ------------------------------------------------------------------
    _section('Cross-Model Comparison')
    plot_cross_model_comparison(rule_triples, dep_triples, full_llm_triples,
                                sample_n, bert_triples, t5_triples, plots_dir)

    # ------------------------------------------------------------------
    # Finding-alignment evaluation (ground truth)
    # ------------------------------------------------------------------
    _section('Finding-Alignment Evaluation (Ground Truth)')

    data_path = cfg.get('paths', 'data_path',
                        fallback='data/clean/cleaned_narritives_and_findings.csv')

    if Path(data_path).exists():
        findings_df = load_findings(data_path)
        print(f'  Findings loaded: {len(findings_df):,} rows, '
              f'{findings_df["ev_id"].nunique():,} unique accidents')

        test_split  = _load_json_file(training_dir / 'test_split.json')
        test_ev_ids = (
            set(test_split.get('test_ev_ids', []))
            if isinstance(test_split, dict) else set()
        )
        unified_mode = bool(test_ev_ids)

        def _filter_triples(triples, ev_ids_set):
            if not ev_ids_set:
                return triples
            return [t for t in triples if str(t.get('ev_id', '')) in ev_ids_set]

        alignment_results = []
        _section('Finding-Alignment — Full Dataset')
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
            print(f'    Category alignment:       {res["category_alignment_score"]:.1%}')
            print(f'    Cause-confirmed coverage: {res["cause_confirmed_coverage"]:.1%}')
            print(f'    Finding keyword recall:   {res["finding_keyword_recall"]:.1%}')

        if alignment_results:
            print_finding_report(alignment_results)
            plot_finding_alignment(alignment_results, plots_dir)

        unified_results = []
        if unified_mode:
            _section('Finding-Alignment — Unified Test Set')
            print(f'  Unified test-set mode: {len(test_ev_ids)} held-out narratives')
            for label, triples in [
                ('Rule-based',      rule_triples),
                ('Dep-parse',       dep_triples),
                ('BERT Extractor',  bert_triples),
                ('LLM (zero-shot)', full_llm_triples),
                ('LLM (few-shot)',  fewshot_triples),
            ]:
                if not triples:
                    continue
                filtered = _filter_triples(triples, test_ev_ids)
                res = evaluate_finding_alignment(filtered, findings_df, label=label)
                unified_results.append(res)
                print(f'\n  [{label}]  (test set: {len(filtered)} triples)')
                print(f'    Category alignment:       {res["category_alignment_score"]:.1%}')
                print(f'    Cause-confirmed coverage: {res["cause_confirmed_coverage"]:.1%}')
                print(f'    Finding keyword recall:   {res["finding_keyword_recall"]:.1%}')

            if unified_results:
                print_finding_report(unified_results)
                plot_finding_alignment(unified_results, plots_dir, suffix='_unified')

        existing = _load_json_file(eval_dir / 'evaluation_report.json')
        if isinstance(existing, dict):
            existing['finding_alignment'] = {
                r['label']: {k: v for k, v in r.items() if k != 'label'}
                for r in alignment_results
            }
            if unified_mode and unified_results:
                existing['finding_alignment_unified'] = {
                    r['label']: {k: v for k, v in r.items() if k != 'label'}
                    for r in unified_results
                }
            _save_json_file(existing, eval_dir / 'evaluation_report.json')
    else:
        print('  data_path not found — skipping finding-alignment evaluation.')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _section('All Plots Generated')
    plot_files = sorted(plots_dir.glob('*.png'))
    for p in plot_files:
        size_kb = p.stat().st_size / 1024
        print(f'  {p.name:<50}  {size_kb:>6.0f} KB')
    print(f'\n  Total: {len(plot_files)} plots in {plots_dir}')


if __name__ == '__main__':
    main()
