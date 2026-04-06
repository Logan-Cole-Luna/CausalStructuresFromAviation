"""
Plotting utilities for NTSB Causal Chain Extraction.

All functions accept explicit data arguments and a plots_dir Path.
No torch / transformers imports — purely matplotlib + seaborn + numpy.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, plots_dir: Path, filename: str):
    path = plots_dir / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ---------------------------------------------------------------------------
# Model 1 — Traditional NLP
# ---------------------------------------------------------------------------

def plot_traditional_nlp(
    rule_triples: list,
    dep_triples: list,
    sample_n: int,
    plots_dir: Path,
):
    """2×2 grid: pattern counts, density histogram, coverage bars, direction pie."""
    pattern_counts  = Counter(t['relation'] for t in rule_triples)
    direction_counts = Counter(t['direction'] for t in rule_triples)
    per_ev_rule     = Counter(t['ev_id'] for t in rule_triples)
    densities       = list(per_ev_rule.values())
    ev_with_rule    = len(per_ev_rule)
    ev_with_dep     = len({t['ev_id'] for t in dep_triples})

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # [0,0] Rule-based pattern hits
    sorted_pats = sorted(pattern_counts.items(), key=lambda x: -x[1])
    pats_sorted = [p for p, _ in sorted_pats]
    vals_sorted = [v for _, v in sorted_pats]
    bars = axes[0, 0].barh(pats_sorted, vals_sorted, color='#2196F3', alpha=0.85)
    for bar, val in zip(bars, vals_sorted):
        axes[0, 0].text(val + 5, bar.get_y() + bar.get_height() / 2,
                        str(val), va='center', fontsize=8)
    axes[0, 0].set_xlabel('Triple count')
    axes[0, 0].set_title('Rule-based: Pattern Hit Counts')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(True, axis='x', alpha=0.3)

    # [0,1] Triple density histogram with percentile lines
    axes[0, 1].hist(densities, bins=25, color='#4CAF50', edgecolor='white', alpha=0.85)
    mean_d = np.mean(densities)
    med_d  = np.median(densities)
    p90_d  = np.percentile(densities, 90)
    axes[0, 1].axvline(mean_d, color='red',    linestyle='--', linewidth=1.5, label=f'Mean={mean_d:.1f}')
    axes[0, 1].axvline(med_d,  color='orange', linestyle=':',  linewidth=1.5, label=f'Median={med_d:.1f}')
    axes[0, 1].axvline(p90_d,  color='purple', linestyle='-.', linewidth=1.5, label=f'P90={p90_d:.1f}')
    axes[0, 1].set_xlabel('Triples per narrative')
    axes[0, 1].set_ylabel('Narrative count')
    axes[0, 1].set_title('Rule-based: Triple Density Distribution')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # [1,0] Coverage comparison stacked bar (both on same sample_n denominator)
    method_labels = [f'Rule-based\n(n={sample_n})', f'spaCy dep-parse\n(n={sample_n})']
    covered       = [ev_with_rule, ev_with_dep]
    totals        = [sample_n, sample_n]
    not_covered   = [t - c for t, c in zip(totals, covered)]
    x_cov = np.arange(len(method_labels))
    axes[1, 0].bar(x_cov, [c / t * 100 for c, t in zip(covered, totals)],
                   color='#2196F3', alpha=0.85, label='With ≥1 triple')
    axes[1, 0].bar(x_cov, [nc / t * 100 for nc, t in zip(not_covered, totals)],
                   bottom=[c / t * 100 for c, t in zip(covered, totals)],
                   color='#e0e0e0', alpha=0.85, label='No triple extracted')
    for i, (c, t) in enumerate(zip(covered, totals)):
        axes[1, 0].text(i, c / t * 100 / 2, f'{c/t:.1%}',
                        ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    axes[1, 0].set_xticks(x_cov)
    axes[1, 0].set_xticklabels(method_labels)
    axes[1, 0].set_ylabel('% of narratives')
    axes[1, 0].set_ylim(0, 115)
    axes[1, 0].set_title('Extraction Coverage by Method')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, axis='y', alpha=0.3)

    # [1,1] Direction breakdown pie
    direction_labels = list(direction_counts.keys())
    direction_vals   = [direction_counts[k] for k in direction_labels]
    axes[1, 1].pie(direction_vals, labels=direction_labels,
                   autopct='%1.1f%%', colors=['#FF9800', '#3498db'],
                   startangle=90, textprops={'fontsize': 10})
    axes[1, 1].set_title('Rule-based: Causal Direction\n(forward = cause→effect, backward = effect←cause)')

    plt.suptitle('Traditional NLP — Causal Extraction Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_traditional_nlp.png')


# ---------------------------------------------------------------------------
# Model 2 — DistilBERT
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    all_preds: list,
    all_labels: list,
    class_names: list,
    plots_dir: Path,
):
    """Confusion-matrix heatmap."""
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('DistilBERT — Confusion Matrix (Test Set)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_transformer_confusion.png')


def plot_training_curves(history: dict, plots_dir: Path):
    """Loss + val-accuracy curves with early-stop marker."""
    train_loss = history.get('train_loss', [])
    val_acc    = history.get('val_acc',    [])
    best_epoch = history.get('best_epoch', None)

    if not train_loss:
        return

    epochs = list(range(1, len(train_loss) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2, markersize=5, label='Train loss')
    if best_epoch:
        ax1.axvline(best_epoch, color='#2196F3', linestyle='--', linewidth=1.5,
                    label=f'Best epoch ({best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-entropy loss')
    ax1.set_title('DistilBERT — Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [v * 100 for v in val_acc], 's-', color='#4CAF50',
             linewidth=2, markersize=5, label='Val accuracy')
    if best_epoch and best_epoch <= len(val_acc):
        ax2.axvline(best_epoch, color='#2196F3', linestyle='--', linewidth=1.5,
                    label=f'Best epoch ({best_epoch})')
        ax2.scatter([best_epoch], [val_acc[best_epoch - 1] * 100],
                    color='#2196F3', zorder=5, s=80)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('DistilBERT — Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('DistilBERT Training Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_transformer_training_curves.png')


def plot_per_class_metrics(report: dict, label_map: dict, plots_dir: Path):
    """Grouped precision/recall/F1 bars + test-set class distribution."""
    if not report:
        return
    classes = [c for c in sorted(label_map.keys()) if c in report]
    if not classes:
        return

    precision = [report[c]['precision'] for c in classes]
    recall    = [report[c]['recall']    for c in classes]
    f1        = [report[c]['f1-score']  for c in classes]
    support   = [int(report[c]['support']) for c in classes]
    short     = [c.replace(' issues', '') for c in classes]

    x     = np.arange(len(classes))
    width = 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars_p = ax1.bar(x - width, precision, width, label='Precision', color='#2196F3', alpha=0.85)
    bars_r = ax1.bar(x,         recall,    width, label='Recall',    color='#4CAF50', alpha=0.85)
    bars_f = ax1.bar(x + width, f1,        width, label='F1',        color='#FF9800', alpha=0.85)
    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}',
                     ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, rotation=15, ha='right')
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel('Score')
    ax1.set_title('DistilBERT — Per-Class Precision / Recall / F1')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    colors = ['#e74c3c' if s < 50 else '#3498db' for s in support]
    ax2.bar(short, support, color=colors, alpha=0.85, edgecolor='white')
    for i, s in enumerate(support):
        ax2.text(i, s + 2, str(s), ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Test samples')
    ax2.set_title('DistilBERT — Test-Set Class Distribution')
    ax2.set_xticklabels(short, rotation=15, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.legend(handles=[
        mpatches.Patch(color='#e74c3c', label='< 50 samples (rare)'),
        mpatches.Patch(color='#3498db', label='≥ 50 samples'),
    ], fontsize=8)

    plt.suptitle('DistilBERT Per-Class Performance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_transformer_per_class.png')


# ---------------------------------------------------------------------------
# Model 4 — LLM extractor
# ---------------------------------------------------------------------------

def plot_llm_analysis(llm_triples: list, sample_n: int, plots_dir: Path):
    """Relation phrases, density histogram, and coverage donut."""
    if not llm_triples:
        return

    pattern_counts = Counter(t['relation'] for t in llm_triples)
    per_ev         = Counter(t['ev_id'] for t in llm_triples)
    densities      = list(per_ev.values())
    ev_with        = len(per_ev)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # [0] Top-20 relation phrases (negative relations highlighted)
    top_rels   = sorted(pattern_counts.items(), key=lambda x: -x[1])[:20]
    rel_labels = [r for r, _ in top_rels]
    rel_vals   = [v for _, v in top_rels]
    _neg_kw    = ('not', 'fail', 'preclu', 'prevent')
    colors_rel = ['#e74c3c' if any(kw in r.lower() for kw in _neg_kw) else '#2196F3'
                  for r in rel_labels]
    bars = axes[0].barh(rel_labels, rel_vals, color=colors_rel, alpha=0.85)
    for bar, val in zip(bars, rel_vals):
        axes[0].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     str(val), va='center', fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Count')
    axes[0].set_title('LLM: Top Relation Phrases\n(red = negative/barrier relations)')
    axes[0].grid(True, axis='x', alpha=0.3)
    axes[0].legend(handles=[
        mpatches.Patch(color='#e74c3c', label='Negative/barrier'),
        mpatches.Patch(color='#2196F3', label='Positive causal'),
    ], fontsize=8)

    # [1] Triples-per-narrative distribution
    axes[1].hist(densities, bins=min(20, max(densities)), color='#9C27B0',
                 edgecolor='white', alpha=0.85)
    mean_d = np.mean(densities)
    axes[1].axvline(mean_d, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mean_d:.1f}')
    axes[1].axvline(np.median(densities), color='orange', linestyle=':', linewidth=1.5,
                    label=f'Median={np.median(densities):.1f}')
    axes[1].set_xlabel('Triples per narrative')
    axes[1].set_ylabel('Narrative count')
    axes[1].set_title('LLM: Triple Density Distribution')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # [2] Coverage donut
    not_covered = max(0, sample_n - ev_with)
    wedge_vals  = [ev_with, not_covered]
    wedge_cols  = ['#4CAF50', '#e0e0e0']
    wedge_labs  = [f'Extracted\n{ev_with} ({ev_with/sample_n:.1%})',
                   f'No triple\n{not_covered} ({not_covered/sample_n:.1%})']
    wedges, _   = axes[2].pie(wedge_vals, colors=wedge_cols, startangle=90,
                               wedgeprops=dict(width=0.5))
    axes[2].legend(wedges, wedge_labs, loc='lower center', bbox_to_anchor=(0.5, -0.15),
                   fontsize=9, ncol=2)
    axes[2].set_title(f'LLM: Narrative Coverage\n(n={sample_n})')
    axes[2].text(0, 0, f'{ev_with/sample_n:.0%}', ha='center', va='center',
                 fontsize=18, fontweight='bold', color='#4CAF50')

    plt.suptitle('LLM (Mistral-7B) — Causal Extraction Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_llm_extraction.png')


# ---------------------------------------------------------------------------
# Model 3 — Knowledge graph
# ---------------------------------------------------------------------------

def plot_kg_stats(stats_rules: dict, stats_deps: dict, stats_all: dict, plots_dir: Path):
    """2×2: nodes/edges, WCC/density, top-10 causes, top-10 effects."""
    graph_names = ['Rule-based', 'Dep-parse', 'Combined']
    stats_list  = [stats_rules, stats_deps, stats_all]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # [0,0] Nodes & edges grouped bar
    nodes = [s['num_nodes'] for s in stats_list]
    edges = [s['num_edges'] for s in stats_list]
    x, w  = np.arange(len(graph_names)), 0.35
    b1 = axes[0, 0].bar(x - w/2, nodes, w, label='Nodes', color='#2196F3', alpha=0.85)
    b2 = axes[0, 0].bar(x + w/2, edges, w, label='Edges', color='#FF9800', alpha=0.85)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, h + 20, f'{h:,}',
                        ha='center', va='bottom', fontsize=8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(graph_names)
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Graph Size: Nodes & Edges')
    axes[0, 0].legend()
    axes[0, 0].grid(True, axis='y', alpha=0.3)

    # [0,1] WCC count & density (dual axis)
    wccs    = [s['weakly_connected_components'] for s in stats_list]
    density = [s['density'] * 1e4 for s in stats_list]
    ax_wcc  = axes[0, 1]
    ax_den  = ax_wcc.twinx()
    ax_wcc.bar(x - w/2, wccs,    w, label='WCC count',      color='#9C27B0', alpha=0.75)
    ax_den.bar(x + w/2, density, w, label='Density ×10⁻⁴', color='#e74c3c', alpha=0.75)
    ax_wcc.set_xticks(x)
    ax_wcc.set_xticklabels(graph_names)
    ax_wcc.set_ylabel('Weakly connected components', color='#9C27B0')
    ax_den.set_ylabel('Density ×10⁻⁴', color='#e74c3c')
    axes[0, 1].set_title('Connectivity: WCC Count & Density')
    lines1, labels1 = ax_wcc.get_legend_handles_labels()
    lines2, labels2 = ax_den.get_legend_handles_labels()
    ax_wcc.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax_wcc.grid(True, axis='y', alpha=0.3)

    # [1,0] Top-10 causes in combined graph
    top_causes   = stats_all['top_causes'][:10]
    cause_labels = [n[:40] + ('…' if len(n) > 40 else '') for n, _ in top_causes]
    cause_vals   = [d for _, d in top_causes]
    bars = axes[1, 0].barh(cause_labels[::-1], cause_vals[::-1], color='#e74c3c', alpha=0.85)
    for bar, val in zip(bars, cause_vals[::-1]):
        axes[1, 0].text(val + 0.3, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', fontsize=8)
    axes[1, 0].set_xlabel('Out-degree (cause frequency)')
    axes[1, 0].set_title('Combined Graph: Top-10 Cause Nodes')
    axes[1, 0].grid(True, axis='x', alpha=0.3)

    # [1,1] Top-10 effects in combined graph
    top_effects = stats_all['top_effects'][:10]
    eff_labels  = [n[:40] + ('…' if len(n) > 40 else '') for n, _ in top_effects]
    eff_vals    = [d for _, d in top_effects]
    bars = axes[1, 1].barh(eff_labels[::-1], eff_vals[::-1], color='#3498db', alpha=0.85)
    for bar, val in zip(bars, eff_vals[::-1]):
        axes[1, 1].text(val + 0.3, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', fontsize=8)
    axes[1, 1].set_xlabel('In-degree (effect frequency)')
    axes[1, 1].set_title('Combined Graph: Top-10 Effect Nodes')
    axes[1, 1].grid(True, axis='x', alpha=0.3)

    plt.suptitle('Knowledge Graph — Structural Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_kg_stats.png')


# ---------------------------------------------------------------------------
# Cross-model summary
# ---------------------------------------------------------------------------

def plot_cross_model_comparison(
    rule_triples: list,
    dep_triples: list,
    llm_triples: list,
    sample_n: int,
    transformer_results: dict,
    plots_dir: Path,
):
    """Coverage, density, and total-yield bars across all extraction methods."""
    ev_with_rule = len({t['ev_id'] for t in rule_triples})
    ev_with_dep  = len({t['ev_id'] for t in dep_triples})
    ev_with_llm  = len({t['ev_id'] for t in llm_triples}) if llm_triples else 0

    rule_avg = len(rule_triples) / max(1, ev_with_rule)
    dep_avg  = len(dep_triples)  / max(1, ev_with_dep)
    llm_avg  = len(llm_triples)  / max(1, ev_with_llm) if llm_triples else 0

    methods       = [f'Rule-based\n(n={sample_n})', f'spaCy dep\n(n={sample_n})', f'LLM Mistral-7B\n(n={sample_n})']
    coverage_pct  = [ev_with_rule / sample_n * 100, ev_with_dep / sample_n * 100, ev_with_llm / sample_n * 100]
    avg_density   = [rule_avg, dep_avg, llm_avg]
    total_triples = [len(rule_triples), len(dep_triples), len(llm_triples) if llm_triples else 0]
    colors        = ['#2196F3', '#4CAF50', '#9C27B0']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for ax, vals, ylabel, title, fmt in [
        (axes[0], coverage_pct,  'Narratives with ≥1 triple (%)', 'Extraction Coverage',    '{:.1f}%'),
        (axes[1], avg_density,   'Avg triples / narrative (≥1)', 'Extraction Density',      '{:.2f}'),
        (axes[2], total_triples, 'Total triples extracted',       'Total Triple Yield',      '{:,}'),
    ]:
        bars = ax.bar(methods, vals, color=colors, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.02 + 0.5,
                    fmt.format(val), ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        if title == 'Extraction Coverage':
            ax.set_ylim(0, 120)

    acc = transformer_results.get('accuracy', None)
    if acc is not None:
        best_val = transformer_results.get('train_history', {}).get('best_val_acc', 0)
        fig.text(0.5, -0.04,
                 f'DistilBERT classifier (separate task): test acc = {acc*100:.1f}%  |  '
                 f'best val acc = {best_val*100:.1f}%',
                 ha='center', fontsize=10, style='italic',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff9c4', alpha=0.8))

    plt.suptitle(f'Cross-Model Comparison — Causal Extraction (sample n={sample_n})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_cross_model_comparison.png')
