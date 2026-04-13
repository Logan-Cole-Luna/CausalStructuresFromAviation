"""
Plotting utilities for NTSB Causal Chain Extraction.

All functions accept explicit data arguments and a plots_dir Path.
No torch / transformers imports — purely matplotlib + seaborn + numpy.
"""
from __future__ import annotations

import math
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
    print(f"  Plot saved -> {path}")


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
    bert_triples: list,
    t5_triples: list,
    plots_dir: Path,
):
    """Coverage, density, and total-yield bars across all extraction methods."""
    ev_with_rule  = len({t['ev_id'] for t in rule_triples})
    ev_with_dep   = len({t['ev_id'] for t in dep_triples})
    ev_with_bert  = len({t['ev_id'] for t in bert_triples}) if bert_triples else 0
    ev_with_t5    = len({t['ev_id'] for t in t5_triples}) if t5_triples else 0
    ev_with_llm   = len({t['ev_id'] for t in llm_triples})  if llm_triples  else 0

    rule_avg  = len(rule_triples)  / max(1, ev_with_rule)
    dep_avg   = len(dep_triples)   / max(1, ev_with_dep)
    bert_avg  = len(bert_triples)  / max(1, ev_with_bert)  if bert_triples else 0
    t5_avg    = len(t5_triples)    / max(1, ev_with_t5)    if t5_triples else 0
    llm_avg   = len(llm_triples)   / max(1, ev_with_llm)   if llm_triples  else 0

    methods = [
        f'Rule-based\n(n={sample_n})',
        f'spaCy dep\n(n={sample_n})',
        f'BERT Extractor\n(test set)',
        f'T5 Seq2Seq\n(test set)',
        f'LLM Mistral-7B\n(n={sample_n})',
    ]
    coverage_pct  = [
        ev_with_rule  / sample_n * 100,
        ev_with_dep   / sample_n * 100,
        ev_with_bert  / max(1, sample_n) * 100,
        ev_with_t5    / max(1, sample_n) * 100,
        ev_with_llm   / sample_n * 100,
    ]
    avg_density   = [rule_avg, dep_avg, bert_avg, t5_avg, llm_avg]
    total_triples = [
        len(rule_triples),
        len(dep_triples),
        len(bert_triples) if bert_triples else 0,
        len(t5_triples) if t5_triples else 0,
        len(llm_triples)  if llm_triples  else 0,
    ]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#FFC107', '#9C27B0']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, vals, ylabel, title, fmt in [
        (axes[0], coverage_pct,  'Narratives with ≥1 triple (%)', 'Extraction Coverage',    '{:.1f}%'),
        (axes[1], avg_density,   'Avg triples / narrative (≥1)', 'Extraction Density',      '{:.2f}'),
        (axes[2], total_triples, 'Total triples extracted',       'Total Triple Yield',      '{:,}'),
    ]:
        bars = ax.bar(methods, vals, color=colors, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02 + 0.5,
                    fmt.format(val), ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        if title == 'Extraction Coverage':
            ax.set_ylim(0, 120)

    plt.suptitle(f'Cross-Model Comparison — Causal Triple Extraction (sample n={sample_n})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_cross_model_comparison.png')


# ---------------------------------------------------------------------------
# Radar (spider) chart — extraction models
# ---------------------------------------------------------------------------

def plot_radar_extraction(
    rule_triples: list,
    dep_triples: list,
    llm_triples: list,
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
        diversity = n_rels / max(1, len(triples)) * 100   # unique-relation %
        nodes     = kg_stats.get('num_nodes', 0)
        edges     = kg_stats.get('num_edges', 0)
        return coverage, density, diversity, nodes, edges

    r_cov, r_den, r_div, r_nod, r_edg = _metrics(rule_triples, rule_kg_stats)
    d_cov, d_den, d_div, d_nod, d_edg = _metrics(dep_triples,  dep_kg_stats)
    l_cov, l_den, l_div, l_nod, l_edg = _metrics(llm_triples,  llm_kg_stats)

    # Normalise each axis to [0, 1] using per-column max
    raw = np.array([
        [r_cov, r_den, r_div, r_nod, r_edg],
        [d_cov, d_den, d_div, d_nod, d_edg],
        [l_cov, l_den, l_div, l_nod, l_edg],
    ], dtype=float)

    col_max = raw.max(axis=0)
    col_max[col_max == 0] = 1.0       # avoid divide-by-zero
    norm = raw / col_max              # shape (3, 5), values in [0, 1]

    categories = [
        'Coverage\n(%)',
        'Density\n(triples/narr)',
        'Relation\nDiversity (%)',
        'KG Nodes',
        'KG Edges',
    ]
    N      = len(categories)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]              # close the polygon

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


# ---------------------------------------------------------------------------
# Radar — DistilBERT per-class metrics
# ---------------------------------------------------------------------------

def plot_radar_classifier(
    classification_report: dict,
    label_map: dict,
    plots_dir: Path,
):
    """Spider chart of Precision / Recall / F1 per class for DistilBERT."""
    if not classification_report:
        return

    classes = sorted(
        [c for c in label_map if c in classification_report],
        key=lambda c: label_map[c],
    )
    if len(classes) < 2:
        return

    short = [c.replace(' issues', '') for c in classes]
    N      = len(classes)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    metrics = {
        'Precision': [classification_report[c]['precision'] for c in classes],
        'Recall':    [classification_report[c]['recall']    for c in classes],
        'F1':        [classification_report[c]['f1-score']  for c in classes],
    }
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))

    for (metric, vals), color in zip(metrics.items(), colors):
        data = vals + vals[:1]
        ax.plot(angles, data, 'o-', linewidth=2, color=color, label=metric)
        ax.fill(angles, data, alpha=0.12, color=color)

    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7, color='grey')
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_title('DistilBERT — Per-Class Precision / Recall / F1',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    _save(fig, plots_dir, 'eval_radar_classifier.png')


# ---------------------------------------------------------------------------
# Per-source knowledge graph visualizations
# ---------------------------------------------------------------------------

def plot_kg_per_source(
    rule_triples: list,
    dep_triples: list,
    llm_triples: list,
    noise_filter: bool = True,
    normalize: bool = True,
    top_n: int = 30,
    plots_dir: Path = Path('outputs/plots'),
):
    """Three side-by-side network graphs — one per extraction source."""
    try:
        import networkx as nx
    except ImportError:
        print('  networkx not installed — skipping per-source KG plots.')
        return

    from src.knowledge_graph import build_graph, _is_noise, _normalize_entity

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

        # Top-N nodes by total degree
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:top_n]
        sub = G.subgraph(top_nodes)

        pos = nx.spring_layout(sub, seed=42, k=1.8)

        node_sizes  = [200 + 80 * sub.degree(n) for n in sub.nodes()]
        node_colors = [
            color if sub.nodes[n].get('type') == 'cause_node' else '#ecf0f1'
            for n in sub.nodes()
        ]

        nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=node_sizes,
                               node_color=node_colors, alpha=0.88, linewidths=0.5,
                               edgecolors='#555')
        nx.draw_networkx_edges(sub, pos, ax=ax, edge_color='#aaa',
                               arrows=True, arrowsize=12, width=1.0, alpha=0.5)

        # Label top-15 nodes only
        top15 = set(sorted(sub.nodes(), key=lambda n: sub.degree(n), reverse=True)[:15])
        labels = {n: (n[:22] + '…' if len(n) > 22 else n) for n in top15}
        nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax, font_size=6)

        stats_txt = (
            f'Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}\n'
            f'Triples: {len(triples):,}  WCC: {nx.number_weakly_connected_components(G)}'
        )
        ax.set_title(f'{name} Knowledge Graph\n(top {top_n} nodes shown)\n{stats_txt}',
                     fontsize=10, fontweight='bold')
        ax.axis('off')

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=color,     label='Cause node'),
            Patch(facecolor='#ecf0f1', label='Effect node', edgecolor='#555'),
        ], fontsize=7, loc='lower left')

    plt.suptitle('Knowledge Graphs by Extraction Source', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_kg_per_source.png')


def plot_kg_rule_bert_llm(
    rule_triples: list,
    bert_triples: list,
    llm_triples: list,
    noise_filter: bool = True,
    normalize: bool = True,
    top_n: int = 30,
    plots_dir: Path = Path('outputs/plots'),
):
    """
    Three side-by-side knowledge graph panels — one per extraction method:
      Left   — Rule-based causal KG
      Center — BERT Causal Extractor KG
      Right  — LLM (Mistral-7B) causal KG
    All three models now perform the same task (causal triple extraction),
    enabling a direct structural comparison of the extracted knowledge.
    """
    try:
        import networkx as nx
    except ImportError:
        print('  networkx not installed — skipping KG plot.')
        return

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
                               node_color=node_colors, alpha=0.88,
                               linewidths=0.5, edgecolors='#555')
        nx.draw_networkx_edges(sub, pos, ax=ax, edge_color='#aaa',
                               arrows=True, arrowsize=12, width=1.0, alpha=0.5)
        top15 = set(sorted(sub.nodes(), key=lambda n: sub.degree(n), reverse=True)[:15])
        labels = {n: (n[:22] + '...' if len(n) > 22 else n) for n in top15}
        nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax, font_size=6)

        ev_with = len({t['ev_id'] for t in triples})
        stats = (
            f'Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}\n'
            f'Triples: {len(triples):,}  Narratives: {ev_with:,}  '
            f'WCC: {nx.number_weakly_connected_components(G)}'
        )
        ax.set_title(f'{title}\n(top {top_n} nodes shown)\n{stats}',
                     fontsize=10, fontweight='bold')
        ax.axis('off')
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=color, label='Cause node'),
            Patch(facecolor='#ecf0f1', label='Effect node', edgecolor='#555'),
        ], fontsize=7, loc='lower left')

    _draw_kg(axes[0], rule_triples,  '#2196F3', 'Rule-based Knowledge Graph')
    _draw_kg(axes[1], bert_triples,  '#FF9800', 'BERT Extractor Knowledge Graph')
    _draw_kg(axes[2], llm_triples,   '#9C27B0', 'LLM Knowledge Graph')

    plt.suptitle('Knowledge Graphs: Rule-based  |  BERT Extractor  |  LLM',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_kg_rule_bert_llm.png')


# ---------------------------------------------------------------------------
# LLM cache growth / utilisation
# ---------------------------------------------------------------------------

def plot_llm_cache_growth(
    cache: dict,
    llm_triples: list,
    sample_n: int,
    plots_dir: Path,
):
    """Bar chart showing cache size, triples parsed, coverage, and avg density."""
    if not cache:
        return

    total_cached  = len(cache)
    # Parse triples from every cached entry (reuse existing parse logic inline)
    import re, json as _json
    _JSON_RE = re.compile(r'\[.*\]', re.DOTALL)
    _JUNK    = {"the accident", "this accident", "the incident", "this incident",
                "the crash", "an accident", "an incident"}

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
            if c and e and c.lower() not in _JUNK and e.lower() not in _JUNK:
                full_triples.append({'ev_id': ev_id, 'cause': c, 'relation': r, 'effect': e})

    ev_with_full   = len({t['ev_id'] for t in full_triples})
    coverage_full  = ev_with_full / max(1, total_cached) * 100
    density_full   = len(full_triples) / max(1, ev_with_full)

    ev_with_orig   = len({t['ev_id'] for t in llm_triples})
    coverage_orig  = ev_with_orig / max(1, sample_n) * 100
    density_orig   = len(llm_triples) / max(1, ev_with_orig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # [0] Narratives processed: sampled vs full cache
    labels = [f'Training sample\n(n={sample_n})', f'Full cache\n(n={total_cached})']
    covered = [ev_with_orig, ev_with_full]
    colors  = ['#9C27B0', '#4CAF50']
    bars = axes[0].bar(labels, covered, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, covered):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 10,
                     f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[0].set_ylabel('Narratives with ≥1 triple')
    axes[0].set_title('LLM: Narratives Covered')
    axes[0].grid(True, axis='y', alpha=0.3)

    # [1] Total triples
    triple_counts = [len(llm_triples), len(full_triples)]
    bars2 = axes[1].bar(labels, triple_counts, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars2, triple_counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 20,
                     f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('Total triples extracted')
    axes[1].set_title('LLM: Total Triple Yield')
    axes[1].grid(True, axis='y', alpha=0.3)

    # [2] Coverage % and avg density side-by-side
    x = np.arange(2)
    w = 0.35
    ax2 = axes[2]
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


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Top relation phrases — three-model comparison
# ---------------------------------------------------------------------------

def plot_top_relation_phrases(
    rule_triples: list,
    bert_triples: list,
    llm_triples: list,
    plots_dir: Path,
    top_n: int = 10,
):
    """
    Three-panel horizontal bar chart: top relation phrases for each model.
    All three models evaluated on the same test set, so counts are directly
    comparable.
    """
    sources = [
        ('Rule-based', rule_triples, '#2196F3'),
        ('BERT Extractor', bert_triples, '#FF9800'),
        ('LLM (Mistral-7B)', llm_triples, '#9C27B0'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, (name, triples, color) in zip(axes, sources):
        if not triples:
            ax.set_title(f'{name}\n(no triples)')
            ax.axis('off')
            continue

        counts = Counter(t['relation'] for t in triples)
        top    = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
        labels = [p for p, _ in top]
        vals   = [v for _, v in top]
        total  = sum(counts.values())

        bars = ax.barh(labels, vals, color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            pct = val / max(1, total) * 100
            ax.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:,} ({pct:.1f}%)', va='center', fontsize=8)

        ax.invert_yaxis()
        ax.set_xlabel('Triple count')
        n_ev = len({t['ev_id'] for t in triples})
        ax.set_title(
            f'{name}\n{len(triples):,} triples · {n_ev} narratives\n'
            f'top {top_n} of {len(counts)} unique relations',
            fontsize=10, fontweight='bold',
        )
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlim(right=max(vals) * 1.25)

    plt.suptitle('Top Relation Phrases — Test Set (909 narratives)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, 'eval_top_relation_phrases.png')


# ---------------------------------------------------------------------------
# Finding-alignment evaluation plots
# ---------------------------------------------------------------------------

def plot_finding_alignment(
    alignment_results: list,   # list of dicts from finding_evaluator.evaluate_finding_alignment
    plots_dir: Path,
    suffix: str = '',
):
    """
    3-panel figure summarising NTSB finding-alignment metrics across models:
    - Category alignment score
    - Cause-confirmed coverage
    - Finding keyword recall
    Plus a grouped bar chart breaking down category alignment by NTSB top category.
    """
    if not alignment_results:
        return

    labels = [r['label'] for r in alignment_results]
    colors = ['#2196F3', '#4CAF50', '#9C27B0', '#FF9800'][:len(labels)]

    # None means N/A for that model (e.g. keyword_recall for a classifier)
    cat_aln = [r['category_alignment_score'] * 100 for r in alignment_results]
    cc_cov  = [r['cause_confirmed_coverage'] * 100 for r in alignment_results]
    kw_rec  = [r['finding_keyword_recall'] * 100 if r['finding_keyword_recall'] is not None else None
               for r in alignment_results]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    def _bar_with_na(ax, vals, model_labels, model_colors, title, ylabel):
        """Bar chart that skips None values and labels them 'N/A'."""
        x = np.arange(len(model_labels))
        w = 0.65
        for i, (val, lbl, col) in enumerate(zip(vals, model_labels, model_colors)):
            if val is None:
                ax.bar(i, 0, w, color='#e0e0e0', alpha=0.5, edgecolor='white')
                ax.text(i, 3, 'N/A', ha='center', va='bottom', fontsize=10,
                        color='#999', fontweight='bold')
            else:
                ax.bar(i, val, w, color=col, alpha=0.85, edgecolor='white')
                ax.text(i, val + 0.5, f'{val:.1f}%', ha='center', va='bottom',
                        fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=10, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 115)
        ax.grid(True, axis='y', alpha=0.3)

    _bar_with_na(axes[0], cat_aln, labels, colors,
                 'Category Alignment\n(predicted cat == NTSB finding cat)',
                 'Accuracy (%)')
    _bar_with_na(axes[1], cc_cov, labels, colors,
                 'Cause-Confirmed Coverage\n(C-findings only as denominator)',
                 '% of C-finding accidents covered')
    _bar_with_na(axes[2], kw_rec, labels, colors,
                 'Finding Keyword Recall\n(% finding tokens in extracted text)',
                 'Avg recall (%)')

    # Sub-labels: sample sizes
    for ax, n_items in zip(axes, [
        [f'n={r["category_alignment_n"]}' for r in alignment_results],
        [f'{r["cause_confirmed_n"]}/{r["cause_confirmed_denom"]}' for r in alignment_results],
        [f'n={r["keyword_recall_n"]}' if r['finding_keyword_recall'] is not None else 'N/A'
         for r in alignment_results],
    ]):
        for i, label in enumerate(n_items):
            ax.text(i, ax.get_ylim()[0] + 2, label,
                    ha='center', va='bottom', fontsize=7, color='grey')

    title_note = ' (Unified Test Set)' if suffix else ''
    plt.suptitle(f'NTSB Finding-Alignment Evaluation — Ground Truth Comparison{title_note}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, plots_dir, f'eval_finding_alignment{suffix}.png')

    # ------------------------------------------------------------------
    # Per-category alignment breakdown
    # ------------------------------------------------------------------
    all_cats = sorted({cat for r in alignment_results
                       for cat in r.get('per_category_alignment', {})})
    if not all_cats:
        return

    x     = np.arange(len(all_cats))
    width = 0.8 / max(1, len(labels))

    fig2, ax2 = plt.subplots(figsize=(13, 6))
    for i, (r, color) in enumerate(zip(alignment_results, colors)):
        pa    = r.get('per_category_alignment', {})
        vals  = [pa.get(cat, {}).get('score', 0) * 100 for cat in all_cats]
        ns    = [pa.get(cat, {}).get('total', 0) for cat in all_cats]
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
