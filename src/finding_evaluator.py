"""
finding_evaluator.py — Ground-truth evaluation using NTSB finding_description.

The finding_description column contains NTSB's official causal determinations
in a hierarchical taxonomy:
    Category - Subcategory - Sub-sub - ... - Actor - CauseCode

CauseCode:
    'C' = Contributing cause (this IS a cause)
    'F' = Finding (observed but not designated as a cause)
    (other values appear for equipment-type findings with no actor code)

This module computes three extraction-quality metrics against this ground truth:

1. category_alignment_score
   For each ev_id where we extracted >= 1 triple AND a finding exists:
   Classify the cause text into one of the 4 NTSB top-level categories using
   keyword heuristics, then compare to the official finding's top category.
   Score = % of ev_ids where the predicted category matches the official one.

2. cause_confirmed_coverage
   Denominator: ev_ids that have >= 1 'C'-coded finding (accidents with a
                confirmed cause that the model *should* be able to extract).
   Numerator:   of those, how many did the model actually extract >= 1 triple for.
   This is a stricter and more meaningful coverage metric than raw coverage.

3. finding_keyword_recall
   For each ev_id, tokenize the finding hierarchy (split by '-', drop 'C'/'F'
   codes and short tokens) to build a set of ground-truth concept tokens.
   Compute what % of those tokens appear anywhere in the extracted cause or
   effect text.  Average across ev_ids.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Category keyword heuristics
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    'Personnel issues': [
        'pilot', 'crew', 'captain', 'officer', 'student', 'instructor',
        'decision', 'judgment', 'attention', 'situational', 'awareness',
        'fatigue', 'training', 'procedure', 'checklist', 'error',
        'workload', 'distraction', 'scan', 'monitor', 'experience',
        'planning', 'action', 'omission', 'communication', 'coordination',
    ],
    'Aircraft': [
        'engine', 'fuel', 'power', 'propeller', 'rotor', 'blade',
        'gear', 'brake', 'flap', 'control', 'aileron', 'elevator',
        'rudder', 'hydraulic', 'electrical', 'battery', 'circuit',
        'mechanical', 'structural', 'airframe', 'component', 'system',
        'failure', 'malfunction', 'fatigue', 'corrosion', 'wear',
        'carburetor', 'manifold', 'exhaust', 'oil', 'ignition',
        'magneto', 'cylinder', 'piston', 'crankshaft', 'bearing',
    ],
    'Environmental issues': [
        'weather', 'wind', 'gust', 'turbulence', 'icing', 'ice',
        'fog', 'visibility', 'cloud', 'ceiling', 'precipitation',
        'rain', 'snow', 'density', 'altitude', 'terrain', 'obstacle',
        'bird', 'wildlife', 'night', 'dark',
    ],
    'Organizational issues': [
        'maintenance', 'management', 'organization', 'oversight',
        'regulation', 'policy', 'procedure', 'inspection', 'supervision',
        'dispatch', 'scheduling',
    ],
}

# Short tokens to skip when building finding keyword sets
_SKIP_TOKENS = frozenset({
    'c', 'f', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to',
    'not', 'by', 'on', 'at', 'for', 'with', 'issues', 'general',
    'other', 'unknown', 'misc', 'attained', 'maintained',
    'use', 'effect', 'type', 'condition', 'related',
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize_finding(finding: str) -> List[str]:
    """Split a finding string into meaningful concept tokens."""
    raw = re.split(r'[-/]', finding)
    tokens = []
    for part in raw:
        part = part.strip().lower()
        if len(part) >= 4 and part not in _SKIP_TOKENS:
            # Further split on spaces and keep multi-word tokens as is
            for word in re.split(r'\s+', part):
                word = re.sub(r'[^a-z]', '', word)
                if len(word) >= 4 and word not in _SKIP_TOKENS:
                    tokens.append(word)
    return tokens


def _classify_text(text: str) -> str:
    """Heuristic: classify a cause/effect string into an NTSB top category."""
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for cat, kws in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in kws if kw in text_lower)
    best_score = max(scores.values())
    if best_score == 0:
        return 'Unknown'
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def load_findings(data_path: str | Path) -> pd.DataFrame:
    """
    Load finding_description from the dataset and parse into structured columns.

    Returns a DataFrame with columns:
        ev_id, finding_description, category, level1, level2, is_cause
    """
    df = pd.read_csv(data_path, usecols=['ev_id', 'finding_description'])
    df['ev_id'] = df['ev_id'].astype(str)
    df = df.dropna(subset=['finding_description'])

    parts = df['finding_description'].str.split('-')
    df['category'] = parts.str[0].str.strip()
    df['level1']   = parts.str[1].str.strip() if parts.str.len().max() > 1 else ''
    df['level2']   = parts.str[2].str.strip() if parts.str.len().max() > 2 else ''
    df['cause_code'] = parts.str[-1].str.strip()
    df['is_cause'] = df['cause_code'] == 'C'

    return df.reset_index(drop=True)


def evaluate_finding_alignment(
    triples: List[dict],
    findings_df: pd.DataFrame,
    label: str = 'model',
) -> Dict:
    """
    Compute the three finding-alignment metrics for a set of extracted triples.

    Parameters
    ----------
    triples : list of dicts with keys ev_id, cause, effect, relation
    findings_df : output of load_findings()
    label : name of the extraction method (for reporting)

    Returns
    -------
    dict with keys:
        label, total_triples, ev_ids_extracted,
        category_alignment_score,  category_alignment_n,
        cause_confirmed_coverage,  cause_confirmed_n, cause_confirmed_denom,
        finding_keyword_recall,    keyword_recall_n,
        per_category_alignment     (breakdown by NTSB category)
    """
    # Index findings by ev_id
    findings_by_ev: Dict[str, List[pd.Series]] = defaultdict(list)
    for _, row in findings_df.iterrows():
        findings_by_ev[row['ev_id']].append(row)

    # Build per-ev_id triple text (concatenate all causes + effects)
    triple_text_by_ev: Dict[str, str] = defaultdict(str)
    for t in triples:
        eid = str(t.get('ev_id', ''))
        triple_text_by_ev[eid] += ' ' + str(t.get('cause', '')) + ' ' + str(t.get('effect', ''))

    extracted_ev_ids = set(triple_text_by_ev.keys())
    cause_finding_ev_ids = set(
        row['ev_id'] for _, row in findings_df.iterrows() if row['is_cause']
    )

    # ------------------------------------------------------------------
    # Metric 1: Category alignment
    # ------------------------------------------------------------------
    alignment_correct = 0
    alignment_total   = 0
    per_category_correct: Dict[str, int] = defaultdict(int)
    per_category_total:   Dict[str, int] = defaultdict(int)

    for ev_id, text in triple_text_by_ev.items():
        if ev_id not in findings_by_ev:
            continue
        # Use the primary (first) cause finding for comparison
        cause_findings = [r for r in findings_by_ev[ev_id] if r['is_cause']]
        if not cause_findings:
            cause_findings = findings_by_ev[ev_id]   # fall back to all findings
        official_cat = cause_findings[0]['category']

        predicted_cat = _classify_text(text)

        per_category_total[official_cat] += 1
        if predicted_cat == official_cat:
            alignment_correct += 1
            per_category_correct[official_cat] += 1
        alignment_total += 1

    cat_alignment_score = (
        alignment_correct / alignment_total if alignment_total > 0 else 0.0
    )
    per_cat_alignment = {
        cat: {
            'correct': per_category_correct[cat],
            'total':   per_category_total[cat],
            'score':   round(per_category_correct[cat] / max(1, per_category_total[cat]), 4),
        }
        for cat in per_category_total
    }

    # ------------------------------------------------------------------
    # Metric 2: Cause-confirmed coverage
    # ------------------------------------------------------------------
    # Accidents that (a) appear in our dataset AND (b) have a C finding
    eligible_ev_ids = cause_finding_ev_ids & set(findings_by_ev.keys())
    covered_cause_ev_ids = eligible_ev_ids & extracted_ev_ids

    cause_confirmed_cov = (
        len(covered_cause_ev_ids) / len(eligible_ev_ids)
        if eligible_ev_ids else 0.0
    )

    # ------------------------------------------------------------------
    # Metric 3: Finding keyword recall
    # ------------------------------------------------------------------
    recall_scores: List[float] = []
    for ev_id, text in triple_text_by_ev.items():
        if ev_id not in findings_by_ev:
            continue
        # Gather all finding tokens for this accident
        all_tokens: List[str] = []
        for row in findings_by_ev[ev_id]:
            all_tokens.extend(_tokenize_finding(row['finding_description']))
        if not all_tokens:
            continue

        text_lower = text.lower()
        matched = sum(1 for tok in all_tokens if tok in text_lower)
        recall_scores.append(matched / len(all_tokens))

    avg_keyword_recall = (
        sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    )

    return {
        'label':                     label,
        'total_triples':             len(triples),
        'ev_ids_extracted':          len(extracted_ev_ids),
        'category_alignment_score':  round(cat_alignment_score,  4),
        'category_alignment_n':      alignment_total,
        'cause_confirmed_coverage':  round(cause_confirmed_cov,  4),
        'cause_confirmed_n':         len(covered_cause_ev_ids),
        'cause_confirmed_denom':     len(eligible_ev_ids),
        'finding_keyword_recall':    round(avg_keyword_recall,    4),
        'keyword_recall_n':          len(recall_scores),
        'per_category_alignment':    per_cat_alignment,
    }


def evaluate_classifier_alignment(
    predictions: Dict[str, str],
    findings_df: pd.DataFrame,
    label: str = 'DistilBERT',
) -> Dict:
    """
    Evaluate a classifier (DistilBERT) against finding_description ground truth.

    Parameters
    ----------
    predictions : {ev_id: predicted_category_str}  — one prediction per narrative
    findings_df : output of load_findings()
    label       : display name for the model

    Returns a dict with the same keys as evaluate_finding_alignment so it can
    be included in the same table and plots.  finding_keyword_recall is None
    because a classifier has no free-text output to match against.
    """
    findings_by_ev: Dict[str, List] = defaultdict(list)
    for _, row in findings_df.iterrows():
        findings_by_ev[row['ev_id']].append(row)

    cause_finding_ev_ids = set(
        row['ev_id'] for _, row in findings_df.iterrows() if row['is_cause']
    )

    alignment_correct = 0
    alignment_total   = 0
    per_category_correct: Dict[str, int] = defaultdict(int)
    per_category_total:   Dict[str, int] = defaultdict(int)

    for ev_id, predicted_cat in predictions.items():
        if ev_id not in findings_by_ev:
            continue
        cause_findings = [r for r in findings_by_ev[ev_id] if r['is_cause']]
        official_cat   = (cause_findings or findings_by_ev[ev_id])[0]['category']

        per_category_total[official_cat] += 1
        if predicted_cat == official_cat:
            alignment_correct += 1
            per_category_correct[official_cat] += 1
        alignment_total += 1

    cat_alignment_score = alignment_correct / alignment_total if alignment_total > 0 else 0.0

    eligible  = cause_finding_ev_ids & set(findings_by_ev.keys())
    covered   = eligible & set(predictions.keys())

    per_cat_alignment = {
        cat: {
            'correct': per_category_correct[cat],
            'total':   per_category_total[cat],
            'score':   round(per_category_correct[cat] / max(1, per_category_total[cat]), 4),
        }
        for cat in per_category_total
    }

    return {
        'label':                     label,
        'total_triples':             None,          # N/A — classifier, not extractor
        'ev_ids_extracted':          len(predictions),
        'category_alignment_score':  round(cat_alignment_score, 4),
        'category_alignment_n':      alignment_total,
        'cause_confirmed_coverage':  round(len(covered) / max(1, len(eligible)), 4),
        'cause_confirmed_n':         len(covered),
        'cause_confirmed_denom':     len(eligible),
        'finding_keyword_recall':    None,          # N/A — no free-text output
        'keyword_recall_n':          0,
        'per_category_alignment':    per_cat_alignment,
    }


def print_finding_report(results: List[Dict]) -> None:
    """Pretty-print finding-alignment metrics for multiple models."""
    w = 26
    print(f'\n  {"Metric":<{w}}  ' +
          '  '.join(f'{r["label"]:>18}' for r in results))
    print('  ' + '-' * (w + 22 * len(results)))

    rows = [
        ('ev_ids extracted',        lambda r: f'{r["ev_ids_extracted"]:>18,}'),
        ('Cat. alignment score',    lambda r: f'{r["category_alignment_score"]:>17.1%}'),
        ('  (n ev_ids evaluated)',  lambda r: f'{r["category_alignment_n"]:>18,}'),
        ('Cause-confirmed cov.',    lambda r: f'{r["cause_confirmed_coverage"]:>17.1%}'),
        ('  (n covered / denom)',   lambda r: f'{r["cause_confirmed_n"]}/{r["cause_confirmed_denom"]:>14,}'),
        ('Finding keyword recall',  lambda r: f'{"N/A":>18}' if r["finding_keyword_recall"] is None else f'{r["finding_keyword_recall"]:>17.1%}'),
        ('  (n ev_ids scored)',     lambda r: f'{"N/A":>18}' if r["finding_keyword_recall"] is None else f'{r["keyword_recall_n"]:>18,}'),
    ]

    for label, fmt in rows:
        print(f'  {label:<{w}}  ' + '  '.join(fmt(r) for r in results))

    print()
    # Per-category breakdown
    all_cats = sorted({cat for r in results for cat in r['per_category_alignment']})
    print(f'  Category alignment breakdown:')
    print(f'  {"Category":<28}  ' +
          '  '.join(f'{r["label"]:>18}' for r in results))
    print('  ' + '-' * (30 + 22 * len(results)))
    for cat in all_cats:
        short = cat.replace(' issues', '')
        row_str = f'  {short:<28}  '
        for r in results:
            d = r['per_category_alignment'].get(cat, {})
            if d:
                row_str += f'  {d["score"]:>12.1%} ({d["correct"]}/{d["total"]})'
            else:
                row_str += '                     n/a'
        print(row_str)
