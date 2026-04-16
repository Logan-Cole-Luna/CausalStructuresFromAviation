"""
extractor_utils.py — Shared utilities for NTSB causal extractor models.

Re-exports pattern constants from rule_based.py and provides common
functions used by bert_extractor.py, t5_extractor.py, and llm_extractor.py.
"""
from typing import Tuple

from src.rule_based import CAUSAL_FORWARD, CAUSAL_BACKWARD, _ALL_PATTERNS, _PATTERN_RE

# ---------------------------------------------------------------------------
# Re-export pattern constants (imported from rule_based to avoid duplication)
# ---------------------------------------------------------------------------

__all__ = [
    'CAUSAL_FORWARD',
    'CAUSAL_BACKWARD',
    '_ALL_PATTERNS',
    '_PATTERN_RE',
    'infer_relation',
    'log_bias_variance',
    '_JUNK_NODES',
    '_JSON_RE',
]

# ---------------------------------------------------------------------------
# Junk node filter and JSON regex (shared by llm_extractor and plotting)
# ---------------------------------------------------------------------------

import re

_JUNK_NODES = {
    "the accident", "this accident", "the incident", "this incident",
    "the crash", "an accident", "an incident",
}

# Regex to pull the first JSON array out of model output
_JSON_RE = re.compile(r'\[.*\]', re.DOTALL)


# ---------------------------------------------------------------------------
# infer_relation — identical logic previously duplicated in bert and t5
# ---------------------------------------------------------------------------

def infer_relation(sentence: str, cause: str, effect: str) -> Tuple[str, str]:
    """
    Find the first causal connective phrase in sentence.
    Returns (relation_phrase, direction).
    Direction is 'forward' for CAUSE->EFFECT patterns, 'backward' for EFFECT<-CAUSE.
    """
    for pat in _ALL_PATTERNS:
        if _PATTERN_RE[pat].search(sentence):
            direction = 'forward' if pat in CAUSAL_FORWARD else 'backward'
            return pat, direction
    return 'caused', 'forward'


# ---------------------------------------------------------------------------
# log_bias_variance — identical method previously duplicated in BERT and T5
# ---------------------------------------------------------------------------

def log_bias_variance(
    train_loss: float,
    val_loss: float,
    train_f1: float,
    val_f1: float,
    epoch: int,
) -> dict:
    """Analyze bias-variance tradeoff for an epoch."""
    loss_gap = val_loss - train_loss
    f1_gap = val_f1 - train_f1

    # Classify regime
    if loss_gap > 0.1 and f1_gap < -0.05:
        regime = 'high_variance'  # Overfitting
    elif loss_gap < -0.1 or f1_gap > 0.05:
        regime = 'high_bias'      # Underfitting
    else:
        regime = 'balanced'

    return {
        'epoch': epoch,
        'train_loss': round(train_loss, 6),
        'val_loss': round(val_loss, 6),
        'loss_gap': round(loss_gap, 6),
        'train_f1': round(train_f1, 4),
        'val_f1': round(val_f1, 4),
        'f1_gap': round(f1_gap, 4),
        'regime': regime,
    }
