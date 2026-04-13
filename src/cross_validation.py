"""
cross_validation.py - Cross-validation split management for train/val/test.

Provides utilities for creating and managing stratified train/val/test splits
with a 60/20/20 split ratio.
"""
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_cv_split(
    df: pd.DataFrame,
    id_col: str = 'ev_id',
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    random_state: int = 42,
    stratify_col: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Create a train/val/test split with specified fractions.

    Args:
        df: DataFrame with id_col to split
        id_col: Column name containing unique IDs
        train_frac: Fraction for training (default 0.6)
        val_frac: Fraction for validation (default 0.2)
        test_frac: Fraction for testing (default 0.2)
        random_state: Random seed for reproducibility
        stratify_col: Optional column for stratified splitting

    Returns:
        Dict with keys 'train_ev_ids', 'val_ev_ids', 'test_ev_ids'
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, \
        f"Fractions must sum to 1.0, got {train_frac + val_frac + test_frac}"

    all_ids = df[id_col].astype(str).unique().tolist()
    stratify = df.drop_duplicates(subset=[id_col])[stratify_col].values if stratify_col else None

    # Split train+val vs test
    test_size = test_frac
    train_ids, test_ids = train_test_split(
        all_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Split train vs val (from remaining)
    val_rel_size = val_frac / (1.0 - test_frac)
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=val_rel_size,
        random_state=random_state,
        stratify=stratify[np.isin(all_ids, train_ids)] if stratify is not None else None,
    )

    return {
        'train_ev_ids': train_ids,
        'val_ev_ids': val_ids,
        'test_ev_ids': test_ids,
    }


def save_cv_split(split: Dict[str, List[str]], path: Path):
    """Save cross-validation split to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"[CV Split] Saved to {path}")


def load_cv_split(path: Path) -> Dict[str, List[str]]:
    """Load cross-validation split from JSON file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def print_cv_split(split: Dict[str, List[str]], df: Optional[pd.DataFrame] = None):
    """Pretty-print cross-validation split statistics."""
    print("\n[CV Split] Train/Val/Test Distribution:")
    for key in ['train_ev_ids', 'val_ev_ids', 'test_ev_ids']:
        n = len(split.get(key, []))
        frac = n / sum(len(split.get(k, [])) for k in ['train_ev_ids', 'val_ev_ids', 'test_ev_ids'])
        print(f"  {key:20s} {n:6d} ({frac:.1%})")
