"""
hyperparameter_tuning.py - Hyperparameter optimization for extractors.

Uses Optuna for efficient hyperparameter search with Bayesian optimization.
"""
import json
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple

import numpy as np

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


class HyperparameterTuner:
    """Manages hyperparameter optimization for model training."""

    def __init__(
        self,
        param_space: Dict[str, Dict[str, Any]],
        n_trials: int = 20,
        n_jobs: int = 1,
        timeout: Optional[int] = None,
    ):
        """
        Initialize tuner.

        Args:
            param_space: Dict mapping param name to distribution config.
                         Example: {'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-3}}
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs (1 = sequential)
            timeout: Max seconds for study (None = unlimited)
        """
        self.param_space = param_space
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.study = None
        self.trial_history = []

    def optimize(
        self,
        objective: Callable[[optuna.Trial], float],
        direction: str = 'maximize',
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run optimization.

        Args:
            objective: Callable that takes a trial and returns a metric to optimize
            direction: 'maximize' or 'minimize'
            verbose: Print progress

        Returns:
            Dict with 'best_params', 'best_value', 'trials_df'
        """
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()

        self.study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction=direction,
        )

        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            show_progress_bar=verbose,
        )

        best_params = self.study.best_params
        best_value = self.study.best_value

        # Build trials dataframe-like summary
        trials_data = []
        for trial in self.study.trials:
            trials_data.append({
                'trial': trial.number,
                'value': trial.value,
                'status': trial.state.name,
                'params': trial.params,
            })
        self.trial_history = trials_data

        return {
            'best_params': best_params,
            'best_value': best_value,
            'trials': trials_data,
        }

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters based on param_space."""
        params = {}
        for name, config in self.param_space.items():
            ptype = config.get('type')
            if ptype == 'float':
                params[name] = trial.suggest_float(
                    name,
                    low=config['low'],
                    high=config['high'],
                    log=config.get('log', False),
                )
            elif ptype == 'int':
                params[name] = trial.suggest_int(
                    name,
                    low=config['low'],
                    high=config['high'],
                    log=config.get('log', False),
                )
            elif ptype == 'categorical':
                params[name] = trial.suggest_categorical(
                    name,
                    choices=config['choices'],
                )
            else:
                raise ValueError(f"Unknown param type: {ptype}")
        return params

    def save_results(self, path: Path):
        """Save optimization results to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'trials': self.trial_history,
        }
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[Tuner] Saved results to {path}")


def log_bias_variance(
    train_loss: float,
    val_loss: float,
    train_metric: float,
    val_metric: float,
    epoch: int,
) -> Dict[str, Any]:
    """
    Analyze bias-variance tradeoff for an epoch.

    Returns dict with analysis metrics.
    """
    loss_gap = val_loss - train_loss
    metric_gap = val_metric - train_metric

    # Classify regime
    if loss_gap > 0.1 and metric_gap > 0.05:
        regime = 'high_variance'  # Overfitting
    elif loss_gap < -0.1 or metric_gap < -0.05:
        regime = 'high_bias'      # Underfitting
    else:
        regime = 'balanced'

    return {
        'epoch': epoch,
        'train_loss': round(train_loss, 6),
        'val_loss': round(val_loss, 6),
        'loss_gap': round(loss_gap, 6),
        'train_metric': round(train_metric, 4),
        'val_metric': round(val_metric, 4),
        'metric_gap': round(metric_gap, 4),
        'regime': regime,
    }


def print_bias_variance_analysis(logs: list):
    """Pretty-print bias-variance analysis."""
    print("\n[Bias-Variance Analysis]")
    print("  Epoch  Train Loss  Val Loss  Gap(V-T)  Train F1  Val F1  Regime")
    print("  " + "=" * 75)
    for log in logs:
        train_f1 = log.get('train_f1', log.get('train_metric', 0.0))
        val_f1 = log.get('val_f1', log.get('val_metric', 0.0))
        print(f"  {log['epoch']:5d}  {log['train_loss']:9.6f}  "
              f"{log['val_loss']:8.6f}  {log['loss_gap']:8.6f}  "
              f"{train_f1:8.4f}  {val_f1:8.4f}  {log['regime']}")

    # Summary statistics
    regimes = [log['regime'] for log in logs]
    high_var = regimes.count('high_variance')
    high_bias = regimes.count('high_bias')
    balanced = regimes.count('balanced')

    print("\n  Summary:")
    print(f"    High Variance (overfitting):  {high_var} epochs")
    print(f"    High Bias (underfitting):     {high_bias} epochs")
    print(f"    Balanced:                     {balanced} epochs")

    if high_var > high_bias:
        print("\n  Recommendation: Model shows signs of overfitting. "
              "Consider reducing complexity, adding regularization, or early stopping.")
    elif high_bias > high_var:
        print("\n  Recommendation: Model shows signs of underfitting. "
              "Consider increasing model capacity or training longer.")
    else:
        print("\n  Recommendation: Training regime appears balanced.")
