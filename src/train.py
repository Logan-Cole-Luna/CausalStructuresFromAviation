"""
hp_train.py - Hyperparameter tuning and training for BERT and T5 extractors.

Uses Optuna for Bayesian hyperparameter optimization on the validation set,
then trains final models with best hyperparameters on the full training set,
and evaluates on the test set.

Run as:
    python -m src.hp_train
"""
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.data_loader import load_data, preprocess_data
from src.bert_extractor import BERTCausalExtractor
from src.t5_extractor import T5CausalExtractor
from src.cross_validation import load_cv_split


def section(title: str):
    print('\n' + '=' * 80)
    print(f'  {title}')
    print('=' * 80)


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


# ============================================================================
# BERT Hyperparameter Tuning
# ============================================================================

def objective_bert(
    trial: optuna.Trial,
    df: pd.DataFrame,
    rule_triples: list,
    train_ev_ids: list,
    val_ev_ids: list,
    output_dir: Path,
) -> float:
    """Objective function for BERT hyperparameter optimization."""
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 5e-6, 5e-5, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 32, step=8)
    epochs = trial.suggest_int('epochs', 3, 8)

    print(f"\n  [Trial {trial.number}] lr={lr:.2e}, batch_size={batch_size}, epochs={epochs}")

    try:
        extractor = BERTCausalExtractor(model_name='distilbert-base-uncased')

        # Prepare data
        train_ds, val_ds = extractor.prepare_data(
            df=df,
            rule_triples=rule_triples,
            train_ev_ids=train_ev_ids,
            val_fraction=0.15,
        )

        # Train
        history = extractor.train(
            train_ds, val_ds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=None,
            patience=2,
        )

        best_val_f1 = history.get('best_val_f1', 0.0)
        print(f"    Result: Best val F1 = {best_val_f1:.4f}")

        # Report intermediate value for pruning
        trial.report(best_val_f1, step=epochs)

        return best_val_f1

    except Exception as e:
        print(f"    Error during trial: {e}")
        return 0.0


def tune_and_train_bert(
    df: pd.DataFrame,
    rule_triples: list,
    cv_split: Dict[str, list],
    output_dir: Path,
    n_trials: int = 10,
) -> Tuple[BERTCausalExtractor, Dict[str, Any]]:
    """
    Run hyperparameter tuning for BERT, then train final model.

    Returns:
        (trained_extractor, tuning_results)
    """
    section('BERT: Hyperparameter Tuning (Validation Set)')

    train_ev_ids = cv_split['train_ev_ids']
    val_ev_ids = cv_split['val_ev_ids']
    test_ev_ids = cv_split['test_ev_ids']

    print(f"  Train set: {len(train_ev_ids)} narratives")
    print(f"  Val set: {len(val_ev_ids)} narratives")
    print(f"  Test set: {len(test_ev_ids)} narratives")
    print(f"\n  Running {n_trials} trials with Optuna...\n")

    if not OPTUNA_AVAILABLE:
        print("  [WARN] Optuna not available — skipping hyperparameter tuning.")
        print("  Using default hyperparameters: lr=2e-5, batch_size=16, epochs=5")
        best_params = {
            'lr': 2e-5,
            'batch_size': 16,
            'epochs': 5,
        }
    else:
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction='maximize',
        )

        study.optimize(
            lambda trial: objective_bert(trial, df, rule_triples, train_ev_ids, val_ev_ids, output_dir),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_value = study.best_value

        print(f"\n  Best trial: {study.best_trial.number}")
        print(f"  Best params: {best_params}")
        print(f"  Best val F1: {best_value:.4f}")

    # Train final model with best params
    section('BERT: Training Final Model (Full Training Set)')

    bert_dir = output_dir / 'model_bert_extractor_tuned'
    extractor = BERTCausalExtractor(model_name='distilbert-base-uncased')

    # Prepare data on full training + val set
    train_ds, val_ds = extractor.prepare_data(
        df=df,
        rule_triples=rule_triples,
        train_ev_ids=train_ev_ids,
        val_fraction=0.15,
    )

    print(f"  Training with: lr={best_params['lr']:.2e}, "
          f"batch_size={best_params['batch_size']}, epochs={best_params['epochs']}")

    history = extractor.train(
        train_ds, val_ds,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        lr=best_params['lr'],
        save_path=str(bert_dir),
        patience=3,
    )

    # Log bias-variance analysis
    if 'bias_variance_logs' in history:
        from src.hyperparameter_tuning import print_bias_variance_analysis
        print_bias_variance_analysis(history['bias_variance_logs'])

    print(f"\n  Model saved to: {bert_dir}")

    return extractor, {
        'best_params': best_params,
        'training_history': history,
    }


# ============================================================================
# T5 Hyperparameter Tuning
# ============================================================================

def objective_t5(
    trial: optuna.Trial,
    df: pd.DataFrame,
    rule_triples: list,
    train_ev_ids: list,
    val_ev_ids: list,
    output_dir: Path,
) -> float:
    """Objective function for T5 hyperparameter optimization."""
    lr = trial.suggest_float('lr', 5e-6, 5e-5, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 32, step=8)
    epochs = trial.suggest_int('epochs', 3, 8)

    print(f"\n  [Trial {trial.number}] lr={lr:.2e}, batch_size={batch_size}, epochs={epochs}")

    try:
        extractor = T5CausalExtractor(model_name='t5-base')

        # Prepare data
        train_ds, val_ds = extractor.prepare_data(
            df=df,
            rule_triples=rule_triples,
            train_ev_ids=train_ev_ids,
            val_fraction=0.15,
        )

        # Train
        history = extractor.train(
            train_ds, val_ds,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            save_path=None,
            patience=2,
        )

        best_val_loss = history.get('best_val_loss', float('inf'))
        # Convert loss to metric (lower loss = higher score)
        metric = 1.0 / (1.0 + best_val_loss)
        print(f"    Result: Best val loss = {best_val_loss:.4f} (metric={metric:.4f})")

        # Report intermediate value for pruning
        trial.report(metric, step=epochs)

        return metric

    except Exception as e:
        print(f"    Error during trial: {e}")
        return 0.0


def tune_and_train_t5(
    df: pd.DataFrame,
    rule_triples: list,
    cv_split: Dict[str, list],
    output_dir: Path,
    n_trials: int = 10,
) -> Tuple[T5CausalExtractor, Dict[str, Any]]:
    """
    Run hyperparameter tuning for T5, then train final model.

    Returns:
        (trained_extractor, tuning_results)
    """
    section('T5: Hyperparameter Tuning (Validation Set)')

    train_ev_ids = cv_split['train_ev_ids']
    val_ev_ids = cv_split['val_ev_ids']
    test_ev_ids = cv_split['test_ev_ids']

    print(f"  Train set: {len(train_ev_ids)} narratives")
    print(f"  Val set: {len(val_ev_ids)} narratives")
    print(f"  Test set: {len(test_ev_ids)} narratives")
    print(f"\n  Running {n_trials} trials with Optuna...\n")

    if not OPTUNA_AVAILABLE:
        print("  [WARN] Optuna not available — skipping hyperparameter tuning.")
        print("  Using default hyperparameters: lr=1e-4, batch_size=16, epochs=5")
        best_params = {
            'lr': 1e-4,
            'batch_size': 16,
            'epochs': 5,
        }
    else:
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            direction='maximize',
        )

        study.optimize(
            lambda trial: objective_t5(trial, df, rule_triples, train_ev_ids, val_ev_ids, output_dir),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_value = study.best_value

        print(f"\n  Best trial: {study.best_trial.number}")
        print(f"  Best params: {best_params}")
        print(f"  Best metric: {best_value:.4f}")

    # Train final model with best params
    section('T5: Training Final Model (Full Training Set)')

    t5_dir = output_dir / 'model_t5_extractor_tuned'
    extractor = T5CausalExtractor(model_name='t5-base')

    # Prepare data on full training set
    train_ds, val_ds = extractor.prepare_data(
        df=df,
        rule_triples=rule_triples,
        train_ev_ids=train_ev_ids,
        val_fraction=0.15,
    )

    print(f"  Training with: lr={best_params['lr']:.2e}, "
          f"batch_size={best_params['batch_size']}, epochs={best_params['epochs']}")

    history = extractor.train(
        train_ds, val_ds,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        lr=best_params['lr'],
        save_path=str(t5_dir),
        patience=3,
    )

    print(f"\n  Model saved to: {t5_dir}")

    return extractor, {
        'best_params': best_params,
        'training_history': history,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for BERT and T5 extractors')
    parser.add_argument('--bert-trials', type=int, default=15, help='Number of Optuna trials for BERT')
    parser.add_argument('--t5-trials', type=int, default=15, help='Number of Optuna trials for T5')
    parser.add_argument('--config', type=str, default='CONFIG.conf', help='Config file path')
    parser.add_argument('--skip-bert', action='store_true', help='Skip BERT tuning')
    parser.add_argument('--skip-t5', action='store_true', help='Skip T5 tuning')
    args = parser.parse_args()

    section('Hyperparameter Tuning: BERT & T5 Causal Extractors')

    # Load data
    print('\nLoading data...')
    df = load_data('data/clean/cleaned_narritives_and_findings.csv')
    df = preprocess_data(df)
    print(f'  Records: {len(df)}')

    # Load CV split
    training_dir = Path('outputs/training')
    cv_split = load_cv_split(training_dir / 'cv_split.json')
    if not cv_split:
        print('  ERROR: cv_split.json not found. Run eval.py first to create the CV split.')
        return

    # Load rule triples
    rule_triples = _load_json(training_dir / 'rule_triples.json')
    if not rule_triples:
        print('  ERROR: rule_triples.json not found. Run train.py first.')
        return

    output_dir = Path('outputs')
    tuning_results = {}

    # BERT Hyperparameter Tuning
    if not args.skip_bert:
        bert_extractor, bert_results = tune_and_train_bert(
            df, rule_triples, cv_split, output_dir, n_trials=args.bert_trials
        )
        tuning_results['bert'] = bert_results

    # T5 Hyperparameter Tuning
    if not args.skip_t5:
        t5_extractor, t5_results = tune_and_train_t5(
            df, rule_triples, cv_split, output_dir, n_trials=args.t5_trials
        )
        tuning_results['t5'] = t5_results

    # Save tuning results
    section('Saving Tuning Results')
    results_path = output_dir / 'tuning_results.json'
    _save_json(tuning_results, results_path)
    print(f'  Tuning results saved to {results_path}')

    section('Hyperparameter Tuning Complete')
    print('\nNext steps:')
    print('  1. Run evaluation: python -m src.eval')
    print('  2. Check outputs/plots/ for cross-model comparison')
    print('  3. Review outputs/evaluation/evaluation_report.json for detailed metrics')


if __name__ == '__main__':
    main()
