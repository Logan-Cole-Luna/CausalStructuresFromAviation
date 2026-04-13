"""
main.py — Full NTSB pipeline: hyperparameter tuning, training, then evaluation.

Usage:
    python main.py              # Full pipeline (15 Optuna trials per model + evaluation)

This runs the complete causal extraction pipeline:
  1. Hyperparameter tuning (Optuna Bayesian search) on validation set
  2. Final model training on full training set
  3. Evaluation on held-out test set (1,192 narratives)
  4. Comparison across all 6 extraction approaches

All artifacts saved to outputs/ directory.
"""
import sys


def main():
    print("=" * 80)
    print("  NTSB Causal Chain Extraction — Full Pipeline")
    print("  (Hyperparameter Tuning → Training → Evaluation)")
    print("=" * 80)

    # ---- Step 1: Hyperparameter Tuning & Training ----
    print("\n[1/2] TRAINING: Hyperparameter tuning + final model training...")
    print("      Models: BERT (distilbert-base-uncased) + T5 (t5-base)")
    print("      Trials: 15 Optuna trials per model on validation set")
    print()

    from src.train import main as run_train
    sys.argv = ['train.py']  # No arguments - uses defaults (15 trials each)
    try:
        run_train()
    except SystemExit:
        pass  # train.py calls sys.exit() - catch it to continue

    # ---- Step 2: Evaluation ----
    print("\n" + "=" * 80)
    print("[2/2] EVALUATION: Testing trained models on held-out test set...")
    print("      Test set: 1,192 narratives (20% of 6,059 total)")
    print("      Models evaluated: Rule-based, Dep-parse, BERT, T5, LLM (2 variants)")
    print()

    from src.eval import main as run_eval
    sys.argv = ['eval.py']  # No arguments - evaluates on test set
    try:
        run_eval()
    except SystemExit:
        pass  # eval.py calls sys.exit() - catch it

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE")
    print("=" * 80)
    print("\nOutput artifacts:")
    print("  - Tuned models:        outputs/model_{bert,t5}_extractor_tuned/")
    print("  - Extracted triples:   outputs/extractions/{bert,t5}_triples.json")
    print("  - Evaluation report:   outputs/evaluation/evaluation_report.json")
    print("  - Visualizations:      outputs/plots/eval_*.png")
    print("\nNext steps:")
    print("  - View results: cat outputs/evaluation/evaluation_report.json")
    print("  - Open plots: outputs/plots/eval_cross_model_comparison.png")
    print()


if __name__ == '__main__':
    main()
