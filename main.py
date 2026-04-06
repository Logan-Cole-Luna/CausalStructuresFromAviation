"""
main.py — Full NTSB pipeline: training then evaluation.

Usage:
    python main.py                     # uses sample_n from CONFIG.conf [global]
    python main.py --sample 2000       # override sample size for all models
    python main.py --sample 500 --no-llm
    python main.py --eval-only         # skip training, re-plot from existing artifacts
    python main.py --train-only        # training only, no plots

All models use the same --sample count for fair comparison.
"""
import argparse
import configparser
import sys


def _load_cfg(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',))
    cfg.read(path)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='NTSB Causal Chain Extraction — Full Pipeline'
    )
    parser.add_argument('--sample',        type=int, default=None,
                        help='Narratives per model (overrides CONFIG.conf [global] sample_n)')
    parser.add_argument('--config',        type=str, default='CONFIG.conf')
    parser.add_argument('--no-llm',        action='store_true', help='Skip LLM extraction')
    parser.add_argument('--no-distilbert', action='store_true', help='Skip DistilBERT training')
    parser.add_argument('--train-only',    action='store_true', help='Run training only')
    parser.add_argument('--eval-only',     action='store_true', help='Run evaluation only')
    args = parser.parse_args()

    if args.train_only and args.eval_only:
        print('Error: --train-only and --eval-only are mutually exclusive.')
        sys.exit(1)

    cfg      = _load_cfg(args.config)
    sample_n = args.sample if args.sample is not None else \
               int(cfg.get('global', 'sample_n', fallback=2000))

    # ---- Training ----
    if not args.eval_only:
        from src.train import main as run_train
        train_argv = ['train.py', '--sample', str(sample_n), '--config', args.config]
        if args.no_llm:
            train_argv.append('--no-llm')
        if args.no_distilbert:
            train_argv.append('--no-distilbert')
        sys.argv = train_argv
        run_train()

    # ---- Evaluation ----
    if not args.train_only:
        from src.eval import main as run_eval
        eval_argv = ['eval.py', '--sample', str(sample_n), '--config', args.config]
        if args.no_llm:
            eval_argv.append('--no-llm')
        sys.argv = eval_argv
        run_eval()


if __name__ == '__main__':
    main()
