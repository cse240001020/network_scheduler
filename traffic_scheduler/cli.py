"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import rebuild_reports, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare algorithmic and AI network traffic schedulers."
    )
    parser.add_argument("--flows", type=int, default=2500,
                        help="flows per synthetic scenario (default: 2500)")
    parser.add_argument("--train-scenarios", type=int, default=10,
                        help="number of training scenarios (default: 10)")
    parser.add_argument("--test-repetitions", type=int, default=3,
                        help="randomized repetitions per held-out traffic regime (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="root seed for reproducible workloads (default: 42)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"),
                        help="directory for CSV, model, and chart outputs")
    parser.add_argument("--refresh-report", action="store_true",
                        help="recompute summary tables and chart from existing evaluation_results.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.refresh_report:
        rebuild_reports(args.output_dir, args.seed)
        return
    if args.flows < 100:
        raise SystemExit("--flows must be at least 100")
    if args.train_scenarios < 4:
        raise SystemExit("--train-scenarios must be at least 4 for a scenario-level holdout")
    if args.test_repetitions < 2:
        raise SystemExit("--test-repetitions must be at least 2 for uncertainty estimates")
    run_pipeline(
        flows=args.flows,
        train_scenarios=args.train_scenarios,
        test_repetitions=args.test_repetitions,
        seed=args.seed,
        output_dir=args.output_dir,
    )
