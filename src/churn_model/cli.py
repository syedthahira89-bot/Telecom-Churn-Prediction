"""Command-line entry points for model training and monthly scoring."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from .pipeline import load_model, save_model, score_customers, train_model


def _atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=f".{destination.stem}-", dir=destination.parent,
            encoding="utf-8", newline="", delete=False
        ) as temporary:
            temporary_name = temporary.name
            frame.to_csv(temporary, index=False)
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _train(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.input)
    bundle = train_model(frame)
    save_model(bundle, args.model)
    print(json.dumps(bundle.holdout_metrics, indent=2))
    auc_ok = bundle.holdout_metrics["roc_auc"] >= args.minimum_auc
    recall_ok = bundle.holdout_metrics["high_tier_recall"] >= args.minimum_high_recall
    if not auc_ok or not recall_ok:
        print("ALERT: holdout model performance is below the configured threshold.")
        return 2
    return 0


def _score(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.input)
    bundle = load_model(args.model)
    scores, tasks = score_customers(
        frame,
        bundle,
        include_explanations=not args.skip_shap,
        shap_max_evals=args.shap_max_evals,
    )
    if args.top_n is not None:
        tasks = tasks.head(args.top_n)
    _atomic_csv(scores, args.output)
    task_path = Path(args.tasks) if args.tasks else Path(args.output).with_name(
        f"{Path(args.output).stem}_crm_tasks.csv"
    )
    _atomic_csv(tasks, task_path)
    print(f"Scored {len(scores)} customers; {len(tasks)} CRM tasks exported.")
    print(f"Scores: {args.output}\nCRM tasks: {task_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telecom churn model tools")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train and validate the stacked ensemble")
    train.add_argument("--input", required=True, help="Labeled customer snapshot CSV")
    train.add_argument("--model", required=True, help="Output joblib model artifact")
    train.add_argument("--minimum-auc", type=float, default=0.70)
    train.add_argument("--minimum-high-recall", type=float, default=0.75)
    train.set_defaults(handler=_train)

    score = commands.add_parser("score", help="Score customers and export CRM tasks")
    score.add_argument("--input", required=True, help="Current monthly customer snapshot CSV")
    score.add_argument("--model", required=True, help="Trained joblib model artifact")
    score.add_argument("--output", required=True, help="Output score CSV")
    score.add_argument("--tasks", help="Optional CRM task CSV path")
    score.add_argument("--top-n", type=int, help="Maximum tasks, ranked by expected revenue saved")
    score.add_argument("--skip-shap", action="store_true", help="Skip SHAP explanations for a fast run")
    score.add_argument("--shap-max-evals", type=int, default=100)
    score.set_defaults(handler=_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())