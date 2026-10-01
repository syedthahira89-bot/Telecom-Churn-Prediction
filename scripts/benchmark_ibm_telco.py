"""Benchmark the model on Kaggle's static IBM Telco churn dataset.

This dataset's Churn label is not a 60-day outcome and is not production data.
Download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn
and keep the CSV under data/ (the folder is git-ignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from churn_model.pipeline import train_model


DEFAULT_INPUT = Path(
    "data/ibm_telco/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
TARGET = "churn"


def prepare_ibm_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "customerID",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"IBM Telco dataset is missing columns: {sorted(missing)}")

    # Keep product/account attributes only; omit the dataset's demographic fields.
    benchmark = pd.DataFrame(index=frame.index)
    benchmark["customer_id"] = frame["customerID"].astype("string")
    benchmark["tenure_days"] = pd.to_numeric(frame["tenure"], errors="coerce") * 30.4375
    benchmark["contract_type"] = frame["Contract"]
    benchmark["monthly_charges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    benchmark["total_charges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    benchmark[TARGET] = frame["Churn"]

    feature_columns = {
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
    }
    for source, destination in feature_columns.items():
        if source in frame.columns:
            benchmark[destination] = frame[source]
    return benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a static IBM Telco churn benchmark (not a 60-day forecast)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(
            f"Dataset not found: {args.input}\n"
            "Download the CSV from https://www.kaggle.com/datasets/"
            "blastchar/telco-customer-churn and extract it under data/ibm_telco/."
        )

    raw = pd.read_csv(args.input)
    benchmark = prepare_ibm_dataset(raw)
    bundle = train_model(benchmark, target_column=TARGET)
    metrics = bundle.holdout_metrics

    print("IBM Telco static churn benchmark; not a 60-day forecast.")
    print(f"Input rows: {len(raw):,}; scored rows after tenure eligibility: {int(metrics['holdout_rows'] / 0.2):,}")
    print(f"Stratified holdout rows: {int(metrics['holdout_rows']):,}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    high_rows = int(round(metrics["high_tier_share"] * metrics["holdout_rows"]))
    print(
        f"Fixed 0.70 cutoff: {high_rows:,} selected "
        f"({metrics['high_tier_share']:.1%}), precision "
        f"{metrics['high_tier_precision']:.1%}, recall {metrics['high_tier_recall']:.1%}"
    )
    print(
        f"Top 5% by risk: {int(metrics['top_5pct_selected_rows']):,} selected, "
        f"cutoff {metrics['top_5pct_cutoff']:.3f}, "
        f"precision {metrics['top_5pct_precision']:.1%}, "
        f"recall {metrics['top_5pct_recall']:.1%}"
    )
    print("Holdout score distribution:")
    print(f"  min / median / p90 / p95 / max: "
          f"{metrics['score_min']:.3f} / {metrics['score_median']:.3f} / "
          f"{metrics['score_p90']:.3f} / {metrics['score_p95']:.3f} / "
          f"{metrics['score_max']:.3f}")
    print("  Low / Medium / High shares: "
          f"{metrics['low_tier_share']:.1%} / {metrics['medium_tier_share']:.1%} / "
          f"{metrics['high_tier_share']:.1%}")
    print(
        "Limitations: the Kaggle file is a single static snapshot with a general "
        "Churn label; thresholds and metrics are exploratory, not production approval."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())