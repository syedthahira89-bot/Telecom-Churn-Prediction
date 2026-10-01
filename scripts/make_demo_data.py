"""Generate illustrative data for smoke-testing the workflow; not real customers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 1800, size=rows)
    monthly_charges = rng.normal(78, 24, size=rows).clip(20, 180)
    previous_charges = monthly_charges - rng.normal(0, 8, size=rows)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], rows, p=[0.55, 0.25, 0.20])
    support_calls = rng.poisson(1.4, size=rows)
    data_probability = (
        -2.0
        + (contract == "month-to-month") * 1.0
        + (support_calls >= 3) * 0.75
        + (monthly_charges > 100) * 0.5
        - (tenure > 700) * 0.7
    )
    churn_probability = 1 / (1 + np.exp(-data_probability))
    return pd.DataFrame(
        {
            "customer_id": [f"DEMO-{index:06d}" for index in range(rows)],
            "tenure_days": tenure,
            "contract_type": contract,
            "monthly_charges": monthly_charges.round(2),
            "previous_monthly_charges": previous_charges.round(2),
            "support_calls_90d": support_calls,
            "data_usage_gb": rng.gamma(3, 15, size=rows).round(2),
            "customer_lifetime_value": rng.normal(2200, 650, size=rows).clip(100, 9000).round(2),
            "marketing_opt_out": rng.choice([False, False, False, True], rows),
            "last_offer_date": "",
            "age": rng.integers(18, 90, size=rows),
            "churn_60d": rng.binomial(1, churn_probability),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/demo_customers.csv")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    make_demo_data(args.rows, args.seed).to_csv(destination, index=False)
    print(f"Wrote illustrative data to {destination}")


if __name__ == "__main__":
    main()