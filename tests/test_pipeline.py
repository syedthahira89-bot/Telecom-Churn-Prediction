from datetime import date

import pandas as pd

from churn_model.pipeline import (
    MINIMUM_TENURE_DAYS,
    prepare_features,
    risk_tier,
    train_model,
    score_customers,
)
from scripts.make_demo_data import make_demo_data


def test_risk_tier_boundaries() -> None:
    assert risk_tier(0.3999) == "Low"
    assert risk_tier(0.40) == "Medium"
    assert risk_tier(0.6999) == "Medium"
    assert risk_tier(0.70) == "High"


def test_feature_policy_blocks_protected_and_operational_fields() -> None:
    features = prepare_features(
        pd.DataFrame(
            {
                "age": [40],
                "SeniorCitizen": [0],
                "gender_identity": ["x"],
                "customer_id": ["x"],
                "snapshot_date": ["2025-01-01"],
                "marketing_opt_out": [False],
                "monthly_charges": [90],
                "previous_monthly_charges": [80],
            }
        )
    )
    assert list(features.columns) == [
        "monthly_charges",
        "previous_monthly_charges",
        "monthly_charge_change",
    ]


def test_score_obeys_tenure_optout_and_offer_cooldown() -> None:
    history = make_demo_data(300, seed=8)
    bundle = train_model(history)
    snapshot = make_demo_data(12, seed=9).drop(columns="churn_60d")
    snapshot.loc[0, "tenure_days"] = MINIMUM_TENURE_DAYS - 1
    snapshot.loc[1, "marketing_opt_out"] = True
    snapshot.loc[2, "last_offer_date"] = "2026-08-15"
    scores, tasks = score_customers(
        snapshot, bundle, as_of=date(2026, 9, 30), include_explanations=False
    )
    assert scores.loc[0, "risk_tier"] == "Not scored"
    assert not scores.loc[1, "outreach_eligible"]
    assert not scores.loc[2, "outreach_eligible"]
    assert "customer_id" in tasks.columns


def test_train_model_accepts_explicit_benchmark_target() -> None:
    history = make_demo_data(300, seed=15).rename(columns={"churn_60d": "churn"})
    bundle = train_model(history, target_column="churn")
    assert 0 <= bundle.holdout_metrics["score_min"]
    assert bundle.holdout_metrics["score_max"] <= 1
    assert bundle.holdout_metrics["top_5pct_selected_rows"] == 3
    assert 0 <= bundle.holdout_metrics["top_5pct_recall"] <= 1