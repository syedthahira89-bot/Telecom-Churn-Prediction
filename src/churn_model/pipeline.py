"""Training, scoring, policy, and explanation logic for churn risk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TARGET = "churn_60d"
CUSTOMER_ID = "customer_id"
TENURE_DAYS = "tenure_days"
LIFETIME_VALUE = "customer_lifetime_value"
MARKETING_OPTOUT = "marketing_opt_out"
LAST_OFFER_DATE = "last_offer_date"
HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.40
MINIMUM_TENURE_DAYS = 30
RANDOM_STATE = 42

PROTECTED_TOKENS = {
    "age",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "religion",
    "marital",
    "disability",
    "national_origin",
}
NON_FEATURE_COLUMNS = {
    CUSTOMER_ID,
    TARGET,
    LIFETIME_VALUE,
    MARKETING_OPTOUT,
    LAST_OFFER_DATE,
    "churn",
    "churned",
    "churn_date",
    "outcome",
    "outcome_date",
}


@dataclass
class ModelBundle:
    model: StackingClassifier
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    holdout_metrics: dict[str, float]


def _normalized_name(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").split())


def _is_protected(name: str) -> bool:
    tokens = set(_normalized_name(name).split("_"))
    return bool(tokens & PROTECTED_TOKENS)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add simple derived features and remove columns forbidden from modeling."""
    result = frame.copy()
    if {"monthly_charges", "previous_monthly_charges"}.issubset(result.columns):
        result["monthly_charge_change"] = (
            pd.to_numeric(result["monthly_charges"], errors="coerce")
            - pd.to_numeric(result["previous_monthly_charges"], errors="coerce")
        )
    blocked = {
        column
        for column in result.columns
        if _normalized_name(column) in NON_FEATURE_COLUMNS or _is_protected(column)
    }
    return result.drop(columns=list(blocked), errors="ignore")


def _encode_target(target: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(target):
        values = set(target.dropna().unique())
        if values.issubset({0, 1}):
            return target.astype("Int64").astype(int)

    positive = {"1", "true", "yes", "y", "churn", "churned"}
    negative = {"0", "false", "no", "n", "stay", "retained"}
    normalized = target.astype("string").str.strip().str.lower()
    unknown = set(normalized.dropna().unique()) - positive - negative
    if unknown:
        raise ValueError(f"Unrecognized {TARGET} labels: {sorted(unknown)}")
    return normalized.isin(positive).astype(int)


def risk_tier(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def _make_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        remainder="drop",
    )


def _make_estimator(
    numeric_columns: list[str], categorical_columns: list[str], positive_weight: float
) -> StackingClassifier:
    def branch(classifier: Any) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocess", _make_preprocessor(numeric_columns, categorical_columns)),
                ("classifier", classifier),
            ]
        )

    estimators = [
        (
            "random_forest",
            branch(
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
            ),
        ),
        (
            "xgboost",
            branch(
                XGBClassifier(
                    n_estimators=140,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=positive_weight,
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=RANDOM_STATE,
                )
            ),
        ),
        (
            "logistic_regression",
            branch(
                LogisticRegression(
                    class_weight="balanced", max_iter=1500, random_state=RANDOM_STATE
                )
            ),
        ),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1500, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
    )


def train_model(frame: pd.DataFrame) -> ModelBundle:
    required = {CUSTOMER_ID, TARGET, TENURE_DAYS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {sorted(missing)}")

    eligible = frame.loc[
        pd.to_numeric(frame[TENURE_DAYS], errors="coerce") >= MINIMUM_TENURE_DAYS
    ].copy()
    eligible = eligible.loc[eligible[TARGET].notna()].reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No labeled training rows meet the 30-day tenure rule.")

    labels = _encode_target(eligible[TARGET])
    if labels.nunique() != 2 or labels.value_counts().min() < 10:
        raise ValueError("Training requires at least 10 examples from each churn class.")

    features = prepare_features(eligible).drop(
        columns=[TARGET], errors="ignore"
    )
    feature_columns = list(features.columns)
    if not feature_columns:
        raise ValueError("No eligible model features remain after policy exclusions.")

    numeric_columns = list(features.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]
    train_x, valid_x, train_y, valid_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    positive_weight = float((train_y == 0).sum() / max((train_y == 1).sum(), 1))
    model = _make_estimator(numeric_columns, categorical_columns, positive_weight)
    model.fit(train_x, train_y)

    probabilities = model.predict_proba(valid_x)[:, 1]
    high_risk = probabilities >= HIGH_RISK_THRESHOLD
    metrics = {
        "roc_auc": float(roc_auc_score(valid_y, probabilities)),
        "high_tier_recall": float(recall_score(valid_y, high_risk, zero_division=0)),
        "high_tier_precision": float(precision_score(valid_y, high_risk, zero_division=0)),
        "high_tier_share": float(high_risk.mean()),
        "holdout_rows": float(len(valid_y)),
    }
    return ModelBundle(model, feature_columns, numeric_columns, categorical_columns, metrics)


def save_model(bundle: ModelBundle, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)


def load_model(path: str | Path) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise ValueError("Model artifact is not a compatible churn model bundle.")
    return bundle


def _as_feature_frame(frame: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    prepared = prepare_features(frame)
    missing = set(bundle.feature_columns) - set(prepared.columns)
    if missing:
        raise ValueError(f"Scoring data is missing model features: {sorted(missing)}")
    features = prepared.loc[:, bundle.feature_columns].copy()
    for column in bundle.numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in bundle.categorical_columns:
        features[column] = features[column].astype("object")
    return features


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "opted_out"}
    )


def _outreach_eligibility(frame: pd.DataFrame, as_of: date) -> pd.Series:
    opted_out = (
        _truthy(frame[MARKETING_OPTOUT])
        if MARKETING_OPTOUT in frame
        else pd.Series(False, index=frame.index)
    )
    if LAST_OFFER_DATE in frame:
        last_offer = pd.to_datetime(frame[LAST_OFFER_DATE], errors="coerce", utc=True)
        cutoff = pd.Timestamp(as_of, tz="UTC") - pd.Timedelta(days=90)
        recent_offer = last_offer.notna() & (last_offer > cutoff)
    else:
        recent_offer = pd.Series(False, index=frame.index)
    return ~opted_out & ~recent_offer


def _explain_drivers(
    bundle: ModelBundle, features: pd.DataFrame, max_evals: int
) -> list[list[str]]:
    import shap

    if features.empty:
        return []
    background = features.sample(min(len(features), 40), random_state=RANDOM_STATE)
    category_values: dict[str, list[Any]] = {}
    encoded_features = features.copy()
    for column in bundle.categorical_columns:
        categories = list(pd.unique(features[column].dropna()))
        category_values[column] = categories
        category_codes = {value: index for index, value in enumerate(categories)}
        encoded_features[column] = features[column].map(category_codes).astype(float)
    encoded_background = encoded_features.loc[background.index]

    def churn_probability(values: Any) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            encoded_batch = values.copy()
        else:
            encoded_batch = pd.DataFrame(values, columns=bundle.feature_columns)
        batch = encoded_batch.copy()
        for column, categories in category_values.items():
            codes = pd.to_numeric(encoded_batch[column], errors="coerce")
            batch[column] = codes.map(
                lambda code: categories[int(round(code))]
                if pd.notna(code) and 0 <= int(round(code)) < len(categories)
                else np.nan
            )
        for column in bundle.numeric_columns:
            batch[column] = pd.to_numeric(batch[column], errors="coerce")
        for column in bundle.categorical_columns:
            batch[column] = batch[column].astype("object")
        return bundle.model.predict_proba(batch.loc[:, bundle.feature_columns])[:, 1]

    explainer = shap.Explainer(
        churn_probability,
        shap.maskers.Independent(encoded_background, max_samples=len(encoded_background)),
        algorithm="permutation",
        feature_names=bundle.feature_columns,
    )
    explanation = explainer(
        encoded_features,
        max_evals=max(max_evals, 2 * len(bundle.feature_columns) + 1),
    )
    values = np.asarray(explanation.values)
    drivers: list[list[str]] = []
    for row in values:
        positive_indices = np.flatnonzero(row > 0)
        ranked = sorted(positive_indices, key=lambda index: row[index], reverse=True)
        if len(ranked) < 3:
            remaining = [index for index in np.argsort(np.abs(row))[::-1] if index not in ranked]
            ranked.extend(remaining[: 3 - len(ranked)])
        drivers.append(
            [f"{bundle.feature_columns[index]} ({row[index]:+.3f})" for index in ranked[:3]]
        )
    return drivers


def score_customers(
    frame: pd.DataFrame,
    bundle: ModelBundle,
    as_of: date | None = None,
    include_explanations: bool = True,
    shap_max_evals: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {CUSTOMER_ID, TENURE_DAYS, LIFETIME_VALUE}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Scoring data is missing required columns: {sorted(missing)}")
    if frame[CUSTOMER_ID].isna().any() or frame[CUSTOMER_ID].duplicated().any():
        raise ValueError("customer_id must be present and unique for every scoring row.")

    output = frame.copy()
    output["churn_probability"] = np.nan
    output["risk_tier"] = "Not scored"
    eligible = pd.to_numeric(output[TENURE_DAYS], errors="coerce") >= MINIMUM_TENURE_DAYS
    eligible_rows = output.loc[eligible]
    features = _as_feature_frame(eligible_rows, bundle)
    if not features.empty:
        probabilities = bundle.model.predict_proba(features)[:, 1]
        output.loc[eligible, "churn_probability"] = probabilities
        output.loc[eligible, "risk_tier"] = [risk_tier(value) for value in probabilities]
        if include_explanations:
            drivers = _explain_drivers(bundle, features, shap_max_evals)
        else:
            drivers = [[] for _ in range(len(features))]
        output.loc[eligible, "top_churn_drivers"] = [
            __import__("json").dumps(row) for row in drivers
        ]
    else:
        output["top_churn_drivers"] = "[]"

    output["outreach_eligible"] = _outreach_eligibility(output, as_of or date.today())
    output["expected_revenue_saved"] = (
        pd.to_numeric(output["churn_probability"], errors="coerce").fillna(0)
        * pd.to_numeric(output[LIFETIME_VALUE], errors="coerce").fillna(0)
    )
    candidates = output.loc[
        (output["risk_tier"] == "High") & output["outreach_eligible"]
    ].copy()
    candidates["recommended_action"] = candidates["top_churn_drivers"].map(
        _recommended_action
    )
    candidates = candidates.sort_values("expected_revenue_saved", ascending=False)
    task_columns = [
        CUSTOMER_ID,
        "churn_probability",
        "expected_revenue_saved",
        "top_churn_drivers",
        "recommended_action",
    ]
    return output, candidates[task_columns].reset_index(drop=True)


def _recommended_action(serialized_drivers: str) -> str:
    drivers = serialized_drivers.lower()
    if any(term in drivers for term in ("charge", "price", "monthly")):
        return "Review price or discount offer"
    if any(term in drivers for term in ("support", "complaint", "ticket", "call")):
        return "Customer care service-recovery call"
    return "Analyst review"