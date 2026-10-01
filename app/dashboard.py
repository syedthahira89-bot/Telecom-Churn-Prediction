"""Local analyst dashboard for churn scoring and retention prioritization."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

from churn_model.pipeline import (
    CUSTOMER_ID,
    LIFETIME_VALUE,
    TARGET,
    load_model,
    save_model,
    score_customers,
    train_model,
)

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Retention Desk | Telecom Churn",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
    :root {
        --ink: #192c2b;
        --muted: #657674;
        --line: #dce5e1;
        --paper: #f4f7f4;
        --white: #ffffff;
        --green: #1c6553;
        --lime: #d4e96a;
        --coral: #db705b;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    [data-testid="stSidebar"] { background: #eaf0ec; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    h1, h2, h3 { color: var(--ink); }
    h1, .editorial { font-family: 'Newsreader', Georgia, serif !important; letter-spacing: 0; }
    h1 { font-size: 2.55rem !important; font-weight: 500 !important; }
    h2 { font-size: 1.45rem !important; }
    [data-testid="stMetric"] { background: var(--white); border: 1px solid var(--line); padding: 1rem 1.1rem; border-radius: 6px; }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .78rem; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'DM Mono', monospace; font-size: 1.65rem; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
    div.stButton > button[kind="primary"] { background: var(--green); border-color: var(--green); }
    div.stButton > button { border-radius: 5px; }
    .brand-mark { color: var(--green); font: 500 .75rem 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
    .run-stamp { color: var(--muted); font: .76rem 'DM Mono', monospace; }
    .section-rule { border-top: 1px solid var(--line); margin: 1.2rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _state_defaults() -> None:
    defaults = {
        "bundle": None,
        "model_metrics": None,
        "model_origin": None,
        "scores": None,
        "tasks": None,
        "snapshot": None,
        "scored_at": None,
        "explanation_customer": None,
        "explanation": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_uploaded_csv(upload: object) -> pd.DataFrame | None:
    if upload is None:
        return None
    try:
        return pd.read_csv(upload)
    except Exception as error:
        st.error(f"Could not read CSV: {error}")
        return None


def _handle_model_controls() -> None:
    st.sidebar.markdown('<div class="brand-mark">Model & data</div>', unsafe_allow_html=True)
    model_mode = st.sidebar.radio("Model source", ["Train from history", "Load saved model"])
    if model_mode == "Train from history":
        training_upload = st.sidebar.file_uploader(
            "Labeled training CSV",
            type=["csv"],
            key="training_csv",
            help="Historical training data must include churn_60d, the known 60-day churn outcome.",
        )
        st.sidebar.caption(
            "Training demo: data/demo_customers.csv · Scoring only: data/demo_scoring.csv"
        )
        if training_upload and st.sidebar.button("Train model", type="primary", use_container_width=True):
            try:
                history = pd.read_csv(training_upload)
                if TARGET not in history.columns:
                    raise ValueError(
                        f"'{training_upload.name}' has no '{TARGET}' column. "
                        "Training needs labeled historical outcomes. For the demo, upload "
                        "data/demo_customers.csv here; data/demo_scoring.csv is unlabeled "
                        "and belongs under Active customer CSV."
                    )
                with st.spinner("Fitting the stacked ensemble and checking holdout performance..."):
                    bundle = train_model(history)
                    save_model(bundle, ROOT / "models" / "churn.joblib")
                st.session_state.bundle = bundle
                st.session_state.model_metrics = bundle.holdout_metrics
                st.session_state.model_origin = "Newly trained model"
                st.session_state.scores = None
                st.session_state.tasks = None
                st.session_state.snapshot = None
                st.session_state.explanation = None
                st.sidebar.success("Model trained and saved to models/churn.joblib")
            except Exception as error:
                st.sidebar.error(f"Training failed: {error}")
    else:
        model_path = st.sidebar.text_input("Model artifact path", "models/churn.joblib")
        resolved_path = Path(model_path)
        if not resolved_path.is_absolute():
            resolved_path = ROOT / resolved_path
        model_exists = resolved_path.is_file()
        if not model_exists:
            st.sidebar.info(
                "No model file at this path. Train from a labeled history CSV, "
                "or enter the path to an existing .joblib model."
            )
        if st.sidebar.button(
            "Load model",
            type="primary",
            use_container_width=True,
            disabled=not model_exists,
        ):
            try:
                st.session_state.bundle = load_model(resolved_path)
                st.session_state.model_metrics = st.session_state.bundle.holdout_metrics
                st.session_state.model_origin = str(resolved_path.relative_to(ROOT)) if resolved_path.is_relative_to(ROOT) else str(resolved_path)
                st.session_state.scores = None
                st.session_state.tasks = None
                st.session_state.snapshot = None
                st.session_state.explanation = None
                st.sidebar.success("Model loaded")
            except Exception as error:
                st.sidebar.error(f"Could not load model: {error}")

    if st.session_state.bundle is not None:
        st.sidebar.caption(f"Active: {st.session_state.model_origin}")
        metrics = st.session_state.model_metrics or {}
        if metrics:
            st.sidebar.caption(
                f"Holdout AUC {metrics.get('roc_auc', float('nan')):.2f} · "
                f"High recall {metrics.get('high_tier_recall', float('nan')):.0%}"
            )
            if metrics.get("roc_auc", 0) < 0.70 or metrics.get("high_tier_recall", 0) < 0.75:
                st.sidebar.warning(
                    "Training succeeded and the model was saved, but it failed approval checks: "
                    f"AUC {metrics.get('roc_auc', 0):.3f} (target >= 0.70), "
                    f"High-tier recall {metrics.get('high_tier_recall', 0):.0%} "
                    "(target >= 75%). Do not use it for real campaigns; validate a model "
                    "trained on representative historical data."
                )


def _handle_scoring_controls() -> None:
    st.sidebar.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="brand-mark">Monthly snapshot</div>', unsafe_allow_html=True)
    scoring_upload = st.sidebar.file_uploader(
        "Active customer CSV", type=["csv"], key="scoring_csv"
    )
    if scoring_upload and st.session_state.bundle is not None:
        if st.sidebar.button("Score customers", type="primary", use_container_width=True):
            try:
                snapshot = pd.read_csv(scoring_upload)
                with st.spinner("Scoring active customers..."):
                    scores, tasks = score_customers(
                        snapshot, st.session_state.bundle, include_explanations=False
                    )
                st.session_state.snapshot = snapshot
                st.session_state.scores = scores
                st.session_state.tasks = tasks
                st.session_state.scored_at = datetime.now().astimezone()
                st.session_state.explanation = None
                st.sidebar.success(f"Scored {len(scores):,} customer records")
            except Exception as error:
                st.sidebar.error(f"Scoring failed: {error}")
    elif scoring_upload:
        st.sidebar.info("Load or train a model before scoring.")


def _format_money(value: float) -> str:
    return f"${value:,.0f}"


def _render_header() -> None:
    st.markdown('<div class="brand-mark">Telecom · Retention operations</div>', unsafe_allow_html=True)
    st.title("Retention desk")
    st.caption("Monthly churn risk and outreach prioritization")
    if st.session_state.scored_at:
        st.markdown(
            f'<div class="run-stamp">LAST SCORED · {st.session_state.scored_at:%d %b %Y, %H:%M %Z}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="run-stamp">NO MONTHLY RUN LOADED</div>', unsafe_allow_html=True)


def _render_overview(scores: pd.DataFrame) -> None:
    scored = scores.loc[scores["risk_tier"] != "Not scored"]
    high = scores.loc[scores["risk_tier"] == "High"]
    eligible_high = high.loc[high["outreach_eligible"]]
    top_value = eligible_high["expected_revenue_saved"].sum()

    metrics = st.columns(4)
    metrics[0].metric("Customers scored", f"{len(scored):,}")
    metrics[1].metric("High risk", f"{len(high):,}", f"{len(high) / max(len(scored), 1):.1%} of scored")
    metrics[2].metric("Eligible high risk", f"{len(eligible_high):,}", f"{len(high) - len(eligible_high):,} suppressed")
    metrics[3].metric("At-risk value", _format_money(top_value), "risk × lifetime value")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    tier_order = ["High", "Medium", "Low", "Not scored"]
    tier_colors = alt.Scale(
        domain=tier_order,
        range=["#db705b", "#d5a746", "#1c6553", "#a6b2ad"],
    )
    chart_columns = st.columns(3, gap="large")

    with chart_columns[0]:
        st.subheader("Risk mix")
        tier_counts = (
            scores.groupby("risk_tier", as_index=False)
            .size()
            .rename(columns={"size": "Customers"})
        )
        risk_mix = (
            alt.Chart(tier_counts)
            .mark_arc(innerRadius=56, outerRadius=88, stroke="#f4f7f4", strokeWidth=2)
            .encode(
                theta=alt.Theta("Customers:Q", stack=True),
                color=alt.Color("risk_tier:N", scale=tier_colors, title="Tier"),
                tooltip=["risk_tier:N", "Customers:Q"],
            )
            .properties(height=230)
        )
        st.altair_chart(risk_mix, width="stretch")
        st.caption(f"{len(scores) - len(scored):,} are below 30 days' tenure.")

    with chart_columns[1]:
        st.subheader("Churn score spread")
        probability_data = scored[["churn_probability"]].dropna().copy()
        if probability_data.empty:
            st.info("No customers meet the scoring tenure minimum.")
        else:
            probability_data["Churn probability (%)"] = probability_data["churn_probability"] * 100
            score_spread = (
                alt.Chart(probability_data)
                .mark_bar(color="#1c6553", opacity=0.85)
                .encode(
                    x=alt.X(
                        "Churn probability (%):Q",
                        bin=alt.Bin(maxbins=14),
                        title="Churn probability (%)",
                    ),
                    y=alt.Y("count():Q", title="Customers"),
                    tooltip=[alt.Tooltip("count():Q", title="Customers")],
                )
                .properties(height=230)
            )
            st.altair_chart(score_spread, width="stretch")
        st.caption("Distribution of model probabilities across scored customers.")

    with chart_columns[2]:
        st.subheader("Expected value by tier")
        value_by_tier = (
            scores.groupby("risk_tier", as_index=False)["expected_revenue_saved"]
            .sum()
            .rename(columns={"expected_revenue_saved": "Expected value ($)"})
        )
        value_chart = (
            alt.Chart(value_by_tier)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Expected value ($):Q", title="Risk × lifetime value ($)"),
                y=alt.Y("risk_tier:N", sort=tier_order, title=None),
                color=alt.Color("risk_tier:N", scale=tier_colors, legend=None),
                tooltip=["risk_tier:N", alt.Tooltip("Expected value ($):Q", format="$,.0f")],
            )
            .properties(height=230)
        )
        st.altair_chart(value_chart, width="stretch")
        st.caption("Prioritization estimate, not guaranteed savings.")

    if "contract_type" in scores.columns and not scored.empty:
        contract_data = scored.groupby("contract_type", as_index=False).agg(
            customers=(CUSTOMER_ID, "count"),
            average_risk=("churn_probability", "mean"),
        )
        contract_data["Average churn risk (%)"] = contract_data["average_risk"] * 100
        st.subheader("Average churn risk by contract")
        contract_chart = (
            alt.Chart(contract_data)
            .mark_bar(color="#1c6553", cornerRadiusEnd=4)
            .encode(
                x=alt.X("Average churn risk (%):Q", title="Average churn probability (%)"),
                y=alt.Y("contract_type:N", sort="-x", title=None),
                tooltip=[
                    "contract_type:N",
                    "customers:Q",
                    alt.Tooltip("Average churn risk (%):Q", format=".1f"),
                ],
            )
            .properties(height=max(135, min(300, len(contract_data) * 45)))
        )
        st.altair_chart(contract_chart, width="stretch")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.subheader("Customers to review")
    with st.container():
        controls = st.columns([1, 1.2])
        selected_tiers = controls[0].multiselect(
            "Risk tier", tier_order, default=["High", "Medium"]
        )
        minimum_value = controls[1].number_input(
            "Minimum lifetime value ($)", min_value=0, value=0, step=250
        )
        filtered = scores.loc[
            scores["risk_tier"].isin(selected_tiers)
            & (pd.to_numeric(scores[LIFETIME_VALUE], errors="coerce").fillna(0) >= minimum_value)
        ].sort_values("expected_revenue_saved", ascending=False)
        display_columns = [
            CUSTOMER_ID,
            "churn_probability",
            "risk_tier",
            LIFETIME_VALUE,
            "outreach_eligible",
            "expected_revenue_saved",
        ]
        display_scores = filtered[display_columns].copy()
        display_scores["churn_probability"] *= 100
        st.dataframe(
            display_scores,
            hide_index=True,
            width="stretch",
            height=380,
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn risk", format="%.0f%%", min_value=0, max_value=100
                ),
                LIFETIME_VALUE: st.column_config.NumberColumn(format="$%.0f"),
            },
        )
        st.download_button(
            "Download filtered scores",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="churn_scores.csv",
            mime="text/csv",
            icon=":material/download:",
        )


def _render_campaign_queue(scores: pd.DataFrame, tasks: pd.DataFrame) -> None:
    st.subheader("Retention campaign queue")
    high_count = len(tasks)
    default_cap = min(high_count, max(1, math.ceil(len(scores) * 0.05)))
    budget = st.number_input(
        "Monthly outreach budget · customers",
        min_value=0,
        max_value=high_count,
        value=default_cap,
        step=1,
        help="Defaults to approximately 5% of the scored base and ranks by expected revenue saved.",
    )
    queue = tasks.head(int(budget)).copy()
    metrics = st.columns(3)
    metrics[0].metric("Eligible High tier", f"{high_count:,}")
    metrics[1].metric("Tasks in budget", f"{len(queue):,}")
    metrics[2].metric("Expected value in queue", _format_money(queue["expected_revenue_saved"].sum()))

    display_queue = queue.copy()
    display_queue["churn_probability"] *= 100
    st.dataframe(
        display_queue,
        hide_index=True,
        width="stretch",
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "Churn risk", format="%.0f%%", min_value=0, max_value=100
            ),
            "expected_revenue_saved": st.column_config.NumberColumn(format="$%.0f"),
        },
    )
    st.download_button(
        "Export CRM task CSV",
        data=queue.to_csv(index=False).encode("utf-8"),
        file_name="retention_tasks.csv",
        mime="text/csv",
        icon=":material/download:",
        disabled=queue.empty,
    )


def _render_customer_explorer(scores: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    st.subheader("Customer explanation")
    query = st.text_input("Find customer ID", placeholder="Start typing an ID")
    candidates = scores
    if query:
        candidates = scores.loc[
            scores[CUSTOMER_ID].astype(str).str.contains(query, case=False, regex=False)
        ]
    if candidates.empty:
        st.info("No matching customer in this scoring run.")
        return

    customer_ids = candidates[CUSTOMER_ID].astype(str).tolist()
    selected_id = st.selectbox("Customer", customer_ids, label_visibility="collapsed")
    score_row = scores.loc[scores[CUSTOMER_ID].astype(str) == selected_id].iloc[0]
    source_row = snapshot.loc[snapshot[CUSTOMER_ID].astype(str) == selected_id].head(1)
    details = st.columns(4)
    details[0].metric("Risk tier", str(score_row["risk_tier"]))
    probability = score_row["churn_probability"]
    details[1].metric("Churn probability", "—" if pd.isna(probability) else f"{probability:.1%}")
    details[2].metric("Lifetime value", _format_money(float(score_row[LIFETIME_VALUE] or 0)))
    details[3].metric("Outreach", "Eligible" if score_row["outreach_eligible"] else "Suppressed")

    if score_row["risk_tier"] == "Not scored":
        st.info("This customer has not met the 30-day tenure minimum.")
        return
    if st.button("Calculate SHAP drivers", type="primary", icon=":material/insights:"):
        try:
            with st.spinner("Calculating customer-level explanation..."):
                explained, _ = score_customers(
                    source_row,
                    st.session_state.bundle,
                    include_explanations=True,
                    shap_max_evals=100,
                )
            serialized = explained.iloc[0]["top_churn_drivers"]
            st.session_state.explanation_customer = selected_id
            st.session_state.explanation = json.loads(serialized)
        except Exception as error:
            st.error(f"Could not calculate SHAP drivers: {error}")

    if st.session_state.explanation_customer == selected_id and st.session_state.explanation:
        rows = []
        for item in st.session_state.explanation:
            match = re.match(r"(.+) \(([+-]?\d+(?:\.\d+)?)\)$", item)
            if match:
                rows.append({"Driver": match.group(1), "Contribution": float(match.group(2))})
        if rows:
            st.caption("Positive contribution raises the churn score; negative contribution lowers it.")
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True,
                column_config={"Contribution": st.column_config.NumberColumn(format="%+.3f")},
            )


def main() -> None:
    _state_defaults()
    _handle_model_controls()
    _handle_scoring_controls()
    _render_header()

    if st.session_state.scores is None:
        st.info("Load a model and monthly customer snapshot from the left panel to view a scoring run.")
        return

    scores = st.session_state.scores
    tasks = st.session_state.tasks
    snapshot = st.session_state.snapshot
    overview_tab, campaign_tab, customer_tab = st.tabs(
        ["Risk overview", "Campaign queue", "Customer explorer"]
    )
    with overview_tab:
        _render_overview(scores)
    with campaign_tab:
        _render_campaign_queue(scores, tasks)
    with customer_tab:
        _render_customer_explorer(scores, snapshot)


if __name__ == "__main__":
    main()