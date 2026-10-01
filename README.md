# Telecom Churn Prediction

A Python starter project for identifying telecom customers at risk of cancelling within 60 days. It trains a stacked ensemble of Random Forest, XGBoost, and Logistic Regression models, assigns risk tiers, calculates customer-level SHAP explanations, and prepares budget-ranked retention task exports.

> **Demo data only:** This repository does not include real customer or warehouse data. The included generator produces synthetic examples so you can run the workflow. Demo model scores are not production evidence and must not be used to make customer outreach decisions.

## Features

- Stacked churn classifier and holdout AUC / High-tier recall reporting.
- Risk tiers: High (`>= 0.70`), Medium (`0.40` to `< 0.70`), and Low (`< 0.40`).
- Minimum-tenure rule: customers under 30 days are not scored.
- Protected demographic fields are excluded from model features.
- SHAP-based top drivers, calculated by default for CLI scoring and on demand in the dashboard.
- Outreach suppression for marketing opt-outs and customers with an offer in the previous 90 days.
- Expected-value ranking using churn probability multiplied by customer lifetime value.
- Streamlit analyst dashboard with graphical risk summaries, filters, customer explanations, and CSV exports.

## Requirements

- Windows PowerShell instructions are shown below; Python 3.10 or later is required.
- Git is only needed if you plan to publish the project to GitHub.

## Quick Start

Run these commands from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Generate synthetic training and scoring CSVs:

```powershell
python scripts/make_demo_data.py --output data/demo_customers.csv
Import-Csv data/demo_customers.csv |
    Select-Object * -ExcludeProperty churn_60d |
    Export-Csv data/demo_scoring.csv -NoTypeInformation
```

Train, then score the separate unlabeled snapshot:

```powershell
churn-model train --input data/demo_customers.csv --model models/churn.joblib
churn-model score --input data/demo_scoring.csv --model models/churn.joblib --output outputs/scores.csv --top-n 60
```

Training saves the model at `models/churn.joblib`. The CLI exits with code `2` and prints an alert if holdout AUC is below `0.70` or High-tier recall is below `0.75`. This is an approval warning, not necessarily a training failure. The synthetic data may fail these targets.

## IBM Kaggle Benchmark

The [IBM Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) can be used for an exploratory benchmark. Download and extract `WA_Fn-UseC_-Telco-Customer-Churn.csv` to `data/ibm_telco/`, then run:

```powershell
python scripts/benchmark_ibm_telco.py
```

The benchmark adapter excludes demographic fields, cleans `TotalCharges`, and converts tenure months to approximate days for the existing 30-day eligibility rule. The downloaded data remains local: all of `data/` is ignored by Git and is not included in this repository.

**This is not a 60-day churn evaluation.** Kaggle describes one row per customer and a general `Churn` label; it provides no monthly snapshot date or measured 60-day outcome. On the downloaded 7,043-row file, the current stratified random holdout had 1,407 rows and produced AUC **0.840**. The fixed 0.70 cutoff selected 102 customers (**7.2%**) with **81.4% precision** and **22.2% recall**. Selecting exactly the top 5% by score (71 holdout customers) yielded **83.1% precision** and **15.8% recall**. The holdout score distribution was min **0.027**, median **0.169**, p90 **0.664**, p95 **0.721**, max **0.793**; tier shares were Low **68.2%**, Medium **24.6%**, High **7.2%**. These figures are for a public static benchmark only; do not treat them as approval or expected production performance.

### Benchmark Insights and Recommendations

- The AUC indicates useful ranking on this static dataset, but does not establish calibrated 60-day probabilities or campaign value.
- The fixed 0.70 cutoff exceeds a 5% contact capacity (7.2% selected). Restricting selection to the top 5% slightly increased precision (81.4% to 83.1%) but reduced recall (22.2% to 15.8%). This is a capacity-versus-capture tradeoff, not evidence that either policy is optimal.
- Before selecting a threshold, compare precision, recall, and selected volume across multiple validation splits; calibrate probabilities on separate validation data. Do not tune and report performance on the same holdout.
- For the stated 60-day use case, replace this benchmark evaluation with approved, dated historical snapshots and mature 60-day outcomes. Use a chronological holdout; if customers have repeated snapshots, keep each customer isolated between training and validation.
- Before claiming retention impact or ROI, run a controlled campaign experiment with an appropriate control group and track offer cost, incremental retention, and revenue outcomes.

## Dashboard

### Demo Screenshot

The screenshot below shows the dashboard populated with synthetic demo data. Its model metrics are illustrative only and are not approved for customer outreach.

![Telecom churn retention dashboard demo](docs/images/churn-dashboard-demo.png)

Start the local dashboard:

```powershell
streamlit run app/dashboard.py
```

Open the local URL printed by Streamlit. In the sidebar:

1. Choose **Train from history** and upload `data/demo_customers.csv`, or choose **Load saved model** to load `models/churn.joblib`.
2. Upload `data/demo_scoring.csv` under **Active customer CSV**.
3. Select **Score customers**.

The dashboard provides a risk overview with tier mix, churn-score distribution, estimated value by tier, and average churn risk by contract. The campaign queue ranks eligible High-tier customers under a monthly budget cap. The customer explorer calculates SHAP drivers for a selected customer. Score and CRM task tables can be downloaded as CSV files.

Use `demo_customers.csv` for **training** because it contains the `churn_60d` outcome. Use `demo_scoring.csv` for **scoring** because it intentionally has no outcome label. The dashboard reports a clear message if the unlabeled scoring file is mistakenly selected for training.

## CSV Data Contract

### Preparing Historical Customer Snapshots

Use [`templates/historical_customer_snapshots.csv`](templates/historical_customer_snapshots.csv) as a header-only starting point for an approved warehouse extract. It contains no customer records. Populate one row per customer per monthly snapshot, keeping the same pseudonymous `customer_id` across that customer's snapshots.

- Set `snapshot_date` to the date the features describe. Feature values must be known as of that date; do not include information recorded afterward.
- Set `churn_60d` to `1` only when the customer churned during the 60 days after that snapshot, and `0` when they remained active for the full window. Leave out snapshots whose full 60-day outcome window has not elapsed.
- Keep `tenure_days` and the feature values as of the snapshot. `last_offer_date` must also be the date known at that time.
- Use the approved features your organization permits. Do not include direct identifiers or protected demographic attributes. The template's `customer_id` should be pseudonymized consistently, not replaced with names, phone numbers, or account numbers.
- Keep the populated file local and access-controlled. The entire `data/` directory is ignored by Git; do not commit customer data or upload it to a public repository.

`snapshot_date` is treated as metadata and excluded from the model features. **Important:** the current trainer evaluates with a stratified random row split. If multiple monthly rows for each customer are included, snapshots from the same customer can appear in both training and holdout sets. Do not treat those metrics as production validation; a customer-grouped or time-based holdout must be implemented and used before approving a model trained on repeated snapshots.

Training data must contain:

| Column | Description |
| --- | --- |
| `customer_id` | Unique customer identifier |
| `tenure_days` | Customer tenure at the historical snapshot |
| `churn_60d` | Whether the customer churned within 60 days of that snapshot; accepts binary or common yes/no labels |

At least 10 examples of each outcome must remain after applying the 30-day tenure rule. Multiple historical snapshots can improve coverage across time, but use a customer-grouped or chronological holdout before trusting evaluation metrics when customers have repeated rows.

Scoring data must contain `customer_id`, `tenure_days`, `customer_lifetime_value`, and every model feature used at training time. Useful feature examples include `contract_type`, `monthly_charges`, `previous_monthly_charges`, `support_calls_90d`, and usage aggregates. `monthly_charge_change` is derived when both current and previous monthly charges are present.

Optional scoring columns:

| Column | Purpose |
| --- | --- |
| `marketing_opt_out` | Excludes opted-out customers from outreach tasks, but not scoring |
| `last_offer_date` | Suppresses outreach if a retention offer was sent within 90 days |

Identifiers, labels/outcomes, customer lifetime value, opt-out status, offer dates, and detected protected demographic columns are not used as model features. Review the protected-column denylist against your organization’s compliance requirements before deployment.

## Validation and Tests

Run the test suite:

```powershell
python -m pytest -q
```

The holdout metrics are a basic development check. The current training implementation uses a stratified random split; production approval should use a time-separated holdout, representative data, and agreed monitoring thresholds. Recall/AUC do not establish campaign lift or ROI; use a controlled campaign experiment to measure those outcomes.

## Outputs

- `models/churn.joblib`: trained model artifact.
- `outputs/scores.csv`: customer scores, tiers, drivers, eligibility, and expected-value estimates.
- `outputs/scores_crm_tasks.csv`: eligible High-tier CRM task candidates, ranked by expected value.

The dashboard’s download buttons save files through the browser rather than automatically writing them to `outputs/`. The CLI writes its outputs to the paths passed on the command line. CRM, warehouse, dashboard hosting, alerting, and campaign outcome integrations are not included in this local starter project.

## Publish to GitHub

Generated customer CSVs, trained model artifacts, and output files are ignored by `.gitignore`. Do not commit real customer data, model artifacts containing sensitive information, credentials, or warehouse exports.

Create an empty GitHub repository, then from this project folder run:

```powershell
git init
git add .
git status
git commit -m "Add telecom churn prediction starter"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repository>.git
git push -u origin main
```

Replace the remote URL with your repository URL. Review `git status` before committing to confirm no private data or generated artifacts are staged.
