# SBC Workshop — Bank Churn Prediction

An end-to-end machine-learning workshop for **retail-bank customer attrition (churn)**.
It walks through the full Databricks ML lifecycle — data generation, feature engineering,
training, inference, observability, monitoring, retraining, and deployment.

| | |
| --- | --- |
| **Catalog / schema** | `solution_builder.sbc_churn_prediction` |
| **Compute** | Serverless, **ML Environment Version 5** (declared in each notebook's `# /// script` header) |
| **Target** | `churned` (binary, ~8.6% positive — realistic, imbalanced) |
| **Cleanup tag** | every asset tagged `sbc` + `churn-prediction` |

**The goal:** flag high-value, at-risk customers **before** they leave, so the bank can
intervene. Attrition skews toward valuable, longer-tenured customers holding maturing CDs.
A representative hero account is **`CUST-0000214`** (12-year affluent, ~$650K CD maturing
in days). We build a model to identify these customers early.

**10 model features:** `tenure_years, total_balance_usd, num_products, num_deposit_products,
has_maturing_cd, txn_count_60d, total_outflow_60d_usd, withdrawal_count_60d, tier_rank, balanceCategory`.
`attrition_risk_score` / `balance_outflow_30d_usd` derive the label, so they are **excluded** (leakage guard).

## ML lifecycle → notebook agenda

The **ML lifecycle** (left column) maps to the notebook that implements it. Notebooks are
numbered `00`–`09` in run order; notebook `07` spans three lifecycle stages (its sections
A–C, D, E). Stages tagged **② BAU** run against the deployed endpoint's live traffic;
the rest are **① dev / pre-deployment**, against the registered `@dev` model.

| ML lifecycle stage | # | Notebook | Topics | Key ML features |
| --- | --- | --- | --- | --- |
| **Ingest & prepare data** | 00 | **Setup** | Generates raw data inline, flattens 6 datasets → `customer_churn`, builds the `DA` config + shared imports + loads `df` | Unity Catalog, Delta, synthetic data |
| **Explore data** | 01 | **Overview** | Notebook UI, multi-language, Genie Code intro | Workspace basics |
| **Explore data** | 02 | **EDA with Genie Code** | Profiling, target distribution, correlation, outliers | Agent-mode EDA |
| **Engineer features** | 03 | **Feature Engineering** | Feature variables, engineered `balanceCategory`, publish feature table, Lakebase synced table | Feature Store, FeatureLookup, Lakebase |
| **Train & register** | 04 | **Model Training** | Train **RandomForest + XGBoost**, log both runs, register **best by PR-AUC** | MLflow tracking + UC registry |
| **Validate** ① dev | 07 A–C | **Observability** | MLflow compare · feature importance + SHAP · held-out eval → `customer_churn_eval_log` | SHAP, confusion matrix |
| **Batch inference** | 05 | **Batch Inference** | Score the book via `fe.score_batch` | Feature-store batch scoring |
| **Deploy & serve** ② BAU | 06 | **Real-Time Inference** | Serving endpoint (champion version) + AI Gateway inference logging | Model Serving, inference tables |
| **Monitor** ② BAU | 07 D | **Observability** | Lakehouse Monitoring — unpack payloads + create monitor | Drift monitoring |
| **Retrain** ② BAU | 07 E | **Observability** | Retrain-on-drift gate + job → runs `04` | Continuous training |
| **Deploy (CD)** ② BAU | 08 | **MLflow (Deployment Jobs)** | Deployment job + `08-MLFlow-Evaluate/Approve/Deploy` tasks | Evaluate → Approve → Deploy, champion alias, endpoint migration |
| **Others** | 09 | **Migration** | Placeholder (TBC) | — |
| **Others** | 10 | **Databricks Apps** | Retention Cockpit app — churn prediction + Lakebase + Databricks Apps | Model Serving, Lakebase, Apps |
| **Others** | 11 | **Lakeflow Designer** | Visual pipeline authoring in Lakeflow Designer | Lakeflow, declarative pipelines |

**Metrics:** PR-AUC is primary (imbalanced target); `recall_pos`/`precision_pos` track the
churn class; `test_f1` (macro) is the number the Approve gate reads.

## How to run

1. Open **`00-Setup`** on Serverless **ML v5** and run it first.
2. Run **`01` → `07`** in order, then **`08`**.

Every notebook starts with `%run "./00-Setup"`, which provides the `DA` config, shared imports, and the loaded `df`.

> ⚠️ **Some exercises are done only in the UI** (e.g. Genie Code EDA, enabling the
> Lakehouse monitor, approving in the deployment job). If you use **Run All**, be careful
> not to skip these UI actions — the notebook cells alone won't complete them.

## Assets created (all tagged `sbc` + `churn-prediction`)

| Type | Name |
| --- | --- |
| Tables | `customer_churn`, `customer_churn_features`, `customer_churn_eval_log` |
| Volume | `raw_data` (6 raw parquet datasets) |
| Feature synced table | `customer_churn_features_synced` (Lakebase) |
| Model | `bank_churn_model` (aliases `dev`, `champion`) |
| Experiment | `experiments/bank-churn-training` |
| Endpoint | `sbc-bank-churn-<user>` (+ AI Gateway logging → `churn_endpoint_payload`) |
| Jobs | Deployment job (Evaluate→Approve→Deploy); Retrain-on-Drift job (paused) |

**Cleanup:** find everything by the `sbc` / `churn-prediction` tags, then drop.

---

### Related
A **Retention Cockpit** Streamlit app (`sbc-churn-cockpit`) consumes this model — reading
features from Lakebase and scoring on the endpoint. See the app project's own README.

### Notes
- The endpoint returns a **binary** class (0/1). A probability gauge would need a `predict_proba` pyfunc (deferred).
- Autoscaling **Lakebase** synced tables / app resources are attached via the **UI** (CLI recognizes only provisioned instances).
- Inference logging uses **AI Gateway inference tables** (legacy `auto_capture_config` is deprecated).
