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

The model trains on features from the churn feature table; the fields used to derive the churn label are excluded (leakage guard).

## ML lifecycle → notebook agenda

The **ML lifecycle** (left column) maps to the notebook that implements it. Notebooks are
numbered 00–09 in run order; Notebook 07 spans three lifecycle stages (its sections
A–C, D, E). Stages tagged **② BAU** run against the deployed endpoint's live traffic;
the rest are **① dev / pre-deployment**, against the registered `@dev` model.

| ML lifecycle stage | # | Notebook | Topics | Key ML features |
| --- | --- | --- | --- | --- |
| **Ingest & prepare data** | 00 | **Setup** | Generates raw data inline, flattens 6 datasets → `customer_churn`, builds the `DA` config + shared imports + loads `df` | Unity Catalog, Delta, synthetic data |
| **Explore data** | 01 | **Overview** | Notebook UI, multi-language, Genie Code intro | Workspace basics |
| **Explore data** | 02 | **EDA with Genie Code** | Profiling, target distribution, correlation, outliers | Agent-mode EDA |
| **Engineer features** | 03 | **Feature Engineering** | Feature variables, engineered `balanceCategory`, publish feature table, Lakebase synced table | Feature Store, FeatureLookup, Lakebase |
| **Train & register** | 04 | **Model Training** | Train **RandomForest + XGBoost**, log both runs, register **best by PR-AUC** | MLflow tracking + UC registry |
| **Validate** ① dev | 07 A–C | **Observability & Continuous Retrain** | **A.** MLflow Experiment Tracking & Run Comparison · **B.** Feature Importance & Model Explainability (SHAP) · **C.** Offline Quality Validation (held-out eval) → `customer_churn_eval_log` | SHAP, confusion matrix |
| **Batch inference** | 05 | **Batch Inference** | Score the book via `fe.score_batch` | Feature-store batch scoring |
| **Deploy & serve** ② BAU | 06 | **Real-Time Inference** | **A.** Introduction to Databricks Model Serving · **B.** Create a Model Serving Endpoint (champion) · **C.** Query the Serving Endpoint · **D.** Conclusion — with AI Gateway inference logging | Model Serving, inference tables |
| **Monitor** ② BAU | 07 D | **Observability & Continuous Retrain** | **D.** Lakehouse Monitoring of Live Predictions — D1 unpack payloads · D2 enable monitor · D3 refresh | Drift monitoring |
| **Retrain** ② BAU | 07 E | **Observability & Continuous Retrain** | **E.** Retrain on Drift — drift gate + job → runs Notebook 04 | Continuous training |
| **Deploy (CD)** ② BAU | 08 | **Continuous Deployment** | **A.** Create a Deployment Job — Evaluate→Approve→Deploy tasks (`08-MLFlow-Evaluate/Approve/Deploy`) | Evaluate → Approve → Deploy, champion alias, endpoint migration |
| **Others** | 09 | **Compute Resource Selection** | Recommended compute per workload: interactive, batch/scheduled, feature store, online store, serving | Serverless, Jobs, Lakebase, Model Serving |
| **Others** | 10 | **Databricks Apps** | Retention Cockpit app — churn prediction + Lakebase + Databricks Apps | Model Serving, Lakebase, Apps |
| **Others** | 11 | **Lakeflow Designer** | Visual pipeline authoring in Lakeflow Designer | Lakeflow, declarative pipelines |

**Metrics:** PR-AUC is primary (imbalanced target); `recall_pos`/`precision_pos` track the
churn class; `test_f1` (macro) is the number the Approve gate reads.

## How to run

1. Run **Notebook 01 → 06** in order.
2. Then **Notebook `07-Observability-and-Continuous-Retrain`**.
3. Then **Notebook `08-Continuous-Deployment`**.

Every notebook first runs Notebook `00-Setup`, which provides the shared config, imports, and the loaded data.

> ⚠️ **Steps done in the UI** — these require manual creation in the UI, so don't skip them if you Run All:
> - **Notebook 02** — Genie Code EDA in the Agent-mode panel
> - **Notebook 03** — create the Lakebase synced (online) feature table
>
> Notebooks 06, 07, and 08 also have UI equivalents (serving endpoint, Lakehouse monitor, model approval), but those are **supplemental** — the notebook scripts create them for you.

## Two workshop scenarios: with vs. without Lakebase

Real-time inference (Notebook 06) adapts to whether an **online feature store in Lakebase** exists.
A single online-store flag — checked once in Notebook 06's *Check Online Feature Store* cell
(does the synced feature table exist?) — drives **both deployment and querying**:

**① With Lakebase (recommended)** — the synced/online feature table exists:
- Notebook 06 deploys the **feature-store model** (`customer_churn_model`); the endpoint looks features up by `customer_id` in Lakebase (low latency).
- You query by passing just `customer_id`.

**② Without Lakebase (demo workaround)** — no synced table:
- Notebook 06 registers + deploys the **no-feature-store model** (`customer_churn_model_no_feature_store`) — a plain model whose inputs are the 10 feature columns.
- You query by passing the **10 feature values directly** (read from the offline feature table).

**How the flag works:** the same online-store flag chooses (a) which model the deploy cell (B1) deploys and
(b) how the query cell (C1) queries — so toggling Lakebase flips the whole notebook consistently. Create the
synced feature table in Notebook 03 to move from Scenario ② → ①.

---

## Assets created (all tagged `sbc` + `churn-prediction`)

| Type | Name |
| --- | --- |
| Tables | `customer_churn`, `customer_churn_features`, `customer_churn_eval_log` |
| Volume | `raw_data` (6 raw parquet datasets) |
| Feature synced table | `customer_churn_features_synced` (Lakebase) |
| Model | `customer_churn_model` (aliases `dev`, `champion`) |
| Experiment | `experiments/bank-churn-training` |
| Endpoint | `customer_churn_endpoint` (+ AI Gateway logging → `churn_endpoint_payload`) |
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
