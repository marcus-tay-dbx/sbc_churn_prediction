# Databricks notebook source
# MAGIC %md
# MAGIC # SBC Workshop — Bank Churn Prediction
# MAGIC
# MAGIC An end-to-end machine-learning workshop for **retail-bank customer attrition (churn)**.
# MAGIC It walks through the full Databricks ML lifecycle — data generation, feature engineering,
# MAGIC training, inference, observability, monitoring, retraining, and deployment.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Catalog / schema** | `solution_builder.sbc_churn_prediction` |
# MAGIC | **Compute** | Serverless, **ML Environment Version 5** (declared in each notebook's `# /// script` header) |
# MAGIC | **Target** | `churned` (binary, ~8.6% positive — realistic, imbalanced) |
# MAGIC | **Cleanup tag** | every asset tagged `sbc` + `churn-prediction` |
# MAGIC
# MAGIC **The goal:** flag high-value, at-risk customers **before** they leave, so the bank can
# MAGIC intervene. Attrition skews toward valuable, longer-tenured customers holding maturing CDs.
# MAGIC A representative hero account is **`CUST-0000214`** (12-year affluent, ~$650K CD maturing
# MAGIC in days). We build a model to identify these customers early.
# MAGIC
# MAGIC The model trains on features from the churn feature table; the fields used to derive the churn label are excluded (leakage guard).

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML lifecycle → notebook agenda
# MAGIC
# MAGIC The **ML lifecycle** (left column) maps to the notebook that implements it. Notebooks are
# MAGIC numbered 00–09 in run order; Notebook 07 spans three lifecycle stages (its sections
# MAGIC A–C, D, E). Stages tagged **② BAU** run against the deployed endpoint's live traffic;
# MAGIC the rest are **① dev / pre-deployment**, against the registered `@dev` model.
# MAGIC
# MAGIC | ML lifecycle stage | # | Notebook | Topics | Key ML features |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | **Ingest & prepare data** | 00 | **Setup** | Generates raw data inline, flattens 6 datasets → `customer_churn`, builds the `DA` config + shared imports + loads `df` | Unity Catalog, Delta, synthetic data |
# MAGIC | **Explore data** | 01 | **Overview** | Notebook UI, multi-language, Genie Code intro | Workspace basics |
# MAGIC | **Explore data** | 02 | **EDA with Genie Code** | Profiling, target distribution, correlation, outliers | Agent-mode EDA |
# MAGIC | **Engineer features** | 03 | **Feature Engineering** | Feature variables, engineered `balanceCategory`, publish feature table, Lakebase synced table | Feature Store, FeatureLookup, Lakebase |
# MAGIC | **Train & register** | 04 | **Model Training** | Train **RandomForest + XGBoost**, log both runs, register **best by PR-AUC** | MLflow tracking + UC registry |
# MAGIC | **Validate** ① dev | 07 A–C | **Observability & Continuous Retrain** | MLflow compare · feature importance + SHAP · held-out eval → `customer_churn_eval_log` | SHAP, confusion matrix |
# MAGIC | **Batch inference** | 05 | **Batch Inference** | Score the book via `fe.score_batch` | Feature-store batch scoring |
# MAGIC | **Deploy & serve** ② BAU | 06 | **Real-Time Inference** | Serving endpoint (champion version) + AI Gateway inference logging | Model Serving, inference tables |
# MAGIC | **Monitor** ② BAU | 07 D | **Observability & Continuous Retrain** | Lakehouse Monitoring — unpack payloads + create monitor | Drift monitoring |
# MAGIC | **Retrain** ② BAU | 07 E | **Observability & Continuous Retrain** | Retrain-on-drift gate + job → runs Notebook 04 | Continuous training |
# MAGIC | **Deploy (CD)** ② BAU | 08 | **Continuous Deployment** | Deployment job + `08-MLFlow-Evaluate/Approve/Deploy` tasks | Evaluate → Approve → Deploy, champion alias, endpoint migration |
# MAGIC | **Others** | 09 | **Compute Resource Selection** | Recommended compute per workload: interactive, batch/scheduled, feature store, online store, serving | Serverless, Jobs, Lakebase, Model Serving |
# MAGIC | **Others** | 10 | **Databricks Apps** | Retention Cockpit app — churn prediction + Lakebase + Databricks Apps | Model Serving, Lakebase, Apps |
# MAGIC | **Others** | 11 | **Lakeflow Designer** | Visual pipeline authoring in Lakeflow Designer | Lakeflow, declarative pipelines |
# MAGIC
# MAGIC **Metrics:** PR-AUC is primary (imbalanced target); `recall_pos`/`precision_pos` track the
# MAGIC churn class; `test_f1` (macro) is the number the Approve gate reads.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to run
# MAGIC
# MAGIC 1. Run **Notebook 01 → 06** in order.
# MAGIC 2. Then **Notebook `07-Observability-and-Continuous-Retrain`**.
# MAGIC 3. Then **Notebook `08-Continuous-Deployment`**.
# MAGIC
# MAGIC Every notebook first runs Notebook `00-Setup`, which provides the shared config, imports, and the loaded data.
# MAGIC
# MAGIC > ⚠️ **Steps done in the UI** — the notebook cells alone won't complete these, so don't skip them if you Run All:
# MAGIC > - **Notebook 02** — Genie Code EDA in the Agent-mode panel
# MAGIC > - **Notebook 03** — create the Lakebase synced (online) feature table
# MAGIC > - **Notebook 06** — (optional) create the serving endpoint from the Serving UI
# MAGIC > - **Notebook 07** — enable the Lakehouse monitor (table's Quality tab) and open its dashboard
# MAGIC > - **Notebook 08** — approve the model in the deployment job's **Approve** task

# COMMAND ----------

# MAGIC %md
# MAGIC ## Two workshop scenarios: with vs. without Lakebase
# MAGIC
# MAGIC Real-time inference (Notebook 06) adapts to whether an **online feature store in Lakebase** exists.
# MAGIC A single online-store flag — checked once in Notebook 06's *Check Online Feature Store* cell
# MAGIC (does the synced feature table exist?) — drives **both deployment and querying**:
# MAGIC
# MAGIC **① With Lakebase (recommended)** — the synced/online feature table exists:
# MAGIC - Notebook 06 deploys the **feature-store model** (`customer_churn_model`); the endpoint looks features up by `customer_id` in Lakebase (low latency).
# MAGIC - You query by passing just `customer_id`.
# MAGIC
# MAGIC **② Without Lakebase (demo workaround)** — no synced table:
# MAGIC - Notebook 06 registers + deploys the **no-feature-store model** (`customer_churn_model_no_feature_store`) — a plain model whose inputs are the 10 feature columns.
# MAGIC - You query by passing the **10 feature values directly** (read from the offline feature table).
# MAGIC
# MAGIC **How the flag works:** the same online-store flag chooses (a) which model the deploy cell (B1) deploys and
# MAGIC (b) how the query cell (C1) queries — so toggling Lakebase flips the whole notebook consistently. Create the
# MAGIC synced feature table in Notebook 03 to move from Scenario ② → ①.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Assets created (all tagged `sbc` + `churn-prediction`)
# MAGIC
# MAGIC | Type | Name |
# MAGIC | --- | --- |
# MAGIC | Tables | `customer_churn`, `customer_churn_features`, `customer_churn_eval_log` |
# MAGIC | Volume | `raw_data` (6 raw parquet datasets) |
# MAGIC | Feature synced table | `customer_churn_features_synced` (Lakebase) |
# MAGIC | Model | `customer_churn_model` (aliases `dev`, `champion`) |
# MAGIC | Experiment | `experiments/bank-churn-training` |
# MAGIC | Endpoint | `customer_churn_endpoint` (+ AI Gateway logging → `churn_endpoint_payload`) |
# MAGIC | Jobs | Deployment job (Evaluate→Approve→Deploy); Retrain-on-Drift job (paused) |
# MAGIC
# MAGIC **Cleanup:** find everything by the `sbc` / `churn-prediction` tags, then drop.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Related
# MAGIC A **Retention Cockpit** Streamlit app (`sbc-churn-cockpit`) consumes this model — reading
# MAGIC features from Lakebase and scoring on the endpoint. See the app project's own README.
# MAGIC
# MAGIC ### Notes
# MAGIC - The endpoint returns a **binary** class (0/1). A probability gauge would need a `predict_proba` pyfunc (deferred).
# MAGIC - Autoscaling **Lakebase** synced tables / app resources are attached via the **UI** (CLI recognizes only provisioned instances).
# MAGIC - Inference logging uses **AI Gateway inference tables** (legacy `auto_capture_config` is deprecated).
