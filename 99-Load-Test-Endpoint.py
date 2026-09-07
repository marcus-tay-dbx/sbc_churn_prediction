# Databricks notebook source
# /// script
# dependencies = [
#   "databricks-feature-engineering",
# ]
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "databricks-feature-engineering",
# ]
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 99 - Endpoint Load Test (Inference Traffic Generator)
# MAGIC
# MAGIC This notebook **generates inference traffic** against the deployed churn prediction endpoint.
# MAGIC It is purpose-built for **demo preparation**: sending enough requests — with real feature
# MAGIC values in the payload — so that:
# MAGIC
# MAGIC 1. The **AI Gateway inference table** (`churn_endpoint_payload`) fills up with scored records
# MAGIC 2. The **D1 unpack cell** in `07-Observability-and-Continuous-Retrain` can extract feature + prediction rows
# MAGIC 3. The **Lakehouse Monitor** has enough data to produce `_drift_metrics`
# MAGIC 4. The **drift gate** in `07-Observability-Retrain-Drift-Gate` trips and triggers the **Deployment Job** (Evaluate → Approve → Deploy)
# MAGIC
# MAGIC > **Prerequisites**: Run `06-Real-Time-Inference` first (endpoint deployed, AI Gateway inference
# MAGIC > logging enabled). After this notebook completes, wait ~10–30 min for the AI Gateway to flush
# MAGIC > payloads, then re-run Section D1 in `07-Observability-and-Continuous-Retrain` to unpack them.
# MAGIC
# MAGIC ### Why features must be in the payload
# MAGIC The Lakehouse Monitor detects **feature drift** by comparing the distribution of input features
# MAGIC across time windows. That means the features must be present in the request payload. This
# MAGIC notebook always uses **Option 2** (pass features directly) — every request contains all 10
# MAGIC feature values, so the D1 unpack cell can extract them even without an online feature store.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Widgets
# Configure how many requests to generate and how to vary them.
dbutils.widgets.text("sample_size",           "1000",  "Customers to sample from feature table")
dbutils.widgets.text("batch_size",            "25",    "Customers per endpoint request")
dbutils.widgets.text("n_rounds",              "3",     "Rounds (each round = full sample cycle)")
dbutils.widgets.dropdown("add_drift_cohort", "true",  ["true", "false"],
                         "Add drifted cohort (simulate distribution shift)")
dbutils.widgets.text("mlflow_run_name", "load-test-inference", "MLflow run name")

sample_size      = int(dbutils.widgets.get("sample_size") or "1000")
batch_size       = int(dbutils.widgets.get("batch_size") or "25")
n_rounds         = int(dbutils.widgets.get("n_rounds") or "3")
add_drift_cohort = dbutils.widgets.get("add_drift_cohort").lower() == "true"
run_name         = dbutils.widgets.get("mlflow_run_name") or "load-test-inference"

print(f"sample_size:      {sample_size}")
print(f"batch_size:       {batch_size}")
print(f"n_rounds:         {n_rounds}")
print(f"add_drift_cohort: {add_drift_cohort}")
print(f"Total requests ≈  {sample_size * n_rounds // batch_size} batches "
      f"({sample_size * n_rounds:,} predictions)")

# COMMAND ----------

# DBTITLE 1,Validate Endpoint
from mlflow.deployments import get_deploy_client
import time

client = get_deploy_client("databricks")
endpoint_name = DA.endpoint_name

try:
    ep = client.get_endpoint(endpoint_name)
    state = ep.get("state", {}).get("ready", "unknown")
    print(f"Endpoint '{endpoint_name}' → state: {state}")
    if state != "READY":
        print("⚠️  Endpoint is not READY yet. Wait for it to finish scaling, then re-run.")
        dbutils.notebook.exit("endpoint_not_ready")
except Exception as e:
    print(f"❌ Cannot reach endpoint: {e}")
    print("   Run 06-Real-Time-Inference first to create the endpoint.")
    dbutils.notebook.exit("endpoint_not_found")

# COMMAND ----------

# DBTITLE 1,Load Feature Data
import pandas as pd
import numpy as np

feature_table = DA.feature_table_name
feature_cols  = DA.feature_columns

print(f"Loading features from: {feature_table}")
all_features = (
    spark.table(feature_table)
    .select("customer_id", *feature_cols)
    .toPandas()
)
print(f"Total customers available: {len(all_features):,}")

# Sample `sample_size` customers (deterministic seed for reproducibility).
if sample_size < len(all_features):
    sample_df = all_features.sample(n=sample_size, random_state=42).reset_index(drop=True)
else:
    sample_df = all_features.copy()
print(f"Working sample:  {len(sample_df):,} customers")
print(f"Features:        {feature_cols}")

# COMMAND ----------

# DBTITLE 1,Build Drifted Cohort (Distribution Shift Simulation)
# MAGIC %md
# MAGIC ### Why add a drifted cohort?
# MAGIC
# MAGIC The Lakehouse Monitor measures drift by comparing the **distribution of features and
# MAGIC predictions** in recent time windows against a baseline. If every request uses the same
# MAGIC feature values from the training set, drift will be close to zero and the gate never trips.
# MAGIC
# MAGIC To make drift visible in the metrics, we inject a cohort with **shifted feature values** —
# MAGIC higher balances, more recent outflow, higher tier — that push the prediction distribution
# MAGIC toward churn. This simulates what real drift looks like (a competitor rate promotion that
# MAGIC changes the at-risk population) and ensures the `js_distance` in the drift-metrics table
# MAGIC is non-zero.

# COMMAND ----------

# DBTITLE 1,Drifted Cohort Builder
if add_drift_cohort:
    drifted = sample_df.copy()

    # Simulate a competitor-promo scenario: higher-value customers show elevated outflow.
    drifted["total_outflow_60d_usd"]  = drifted["total_outflow_60d_usd"] * 3.5 - 50_000
    drifted["total_balance_usd"]      = drifted["total_balance_usd"] * 1.6
    drifted["has_maturing_cd"]        = 1
    drifted["tier_rank"]              = drifted["tier_rank"].clip(lower=1) + 1
    drifted["withdrawal_count_60d"]   = drifted["withdrawal_count_60d"] + 8
    drifted["txn_count_60d"]          = drifted["txn_count_60d"] + 5
    # Assign fictional "drifted" customer IDs so they appear distinct in the payload table.
    drifted["customer_id"] = drifted["customer_id"].str.replace("CUST-", "CUST-D-", regex=False)

    print(f"Drifted cohort: {len(drifted):,} customers (shifted distributions)")
    print(f"  Avg total_outflow_60d_usd:  original {sample_df['total_outflow_60d_usd'].mean():,.0f} "
          f"→ drifted {drifted['total_outflow_60d_usd'].mean():,.0f}")
    print(f"  Avg total_balance_usd:      original {sample_df['total_balance_usd'].mean():,.0f} "
          f"→ drifted {drifted['total_balance_usd'].mean():,.0f}")
    print(f"  has_maturing_cd = 1 for all drifted rows (was {sample_df['has_maturing_cd'].mean():.1%})")
else:
    drifted = pd.DataFrame(columns=sample_df.columns)
    print("Drifted cohort disabled — using original features only.")

# COMMAND ----------

# DBTITLE 1,Run Load Test
import math
from datetime import datetime

def _send_batch(records: list) -> dict:
    """Send one batch to the endpoint; return raw response dict."""
    payload = {"dataframe_records": records}
    return client.predict(endpoint=endpoint_name, inputs=payload)


def _records_from_df(df: pd.DataFrame, start: int, end: int) -> list:
    """Slice df[start:end] into a list of feature dicts (all values cast to Python native types)."""
    slice_df = df.iloc[start:end].copy()
    # Cast int64/float64 columns to native Python types for JSON serialisation.
    for col in feature_cols:
        if slice_df[col].dtype in (np.int64, np.int32):
            slice_df[col] = slice_df[col].astype(int)
        else:
            slice_df[col] = slice_df[col].astype(float)
    return slice_df[["customer_id"] + feature_cols].to_dict("records")


total_predictions = 0
total_batches     = 0
churn_predictions = 0
errors            = 0
round_summaries   = []

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(DA.experiment_path)

print("=" * 65)
print(f"Starting load test  — {datetime.now().strftime('%H:%M:%S')}")
print(f"  Endpoint:   {endpoint_name}")
print(f"  Rounds:     {n_rounds}  (original + drifted cohorts per round)")
print(f"  Batch size: {batch_size}")
print("=" * 65)

with mlflow.start_run(run_name=run_name, tags={"purpose": "load-test", "sbc": "true", "churn-prediction": "true"}) as run:
    mlflow.log_params({
        "endpoint_name":    endpoint_name,
        "sample_size":      sample_size,
        "batch_size":       batch_size,
        "n_rounds":         n_rounds,
        "add_drift_cohort": add_drift_cohort,
        "feature_table":    feature_table,
    })

    all_preds_log = []   # collect sample predictions for MLflow artifact

    for round_idx in range(1, n_rounds + 1):
        round_preds  = 0
        round_churn  = 0
        round_errors = 0

        # Each round: original cohort first, then drifted cohort (if enabled).
        cohorts = [(sample_df, "original")]
        if add_drift_cohort:
            cohorts.append((drifted, "drifted"))

        for cohort_df, cohort_name in cohorts:
            n = len(cohort_df)
            n_batches = math.ceil(n / batch_size)

            for b in range(n_batches):
                start = b * batch_size
                end   = min(start + batch_size, n)
                records = _records_from_df(cohort_df, start, end)

                try:
                    resp = _send_batch(records)
                    preds = resp.get("predictions", [])
                    n_churn = sum(1 for p in preds if p == 1 or p == 1.0)
                    round_preds  += len(preds)
                    round_churn  += n_churn
                    total_predictions += len(preds)
                    churn_predictions += n_churn
                    total_batches     += 1

                    # Keep a sample of predictions for the MLflow artifact (first 200 per round).
                    if len(all_preds_log) < 200:
                        for rec, pred in zip(records, preds):
                            all_preds_log.append({
                                "round":       round_idx,
                                "cohort":      cohort_name,
                                "customer_id": rec.get("customer_id"),
                                "prediction":  pred,
                            })

                except Exception as e:
                    round_errors += 1
                    errors += 1
                    if round_errors <= 3:
                        print(f"  ⚠️  Batch error (round {round_idx}, batch {b}): {e}")

            # Brief pause between cohorts to spread timestamps.
            time.sleep(0.5)

        churn_rate = round_churn / round_preds if round_preds > 0 else 0
        round_summaries.append({
            "round": round_idx, "predictions": round_preds,
            "churn_rate": churn_rate, "errors": round_errors,
        })
        print(f"  Round {round_idx}/{n_rounds}: {round_preds:,} predictions "
              f"({churn_rate:.1%} churn)  errors={round_errors}")

        # Log per-round metrics.
        mlflow.log_metrics({
            f"round_{round_idx}_predictions": round_preds,
            f"round_{round_idx}_churn_rate":  churn_rate,
        }, step=round_idx)

        # Small delay between rounds so AI Gateway timestamps spread across the window.
        if round_idx < n_rounds:
            time.sleep(1)

    # Log aggregate metrics.
    overall_churn_rate = churn_predictions / total_predictions if total_predictions > 0 else 0
    mlflow.log_metrics({
        "total_predictions":       total_predictions,
        "total_batches":           total_batches,
        "total_churn_predictions": churn_predictions,
        "overall_churn_rate":      overall_churn_rate,
        "total_errors":            errors,
    })

    # Log the prediction sample as a CSV artifact.
    if all_preds_log:
        import tempfile, os
        sample_pdf = pd.DataFrame(all_preds_log)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "prediction_sample.csv")
            sample_pdf.to_csv(path, index=False)
            mlflow.log_artifact(path, artifact_path="inference_samples")
        print(f"\nLogged {len(all_preds_log)} sample predictions to MLflow artifact 'inference_samples'.")

    run_id = run.info.run_id

print("\n" + "=" * 65)
print(f"Load test complete — {datetime.now().strftime('%H:%M:%S')}")
print(f"  Total predictions:   {total_predictions:,}")
print(f"  Overall churn rate:  {overall_churn_rate:.1%}")
print(f"  Total batches sent:  {total_batches:,}")
print(f"  Errors:              {errors}")
print(f"  MLflow run ID:       {run_id}")
print("=" * 65)

# COMMAND ----------

# DBTITLE 1,Round Summary
summary_df = spark.createDataFrame(pd.DataFrame(round_summaries))
display(summary_df)

# COMMAND ----------

# DBTITLE 1,Check Payload Table
# MAGIC %md
# MAGIC ## What happens next?
# MAGIC
# MAGIC The AI Gateway **batches** inference table writes — rows appear in
# MAGIC `churn_endpoint_payload` on a delay (typically **10–30 minutes**).
# MAGIC
# MAGIC Once they land:
# MAGIC 1. Re-run **Section D1** in `07-Observability-and-Continuous-Retrain` to unpack the payload
# MAGIC    into `customer_churn_inference_unpacked`.
# MAGIC 2. The Lakehouse Monitor will refresh on its schedule (or trigger manually from the Quality
# MAGIC    tab) to produce `..._drift_metrics`.
# MAGIC 3. The **drift gate** (`07-Observability-Retrain-Drift-Gate`) will read `js_distance` from
# MAGIC    `_drift_metrics` and, with `drift_threshold = 0.0`, will **always trip** → retraining via
# MAGIC    `04-Model-Training`.
# MAGIC 4. A new model version is registered → the **Deployment Job** auto-fires
# MAGIC    (Evaluate → Approve → Deploy).

# COMMAND ----------

# DBTITLE 1,Payload Table Row Count
payload_table  = f"{DA.catalog_name}.{DA.schema_name}.churn_endpoint_payload"
unpacked_table = f"{DA.catalog_name}.{DA.schema_name}.customer_churn_inference_unpacked"

print("Current payload table status:")
if spark.catalog.tableExists(payload_table):
    n = spark.table(payload_table).count()
    print(f"  ✅ {payload_table}: {n:,} rows")
    if n == 0:
        print("     → Rows appear 10–30 min after requests. Re-check later.")
    else:
        print("     → Re-run Section D1 in 07-Observability-and-Continuous-Retrain to unpack.")
else:
    print(f"  ⏳ {payload_table} not yet created (AI Gateway hasn't flushed yet).")
    print("     → Re-check after 15 minutes.")

if spark.catalog.tableExists(unpacked_table):
    n2 = spark.table(unpacked_table).count()
    print(f"  ✅ {unpacked_table}: {n2:,} rows (already unpacked)")
else:
    print(f"  ⏳ {unpacked_table} — run Section D1 in 07-Observability-and-Continuous-Retrain after payload arrives.")

# COMMAND ----------

# DBTITLE 1,Optional - Trigger Monitor Refresh
# MAGIC %md
# MAGIC ### Optional: Trigger the monitor refresh manually
# MAGIC
# MAGIC After you've unpacked the payload (Section D1), you can trigger the monitor refresh manually
# MAGIC instead of waiting for the scheduled run. Run the cell below **only after** the unpacked table
# MAGIC exists with rows.

# COMMAND ----------

# DBTITLE 1,Trigger Monitor Refresh (run after D1 unpack)
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
unpacked_table_full = f"{DA.catalog_name}.{DA.schema_name}.customer_churn_inference_unpacked"

try:
    monitor = w.quality_monitors.get(unpacked_table_full)
    print(f"Monitor found on {unpacked_table_full}.")
    w.quality_monitors.run_refresh(table_name=unpacked_table_full)
    print("Monitor refresh triggered.")
    print("Check the Quality tab on the table in Catalog Explorer for status.")
    print("Drift metrics will appear in:")
    print(f"  {unpacked_table_full}_profile_metrics")
    print(f"  {unpacked_table_full}_drift_metrics")
except Exception as e:
    print(f"⚠️  Could not trigger monitor refresh: {e}")
    print("   Make sure the monitor is created (Section D2 of 07-Observability-and-Continuous-Retrain)")
    print("   and the unpacked table has rows before running this cell.")

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook sent **inference traffic** to the churn endpoint with feature values
# MAGIC embedded in every payload, enabling the full demo flow:
# MAGIC
# MAGIC | Step | Where | What |
# MAGIC |------|-------|------|
# MAGIC | 1 | This notebook | Generate traffic (original + drifted cohorts) |
# MAGIC | 2 | Wait 10–30 min | AI Gateway flushes to `churn_endpoint_payload` |
# MAGIC | 3 | `07` Section D1 | Unpack payload → `customer_churn_inference_unpacked` |
# MAGIC | 4 | `07` Section D2 | Monitor refresh → `_drift_metrics` populated |
# MAGIC | 5 | Drift Gate notebook | `js_distance ≥ 0.0` → `retrain_needed = true` |
# MAGIC | 6 | Retrain job | `04-Model-Training` runs → new model version registered |
# MAGIC | 7 | Deployment Job | Auto-fires: Evaluate → Approve → Deploy → champion |
# MAGIC
# MAGIC The MLflow run logs:
# MAGIC - Per-round prediction counts and churn rates
# MAGIC - A sample CSV of predictions (artifact: `inference_samples/prediction_sample.csv`)
# MAGIC - Total prediction volume and error count
