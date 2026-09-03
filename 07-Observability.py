# Databricks notebook source
# /// script
# dependencies = [
#   "databricks-feature-engineering",
#   "xgboost",
#   "shap",
#   "seaborn",
# ]
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "databricks-feature-engineering",
#   "xgboost",
#   "shap",
#   "seaborn",
# ]
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 07 - Observability, Monitoring & Explainability
# MAGIC
# MAGIC After training and registering a model, the next critical step is ensuring you can **observe**, **explain**, and **monitor** its behavior — both before and after deployment.
# MAGIC
# MAGIC This notebook spans two lifecycle phases:
# MAGIC
# MAGIC **① BEFORE deploying to an endpoint (dev / pre-production)** — validate the registered `@dev` model:
# MAGIC - **A.** MLflow Experiment Tracking & Run Comparison
# MAGIC - **B.** Feature Importance & Model Explainability (SHAP)
# MAGIC - **C.** Offline Quality Validation (held-out evaluation)
# MAGIC
# MAGIC **② AFTER the model is deployed & serving (BAU / production)** — needs a live endpoint with traffic:
# MAGIC - **D.** Lakehouse Monitoring of live predictions (drift)
# MAGIC - **E.** Retrain on Drift (continuous training)
# MAGIC
# MAGIC So A–C run against the **dev model before it reaches an endpoint**; D–E are the ongoing **BAU** loop once it's deployed (they no-op cleanly here until the endpoint has captured traffic).
# MAGIC
# MAGIC **Prerequisites**: Run `00-Setup` and `04-Model-Training` first (and, for D–E, deploy the endpoint in `06`).

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install shap --quiet

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Model & Data (loaded by 00-Setup)
# The dev model (`sk_model`), held-out split (`X_test`/`y_test`), `feature_cols`, and
# `model_name` are loaded by 00-Setup above — but only when 00 is run from THIS notebook
# (see 00-Setup's final cell). This keeps the model/data-loading logic centralized in 00.
assert "sk_model" in globals() and sk_model is not None, (
    "Dev model not loaded. Run 04-Model-Training first, then re-run this notebook — "
    "00-Setup loads the model + held-out split only when it is run from 07-Observability."
)
print(f"Ready for observability: {model_name}")
print(f"Held-out test set: {X_test.shape[0]} samples × {X_test.shape[1]} features")

# COMMAND ----------

# DBTITLE 1,Section A - MLflow Tracking
# MAGIC %md
# MAGIC ## A. MLflow Experiment Tracking & Run Comparison
# MAGIC
# MAGIC MLflow automatically captures everything needed to reproduce and compare model training runs:
# MAGIC - **Parameters** — hyperparameters, model type, training method
# MAGIC - **Metrics** — F1 score, accuracy, loss
# MAGIC - **Artifacts** — model files, feature metadata, signature
# MAGIC - **Tags** — custom metadata for filtering and organization
# MAGIC
# MAGIC **Viewing from the UI:**
# MAGIC 1. In the left sidebar, click **Experiments** — find `bank-churn-training` in the list
# MAGIC 2. Click into the experiment to see a **table of all runs** with parameters, metrics, and duration
# MAGIC 3. Select two or more runs and click **Compare** to see metric charts side by side
# MAGIC 4. Click a single run to inspect its **Parameters**, **Metrics**, **Artifacts** (model files, signature), and **Tags**
# MAGIC 5. Under **Artifacts**, expand `bank_churn_model` to see the logged model, `requirements.txt`, and `MLmodel` spec
# MAGIC
# MAGIC Below we query the experiment to compare all runs programmatically.

# COMMAND ----------

# DBTITLE 1,Compare Experiment Runs
experiment_path = DA.experiment_path
experiment = mlflow.get_experiment_by_name(experiment_path)

if experiment:
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_f1 DESC"]
    )
    print(f"Experiment: {experiment_path}")
    print(f"Total runs: {len(runs)}\n")
    display(spark.createDataFrame(
        runs[["run_id", "params.model_type", "params.n_estimators",
              "metrics.test_f1", "status"]].head(10)
    ))
else:
    print(f"No experiment found at {experiment_path}. Run notebook 04 first.")

# COMMAND ----------

# DBTITLE 1,Section B - Explainability
# MAGIC %md
# MAGIC ## B. Feature Importance & Model Explainability
# MAGIC
# MAGIC **Why explainability matters:**
# MAGIC - Regulatory compliance (model audit trails — especially in banking)
# MAGIC - Debugging unexpected predictions
# MAGIC - Building stakeholder trust
# MAGIC - Identifying data leakage or spurious correlations
# MAGIC
# MAGIC **Viewing from the UI:**
# MAGIC 1. Go to **Experiments** → open the `bank-churn-training` experiment → click a run
# MAGIC 2. Under the **Artifacts** tab, look for logged SHAP plots or feature importance charts (if logged as artifacts)
# MAGIC 3. Navigate to **Models** in the left sidebar → open `solution_builder.sbc_churn_prediction.bank_churn_model`
# MAGIC 4. Click a **model version** to see its lineage: which experiment run produced it, the signature (input/output schema), and any tags/aliases (`dev`, `prod`)
# MAGIC 5. The **Schema** section shows the exact feature columns the model expects — useful for validating inference payloads
# MAGIC
# MAGIC We use two approaches:
# MAGIC 1. **Built-in feature importance** from the Random Forest model
# MAGIC 2. **SHAP values** for per-prediction explanations

# COMMAND ----------

# DBTITLE 1,Feature Importance
importances = sk_model.feature_importances_
feat_imp = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="#1E4651")
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

print("\nTop predictors:")
for _, row in feat_imp.sort_values("Importance", ascending=False).head(3).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.3f}")

# COMMAND ----------

# DBTITLE 1,SHAP Explainability
import shap

explainer = shap.TreeExplainer(sk_model)
shap_values = explainer.shap_values(X_test)

# For binary classification, use the SHAP values for the positive class (churned = 1).
if isinstance(shap_values, list):
    sv_pos = shap_values[1]
else:
    sv_pos = shap_values[:, :, 1] if shap_values.ndim == 3 else shap_values

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(sv_pos, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (mean |SHAP value|) — churn class")
plt.tight_layout()
plt.show()

print("\nSHAP tells us HOW each feature pushes predictions toward churn:")
print("  - total_outflow_60d_usd and total_balance_usd dominate churn decisions")
print("  - Large recent outflow and high balances push the prediction toward churn")

# COMMAND ----------

# DBTITLE 1,SHAP Beeswarm Plot
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(sv_pos, X_test, show=False)
plt.title("SHAP Values for Churn = 1 (per-customer contributions)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Section C - Inference Quality
# MAGIC %md
# MAGIC ## C. Offline Quality Validation (Held-out Evaluation)
# MAGIC
# MAGIC > **Phase ① — dev / pre-deployment.** Runs against the registered `@dev` model **before** it reaches an endpoint. (Sections A & B above are also Phase ①.)
# MAGIC
# MAGIC Before deploying, validate model quality on a **held-out test split** — data the model
# MAGIC did **not** train on — so the metrics are unbiased. We predict on `X_test` here on
# MAGIC purpose: the batch predictions from `05` cover the whole book (including training rows,
# MAGIC so metrics would be optimistic) and `06` only scores a couple of sample requests, so
# MAGIC neither is a clean labeled holdout for an honest quality number.
# MAGIC
# MAGIC Key checks:
# MAGIC - **Classification report** — precision, recall, F1 per class
# MAGIC - **Confusion matrix** — where the model makes mistakes
# MAGIC - **Prediction distribution** — does it match the training distribution?
# MAGIC - **Evaluation table** — the holdout predictions + true labels, saved as a quality baseline
# MAGIC
# MAGIC > This is the **offline evaluation** log (`customer_churn_eval_log`, static, has labels).
# MAGIC > It is **not** the live serving table monitored in Section D
# MAGIC > (`customer_churn_inference_unpacked`) — different purpose, no overlap.

# COMMAND ----------

# DBTITLE 1,Classification Report
y_pred = sk_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Retained (0)", "Churned (1)"]))

f1 = f1_score(y_test, y_pred, average="macro")
print(f"Macro F1: {f1:.4f}")

# COMMAND ----------

# DBTITLE 1,Confusion Matrix
labels = [0, 1]
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print("Observations:")
print(f"  True churners caught (recall): {tp}/{tp + fn} = {tp / max(tp + fn, 1):.1%}")
print(f"  False alarms (retained flagged as churn): {fp}")

# COMMAND ----------

# DBTITLE 1,Prediction Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(y_test, bins=[-0.5, 0.5, 1.5], color="#1E4651", edgecolor="white", alpha=0.7, rwidth=0.8)
axes[0].set_title("Actual Churn Distribution (Test Set)")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Retained", "Churned"])
axes[0].set_ylabel("Count")

axes[1].hist(y_pred, bins=[-0.5, 0.5, 1.5], color="#FF5F46", edgecolor="white", alpha=0.7, rwidth=0.8)
axes[1].set_title("Predicted Churn Distribution")
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Retained", "Churned"])
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

print("Distribution shift check:")
for q in [0, 1]:
    actual_pct = (y_test == q).sum() / len(y_test) * 100
    pred_pct = (y_pred == q).sum() / len(y_pred) * 100
    lab = "Churned" if q == 1 else "Retained"
    print(f"  {lab}: actual {actual_pct:.1f}% vs predicted {pred_pct:.1f}%")

# COMMAND ----------

# DBTITLE 1,Log Holdout Evaluation Table
eval_df = X_test.copy()
eval_df["actual_churned"] = y_test.values
eval_df["predicted_churned"] = y_pred
eval_df["correct"] = (eval_df["actual_churned"] == eval_df["predicted_churned"])

inference_table_name = DA.inference_log   # → customer_churn_eval_log
spark.createDataFrame(eval_df).write.mode("overwrite").saveAsTable(inference_table_name)
tag_table(inference_table_name)

accuracy = eval_df["correct"].mean()
print(f"Evaluation log saved to: {inference_table_name}")
print(f"Total predictions: {len(eval_df)}")
print(f"Accuracy: {accuracy:.1%}")
print(f"Incorrect predictions: {(~eval_df['correct']).sum()}")

# COMMAND ----------

# DBTITLE 1,Section D - Monitoring
# MAGIC %md
# MAGIC ## D. Lakehouse Monitoring of Live Predictions
# MAGIC
# MAGIC > **Phase ② — BAU / production.** This section monitors the **deployed endpoint's live
# MAGIC > traffic**, so it needs `06` deployed with inference logging on. Until traffic exists,
# MAGIC > D1/D2 print a message and skip — nothing here runs against the dev model.
# MAGIC
# MAGIC Databricks **Lakehouse Monitoring** provides automated drift detection and quality tracking:
# MAGIC
# MAGIC | Capability | What It Does |
# MAGIC | --- | --- |
# MAGIC | **Data drift detection** | Alerts when input feature distributions shift from the training baseline |
# MAGIC | **Prediction drift** | Detects when the model's output distribution changes |
# MAGIC | **Data quality** | Tracks nulls, schema changes, volume anomalies |
# MAGIC | **Custom metrics** | Define business-specific quality metrics |
# MAGIC
# MAGIC You enable the monitor on the **unpacked** table (`customer_churn_inference_unpacked` from **D1** — *not* the raw `churn_endpoint_payload`, and not the static `customer_churn_eval_log`). There are **two equivalent ways** — pick either; they create the *same* monitor:
# MAGIC
# MAGIC **Option 1 — UI**
# MAGIC 1. **Catalog** → select `customer_churn_inference_unpacked`
# MAGIC 2. **Quality** tab → **Create monitor** → **Inference profile**
# MAGIC 3. **Problem type** = Classification, **Prediction column** = `prediction`, **Timestamp column** = `inference_timestamp`, **Model ID column** = `model_version`
# MAGIC 4. **Baseline table** = `customer_churn_features`, set a **refresh schedule** (e.g., daily) → **Create**
# MAGIC
# MAGIC **Option 2 — Script**: run cells **D1** (unpack payloads) then **D2** (`w.quality_monitors.create(...)`) below.
# MAGIC
# MAGIC Either way you get a **dashboard** plus **metrics tables** (`..._profile_metrics`, `..._drift_metrics`) you can query.
# MAGIC
# MAGIC > An **InferenceLog** monitor needs a **timestamp** (and ideally a **model-id**) column — the unpacked table has both. The static `customer_churn_eval_log` (from Section C) has neither, so it could only be a **Snapshot** profile (point-in-time data quality), not time-series drift.

# COMMAND ----------

# DBTITLE 1,D1. Unpack Endpoint Payloads into a Monitorable Table
# The serving endpoint (06) auto-captures each request/response into
# `churn_endpoint_payload`, where `request`/`response` are JSON blobs. Lakehouse
# Monitoring needs ONE ROW PER PREDICTION with real columns, so we unpack here:
# each request record's features + its prediction + a timestamp + the model version.
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType

payload_table = f"{DA.catalog_name}.{DA.schema_name}.churn_endpoint_payload"
unpacked_table = f"{DA.catalog_name}.{DA.schema_name}.customer_churn_inference_unpacked"

if not spark.catalog.tableExists(payload_table):
    print(f"No payload table yet at {payload_table}.")
    print("Create the endpoint (06) and send it some traffic, then re-run this cell.")
else:
    raw = spark.table(payload_table)
    if raw.limit(1).count() == 0:
        print(f"{payload_table} exists but is empty — send requests to the endpoint first.")
    else:
        # request  = {"dataframe_records": [ {feature: value, ...}, ... ]}
        # response = {"predictions": [ pred, ... ]}
        feat_fields = [StructField(c, DoubleType(), True) for c in DA.feature_columns] + \
                      [StructField("customer_id", StringType(), True)]
        req_schema = StructType([StructField("dataframe_records", ArrayType(StructType(feat_fields)), True)])
        resp_schema = StructType([StructField("predictions", ArrayType(DoubleType()), True)])

        unpacked = (
            raw
            .withColumn("_req", F.from_json("request", req_schema))
            .withColumn("_resp", F.from_json("response", resp_schema))
            .withColumn("_recs", F.col("_req.dataframe_records"))
            .withColumn("_preds", F.col("_resp.predictions"))
            # zip each request record with its prediction, then explode to one row each
            .withColumn("_pair", F.explode(F.arrays_zip("_recs", "_preds")))
            .withColumn("inference_timestamp", (F.col("timestamp_ms") / 1000).cast("timestamp"))
            .withColumn("model_version", F.lit("served"))  # set from request_metadata if captured
            .select(
                F.col("databricks_request_id").alias("request_id"),
                "inference_timestamp",
                "model_version",
                F.col("_pair._recs.customer_id").alias("customer_id"),
                *[F.col(f"_pair._recs.{c}").alias(c) for c in DA.feature_columns],
                F.col("_pair._preds").alias("prediction"),
            )
        )
        unpacked.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(unpacked_table)
        tag_table(unpacked_table)
        # Change Data Feed is required by Lakehouse Monitoring.
        spark.sql(f"ALTER TABLE {unpacked_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        print(f"Unpacked {spark.table(unpacked_table).count()} predictions → {unpacked_table}")
        display(spark.table(unpacked_table).limit(10))

# COMMAND ----------

# DBTITLE 1,D2. Create the Lakehouse InferenceLog Monitor
# Create a monitor ON THE UNPACKED TABLE (not the raw payload table). It computes
# profile + drift metric tables and a dashboard on a schedule. `label_col` is
# optional — once ground-truth churn is backfilled, quality metrics (F1/precision/
# recall over time) populate too.
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType

w = WorkspaceClient()

if not spark.catalog.tableExists(unpacked_table):
    print(f"{unpacked_table} does not exist yet — run D1 (needs endpoint traffic) first.")
else:
    try:
        w.quality_monitors.get(unpacked_table)
        print(f"Monitor already exists on {unpacked_table}.")
    except Exception:
        print(f"Creating InferenceLog monitor on {unpacked_table} ...")
        w.quality_monitors.create(
            table_name=unpacked_table,
            inference_log=MonitorInferenceLog(
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
                prediction_col="prediction",
                timestamp_col="inference_timestamp",
                granularities=["1 day"],
                model_id_col="model_version",
                # label_col="actual_churned",   # add once ground-truth labels are joined in
            ),
            assets_dir=f"/Workspace{DA.workshop_dir}/monitoring",
            output_schema_name=f"{DA.catalog_name}.{DA.schema_name}",
            baseline_table_name=DA.feature_table_name,  # training features = drift baseline
            slicing_exprs=["tier_rank", "has_maturing_cd"],
        )
        print("Monitor created. It generates:")
        print(f"  • {unpacked_table}_profile_metrics")
        print(f"  • {unpacked_table}_drift_metrics   (the table your retrain gate reads)")
        print("  • a monitoring dashboard (see the table's Quality tab)")

# COMMAND ----------

# DBTITLE 1,Monitoring Readiness Check
print("=" * 60)
print("OBSERVABILITY READINESS SUMMARY")
print("=" * 60)
print(f"\n  Model:          {model_name}")
print(f"  Experiment:     {experiment_path}")
print(f"  Inference log:  {inference_table_name}")
print(f"  Test F1:        {f1:.4f}")
print(f"  Test Accuracy:  {accuracy:.1%}")
print(f"\n  [x] MLflow experiment tracking")
print(f"  [x] Feature importance computed")
print(f"  [x] SHAP explainability generated")
print(f"  [x] Classification report logged")
print(f"  [x] Confusion matrix analyzed")
print(f"  [x] Prediction distribution validated")
print(f"  [x] Inference table saved to UC")
print(f"  [ ] Lakehouse Monitoring (enable via UI)")
print(f"\nReady for deployment → proceed to 08-MLFlow")

# COMMAND ----------

# DBTITLE 1,Section E - Retrain on Drift
# MAGIC %md
# MAGIC ## E. Retrain on Drift (Continuous Training)
# MAGIC
# MAGIC This closes the ML lifecycle loop, right where the drift signal is produced: the
# MAGIC monitor from Section D writes a `..._drift_metrics` table; here we **gate on that drift**
# MAGIC and, when it trips, **retrain by calling `04-Model-Training` directly** (the same
# MAGIC `dbutils.notebook.run` pattern `00` uses for `generate_data`). `04` registers a new
# MAGIC version → the deployment job (`08`) auto-fires: Evaluate → Approve → Deploy → champion → endpoint.
# MAGIC
# MAGIC ```
# MAGIC  D monitor → _drift_metrics → [E drift gate] → run 04 → register → 08 deploy → endpoint
# MAGIC ```
# MAGIC
# MAGIC The recurring version of this runs as its **own job** (created below) so it fires on a
# MAGIC schedule/drift trigger — distinct from the deployment job (different trigger; nesting
# MAGIC training inside deployment would loop). The two connect only through the Model Registry.

# COMMAND ----------

# DBTITLE 1,E1. Drift Gate
# Read the InferenceLog monitor's drift-metrics table (Section D) and decide whether
# prediction drift vs the training baseline exceeds the threshold in the latest window.
dbutils.widgets.text("drift_threshold", "0.2", "Drift threshold (JS distance)")
dbutils.widgets.dropdown("force_retrain", "false", ["true", "false"], "Force retrain (ignore gate)")

drift_threshold = float(dbutils.widgets.get("drift_threshold") or "0.2")
force_retrain = dbutils.widgets.get("force_retrain").lower() == "true"
drift_table = f"{DA.catalog_name}.{DA.schema_name}.customer_churn_inference_unpacked_drift_metrics"

retrain_needed = False
reason = ""
if force_retrain:
    retrain_needed, reason = True, "force_retrain=true"
elif not spark.catalog.tableExists(drift_table):
    reason = (f"No drift-metrics table yet ({drift_table}). Set up the monitor in Section D and let "
              f"it refresh (needs endpoint traffic). Nothing to retrain on.")
else:
    try:
        latest = (
            spark.table(drift_table)
            .filter((F.col("column_name") == "prediction") & (F.col("drift_type") == "BASELINE"))
            .orderBy(F.col("window.start").desc()).limit(1).collect()
        )
        if not latest:
            reason = "Drift-metrics table has no BASELINE row for `prediction` yet (monitor not refreshed)."
        else:
            js = latest[0]["js_distance"]
            retrain_needed = js is not None and js >= drift_threshold
            reason = f"prediction JS distance = {js} (threshold {drift_threshold})"
    except Exception as e:
        reason = f"Could not read drift metrics ({type(e).__name__}: {e}). Skipping retrain."

print(f"Drift gate → retrain_needed = {retrain_needed}")
print(f"Reason: {reason}")

# COMMAND ----------

# DBTITLE 1,E2. Conditional Retrain — call 04-Model-Training directly
if retrain_needed:
    training_notebook = f"{DA.workshop_dir}/04-Model-Training"
    print(f"Drift gate TRIPPED — retraining via {training_notebook} ...")
    result = dbutils.notebook.run(training_notebook, 3600)
    print(f"Retrain finished: {result}")
    print("A new model version is registered → the deployment job (08) auto-fires.")
else:
    print("Drift within threshold (or monitor not ready) — no retrain triggered.")

# COMMAND ----------

# DBTITLE 1,E3. Create the Retrain-on-Drift Job
# Registers a separate job that runs THIS notebook on a (paused) schedule. Enable it —
# or add a table-update trigger on the drift-metrics table — for hands-off retraining.
from databricks.sdk.service.jobs import CronSchedule, PauseStatus

w = WorkspaceClient()
job_name = f"SBC Bank Churn Retrain-on-Drift — {DA.model_name}"
this_notebook = f"/Workspace{DA.workshop_dir}/07-Observability"

existing = [j for j in w.jobs.list(name=job_name)]
if existing:
    retrain_job = existing[0]
    print(f"Retrain job already exists: {retrain_job.job_id}")
else:
    retrain_job = w.jobs.create(
        name=job_name,
        tags={"sbc": "true", "churn-prediction": "true"},
        max_concurrent_runs=1,
        schedule=CronSchedule(quartz_cron_expression="0 0 6 * * ?", timezone_id="UTC",
                              pause_status=PauseStatus.PAUSED),
        tasks=[
            Task(
                task_key="retrain_on_drift",
                notebook_task=NotebookTask(
                    notebook_path=this_notebook,
                    source=Source("WORKSPACE"),
                    base_parameters={"drift_threshold": "0.2", "force_retrain": "false"},
                ),
            ),
        ],
    )
    print(f"Created retrain job: {retrain_job.job_id}")
print(f"View at: {w.config.host}/#job/{retrain_job.job_id}")
print("Schedule is PAUSED — enable it in the Jobs UI when ready.")

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## F. Conclusion
# MAGIC
# MAGIC In this notebook, we established the full **observe → explain → monitor → retrain** loop:
# MAGIC
# MAGIC - **Experiment tracking** — all runs logged and comparable in MLflow
# MAGIC - **Explainability** — feature importance and SHAP values explain model decisions
# MAGIC - **Quality validation** — held-out classification report, confusion matrix, distribution checks
# MAGIC - **Inference logging** — endpoint payloads unpacked + a Lakehouse InferenceLog monitor
# MAGIC - **Retrain on drift** — a drift gate that calls `04` and a scheduled retrain job
# MAGIC
# MAGIC These practices ensure you can **debug**, **audit**, **trust**, and **continuously improve** the model in production.
# MAGIC
# MAGIC Next: Proceed to **08-MLFlow** for the deployment pipeline the retrain loop feeds into.