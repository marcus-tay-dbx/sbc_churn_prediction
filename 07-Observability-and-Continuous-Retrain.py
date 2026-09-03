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
# MAGIC %md-sandbox
# MAGIC <div style="max-width:960px;margin:0 auto;font-family:sans-serif;color:#0b2026;">
# MAGIC   <div style="font-size:22pt;font-weight:700;margin-bottom:4px;">07 · Observability &amp; Continuous Retrain</div>
# MAGIC   <div style="font-size:13pt;color:#5E7077;margin-bottom:18px;">Observe, explain, and monitor the model — <b>before</b> and <b>after</b> it is deployed.</div>
# MAGIC   <div style="display:flex;gap:20px;flex-wrap:wrap;">
# MAGIC     <div style="flex:1 1 380px;background:#F9F7F4;border-radius:10px;box-shadow:0 2px 8px rgba(27,49,57,0.08);padding:20px 22px;position:relative;overflow:hidden;">
# MAGIC       <div style="position:absolute;top:0;left:0;width:100%;height:8px;background:#1B5162;"></div>
# MAGIC       <div style="display:inline-block;background:#1B5162;color:#fff;font-size:10pt;font-weight:700;padding:4px 12px;border-radius:999px;margin-bottom:10px;">① BEFORE deployment · dev</div>
# MAGIC       <div style="font-size:14pt;font-weight:700;margin-bottom:6px;">Validate the <code>@dev</code> model</div>
# MAGIC       <div style="font-size:11.5pt;color:#5E7077;margin-bottom:12px;">Runs against the registered model, before it reaches an endpoint.</div>
# MAGIC       <div style="font-size:12.5pt;line-height:1.8;">
# MAGIC         <b>A.</b> MLflow tracking &amp; run comparison<br/>
# MAGIC         <b>B.</b> Feature importance + SHAP explainability<br/>
# MAGIC         <b>C.</b> Held-out quality validation
# MAGIC       </div>
# MAGIC     </div>
# MAGIC     <div style="flex:1 1 380px;background:#F9F7F4;border-radius:10px;box-shadow:0 2px 8px rgba(27,49,57,0.08);padding:20px 22px;position:relative;overflow:hidden;">
# MAGIC       <div style="position:absolute;top:0;left:0;width:100%;height:8px;background:#FF5F46;"></div>
# MAGIC       <div style="display:inline-block;background:#FF5F46;color:#fff;font-size:10pt;font-weight:700;padding:4px 12px;border-radius:999px;margin-bottom:10px;">② AFTER deployment · BAU</div>
# MAGIC       <div style="font-size:14pt;font-weight:700;margin-bottom:6px;">Monitor live traffic &amp; retrain</div>
# MAGIC       <div style="font-size:11.5pt;color:#5E7077;margin-bottom:12px;">Needs a live endpoint with traffic (deploy in Notebook 06).</div>
# MAGIC       <div style="font-size:12.5pt;line-height:1.8;">
# MAGIC         <b>D.</b> Lakehouse Monitoring of live predictions (drift)<br/>
# MAGIC         <b>E.</b> Retrain on drift (continuous training)
# MAGIC       </div>
# MAGIC     </div>
# MAGIC   </div>
# MAGIC   <div style="margin-top:16px;font-size:11pt;color:#5E7077;">
# MAGIC     <b>Section color key</b> (each section below is tagged): &nbsp; 🟦 <b>① BEFORE</b> deployment (dev) &nbsp;·&nbsp; 🟧 <b>② AFTER</b> deployment (BAU)
# MAGIC   </div>
# MAGIC   <div style="font-size:11.5pt;color:#5E7077;margin-top:16px;"><b>Prerequisites:</b> run Notebook 00-Setup and Notebook 04 first; for D–E, deploy the endpoint in Notebook 06. Sections D–E no-op cleanly until the endpoint has captured traffic.</div>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install shap --quiet

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Section A - MLflow Tracking
# MAGIC %md
# MAGIC ## A. MLflow Experiment Tracking & Run Comparison
# MAGIC
# MAGIC 🟦 **① BEFORE deployment · dev** — Runs against the registered `@dev` model, before it reaches an endpoint.
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
# MAGIC 5. Under **Artifacts**, expand `customer_churn_model` to see the logged model, `requirements.txt`, and `MLmodel` spec
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
# MAGIC 🟦 **① BEFORE deployment · dev** — Runs against the registered `@dev` model, before it reaches an endpoint.
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
# MAGIC 3. Navigate to **Models** in the left sidebar → open `solution_builder.sbc_churn_prediction.customer_churn_model`
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
# MAGIC
# MAGIC 🟦 **① BEFORE deployment · dev** — Runs against the registered `@dev` model, before it reaches an endpoint.
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
# MAGIC 🟧 **② AFTER deployment · BAU** — Needs the deployed endpoint with live traffic.
# MAGIC
# MAGIC Monitoring the live endpoint is a **three-step pipeline**:
# MAGIC
# MAGIC 1. **Notebook `06`** turned on inference logging, so the endpoint auto-captures every
# MAGIC    request/response into the **payload log** `churn_endpoint_payload` (raw JSON blobs).
# MAGIC 2. **D1** (below) **unpacks** that payload log into a monitorable table
# MAGIC    `customer_churn_inference_unpacked` — one row per prediction, with real feature columns,
# MAGIC    the `prediction`, an `inference_timestamp`, and a `model_version`.
# MAGIC 3. **D2** **enables monitoring** by creating a Lakehouse **InferenceLog monitor** on that
# MAGIC    unpacked table — it produces drift/quality metric tables and a dashboard on a schedule.
# MAGIC
# MAGIC Databricks **Lakehouse Monitoring** provides automated drift detection and quality tracking:
# MAGIC
# MAGIC | Capability | What It Does |
# MAGIC | --- | --- |
# MAGIC | **Data drift detection** | Alerts when input feature distributions shift from the training baseline |
# MAGIC | **Prediction drift** | Detects when the model's output distribution changes |
# MAGIC | **Data quality** | Tracks nulls, schema changes, volume anomalies |
# MAGIC | **Custom metrics** | Define business-specific quality metrics |

# COMMAND ----------

# DBTITLE 1,D1. Unpack Endpoint Payloads into a Monitorable Table
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
            # AI Gateway inference tables carry a `request_time` timestamp column directly
            # (the legacy auto_capture_config used epoch `timestamp_ms` instead).
            .withColumn("inference_timestamp", F.col("request_time"))
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

# DBTITLE 1,Enable the Monitor — Two Options
# MAGIC %md
# MAGIC ### D2. Enable the monitor
# MAGIC
# MAGIC With the unpacked table ready (from **D1**), enable the monitor on it — there are **two
# MAGIC equivalent ways**; pick either, they create the *same* monitor:
# MAGIC
# MAGIC **Option 1 — UI**
# MAGIC 1. **Catalog** → select `customer_churn_inference_unpacked`
# MAGIC 2. **Quality** tab → **Create monitor** → **Inference profile**
# MAGIC 3. **Problem type** = Classification, **Prediction column** = `prediction`, **Timestamp column** = `inference_timestamp`, **Model ID column** = `model_version`
# MAGIC 4. **Baseline table** = `customer_churn_features`, set a **refresh schedule** (e.g., daily) → **Create**
# MAGIC
# MAGIC **Option 2 — Script**: run cell **D2** below (`w.quality_monitors.create(...)`).
# MAGIC
# MAGIC Either way you get a **dashboard** plus **metrics tables** (`..._profile_metrics`, `..._drift_metrics`) you can query.

# COMMAND ----------

# DBTITLE 1,Option 2 - By Script
# Create the InferenceLog monitor on the unpacked table.
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType

w = WorkspaceClient()

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
        # No baseline_table_name: drift is computed across time windows. A baseline is optional
        # and, if set, must share the MONITORED table's schema — including the `prediction` and
        # `model_version` columns — so the raw `customer_churn_features` table can't be used here.
        slicing_exprs=["tier_rank", "has_maturing_cd"],
    )
    print(f"Monitor created on {unpacked_table} — generates profile + drift metric tables and a dashboard.")

# COMMAND ----------

# DBTITLE 1,D3. Refresh the Monitors
# Refresh both monitors so their metric tables + dashboards populate (needs monitor ACTIVE).
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInfoStatus

w = WorkspaceClient()
unpacked_tbl = f"{DA.catalog_name}.{DA.schema_name}.customer_churn_inference_unpacked"
batch_log_table = f"{DA.catalog_name}.{DA.schema_name}.customer_churn_batch_inference_log"
for tbl in [unpacked_tbl, batch_log_table]:
    if not spark.catalog.tableExists(tbl):
        print(f"Skip {tbl} — table not found.")
        continue
    try:
        for attempt in range(40):
            if w.quality_monitors.get(tbl).status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE:
                break
            time.sleep(15)
        run = w.quality_monitors.run_refresh(table_name=tbl)
        print(f"Refresh triggered on {tbl} (id {run.refresh_id}).")
    except Exception as e:
        print(f"No monitor on {tbl} yet ({type(e).__name__}).")

# COMMAND ----------

# DBTITLE 1,Section E - Retrain on Drift
# MAGIC %md-sandbox
# MAGIC <div class="pipe-wrap">
# MAGIC <style>
# MAGIC .pipe-wrap { max-width: 940px; margin: 0 auto; font-family: sans-serif; color:#0b2026; padding:8px; box-sizing:border-box; }
# MAGIC .pipe-header { background:#1B5162; color:#fff; border-radius:10px; padding:18px 24px; text-align:center; margin-bottom:20px; box-shadow:0 2px 8px rgba(27,49,57,0.10); }
# MAGIC .pipe-header .t { font-size:17pt; font-weight:700; margin:0 0 4px 0; }
# MAGIC .pipe-header .s { font-size:12pt; opacity:0.92; margin:0; }
# MAGIC .flow { display:flex; flex-wrap:wrap; align-items:stretch; justify-content:center; gap:8px; }
# MAGIC .node { flex:0 0 150px; background:#F9F7F4; border-radius:8px; box-shadow:0 2px 8px rgba(27,49,57,0.06); padding:14px 10px 12px 10px; text-align:center; position:relative; box-sizing:border-box; }
# MAGIC .node::before { content:""; position:absolute; top:0; left:0; width:100%; height:6px; border-radius:8px 8px 0 0; }
# MAGIC .node.mon::before { background:#1B5162; } .node.task::before { background:#00A972; } .node.deploy::before { background:#FFAB00; } .node.ep::before { background:#4299E0; }
# MAGIC .node .lbl { font-size:8.5pt; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; margin:2px 0 4px 0; color:#5A6F77; }
# MAGIC .node .nm { font-size:10.5pt; font-weight:700; color:#0b2026; line-height:1.25; margin:0 0 4px 0; }
# MAGIC .node .ds { font-size:9pt; color:#5E7077; line-height:1.35; margin:0; }
# MAGIC .arrow { display:flex; align-items:center; justify-content:center; color:#5E7077; font-size:18pt; font-weight:700; flex:0 0 16px; }
# MAGIC .steps { max-width:900px; margin:18px auto 0 auto; font-size:12.5pt; line-height:1.6; color:#0b2026; padding-left:22px; }
# MAGIC .steps li { margin-bottom:6px; }
# MAGIC .loopback { max-width:900px; margin:12px auto 0 auto; border:2px dashed #00A972; border-radius:8px; background:rgba(0,169,114,0.06); padding:10px 16px; text-align:center; font-size:11pt; font-weight:600; color:#1B5162; }
# MAGIC .loopback .a { color:#00A972; font-weight:800; font-size:15pt; }
# MAGIC </style>
# MAGIC
# MAGIC <div class="pipe-header">
# MAGIC   <div style="display:inline-block;background:#FF5F46;color:#fff;padding:3px 11px;border-radius:999px;font-size:9pt;font-weight:700;letter-spacing:0.5px;margin-bottom:8px;">② AFTER deployment · BAU</div>
# MAGIC   <div class="t">E. Retrain on Drift — continuous-training loop</div>
# MAGIC   <div class="s">The monitor's drift signal triggers a gated job that retrains and redeploys — automatically.</div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="flow">
# MAGIC   <div class="node mon"><div class="lbl">Monitor · 07 D</div><div class="nm">Drift metrics</div><div class="ds">writes …_drift_metrics</div></div>
# MAGIC   <div class="arrow">&#8594;</div>
# MAGIC   <div class="node task"><div class="lbl">Task 1</div><div class="nm">drift_gate</div><div class="ds">Notebook 07-…-Retrain-Drift-Gate → sets retrain_needed</div></div>
# MAGIC   <div class="arrow">&#8594;</div>
# MAGIC   <div class="node task"><div class="lbl">Task 2</div><div class="nm">check_drift</div><div class="ds">proceed only if retrain_needed = true</div></div>
# MAGIC   <div class="arrow">&#8594;</div>
# MAGIC   <div class="node task"><div class="lbl">Task 3</div><div class="nm">retrain</div><div class="ds">Notebook 04 → new model version</div></div>
# MAGIC   <div class="arrow">&#8594;</div>
# MAGIC   <div class="node deploy"><div class="lbl">Auto</div><div class="nm">Deployment job</div><div class="ds">Notebook 08 → Evaluate · Approve · Deploy</div></div>
# MAGIC   <div class="arrow">&#8594;</div>
# MAGIC   <div class="node ep"><div class="lbl">Serving</div><div class="nm">Endpoint</div><div class="ds">updated to champion</div></div>
# MAGIC </div>
# MAGIC
# MAGIC <div class="loopback"><span class="a">&#8634;</span> &nbsp;<b>Continuous loop</b> — the served <b>Endpoint</b>'s live traffic produces new <b>Drift metrics</b>, which re-trigger the gate. (Endpoint&nbsp;&#8594;&nbsp;Drift metrics)</div>
# MAGIC
# MAGIC <ol class="steps">
# MAGIC   <li><b>Trigger</b> — the job fires when the monitor writes new rows to the drift-metrics table (paused by default; or use a schedule).</li>
# MAGIC   <li><b>Task 1 · drift_gate</b> — runs <b>Notebook 07-Observability-Retrain-Drift-Gate</b>; reads drift and sets a <code>retrain_needed</code> flag.</li>
# MAGIC   <li><b>Task 2 · check_drift</b> — a condition task that continues <b>only when</b> <code>retrain_needed = true</code>.</li>
# MAGIC   <li><b>Task 3 · retrain</b> — runs <b>Notebook 04 (Model Training)</b>, registering a new version; <b>Notebook 08</b>'s deployment job then auto-promotes it to the endpoint.</li>
# MAGIC </ol>
# MAGIC
# MAGIC <p class="steps" style="padding-left:0; list-style:none;">The retrain job is <b>separate</b> from the deployment job (different trigger; nesting training inside deployment would loop) — they connect only through the Model Registry.</p>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Create the Retrain-on-Drift Job
# Build the gated retrain job shown above: drift_gate → check_drift → retrain (Notebook 04).
from databricks.sdk.service.jobs import (
    Task, NotebookTask, Source, TaskDependency, ConditionTask, ConditionTaskOp,
    TriggerSettings, TableUpdateTriggerConfiguration, PauseStatus,
)

w = WorkspaceClient()
job_name = f"SBC Bank Churn Retrain-on-Drift — {DA.model_name}"
gate_notebook = f"/Workspace{DA.workshop_dir}/07-Observability-Retrain-Drift-Gate"
training_notebook = f"/Workspace{DA.workshop_dir}/04-Model-Training"
drift_table = f"{DA.catalog_name}.{DA.schema_name}.customer_churn_inference_unpacked_drift_metrics"

existing = [j for j in w.jobs.list(name=job_name)]
if existing:
    retrain_job = existing[0]
    print(f"Retrain job already exists: {retrain_job.job_id}")
else:
    retrain_job = w.jobs.create(
        name=job_name,
        tags={"sbc": "true", "churn-prediction": "true"},
        max_concurrent_runs=1,
        # Fire when the monitor writes new drift metrics (paused until you enable it).
        trigger=TriggerSettings(
            pause_status=PauseStatus.PAUSED,
            table_update=TableUpdateTriggerConfiguration(table_names=[drift_table]),
        ),
        tasks=[
            Task(
                task_key="drift_gate",
                notebook_task=NotebookTask(
                    notebook_path=gate_notebook, source=Source("WORKSPACE"),
                    base_parameters={"drift_threshold": "0.2", "force_retrain": "false"},
                ),
            ),
            Task(
                task_key="check_drift",
                depends_on=[TaskDependency(task_key="drift_gate")],
                condition_task=ConditionTask(
                    left="{{tasks.drift_gate.values.retrain_needed}}",
                    op=ConditionTaskOp.EQUAL_TO,
                    right="true",
                ),
            ),
            Task(
                task_key="retrain",
                depends_on=[TaskDependency(task_key="check_drift", outcome="true")],
                notebook_task=NotebookTask(
                    notebook_path=training_notebook, source=Source("WORKSPACE"),
                ),
            ),
        ],
    )
    print(f"Created retrain job: {retrain_job.job_id}")
print(f"View at: {w.config.host}/#job/{retrain_job.job_id}")
print("Trigger is PAUSED — enable it in the Jobs UI (or swap to a CronSchedule) when ready.")

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
# MAGIC - **Retrain on drift** — a gated retrain job (drift gate → **Notebook 04**), triggered by the drift signal
# MAGIC
# MAGIC These practices ensure you can **debug**, **audit**, **trust**, and **continuously improve** the model in production.
# MAGIC
# MAGIC Next: Proceed to **Notebook 08 (Continuous Deployment)** for the deployment pipeline the retrain loop feeds into.