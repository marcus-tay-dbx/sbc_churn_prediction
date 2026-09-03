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
# MAGIC # 07-Observability — Retrain Drift Gate (job task)
# MAGIC
# MAGIC Operational drift gate for the **Retrain-on-Drift job** (created in `07-Observability-and-Continuous-Retrain`,
# MAGIC Section E). It reads the monitor's `..._drift_metrics` table, measures prediction drift,
# MAGIC and emits a job **task value** `retrain_needed` (`true`/`false`). The job's condition task
# MAGIC reads that value and runs `04-Model-Training` only when drift trips the threshold.
# MAGIC
# MAGIC This mirrors the illustrative gate in `07` (E1), but is the version the job actually calls.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Drift Gate → task value
# Decide whether prediction drift in the latest window exceeds the threshold.
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
    reason = f"No drift-metrics table yet ({drift_table}). Set up + refresh the monitor (Section D) first."
else:
    latest = (
        spark.table(drift_table)
        .filter((F.col("column_name") == "prediction") & (F.col("drift_type") == "CONSECUTIVE"))
        .orderBy(F.col("window.start").desc()).limit(1).collect()
    )
    if not latest:
        reason = "No CONSECUTIVE prediction-drift row yet (needs >=2 refreshed windows)."
    else:
        js = latest[0]["js_distance"]
        retrain_needed = js is not None and js >= drift_threshold
        reason = f"prediction JS distance = {js} (threshold {drift_threshold})"

print(f"retrain_needed = {retrain_needed} | {reason}")

# Emit the decision as a job task value for the downstream condition task.
dbutils.jobs.taskValues.set(key="retrain_needed", value=str(retrain_needed).lower())