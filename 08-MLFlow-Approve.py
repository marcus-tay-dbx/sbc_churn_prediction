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
# MAGIC # `08-MLFlow-Approve`
# MAGIC
# MAGIC This notebook is the **Approval** task of the MLflow Deployment Job. It gates deployment using an **automatic threshold-based** check.
# MAGIC
# MAGIC If the candidate model's F1 score meets the minimum threshold, deployment proceeds automatically. Otherwise, the task fails and an approver can review and repair it manually.

# COMMAND ----------

# DBTITLE 1,Get Job Parameters
# Get job parameters (auto-populated by the deployment job)

dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

print(f"Approval gate for: {model_name} version {model_version}")

# COMMAND ----------

# DBTITLE 1,Review Evaluation Metrics
# Review evaluation metrics from the Evaluate task

from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

# Fetch the model version details
mv = client.get_model_version(name=model_name, version=model_version)
print(f"Model:   {model_name}")
print(f"Version: {model_version}")
print(f"Status:  {mv.status}")
print(f"Source:  {mv.source}")

# Display any logged metrics (from the Evaluate task)
print("\n--- Evaluation Metrics ---")
try:
    # Model version tags may contain metrics logged by the evaluation task
    for key, value in (mv.tags or {}).items():
        print(f"  {key}: {value}")
except Exception:
    print("  (No tags found — check the MLflow Experiments UI for run metrics)")

# COMMAND ----------

# DBTITLE 1,Approval Gate
import mlflow

F1_THRESHOLD = 0.40

run = mlflow.get_run(mv.run_id)
eval_f1 = run.data.metrics.get("test_f1", 0.0)

print(f"Candidate F1:  {eval_f1:.4f}")
print(f"Threshold:     {F1_THRESHOLD}")

approved = eval_f1 >= F1_THRESHOLD

if approved:
    print(f"\nAPPROVED: Model {model_name} v{model_version} (F1 {eval_f1:.4f} >= {F1_THRESHOLD})")
else:
    raise Exception(
        f"REJECTED: Model {model_name} v{model_version} "
        f"(F1 {eval_f1:.4f} < {F1_THRESHOLD}). "
        f"Review metrics, then repair this task in the Jobs UI to override."
    )