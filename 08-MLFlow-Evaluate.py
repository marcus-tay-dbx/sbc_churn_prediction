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
# MAGIC # `08-MLFlow-Evaluate`
# MAGIC
# MAGIC This notebook is the **Evaluation** task of the MLflow Deployment Job. It is triggered automatically when a new model version is registered.
# MAGIC
# MAGIC It loads the candidate model version, scores it against a holdout set, and logs evaluation metrics back to MLflow.

# COMMAND ----------

# DBTITLE 1,Get Job Parameters
# ---------------------------------------------------------------
# Get job parameters (auto-populated by the deployment job)
# ---------------------------------------------------------------
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

print(f"Evaluating model: {model_name} version {model_version}")

# COMMAND ----------

# DBTITLE 1,Load Model and Evaluate
# ---------------------------------------------------------------
# Load the candidate model version and evaluate
# ---------------------------------------------------------------
import mlflow
import pickle, os
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd

from mlflow.tracking.client import MlflowClient

uc_client = MlflowClient(registry_uri="databricks-uc")
mv = uc_client.get_model_version(name=model_name, version=model_version)
run_id = mv.run_id

artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="customer_churn_model")

sk_model = None
for root, dirs, files in os.walk(artifact_path):
    for f in files:
        if f.endswith(".pkl"):
            with open(os.path.join(root, f), "rb") as fh:
                sk_model = pickle.load(fh)
            break
    if sk_model:
        break

print(f"Loaded model: {model_name} version {model_version} (run {run_id})")
print(f"Model type: {type(sk_model).__name__}")

catalog_schema = ".".join(model_name.split(".")[:2])
feature_table = f"{catalog_schema}.customer_churn_features"
label_table = f"{catalog_schema}.customer_churn"

feature_cols = ["tenure_years", "total_balance_usd", "num_products", "num_deposit_products",
                "has_maturing_cd", "txn_count_60d", "total_outflow_60d_usd", "withdrawal_count_60d",
                "tier_rank", "balanceCategory"]

features_df = spark.table(feature_table).select("customer_id", *feature_cols).toPandas()
labels_df = spark.table(label_table).select("customer_id", "churned").toPandas()

eval_df = labels_df.merge(features_df, on="customer_id", how="inner")

y_true = eval_df["churned"]
X_eval = eval_df[feature_cols]

y_pred = sk_model.predict(X_eval)

# Compute metrics
metrics = {
    "eval_f1_macro": f1_score(y_true, y_pred, average="macro"),
    "eval_accuracy": accuracy_score(y_true, y_pred),
}

for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# COMMAND ----------

# DBTITLE 1,Log Metrics to Model Version
# ---------------------------------------------------------------
# Log evaluation metrics back to the model version
# ---------------------------------------------------------------
for metric_name, metric_value in metrics.items():
    uc_client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key=metric_name,
        value=str(round(metric_value, 4))
    )

mlflow.log_metrics(metrics)

print(f"Logged {len(metrics)} metrics to model version {model_version}")
print("Evaluation task complete.")
