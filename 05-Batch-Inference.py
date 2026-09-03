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
# MAGIC # 05 - Batch Inference
# MAGIC
# MAGIC In this notebook, we perform **batch inference** using a registered model from Unity Catalog with **Feature Store lookups**. This demonstrates how to score customers at scale using features stored in the Feature Store.
# MAGIC
# MAGIC **Prerequisites**: Run `00-Setup`, `03-Feature-Engineering`, and `04-Model-Training` first.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Batch Inference Overview
# MAGIC %md
# MAGIC ## A. Batch Inference with Feature Store
# MAGIC
# MAGIC **Batch inference** is ideal for high-latency scenarios where you score a batch of records at once. It leverages:
# MAGIC - Data saved in Delta tables or Feature Store
# MAGIC - A registered model from Unity Catalog
# MAGIC - `FeatureEngineeringClient.score_batch()` for automatic feature lookup
# MAGIC
# MAGIC The key advantage of using Feature Store for inference is that the model automatically looks up the correct features by primary key, ensuring **consistency** between training and scoring.

# COMMAND ----------

# DBTITLE 1,Batch Inference with Feature Store
fe = FeatureEngineeringClient()

feature_table_name = DA.feature_table_name
model_name = DA.model_name
model_uri = DA.model_uri

print(f"Feature table: {feature_table_name}")
print(f"Model URI: {model_uri}")

# Score the held-out TEST set (20%, stratified on churn, seed 42) — not the full book.
labels_df = spark.table("customer_churn").select("customer_id", "churned").toPandas()
train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df["churned"])
scoring_df = spark.createDataFrame(test_df[["customer_id"]])
print(f"Scoring {scoring_df.count()} held-out test customers")

predictions = fe.score_batch(model_uri=model_uri, df=scoring_df)
print(f"Predictions complete")
display(predictions.select("customer_id", "prediction").limit(20))

# COMMAND ----------

# DBTITLE 1,Log & Monitor Intro
# MAGIC %md
# MAGIC ## B. Log Predictions & Enable Monitoring
# MAGIC
# MAGIC Log predictions to a table, append a small **drifted second window**, then enable a **Lakehouse
# MAGIC InferenceLog monitor** (refreshed in Notebook 07, Section D).

# COMMAND ----------

# DBTITLE 1,Log Batch Inference Results
# Append predictions + timestamp + model version; enable CDF (required by monitoring).
from pyspark.sql import functions as F

mlflow_client = MlflowClient(registry_uri="databricks-uc")
try:
    model_version = mlflow_client.get_model_version_by_alias(model_name, "champion").version
except Exception:
    model_version = mlflow_client.get_model_version_by_alias(model_name, "dev").version

log_table = f"{DA.catalog_name}.{DA.schema_name}.customer_churn_batch_inference_log"
(predictions
    .withColumn("inference_timestamp", F.current_timestamp())
    .withColumn("model_version", F.lit(str(model_version)))
    .write.mode("append").option("mergeSchema", "true").saveAsTable(log_table))
tag_table(log_table)
spark.sql(f"ALTER TABLE {log_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Logged {predictions.count()} predictions → {log_table}")

# COMMAND ----------

# DBTITLE 1,Why a Second Batch
# MAGIC %md
# MAGIC ### Enable Monitoring
# MAGIC
# MAGIC Why a second batch?
# MAGIC
# MAGIC Drift is measured **between time windows**, so a single batch gives the monitor nothing to compare.
# MAGIC Below we append a small **drifted** batch dated to an *earlier* window — now the monitor has two
# MAGIC windows (today's real batch + the earlier drifted one) and can compute a drift signal. This is a
# MAGIC demo shortcut; in production the second window arrives naturally as the endpoint scores over days.

# COMMAND ----------

# DBTITLE 1,Mock a Second (Drifted) Batch → Show Drift
# Append a few drifted rows in an earlier day-window so drift_metrics has 2 windows to compare.
mock = [
    # customer_id, + DA.feature_columns (10), prediction
    ("MOCK-0001", 10, 25000.0, 2, 1, 1, 45, -300000.0, 35, 2, 1.0, 1),
    ("MOCK-0002",  8, 18000.0, 1, 1, 1, 38, -220000.0, 28, 1, 0.0, 1),
    ("MOCK-0003", 12, 42000.0, 3, 2, 1, 50, -350000.0, 40, 3, 2.0, 1),
    ("MOCK-0004",  6,  9000.0, 1, 1, 0, 20,  -80000.0, 15, 0, 0.0, 1),
    ("MOCK-0005",  9, 15000.0, 2, 1, 1, 33, -190000.0, 25, 1, 1.0, 1),
    ("MOCK-0006", 11, 60000.0, 2, 2, 1, 48, -400000.0, 42, 2, 2.0, 0),
]
mock_pdf = pd.DataFrame(mock, columns=["customer_id", *DA.feature_columns, "prediction"])
mock_sdf = (spark.createDataFrame(mock_pdf)
    .withColumn("inference_timestamp", F.current_timestamp() - F.expr("INTERVAL 3 DAYS"))
    .withColumn("model_version", F.lit(str(model_version))))
log_schema = spark.table(log_table).schema
mock_sdf = mock_sdf.select([F.col(f.name).cast(f.dataType) for f in log_schema])
mock_sdf.write.mode("append").option("mergeSchema", "true").saveAsTable(log_table)
print(f"Appended {len(mock)} drifted rows in an earlier window → {log_table}")

# COMMAND ----------

# DBTITLE 1,Enable Monitoring on the Batch Log
# Create the InferenceLog monitor (same as 07 Section D, no unpack). It is refreshed in Notebook 07.
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType

w = WorkspaceClient()
try:
    w.quality_monitors.get(log_table)
    print(f"Monitor already exists on {log_table}.")
except Exception:
    w.quality_monitors.create(
        table_name=log_table,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction",
            timestamp_col="inference_timestamp",
            granularities=["1 day"],
            model_id_col="model_version",
        ),
        assets_dir=f"/Workspace{DA.workshop_dir}/monitoring",
        output_schema_name=f"{DA.catalog_name}.{DA.schema_name}",
        slicing_exprs=["tier_rank", "has_maturing_cd"],
    )
    print(f"Monitor created on {log_table}. Refresh it in Notebook 07 (Section D) to populate metrics.")

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## C. Conclusion
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC - **Batch inference with Feature Store** lookups using `fe.score_batch()`
# MAGIC - Ranking the highest-value at-risk customers for retention outreach
# MAGIC - **Logging predictions** to a Delta table and attaching a **Lakehouse InferenceLog monitor**
# MAGIC - How Feature Store ensures **consistent feature retrieval** between training and inference
# MAGIC
# MAGIC Key takeaways:
# MAGIC - `score_batch()` automatically resolves features by primary key
# MAGIC - The `@dev` alias lets you reference specific model versions without hardcoding version numbers
# MAGIC - Batch inference is ideal for scheduled scoring pipelines
# MAGIC
# MAGIC Next: Proceed to `06-Real-Time-Inference` for deploying models as serving endpoints.