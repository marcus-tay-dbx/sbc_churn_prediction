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

# Score the full customer book — pass only the lookup key; features are joined automatically.
scoring_df = spark.table("customer_churn").select("customer_id")
print(f"Scoring {scoring_df.count()} customers")

predictions = fe.score_batch(model_uri=model_uri, df=scoring_df)
print(f"Predictions complete")
display(predictions.select("customer_id", "prediction").limit(20))

# COMMAND ----------

# DBTITLE 1,Highest-Risk Customers
# The customers the model flags as likely to churn — the retention team's call list.
# score_batch already returns the looked-up features (tenure_years, total_balance_usd, ...),
# so we only join the base table for `tier` (a string attribute, not a model feature).
at_risk = predictions.filter("prediction = 1")
print(f"Customers predicted to churn: {at_risk.count()}")
tier_lookup = spark.table("customer_churn").select("customer_id", "tier")
display(
    at_risk.join(tier_lookup, "customer_id")
    .select("customer_id", "tier", "tenure_years", "total_balance_usd", "prediction")
    .orderBy("total_balance_usd", ascending=False)
    .limit(20)
)

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## B. Conclusion
# MAGIC
# MAGIC In this notebook, we covered:
# MAGIC - **Batch inference with Feature Store** lookups using `fe.score_batch()`
# MAGIC - Ranking the highest-value at-risk customers for retention outreach
# MAGIC - How Feature Store ensures **consistent feature retrieval** between training and inference
# MAGIC
# MAGIC Key takeaways:
# MAGIC - `score_batch()` automatically resolves features by primary key
# MAGIC - The `@dev` alias lets you reference specific model versions without hardcoding version numbers
# MAGIC - Batch inference is ideal for scheduled scoring pipelines
# MAGIC
# MAGIC Next: Proceed to **06-Real-Time-Inference** for deploying models as serving endpoints.