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
# MAGIC # 08-MLFlow-Deploy
# MAGIC
# MAGIC This notebook is the **Deployment** task of the MLflow Deployment Job. After the model version passes evaluation and approval, this task promotes it.
# MAGIC
# MAGIC It sets the `champion` alias on the approved version and optionally updates a model serving endpoint.

# COMMAND ----------

# DBTITLE 1,Get Job Parameters
# ---------------------------------------------------------------
# Get job parameters (auto-populated by the deployment job)
# ---------------------------------------------------------------
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("model_version", "")

model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")

print(f"Deploying model: {model_name} version {model_version}")

# COMMAND ----------

# DBTITLE 1,Set Champion Alias
# ---------------------------------------------------------------
# Promote: Set the "champion" alias on the approved version
# ---------------------------------------------------------------
from mlflow.tracking.client import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=model_version
)

print(f"Set alias 'champion' on {model_name} version {model_version}")

# COMMAND ----------

# DBTITLE 1,Migrate Serving Endpoint to the Champion Version
# ---------------------------------------------------------------
# Point the Model Serving endpoint at the newly promoted version.
# Runs only if the endpoint already exists (otherwise it is created later by
# 06-Real-Time-Inference). The endpoint name matches 06 / DA.endpoint_name.
# ---------------------------------------------------------------
import re
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

w = WorkspaceClient()
username = spark.sql("SELECT current_user()").collect()[0][0]
endpoint_name = "sbc-bank-churn-" + re.sub(r'[^a-zA-Z0-9-]', '-', username)

try:
    w.serving_endpoints.get(endpoint_name)
    endpoint_exists = True
except Exception:
    endpoint_exists = False

if endpoint_exists:
    print(f"Migrating endpoint '{endpoint_name}' to version {model_version} (champion)...")
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=[
            ServedEntityInput(
                name="my-model",
                entity_name=model_name,
                entity_version=model_version,
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ],
    )
    print(f"Update requested — endpoint '{endpoint_name}' is rolling out version {model_version}. "
          f"This takes a few minutes; watch the Serving UI for READY.")
else:
    print(f"Endpoint '{endpoint_name}' does not exist yet — skipping endpoint update. "
          f"Create it via 06-Real-Time-Inference; it will serve version {model_version} once created.")

# COMMAND ----------

# DBTITLE 1,Done
print("Deployment task complete.")
print(f"Model {model_name} v{model_version} is now the 'champion' and is being served by the endpoint.")