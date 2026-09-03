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
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 06 - Real-Time Inference
# MAGIC
# MAGIC In this notebook, we deploy a registered model as a **Databricks Model Serving endpoint** for real-time inference, and show how to query it from both **code** and the **UI**.
# MAGIC
# MAGIC **Prerequisites**: Run `00-Setup` and `04-Model-Training` first.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Model Serving Overview
# MAGIC %md
# MAGIC ## A. Introduction to Databricks Model Serving
# MAGIC
# MAGIC **Databricks Model Serving** provides a unified, production-ready path from a registered model to a live serving endpoint.
# MAGIC
# MAGIC Serving modes:
# MAGIC - **Real-time**: Low-latency REST API for individual predictions
# MAGIC - **Batch**: High-throughput scoring of large datasets (covered in Notebook 05)
# MAGIC
# MAGIC Key features:
# MAGIC - Automatic scaling (including scale-to-zero)
# MAGIC - Built-in monitoring and metrics
# MAGIC - A/B testing and traffic routing
# MAGIC - Integration with Unity Catalog for governance

# COMMAND ----------

# DBTITLE 1,Create Serving Endpoint
# MAGIC %md
# MAGIC ## B. Create a Model Serving Endpoint
# MAGIC
# MAGIC ### B1. From Code
# MAGIC
# MAGIC The cell below creates a serving endpoint using the `mlflow.deployments` API. If the endpoint already exists, it will be reused.

# COMMAND ----------

# DBTITLE 1,Check Online Feature Store
# This model was packaged with the Feature Engineering client, so it can look up features by
# customer_id at serving time — but only if an ONLINE (synced) feature table exists. We check
# for it here and fall back gracefully to the OFFLINE feature table if it's missing (no hard fail).
synced_table_name = f"{DA.feature_table_name}_synced"
online_store_available = spark.catalog.tableExists(synced_table_name)

if online_store_available:
    print(f"✅ Online feature store found: {synced_table_name}")
    print(f"   Row count: {spark.table(synced_table_name).count():,}")
    print("   The endpoint can look up features by customer_id (Option 1 in Section C).")
else:
    print(f"⚠️  Online feature store NOT found: {synced_table_name}")
    print("   Deploying the endpoint against the OFFLINE feature table instead.")
    print("   Real-time lookup by customer_id is unavailable — pass features directly (Option 2 in Section C).")
    print("   To enable online lookup, create the synced table in Notebook 03 (Section D).")

# COMMAND ----------

# MAGIC %md
# MAGIC > ⚠️ **Online feature store fallback**
# MAGIC >
# MAGIC > This model was logged with the Feature Engineering client, so it can **automatically
# MAGIC > look up features by `customer_id`** at query time — but only when an **online (synced)
# MAGIC > feature table** exists.
# MAGIC >
# MAGIC > - **If the synced table exists** → the endpoint serves lookups by `customer_id` (**Option 1** in Section C).
# MAGIC > - **If it does not** → the endpoint is still deployed against the **offline feature table**, but
# MAGIC >   online lookup is unavailable. You must pass all feature values directly in the request
# MAGIC >   (**Option 2** in Section C). Create the synced table in **Notebook 03 (Section D)** to enable online lookup.
# MAGIC >
# MAGIC > Either way the deployment below succeeds — the difference is only in *how* you query the endpoint.

# COMMAND ----------

# DBTITLE 1,Deploy Endpoint
mlflow.set_registry_uri("databricks-uc")
client = get_deploy_client("databricks")

# Create a unique endpoint name with structured prefix
endpoint_name = DA.endpoint_name
model_name = DA.model_name

# Serve the current champion (fallback dev) version — never hardcode a version,
# so the endpoint always launches on the promoted model.
_mc = MlflowClient(registry_uri="databricks-uc")
try:
    serving_version = str(_mc.get_model_version_by_alias(model_name, "champion").version)
except Exception:
    serving_version = str(_mc.get_model_version_by_alias(model_name, "dev").version)
print(f"Serving version: {serving_version}")

# Check if endpoint exists, create if not
try:
    existing_endpoint = client.get_endpoint(endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists.")
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"Creating endpoint: {endpoint_name}")
        endpoint = client.create_endpoint(
            name=endpoint_name,
            config={
                "served_entities": [
                    {
                        "name": "my-model",
                        "entity_name": model_name,
                        "entity_version": serving_version,
                        "workload_size": "Small",
                        "scale_to_zero_enabled": True
                    }
                ],
                "traffic_config": {
                    "routes": [
                        {
                            "served_model_name": "my-model",
                            "traffic_percentage": 100
                        }
                    ]
                },
                "tags": [
                    {"key": "sbc", "value": "true"},
                    {"key": "churn-prediction", "value": "true"}
                ]
            }
        )
        print(f"Endpoint '{endpoint_name}' created. It may take 5-10 minutes to become ready.")
    else:
        print(f"Error: {e}")

# COMMAND ----------

# DBTITLE 1,Enable Inference Logging (AI Gateway)
# Legacy `auto_capture_config` is deprecated — enable logging via AI Gateway inference
# tables. The mlflow.deployments client ignores the `ai_gateway` config key, so we set it
# with the dedicated AI Gateway API (idempotent; works whether the endpoint is new or existing).
# Captures each request/response to `<prefix>_payload` for monitoring & drift detection in 07.
from databricks.sdk import WorkspaceClient

_w = WorkspaceClient()
_w.api_client.do(
    "PUT",
    f"/api/2.0/serving-endpoints/{endpoint_name}/ai-gateway",
    body={
        "inference_table_config": {
            "catalog_name": DA.catalog_name,
            "schema_name": DA.schema_name,
            "table_name_prefix": "churn_endpoint",
            "enabled": True,
        }
    },
)
print(f"Inference logging (AI Gateway) enabled → {DA.catalog_name}.{DA.schema_name}.churn_endpoint_payload")
print("Payload rows appear after the endpoint serves requests (batched; ~10-30 min).")

# COMMAND ----------

# DBTITLE 1,Serve from Offline Feature Store
# MAGIC %md
# MAGIC ### B2. Serve Using the Offline Feature Store
# MAGIC
# MAGIC When **no online (synced) feature table exists**, the endpoint can't look up features by
# MAGIC `customer_id` on its own. You can still serve real-time predictions by **reading the features
# MAGIC from the offline feature table at request time** and passing them directly to the endpoint.
# MAGIC
# MAGIC This is the offline-feature-store serving pattern — no online store required. The offline
# MAGIC table (`customer_churn_features`) is the same one the model was logged against in Notebook 04,
# MAGIC so the feature values match training exactly.

# COMMAND ----------

# DBTITLE 1,Query with Offline Features
# Look up the target customers' features straight from the OFFLINE feature table,
# then pass them directly to the endpoint (works with or without an online store).
target_ids = ["CUST-0000214", "CUST-0000001", "CUST-0000002"]

offline_features = (
    spark.table(DA.feature_table_name)
         .filter(F.col("customer_id").isin(target_ids))
         .select("customer_id", *DA.feature_columns)
         .toPandas()
)
print(f"Fetched {len(offline_features)} rows from offline feature table: {DA.feature_table_name}")

offline_payload = {"dataframe_records": offline_features.to_dict("records")}
response = client.predict(endpoint=endpoint_name, inputs=offline_payload)
print(json.dumps(response, indent=2))

# COMMAND ----------

# DBTITLE 1,Query from UI
# MAGIC %md
# MAGIC ### B3. From the UI
# MAGIC
# MAGIC You can also create and manage serving endpoints from the Databricks UI:
# MAGIC
# MAGIC 1. In the left sidebar, click **Serving**
# MAGIC 2. Click **Create serving endpoint**
# MAGIC 3. Select your registered model from Unity Catalog
# MAGIC 4. Configure scaling options (workload size, scale-to-zero)
# MAGIC 5. Click **Create**
# MAGIC
# MAGIC Once the endpoint is ready, you can test it directly from the **Query endpoint** tab in the UI.

# COMMAND ----------

# DBTITLE 1,Query Endpoint from Notebook
# MAGIC %md
# MAGIC ## C. Query the Serving Endpoint
# MAGIC
# MAGIC ### C1. From a Notebook
# MAGIC
# MAGIC Send a REST API request to the serving endpoint with sample customer data.

# COMMAND ----------

# DBTITLE 1,Query Endpoint Code
# --- Option 1: Pass customer_id — endpoint looks up features from the online store ---
lookup_data = {
    "dataframe_records": [
        {"customer_id": "CUST-0000214"},
        {"customer_id": "CUST-0000001"},
        {"customer_id": "CUST-0000002"}
    ]
}

print("Option 1: Feature Store lookup by customer_id")
try:
    response = client.predict(endpoint=endpoint_name, inputs=lookup_data)
    print(json.dumps(response, indent=2))
except Exception as e:
    print(f"Error (needs an online/synced feature table — see 03 Section D): {e}")

# --- Option 2: Pass features directly — no online store needed ---
feature_data = {
    "dataframe_records": [
        {
            "customer_id": "CUST-0000214",
            "tenure_years": 12,
            "total_balance_usd": 662000.0,
            "num_products": 3,
            "num_deposit_products": 2,
            "has_maturing_cd": 1,
            "txn_count_60d": 18,
            "total_outflow_60d_usd": -420000.0,
            "withdrawal_count_60d": 18,
            "tier_rank": 2,
            "balanceCategory": 2.0
        },
        {
            "customer_id": "CUST-0000001",
            "tenure_years": 4,
            "total_balance_usd": 12500.0,
            "num_products": 2,
            "num_deposit_products": 1,
            "has_maturing_cd": 0,
            "txn_count_60d": 3,
            "total_outflow_60d_usd": 0.0,
            "withdrawal_count_60d": 0,
            "tier_rank": 0,
            "balanceCategory": 0.0
        }
    ]
}

print("\nOption 2: Pass features directly")
try:
    response = client.predict(endpoint=endpoint_name, inputs=feature_data)
    print(json.dumps(response, indent=2))
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

# DBTITLE 1,Query from UI Instructions
# MAGIC %md
# MAGIC ### C2. From the Serving UI
# MAGIC
# MAGIC To test the endpoint from the Databricks UI:
# MAGIC
# MAGIC 1. Navigate to **Serving** in the left sidebar
# MAGIC 2. Click on your endpoint name
# MAGIC 3. Click the **Query endpoint** tab
# MAGIC 4. Paste a JSON payload (like the sample above) into the request body
# MAGIC 5. Click **Send request**
# MAGIC 6. View the prediction response
# MAGIC
# MAGIC The UI also provides:
# MAGIC - **Metrics** tab for monitoring latency, throughput, and error rates
# MAGIC - **Logs** tab for debugging
# MAGIC - **Events** tab for endpoint lifecycle events

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## D. Conclusion
# MAGIC
# MAGIC In this notebook, we:
# MAGIC - Created a **Model Serving endpoint** from a Unity Catalog registered model
# MAGIC - Queried the endpoint **programmatically** from a notebook
# MAGIC - Served predictions from the **offline feature store** when no online store is available
# MAGIC - Demonstrated how to use the **Serving UI** for testing and monitoring
# MAGIC
# MAGIC Key takeaways:
# MAGIC - Model Serving provides **low-latency REST API** access to your models
# MAGIC - Endpoints support **auto-scaling** and **scale-to-zero** for cost efficiency
# MAGIC - The **Serving UI** enables testing, monitoring, and A/B testing without code
# MAGIC - Integration with Unity Catalog ensures **governance** and **lineage** in production
# MAGIC
# MAGIC Next: Proceed to **07-Observability** for monitoring and explainability.
