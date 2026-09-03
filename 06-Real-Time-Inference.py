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
# MAGIC **Recommended pattern:** serve a model **registered with the Feature Store**, backed by an
# MAGIC **online feature store in Lakebase** — the endpoint looks features up by `customer_id` with low
# MAGIC latency, so callers send just the key.
# MAGIC
# MAGIC When Lakebase isn't available, there's a **technical workaround (for demo purposes)**: serve a
# MAGIC model **registered without the Feature Store** and pass the feature values directly in the request.
# MAGIC
# MAGIC The cells below **check for the online feature store**, then **deploy the matching model
# MAGIC automatically** — Feature Store model if present, otherwise the no-feature-store model.

# COMMAND ----------

# DBTITLE 1,Check Online Feature Store
# Check whether the online (synced) feature store exists — B1 uses this to pick the model.
synced_table_name = f"{DA.feature_table_name}_synced"
online_store_available = spark.catalog.tableExists(synced_table_name)
print(("✅ Online feature store found: " if online_store_available
       else "⚠️  No online feature store: ") + synced_table_name)

# COMMAND ----------

# DBTITLE 1,B1a. Select the Model to Serve
# Online store → serve the feature-store model; otherwise register + serve a no-feature-store model.
import os, pickle
from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")
client = get_deploy_client("databricks")
endpoint_name = DA.endpoint_name
_mc = MlflowClient(registry_uri="databricks-uc")

if online_store_available:
    model_name = DA.model_name          # fe-model → online lookup by customer_id
else:
    # Register a no-feature-store model from the fe-model's estimator.
    model_name = DA.model_name_no_fs
    try:
        _mc.get_model_version_by_alias(model_name, "dev")
        print(f"No-feature-store model already registered: {model_name}")
    except Exception:
        art = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{DA.model_name}@dev")
        sk_model = None
        for root, _dirs, files in os.walk(art):
            for fn in files:
                if fn.endswith(".pkl"):
                    with open(os.path.join(root, fn), "rb") as fh:
                        sk_model = pickle.load(fh)
                    break
            if sk_model is not None:
                break
        sample = spark.table(DA.feature_table_name).select(*DA.feature_columns).limit(5).toPandas()
        with mlflow.start_run(run_name="customer_churn-no-feature-store"):
            info = mlflow.sklearn.log_model(
                sk_model=sk_model, artifact_path="customer_churn_model_no_feature_store",
                signature=infer_signature(sample, sk_model.predict(sample)), input_example=sample,
            )
        reg = mlflow.register_model(model_uri=info.model_uri, name=model_name)
        _mc.set_registered_model_alias(name=model_name, alias="dev", version=reg.version)
        for k, v in DA.tags.items():
            _mc.set_registered_model_tag(name=model_name, key=k, value=v)
        print(f"Registered no-feature-store model: {model_name} v{reg.version}")

# COMMAND ----------

# DBTITLE 1,B1b. Deploy the Endpoint
# Serve the champion (else dev) version; create the endpoint if it doesn't exist yet.
try:
    serving_version = str(_mc.get_model_version_by_alias(model_name, "champion").version)
except Exception:
    serving_version = str(_mc.get_model_version_by_alias(model_name, "dev").version)
print(f"Serving {model_name} version {serving_version}")

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
# Captures each request/response to `<prefix>_payload` for monitoring & drift detection in Notebeook 07.
from databricks.sdk import WorkspaceClient

_w = WorkspaceClient()
try:
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
except Exception as e:
    if "already exists" in str(e):
        print("Inference logging table already exists (prior run) — leaving it in place.")
    else:
        raise

# COMMAND ----------

# DBTITLE 1,Deploy from UI
# MAGIC %md
# MAGIC ### B2. From the UI
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
# MAGIC The query **matches how the model was deployed** (the `online_store_available` flag): with an
# MAGIC online store → pass just `customer_id`; without one → pass the feature values directly.

# COMMAND ----------

# DBTITLE 1,Query Endpoint Code
if online_store_available:
    # Feature-store model → pass only customer_id; the endpoint looks features up online.
    payload = {"dataframe_records": [{"customer_id": c}
                                     for c in ["CUST-0000214", "CUST-0000001", "CUST-0000002"]]}
    print("Querying by customer_id (online feature lookup)")
else:
    # No-feature-store model → pass the 10 feature columns directly (read from the offline table).
    sample = spark.table(DA.feature_table_name).select(*DA.feature_columns).limit(3).toPandas()
    payload = {"dataframe_records": sample.to_dict("records")}
    print("Querying by passing the 10 features directly (no online store)")

response = client.predict(endpoint=endpoint_name, inputs=payload)
print(json.dumps(response, indent=2))

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
# MAGIC Next: Proceed to **Notebook 07 (Observability & Continuous Retrain)** for monitoring and explainability.