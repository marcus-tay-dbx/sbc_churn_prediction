# Databricks notebook source
# /// script
# dependencies = [
#   "dataiku-scoring==14.7.0",
#   "xgboost==2.1.4",
# ]
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "dataiku-scoring==14.7.0",
#   "xgboost==2.1.4",
# ]
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # BONUS — Migrating a Dataiku Model to Databricks
# MAGIC
# MAGIC <div style="background:#FFF8F0;border-left:4px solid #FF5F46;padding:14px 18px;border-radius:6px;margin-bottom:8px;">
# MAGIC <b>What this notebook does</b><br/>
# MAGIC Takes a model exported from <b>Dataiku DSS</b> as an MLflow bundle and brings it into
# MAGIC the Databricks ML lifecycle end-to-end — validating parity, registering in Unity Catalog,
# MAGIC deploying to a Model Serving endpoint, and running batch inference on new data.
# MAGIC </div>
# MAGIC
# MAGIC ## Background
# MAGIC
# MAGIC Dataiku DSS can export any trained model as an **MLflow bundle** (a `.zip` file containing
# MAGIC an `MLmodel` spec plus the serialised scoring artefacts). Because the export uses the
# MAGIC standard MLflow `python_function` interface, Databricks can load, register, and serve it
# MAGIC **without needing a Dataiku licence or installation** — only the lightweight
# MAGIC `dataiku-scoring` pip package is required.
# MAGIC
# MAGIC ### The model we're importing
# MAGIC | Property | Value |
# MAGIC |---|---|
# MAGIC | Task | Binary classification — predict customer churn (`Target`) |
# MAGIC | Algorithm | XGBoost (`trees=6, max_depth=4`) |
# MAGIC | Features used | 25 (selected by Dataiku from 163 input columns) |
# MAGIC | Optimal threshold | 0.325 |
# MAGIC | Training runtime | Dataiku DSS 14.7.0 / `dataikuscoring==14.7.0` |
# MAGIC
# MAGIC ### Migration path
# MAGIC ```
# MAGIC  Dataiku DSS                Databricks
# MAGIC  ─────────────              ──────────────────────────────────────────────
# MAGIC  Export model               Upload to UC Volume
# MAGIC  (MLflow bundle .zip)  →    Load + validate parity
# MAGIC                        →    Register in Unity Catalog Model Registry
# MAGIC                        →    Deploy to Model Serving endpoint
# MAGIC                        →    Batch inference via Feature Store / direct scoring
# MAGIC ```
# MAGIC
# MAGIC **Prerequisites**
# MAGIC 1. Upload `sample_mlflow_model.zip` to a UC Volume (see **Section B** for the path widget)
# MAGIC 2. Upload `sample_data_for_model.csv` to the same Volume
# MAGIC 3. Run this notebook on **Serverless ML v5** compute

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install "dataiku-scoring==14.7.0" "xgboost==2.1.4" --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup
import os, re, json, zipfile, warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
import dataikuscoring.mlflow
from mlflow.tracking.client import MlflowClient
from databricks.sdk import WorkspaceClient

warnings.filterwarnings("ignore")

# User identity (used for schema and endpoint naming)
username   = spark.sql("SELECT current_user()").collect()[0][0]
name_part  = username.split("@")[0]
initials   = "".join(p[0] for p in name_part.replace("-", ".").split(".") if p)

print(f"Username:  {username}")
print(f"Initials:  {initials}")

# COMMAND ----------

# DBTITLE 1,Widgets
# ──────────────────────────────────────────────────────────────────────────
# 👉  Set these before running the notebook.
# ──────────────────────────────────────────────────────────────────────────
dbutils.widgets.text("catalog_name",    "solution_builder",        "UC Catalog")
dbutils.widgets.text("schema_name",     f"{initials}_dataiku_migration", "UC Schema")
dbutils.widgets.text("model_zip_path",
    "",  # e.g. /Volumes/solution_builder/shared/sample_mlflow_model.zip
    "Path to sample_mlflow_model.zip (UC Volume)")
dbutils.widgets.text("sample_csv_path",
    "",  # e.g. /Volumes/solution_builder/shared/sample_data_for_model.csv
    "Path to sample_data_for_model.csv (UC Volume)")

catalog_name    = dbutils.widgets.get("catalog_name")
schema_name     = dbutils.widgets.get("schema_name")
model_zip_path  = dbutils.widgets.get("model_zip_path").strip()
sample_csv_path = dbutils.widgets.get("sample_csv_path").strip()

model_name      = f"{catalog_name}.{schema_name}.dataiku_churn_model"
endpoint_name   = "dataiku-churn-" + re.sub(r"[^a-zA-Z0-9-]", "-", username)

# Create the schema if it doesn't exist yet.
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")

print(f"Catalog:        {catalog_name}")
print(f"Schema:         {schema_name}")
print(f"Model:          {model_name}")
print(f"Endpoint:       {endpoint_name}")
print(f"Model zip:      {model_zip_path or '⚠️  NOT SET — fill in the widget above'}")
print(f"Sample CSV:     {sample_csv_path or '⚠️  NOT SET — fill in the widget above'}")

# COMMAND ----------

# DBTITLE 1,Section A — How Dataiku MLflow Exports Work
# MAGIC %md
# MAGIC ## A. How Dataiku MLflow Exports Work
# MAGIC
# MAGIC When you export a model from Dataiku DSS 14+, you get a `.zip` file with this structure:
# MAGIC
# MAGIC ```
# MAGIC sample_mlflow_model.zip
# MAGIC └── saved_models/
# MAGIC     └── Predict_Target__binary_/
# MAGIC         └── active_model/
# MAGIC             └── mlflow_export/
# MAGIC                 ├── scoring_mlflow.zip   ← the actual MLflow bundle
# MAGIC                 ├── requirements.txt     ← dataiku-scoring==14.7.0
# MAGIC                 └── use_databricks_mlflow.py  ← Dataiku's own helper
# MAGIC ```
# MAGIC
# MAGIC Inside `scoring_mlflow.zip`:
# MAGIC ```
# MAGIC MLmodel          ← defines two flavors: 'dss' and 'python_function'
# MAGIC conda.yaml       ← environment spec
# MAGIC model.zip        ← the serialised model + preprocessing pipeline
# MAGIC ```
# MAGIC
# MAGIC **Two loading options:**
# MAGIC | Option | Code | Best for |
# MAGIC |---|---|---|
# MAGIC | **Generic pyfunc** | `mlflow.pyfunc.load_model(path)` | Standard MLflow predict; no probabilities |
# MAGIC | **DSS flavor** ✅ | `dataikuscoring.mlflow.load_model(path)` | Full parity with Dataiku; `predict_proba()` available |
# MAGIC
# MAGIC We'll use the **DSS flavor** — it calls the exact same scoring code Dataiku uses in
# MAGIC production, so predictions are byte-for-byte identical.

# COMMAND ----------

# DBTITLE 1,Section B — Extract the MLflow Bundle
# MAGIC %md
# MAGIC ## B. Extract the MLflow Bundle
# MAGIC
# MAGIC The model zip arrives with a nested structure (Dataiku wraps the MLflow bundle inside
# MAGIC a project-level zip). We unwrap it in two stages:
# MAGIC 1. Outer zip → locate `scoring_mlflow.zip`
# MAGIC 2. Inner zip → the `MLmodel` directory we actually load

# COMMAND ----------

# DBTITLE 1,Locate and Extract Model
# ── Step 1: resolve the model zip (Volume path or local /tmp) ────────────
if not model_zip_path:
    raise ValueError(
        "Set the 'model_zip_path' widget to the UC Volume path of sample_mlflow_model.zip.\n"
        "Example: /Volumes/solution_builder/shared/sample_mlflow_model.zip"
    )

# UC Volume paths start with /Volumes/; DBFS paths start with /dbfs/.
# Both are accessible as local filesystem paths on Databricks.
if model_zip_path.startswith("/Volumes/"):
    local_zip = model_zip_path          # direct access on Databricks
elif model_zip_path.startswith("dbfs:/"):
    local_zip = model_zip_path.replace("dbfs:/", "/dbfs/")
else:
    local_zip = model_zip_path

if not os.path.exists(local_zip):
    raise FileNotFoundError(
        f"Cannot find the model zip at: {local_zip}\n"
        "Upload the file to a UC Volume and update the widget."
    )

# ── Step 2: unpack outer zip → find scoring_mlflow.zip ───────────────────
work_dir = "/tmp/dataiku_model_migration"
os.makedirs(work_dir, exist_ok=True)

print(f"Extracting outer zip: {local_zip}")
with zipfile.ZipFile(local_zip, "r") as z:
    z.extractall(work_dir)

scoring_zip = None
for root, dirs, files in os.walk(work_dir):
    for f in files:
        if f == "scoring_mlflow.zip":
            scoring_zip = os.path.join(root, f)
            break
    if scoring_zip:
        break

if not scoring_zip:
    raise FileNotFoundError(
        "scoring_mlflow.zip not found inside the outer zip. "
        "Make sure you're using the full Dataiku MLflow export package."
    )
print(f"Found scoring_mlflow.zip: {scoring_zip}")

# ── Step 3: unpack inner zip → the MLmodel directory ─────────────────────
mlflow_extract_dir = os.path.join(work_dir, "mlflow_extracted")
os.makedirs(mlflow_extract_dir, exist_ok=True)

with zipfile.ZipFile(scoring_zip, "r") as z:
    z.extractall(mlflow_extract_dir)

# Locate the directory that contains the MLmodel file.
model_dir = None
for root, dirs, files in os.walk(mlflow_extract_dir):
    if "MLmodel" in files:
        model_dir = root
        break

if model_dir is None:
    raise FileNotFoundError("MLmodel not found in the extracted bundle.")

print(f"MLmodel directory:    {model_dir}")
print(f"\nBundle contents:")
for f in sorted(os.listdir(model_dir)):
    print(f"  {f}")

# COMMAND ----------

# DBTITLE 1,Read MLmodel Spec
with open(os.path.join(model_dir, "MLmodel")) as fh:
    mlmodel_text = fh.read()

print("MLmodel spec:")
print("─" * 50)
print(mlmodel_text)

# COMMAND ----------

# DBTITLE 1,Section C — Load the Model (DSS Flavor)
# MAGIC %md
# MAGIC ## C. Load the Model — DSS Flavor
# MAGIC
# MAGIC We load using `dataikuscoring.mlflow.load_model()` — the **preferred path** because:
# MAGIC - It uses Dataiku's exact preprocessing pipeline (same scaling, encoding, imputation)
# MAGIC - It exposes `predict_proba()` for probabilities (not available in generic pyfunc)
# MAGIC - It guarantees **byte-for-byte parity** with scores produced in Dataiku DSS

# COMMAND ----------

# DBTITLE 1,Load with DSS Flavor
print("Loading model with Dataiku DSS flavor …")
dss_model = dataikuscoring.mlflow.load_model(model_dir)
print(f"✅ Loaded: {type(dss_model)}")

# Also load the generic pyfunc (for side-by-side comparison later).
pyfunc_model = mlflow.pyfunc.load_model(model_dir)
print(f"✅ Loaded pyfunc:  {type(pyfunc_model)}")

# COMMAND ----------

# DBTITLE 1,Section D — Validate on Sample Data
# MAGIC %md
# MAGIC ## D. Validate — Score the Sample Data
# MAGIC
# MAGIC We load the customer CSV provided alongside the model export and run it through
# MAGIC **both** loading options to confirm they produce the same predictions.
# MAGIC
# MAGIC > The CSV contains a `Target` column (ground-truth labels). We'll hold that aside
# MAGIC > and pass the remaining columns to the model, then compare predictions vs actuals.

# COMMAND ----------

# DBTITLE 1,Load Sample CSV
if not sample_csv_path:
    raise ValueError(
        "Set the 'sample_csv_path' widget to the UC Volume path of sample_data_for_model.csv.\n"
        "Example: /Volumes/solution_builder/shared/sample_data_for_model.csv"
    )

if sample_csv_path.startswith("/Volumes/"):
    local_csv = sample_csv_path
elif sample_csv_path.startswith("dbfs:/"):
    local_csv = sample_csv_path.replace("dbfs:/", "/dbfs/")
else:
    local_csv = sample_csv_path

if not os.path.exists(local_csv):
    raise FileNotFoundError(f"Cannot find sample CSV at: {local_csv}")

sample_df = pd.read_csv(local_csv)
print(f"Sample data:  {sample_df.shape[0]:,} rows × {sample_df.shape[1]} columns")
print(f"Label column: 'Target'  (1 = churn, 0 = retained)")
print(f"Churn rate:   {sample_df['Target'].mean():.1%}")

# Separate features from label.
TARGET_COL = "Target"
X_sample   = sample_df.drop(columns=[TARGET_COL])
y_sample   = sample_df[TARGET_COL]

display(sample_df.head(5))

# COMMAND ----------

# DBTITLE 1,Score with DSS Flavor
# DSS flavor: returns binary predictions (0/1) at the optimal threshold (0.325).
dss_preds  = dss_model.predict(X_sample)
dss_probas = dss_model.predict_proba(X_sample)   # shape: (n, 2) → [p_retain, p_churn]

print(f"predict()      → shape {np.array(dss_preds).shape},  dtype {np.array(dss_preds).dtype}")
print(f"predict_proba()→ shape {np.array(dss_probas).shape}, dtype {np.array(dss_probas).dtype}")
print(f"\nFirst 5 predictions (binary):       {list(dss_preds[:5])}")
print(f"First 5 churn probabilities (col 1): {[round(float(p[1]), 4) for p in dss_probas[:5]]}")

# COMMAND ----------

# DBTITLE 1,Score with Generic Pyfunc
# Generic pyfunc: also works, returns the same binary predictions.
pyfunc_preds = pyfunc_model.predict(X_sample)

print(f"pyfunc predict() → shape {np.array(pyfunc_preds).shape}")
print(f"First 5: {list(np.array(pyfunc_preds).flatten()[:5])}")

# COMMAND ----------

# DBTITLE 1,Parity Check
# Both loaders must produce identical binary predictions.
dss_arr    = np.array(dss_preds).flatten().astype(int)
pyfunc_arr = np.array(pyfunc_preds).flatten().astype(int)

match_pct = (dss_arr == pyfunc_arr).mean()
print(f"DSS vs pyfunc agreement: {match_pct:.1%}  ({(dss_arr == pyfunc_arr).sum()} / {len(dss_arr)} predictions identical)")

if match_pct == 1.0:
    print("✅ Perfect parity — both loaders produce identical predictions.")
else:
    print(f"⚠️  {(dss_arr != pyfunc_arr).sum()} predictions differ (likely floating-point rounding).")

# COMMAND ----------

# DBTITLE 1,Quality Metrics vs Ground Truth
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report
)

churn_probas = np.array([float(p[1]) for p in dss_probas])
THRESHOLD    = 0.325   # Dataiku's optimal threshold

y_pred_thresh = (churn_probas >= THRESHOLD).astype(int)

print(f"Metrics at Dataiku threshold ({THRESHOLD}):")
print(f"  Accuracy:  {accuracy_score(y_sample, y_pred_thresh):.4f}")
print(f"  Precision: {precision_score(y_sample, y_pred_thresh, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_sample, y_pred_thresh, zero_division=0):.4f}")
print(f"  F1 (macro):{f1_score(y_sample, y_pred_thresh, average='macro'):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_sample, churn_probas):.4f}")
print()
print(classification_report(y_sample, y_pred_thresh, target_names=["Retained", "Churned"]))

# COMMAND ----------

# DBTITLE 1,Section E — Register in Unity Catalog
# MAGIC %md
# MAGIC ## E. Register in Unity Catalog Model Registry
# MAGIC
# MAGIC Now that we've validated the model produces correct predictions, we register it
# MAGIC permanently in **Unity Catalog** so it can be governed, versioned, and served.
# MAGIC
# MAGIC **Why log with `dataikuscoring.mlflow.log_model` instead of `mlflow.pyfunc.log_model`?**
# MAGIC
# MAGIC Dataiku's log function preserves the full DSS artifact bundle (the `model.zip` with
# MAGIC its preprocessing pipeline) and sets the correct `dataikuscoring` dependency in the
# MAGIC model's `requirements.txt`. This means anyone who loads the model from the registry
# MAGIC gets the same environment and the same `predict_proba()` capability.

# COMMAND ----------

# DBTITLE 1,Log and Register Model
mlflow.set_registry_uri("databricks-uc")
uc_client = MlflowClient(registry_uri="databricks-uc")

experiment_path = f"/Users/{username}/dataiku-migration"
mlflow.set_experiment(experiment_path)

with mlflow.start_run(run_name="dataiku-import") as run:
    # Log the key metrics we computed above so they appear in the Experiments UI.
    mlflow.log_params({
        "source":          "Dataiku DSS 14.7.0",
        "algorithm":       "XGBoost",
        "trees":           6,
        "max_depth":       4,
        "threshold":       THRESHOLD,
        "n_features_used": 25,
        "n_features_total": X_sample.shape[1],
        "target_column":   TARGET_COL,
    })
    mlflow.log_metrics({
        "accuracy":  round(accuracy_score(y_sample, y_pred_thresh), 4),
        "f1_macro":  round(f1_score(y_sample, y_pred_thresh, average="macro"), 4),
        "roc_auc":   round(roc_auc_score(y_sample, churn_probas), 4),
        "precision": round(precision_score(y_sample, y_pred_thresh, zero_division=0), 4),
        "recall":    round(recall_score(y_sample, y_pred_thresh, zero_division=0), 4),
    })

    # Log the model with Dataiku's DSS flavor.
    # This bundles the scoring_mlflow.zip and sets the correct pip requirements.
    dataikuscoring.mlflow.log_model(
        dss_model,
        artifact_path="model",
        registered_model_name=model_name,
    )

    model_uri = f"runs:/{run.info.run_id}/model"
    run_id    = run.info.run_id

print(f"✅ Logged run:      {run_id}")
print(f"   Experiment:     {experiment_path}")
print(f"   Model URI:      {model_uri}")
print(f"   Registered as:  {model_name}")

# COMMAND ----------

# DBTITLE 1,Set Champion Alias and Tags
# Retrieve the version that was just registered.
versions = uc_client.search_model_versions(f"name='{model_name}'")
latest   = sorted(versions, key=lambda v: int(v.version))[-1]
version  = latest.version

# Tag the version with provenance information.
for key, value in {
    "source":       "Dataiku DSS 14.7.0",
    "algorithm":    "XGBoost",
    "threshold":    str(THRESHOLD),
    "imported_by":  username,
}.items():
    uc_client.set_model_version_tag(model_name, version, key, value)

# Promote to 'champion' — this is the version the endpoint will serve.
uc_client.set_registered_model_alias(model_name, "champion", version)

print(f"✅ Model {model_name} v{version} tagged and aliased as 'champion'")
print(f"\nView in Catalog Explorer:")
print(f"  Catalog → {model_name.split('.')[0]} → {model_name.split('.')[1]} → {model_name.split('.')[2]}")

# COMMAND ----------

# DBTITLE 1,Section F — Deploy to Model Serving
# MAGIC %md
# MAGIC ## F. Deploy to Model Serving
# MAGIC
# MAGIC With the model registered as **champion**, we deploy it as a low-latency REST endpoint.
# MAGIC
# MAGIC The serving runtime will automatically install `dataiku-scoring==14.7.0` from the
# MAGIC model's `requirements.txt`, so no manual environment setup is needed.
# MAGIC
# MAGIC > **Tip**: The endpoint takes ~5–10 minutes to reach `READY` state while Databricks
# MAGIC > provisions the container and installs dependencies. The cell below waits for it.

# COMMAND ----------

# DBTITLE 1,Create or Update Serving Endpoint
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, TrafficConfig, Route
from databricks.sdk.service.serving import EndpointStateReady

w = WorkspaceClient()

served_entity = ServedEntityInput(
    name          = "champion",
    entity_name   = model_name,
    entity_version= str(version),
    workload_size = "Small",
    scale_to_zero_enabled=True,
)

try:
    ep = w.serving_endpoints.get(endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists — updating to version {version} …")
    w.serving_endpoints.update_config(
        name           = endpoint_name,
        served_entities= [served_entity],
    )
except Exception as e:
    if "NOT_FOUND" in str(e) or "does not exist" in str(e).lower():
        print(f"Creating endpoint '{endpoint_name}' …")
        w.serving_endpoints.create_and_wait(
            name   = endpoint_name,
            config = EndpointCoreConfigInput(
                served_entities=[served_entity],
                traffic_config=TrafficConfig(
                    routes=[Route(served_model_name="champion", traffic_percentage=100)]
                ),
            ),
        )
    else:
        raise

# Wait until READY.
import time
print("Waiting for endpoint to be READY ", end="")
for _ in range(60):
    state = w.serving_endpoints.get(endpoint_name).state.ready
    if state == EndpointStateReady.READY:
        print(" ✅")
        break
    print(".", end="", flush=True)
    time.sleep(10)
else:
    print("\n⚠️  Timed out waiting — check the Serving UI for status.")

print(f"\nEndpoint URL: {w.config.host}/serving-endpoints/{endpoint_name}/invocations")

# COMMAND ----------

# DBTITLE 1,Section G — Query the Endpoint
# MAGIC %md
# MAGIC ## G. Query the Endpoint
# MAGIC
# MAGIC The endpoint accepts JSON in the standard MLflow `dataframe_records` format —
# MAGIC the same format the model used in Dataiku.
# MAGIC
# MAGIC We'll send a **small sample** first to verify the endpoint responds correctly,
# MAGIC then run the full batch.

# COMMAND ----------

# DBTITLE 1,Query — Single Customer
import json

def _safe_val(v):
    """Cast numpy/pandas scalars to plain Python for JSON serialisation."""
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v

def _df_to_records(df: pd.DataFrame) -> list:
    return [{k: _safe_val(v) for k, v in row.items()} for row in df.to_dict("records")]


from mlflow.deployments import get_deploy_client
deploy_client = get_deploy_client("databricks")

# Score a single customer — the hero customer from the sample dataset.
single_row = X_sample.iloc[[0]]
payload    = {"dataframe_records": _df_to_records(single_row)}

response   = deploy_client.predict(endpoint=endpoint_name, inputs=payload)
print("Single-customer endpoint response:")
print(json.dumps(response, indent=2))
print(f"\nGround truth (Target): {int(y_sample.iloc[0])}")

# COMMAND ----------

# DBTITLE 1,Query — Batch (all sample rows)
BATCH_SIZE = 50   # keep each request small; iterate over the full dataset

all_preds_endpoint = []

for start in range(0, len(X_sample), BATCH_SIZE):
    batch   = X_sample.iloc[start : start + BATCH_SIZE]
    payload = {"dataframe_records": _df_to_records(batch)}
    resp    = deploy_client.predict(endpoint=endpoint_name, inputs=payload)
    preds   = resp.get("predictions", [])
    all_preds_endpoint.extend(preds)

all_preds_endpoint = np.array(all_preds_endpoint).flatten().astype(int)

# Sanity: endpoint predictions should match the locally scored ones.
match_pct = (all_preds_endpoint == dss_arr).mean()
print(f"Rows scored via endpoint:     {len(all_preds_endpoint):,}")
print(f"Match vs local DSS scores:    {match_pct:.1%}  ({'✅ perfect parity' if match_pct == 1.0 else '⚠️ check for rounding'})")
print(f"Churn rate (endpoint preds):  {all_preds_endpoint.mean():.1%}")

# COMMAND ----------

# DBTITLE 1,Section H — Save Results to Delta Table
# MAGIC %md
# MAGIC ## H. Save Scored Results to a Delta Table
# MAGIC
# MAGIC Best practice is to persist the scored output in Unity Catalog so downstream
# MAGIC teams (risk, retention, marketing) can query it directly via SQL or a dashboard.

# COMMAND ----------

# DBTITLE 1,Write Scored Table
results_df = X_sample[["bbnumber", "customercategory", "clientsegment",
                        "segmenttype", "economicactivity"]].copy()
results_df["churn_probability"] = [round(float(p[1]), 6) for p in dss_probas]
results_df["predicted_churn"]   = dss_arr
results_df["actual_churn"]      = y_sample.values
results_df["correct"]           = (results_df["predicted_churn"] == results_df["actual_churn"])
results_df["scored_at"]         = pd.Timestamp.now()

output_table = f"{catalog_name}.{schema_name}.dataiku_churn_scored"
spark.createDataFrame(results_df).write.mode("overwrite").saveAsTable(output_table)

n = spark.table(output_table).count()
print(f"✅ Saved {n:,} scored rows → {output_table}")
display(spark.table(output_table).limit(10))

# COMMAND ----------

# DBTITLE 1,Section I — Architecture Summary
# MAGIC %md
# MAGIC ## I. What We Built
# MAGIC
# MAGIC <div style="background:#F4FBF8;border-left:4px solid #00A972;padding:14px 18px;border-radius:6px;">
# MAGIC <b>Migration complete.</b> Here is what now lives in Databricks:
# MAGIC </div>
# MAGIC <br/>
# MAGIC
# MAGIC | Asset | Where | Notes |
# MAGIC |---|---|---|
# MAGIC | **Model versions** | UC Model Registry | Versioned, with `champion` alias, provenance tags |
# MAGIC | **Training metrics** | MLflow Experiment | AUC, F1, precision, recall logged against the registered run |
# MAGIC | **Serving endpoint** | Model Serving | Auto-scales, scale-to-zero; identical predictions to Dataiku |
# MAGIC | **Scored output** | Delta table in UC | Queryable via SQL, ready for dashboards |
# MAGIC
# MAGIC ### What Databricks now gives you on top of the imported model
# MAGIC
# MAGIC | Capability | How to use it |
# MAGIC |---|---|
# MAGIC | **Lineage** | Catalog Explorer → Model → Lineage tab shows the run that produced it |
# MAGIC | **Access control** | UC grants on the model, table, and endpoint |
# MAGIC | **Drift monitoring** | Create a Lakehouse Monitor on `dataiku_churn_scored` (see Notebook 07) |
# MAGIC | **Deployment job** | Wire the model to an Evaluate→Approve→Deploy job (see Notebook 08) |
# MAGIC | **Retraining** | Train the next version natively in Databricks (Notebooks 03–04) and register under the same model name — the `champion` alias makes the endpoint update seamless |
# MAGIC
# MAGIC ### Key takeaway for your team
# MAGIC > Moving from Dataiku to Databricks does **not** require retraining the model.
# MAGIC > The MLflow export is the bridge — you bring the existing model across, validate parity,
# MAGIC > register it under UC governance, and serve it immediately. Retraining is an optional
# MAGIC > next step, not a prerequisite.

# COMMAND ----------

# DBTITLE 1,Final Summary
w2 = WorkspaceClient()
ep_url = f"{w2.config.host}/serving-endpoints/{endpoint_name}"

print("=" * 65)
print("DATAIKU → DATABRICKS MIGRATION COMPLETE")
print("=" * 65)
print(f"  Registered model:  {model_name}  (v{version}, alias=champion)")
print(f"  Experiment:        {experiment_path}")
print(f"  Scored table:      {output_table}")
print(f"  Endpoint:          {ep_url}")
print()
print(f"  Rows validated:    {len(dss_arr):,}")
print(f"  ROC-AUC:           {round(roc_auc_score(y_sample, churn_probas), 4)}")
print(f"  Parity (endpoint): {match_pct:.1%}")
print("=" * 65)
