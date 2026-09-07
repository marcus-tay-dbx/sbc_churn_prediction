# Databricks notebook source
# /// script
# dependencies = [
#   "databricks-feature-engineering",
# ]
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "databricks-feature-engineering",
# ]
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 00 - Cleanup (Full Workshop Reset)
# MAGIC
# MAGIC This notebook **tears down every asset** created by the SBC Churn Prediction workshop
# MAGIC so you can do a clean re-run from `00-Setup`.
# MAGIC
# MAGIC **What is deleted**
# MAGIC | Asset type | Examples |
# MAGIC |---|---|
# MAGIC | Schema + all tables | `{schema}.customer_churn`, `…_features`, `…_eval_log`, `…_inference_unpacked`, `churn_endpoint_payload`, drift/profile metrics tables |
# MAGIC | Volume | `raw_data` (6 raw parquet datasets) |
# MAGIC | UC Model + all versions | `bank_churn_model` (aliases `dev`, `champion`) |
# MAGIC | Lakehouse Monitors | on inference_unpacked, on batch inference log |
# MAGIC | Model Serving endpoint | `sbc-bank-churn-<user>` |
# MAGIC | Jobs | Deployment job (Evaluate→Approve→Deploy); Retrain-on-Drift job |
# MAGIC | MLflow experiment | `experiments/bank-churn-training` + all runs |
# MAGIC | Workspace monitoring dir | `{workshop_dir}/monitoring/` (dashboard files) |
# MAGIC | Schema | Entire `{catalog}.{schema}` (DROP SCHEMA CASCADE) |
# MAGIC
# MAGIC **What is NOT deleted**
# MAGIC - The `solution_builder` catalog
# MAGIC - The **Lakebase project / Postgres instance** (your prerequisite for re-running)
# MAGIC - Any assets belonging to other users
# MAGIC
# MAGIC > **Safe by default**: run with `dry_run = true` (the default) to preview every deletion
# MAGIC > without touching anything. Set `dry_run = false` **and** `confirm = yes` to execute.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.dropdown("dry_run", "true", ["true", "false"],
                         "Dry run (preview only — nothing is deleted)")
dbutils.widgets.text("confirm", "",
                     "Type YES to confirm deletion (only checked when dry_run=false)")

dry_run = dbutils.widgets.get("dry_run").lower() == "true"
confirm = dbutils.widgets.get("confirm").strip().upper()

if dry_run:
    print("=" * 60)
    print("DRY RUN — preview mode. Nothing will be deleted.")
    print("Set dry_run = false and confirm = YES to actually delete.")
    print("=" * 60)
elif confirm != "YES":
    raise Exception(
        "Safety check: set the 'confirm' widget to YES (uppercase) to proceed with deletion."
    )
else:
    print("=" * 60)
    print("⚠️  LIVE RUN — deletions are IRREVERSIBLE.")
    print("=" * 60)

# COMMAND ----------

# DBTITLE 1,Helpers
import re, time

deleted = []
skipped = []

def _log(action: str, name: str, detail: str = ""):
    tag = "  [DRY RUN]" if dry_run else "  ✅ DELETED"
    msg = f"{tag} {action}: {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    deleted.append({"action": action, "name": name})

def _skip(action: str, name: str, reason: str = "not found"):
    print(f"  ⏭️  SKIP    {action}: {name}  — {reason}")
    skipped.append({"action": action, "name": name, "reason": reason})

def _err(action: str, name: str, e):
    print(f"  ⚠️  ERROR   {action}: {name}  — {type(e).__name__}: {e}")

# COMMAND ----------

# DBTITLE 1,Step 1 — Delete Lakehouse Monitors
# MAGIC %md
# MAGIC ## 1. Lakehouse Monitors
# MAGIC The monitor must be deleted before dropping the table it sits on.

# COMMAND ----------

# DBTITLE 1,Delete Monitors
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

monitor_tables = [
    f"{DA.catalog_name}.{DA.schema_name}.customer_churn_inference_unpacked",
    f"{DA.catalog_name}.{DA.schema_name}.customer_churn_batch_inference_log",
    f"{DA.catalog_name}.{DA.schema_name}.customer_churn_eval_log",
]

print("\n[1] Lakehouse Monitors")
for tbl in monitor_tables:
    try:
        w.quality_monitors.get(tbl)
        _log("monitor", tbl)
        if not dry_run:
            w.quality_monitors.delete(table_name=tbl, purge_artifacts=True)
    except Exception as e:
        if "NOT_FOUND" in str(e) or "RESOURCE_DOES_NOT_EXIST" in str(e):
            _skip("monitor", tbl)
        else:
            _err("monitor", tbl, e)

# COMMAND ----------

# DBTITLE 1,Step 2 — Delete Serving Endpoint
# MAGIC %md
# MAGIC ## 2. Model Serving Endpoint

# COMMAND ----------

# DBTITLE 1,Delete Endpoint
print("\n[2] Model Serving Endpoint")
endpoint_name = DA.endpoint_name

try:
    w.serving_endpoints.get(endpoint_name)
    _log("endpoint", endpoint_name)
    if not dry_run:
        w.serving_endpoints.delete(name=endpoint_name)
        # Wait for deletion to propagate before removing the model.
        for _ in range(12):
            try:
                w.serving_endpoints.get(endpoint_name)
                time.sleep(5)
            except Exception:
                break
except Exception as e:
    if "NOT_FOUND" in str(e) or "RESOURCE_DOES_NOT_EXIST" in str(e) or "does not exist" in str(e).lower():
        _skip("endpoint", endpoint_name)
    else:
        _err("endpoint", endpoint_name, e)

# COMMAND ----------

# DBTITLE 1,Step 3 — Delete Jobs
# MAGIC %md
# MAGIC ## 3. Workflow Jobs (Deployment + Retrain-on-Drift)

# COMMAND ----------

# DBTITLE 1,Delete Jobs
print("\n[3] Workflow Jobs")
job_name_patterns = [
    f"SBC Bank Churn Deployment — {DA.model_name}",
    f"SBC Bank Churn Retrain-on-Drift — {DA.model_name}",
]

for pattern in job_name_patterns:
    found = [j for j in w.jobs.list(name=pattern)]
    if not found:
        _skip("job", pattern)
    else:
        for job in found:
            _log("job", f"{pattern} (id={job.job_id})")
            if not dry_run:
                try:
                    w.jobs.delete(job_id=job.job_id)
                except Exception as e:
                    _err("job", pattern, e)

# COMMAND ----------

# DBTITLE 1,Step 4 — Delete Registered Model
# MAGIC %md
# MAGIC ## 4. UC Registered Model (all versions + aliases)

# COMMAND ----------

# DBTITLE 1,Delete Model
from mlflow.tracking.client import MlflowClient

print("\n[4] UC Registered Model")
mlflow.set_registry_uri("databricks-uc")
uc_client = MlflowClient(registry_uri="databricks-uc")
model_name = DA.model_name

try:
    versions = uc_client.search_model_versions(f"name='{model_name}'")
    if not versions:
        _skip("model", model_name, "no versions found")
    else:
        print(f"     Found {len(versions)} version(s) of {model_name}")
        for v in versions:
            _log("model version", f"{model_name} v{v.version} (aliases={v.aliases})")
            if not dry_run:
                # Remove all aliases first, then delete the version.
                for alias in list(v.aliases or []):
                    try:
                        uc_client.delete_registered_model_alias(name=model_name, alias=alias)
                    except Exception:
                        pass
                try:
                    uc_client.delete_model_version(name=model_name, version=v.version)
                except Exception as e:
                    _err("model version", f"{model_name} v{v.version}", e)

        _log("registered model", model_name)
        if not dry_run:
            try:
                uc_client.delete_registered_model(model_name)
            except Exception as e:
                _err("registered model", model_name, e)
except Exception as e:
    if "NOT_FOUND" in str(e) or "RESOURCE_DOES_NOT_EXIST" in str(e):
        _skip("model", model_name)
    else:
        _err("model", model_name, e)

# COMMAND ----------

# DBTITLE 1,Step 5 — Delete MLflow Experiment
# MAGIC %md
# MAGIC ## 5. MLflow Experiment + All Runs

# COMMAND ----------

# DBTITLE 1,Delete Experiment
print("\n[5] MLflow Experiment")
experiment_path = DA.experiment_path

try:
    exp = mlflow.get_experiment_by_name(experiment_path)
    if exp is None:
        _skip("experiment", experiment_path)
    else:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="list")
        _log("experiment", f"{experiment_path}  ({len(runs)} runs)")
        if not dry_run:
            # Delete runs first.
            client_std = MlflowClient()
            for r in runs:
                try:
                    client_std.delete_run(r.info.run_id)
                except Exception:
                    pass
            mlflow.delete_experiment(exp.experiment_id)
except Exception as e:
    _err("experiment", experiment_path, e)

# COMMAND ----------

# DBTITLE 1,Step 6 — Delete Workspace Monitoring Directory
# MAGIC %md
# MAGIC ## 6. Workspace Files (Monitoring Dashboards)

# COMMAND ----------

# DBTITLE 1,Delete Monitoring Directory
print("\n[6] Workspace Monitoring Directory")
monitoring_dir = f"/Workspace{DA.workshop_dir}/monitoring"

try:
    dbutils.fs.ls(monitoring_dir.replace("/Workspace", ""))
    _log("workspace dir", monitoring_dir, "monitoring dashboards")
    if not dry_run:
        try:
            w.workspace.delete(path=monitoring_dir, recursive=True)
        except Exception as e:
            _err("workspace dir", monitoring_dir, e)
except Exception:
    _skip("workspace dir", monitoring_dir)

# COMMAND ----------

# DBTITLE 1,Step 7 — Drop Schema (CASCADE)
# MAGIC %md
# MAGIC ## 7. Drop Schema (CASCADE)
# MAGIC
# MAGIC Dropping the schema with `CASCADE` removes **everything inside it** in one shot:
# MAGIC all tables (including the feature table, base table, eval log, inference tables,
# MAGIC drift/profile metrics tables, payload table) and the `raw_data` volume.
# MAGIC
# MAGIC > The Lakebase synced table (`customer_churn_features_synced`) lives inside this
# MAGIC > schema as a Delta table. Dropping the schema removes the Delta table and its UC
# MAGIC > entry — but **the Lakebase Postgres project/instance itself is not touched**.
# MAGIC > You will need to re-create the synced table from the Lakebase UI after re-running
# MAGIC > notebook `03-Feature-Engineering`.

# COMMAND ----------

# DBTITLE 1,List Tables Before Drop
print("\n[7] Schema Drop")
full_schema = f"{DA.catalog_name}.{DA.schema_name}"

try:
    tables = spark.sql(f"SHOW TABLES IN {full_schema}").collect()
    volumes = spark.sql(f"SHOW VOLUMES IN {full_schema}").collect()
    print(f"     Schema {full_schema} contains:")
    print(f"       Tables/views: {len(tables)}")
    for t in tables:
        print(f"         • {t['tableName']}")
    print(f"       Volumes: {len(volumes)}")
    for v in volumes:
        print(f"         • {v['volume_name']}")
except Exception as e:
    print(f"     Could not list contents: {e}")

_log("schema", f"{full_schema} (CASCADE — drops all tables + volumes)")
if not dry_run:
    try:
        spark.sql(f"DROP SCHEMA IF EXISTS {full_schema} CASCADE")
        print(f"     Schema {full_schema} dropped.")
    except Exception as e:
        _err("schema", full_schema, e)

# COMMAND ----------

# DBTITLE 1,Step 8 — Summary
# MAGIC %md
# MAGIC ## 8. Summary

# COMMAND ----------

# DBTITLE 1,Summary
print("\n" + "=" * 60)
if dry_run:
    print("DRY RUN COMPLETE — nothing was deleted.")
    print(f"  Would delete: {len(deleted)} asset(s)")
    print(f"  Would skip:   {len(skipped)} asset(s) (already absent)")
    print()
    print("To run for real: set dry_run = false and confirm = YES")
else:
    print("CLEANUP COMPLETE")
    print(f"  Deleted: {len(deleted)} asset(s)")
    print(f"  Skipped: {len(skipped)} asset(s) (already absent)")
    print()
    print("Workshop is reset. Re-run from 00-Setup to start fresh.")
    print("Pre-requisite: your Lakebase project/instance is still intact.")
    print("Re-create the synced table from Notebook 03 (Section D) after running the workshop again.")
print("=" * 60)

display(spark.createDataFrame(
    [{"status": "WOULD_DELETE" if dry_run else "DELETED", **r} for r in deleted] +
    [{"status": "SKIPPED", **s, "action": s["action"], "name": s["name"]} for s in skipped]
))
