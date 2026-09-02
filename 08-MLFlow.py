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
# MAGIC # 08 - MLflow (Deployment Jobs)
# MAGIC
# MAGIC In this notebook, we create an **MLflow Deployment Job** and connect it to a registered model in Unity Catalog. When a new model version is registered, the deployment job **auto-triggers** to evaluate, approve, and deploy the model.
# MAGIC
# MAGIC **Prerequisites**: Run `00-Setup` and `04-Model-Training` first (the model must be registered in UC).

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Deployment Jobs Intro
# MAGIC %md-sandbox
# MAGIC ## A. Create a Deployment Job
# MAGIC
# MAGIC An **MLflow Deployment Job** automates the lifecycle of a model version once it is registered in Unity Catalog. When a new model version is created, the deployment job is **auto-triggered** to evaluate, approve, and deploy the model.
# MAGIC
# MAGIC <div style="display: flex; gap: 2rem; align-items: flex-start; margin-top: 16px;">
# MAGIC   <div style="flex: 1; background: #F9F7F4; border-top: 3px solid #FF5F46; border-radius: 8px; padding: 1.5rem;">
# MAGIC     <h4>Deployment Job Tasks</h4>
# MAGIC     <ol>
# MAGIC       <li><strong>Evaluation</strong> — score the new model version against a holdout set and log metrics</li>
# MAGIC       <li><strong>Approval</strong> — pause for a manual review; an approver can accept or reject</li>
# MAGIC       <li><strong>Deployment</strong> — promote the model (e.g., update a serving endpoint or set an alias)</li>
# MAGIC     </ol>
# MAGIC   </div>
# MAGIC   <div style="flex: 1; background: #EEEDE9; border-top: 3px solid #FF5F46; border-radius: 8px; padding: 1.5rem;">
# MAGIC     <h4>Key Requirements</h4>
# MAGIC     <ul>
# MAGIC       <li>Job must have two <strong>job-level parameters</strong>: <code>model_name</code> and <code>model_version</code></li>
# MAGIC       <li>Set <strong>max concurrent runs = 1</strong> to prevent deployment race conditions</li>
# MAGIC       <li>Databricks recommends using a <strong>service principal</strong> as the Run As identity</li>
# MAGIC     </ul>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC > **Docs**: [MLflow Deployment Jobs](https://docs.databricks.com/aws/en/mlflow/deployment-job/)

# COMMAND ----------

# DBTITLE 1,Create Deployment Job
w = WorkspaceClient()

# Step 1: Define paths to the three task notebooks
base_path = f"/Workspace{DA.workshop_dir}"

evaluation_notebook = f"{base_path}/08-MLFlow-Evaluate"
approval_notebook   = f"{base_path}/08-MLFlow-Approve"
deployment_notebook = f"{base_path}/08-MLFlow-Deploy"

# Step 2: Create the deployment job (or reuse an existing one)
job_name = f"SBC Bank Churn Deployment — {DA.model_name}"

existing_jobs = [j for j in w.jobs.list(name=job_name)]
if existing_jobs:
    created_job = existing_jobs[0]
    print(f"Deployment job already exists: {created_job.job_id}")
else:
    created_job = w.jobs.create(
        name=job_name,
        # Tag the job so it can be cleaned up with the rest of the workshop assets.
        tags={"sbc": "true", "churn-prediction": "true"},
        # Job-level parameters — auto-filled when a new model version triggers the job
        parameters=[
            JobParameterDefinition(name="model_name",    default=DA.model_name),
            JobParameterDefinition(name="model_version", default="1"),
        ],
        max_concurrent_runs=1,  # Prevent deployment race conditions
        tasks=[
            # Task 1: Score the candidate model on a holdout set
            Task(
                task_key="Evaluate",
                notebook_task=NotebookTask(
                    notebook_path=evaluation_notebook,
                    source=Source("WORKSPACE"),
                ),
            ),
            # Task 2: Gate — fails by default until an approver repairs it
            Task(
                task_key="Approve",
                depends_on=[TaskDependency(task_key="Evaluate")],
                notebook_task=NotebookTask(
                    notebook_path=approval_notebook,
                    source=Source("WORKSPACE"),
                ),
            ),
            # Task 3: Set the "champion" alias on the approved version
            Task(
                task_key="Deploy",
                depends_on=[TaskDependency(task_key="Approve")],
                notebook_task=NotebookTask(
                    notebook_path=deployment_notebook,
                    source=Source("WORKSPACE"),
                ),
            ),
        ],
    )
    print(f"Created deployment job: {created_job.job_id}")

print(f"View at: {w.config.host}/#job/{created_job.job_id}")

# COMMAND ----------

# DBTITLE 1,Connect Deployment Job to Model
client = MlflowClient(registry_uri="databricks-uc")

# Link the job to the model — new versions will auto-trigger the pipeline
try:
    client.update_registered_model(
        name=DA.model_name,
        deployment_job_id=str(created_job.job_id)
    )
    print(f"Connected deployment job {created_job.job_id} to model '{DA.model_name}'")
    print(f"\nNew model versions will now auto-trigger: Evaluate → Approve → Deploy.")
except Exception as e:
    print(f"Could not connect deployment job: {e}")
    print("You can also connect manually: Catalog UI → Model → Connect deployment job.")

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## B. Conclusion
# MAGIC
# MAGIC In this notebook, we:
# MAGIC - Created a **Deployment Job** with Evaluate → Approve → Deploy tasks using the Databricks SDK
# MAGIC - **Connected** the job to the registered UC model so new versions auto-trigger the pipeline
# MAGIC
# MAGIC The deployment job integrates with **Unity Catalog's CREATE MODEL VERSION ACL** — users with permission can register versions, and the job handles the rest.
# MAGIC
# MAGIC > **Next**: Return to **04-Model-Training** to retrain, or proceed to **09-Migration**.