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
# MAGIC # 04 - Model Training 
# MAGIC
# MAGIC In this notebook, we train **two** models on the churn feature table — a **Random Forest** and an **XGBoost** classifier — track both as separate **MLflow** runs, and register the **best model by PR-AUC** in **Unity Catalog**.
# MAGIC
# MAGIC **Evaluation metrics** (for an imbalanced ~8-12% churn target):
# MAGIC - `pr_auc` — precision-recall AUC, the primary metric for a rare positive class
# MAGIC - `recall_pos` / `precision_pos` — churn-class recall & precision
# MAGIC - `test_f1` — macro-F1, kept as the tracked number the deployment gate reads
# MAGIC
# MAGIC **Prerequisites**: Run `00-Setup` and `03-Feature-Engineering` first.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Install Dependencies
# MAGIC %pip install xgboost --quiet

# COMMAND ----------

# DBTITLE 1,Train with Feature Lookup Intro
# MAGIC %md
# MAGIC ## A. Train with Feature Lookup and MLflow
# MAGIC
# MAGIC Instead of manually loading and joining features, we use **`FeatureLookup`** to let the Feature Store handle it:
# MAGIC
# MAGIC 1. **Labels DataFrame** — contains only the lookup key (`customer_id`) and the label (`churned`)
# MAGIC 2. **`FeatureLookup`** — tells the Feature Store which features to join by primary key
# MAGIC 3. **`fe.create_training_set()`** — automatically joins features to labels
# MAGIC 4. **`fe.log_model()`** — packages feature metadata with the model so `fe.score_batch()` can replay the same lookups at inference time
# MAGIC
# MAGIC This ensures **consistency** between training and inference — the exact same features are used in both.

# COMMAND ----------

# DBTITLE 1,Build the Training Set
fe = FeatureEngineeringClient()

# 1. Labels DataFrame: only the lookup key (customer_id) + the label (churned)
labels_df = spark.table("customer_churn").select("customer_id", "churned")

# 2. Define which features to look up from the feature table
feature_table_name = DA.feature_table_name

feature_lookups = [
    FeatureLookup(
        table_name=feature_table_name,
        feature_names=DA.feature_columns,
        lookup_key="customer_id"
    )
]

# 3. Create a training set — Feature Store joins features by customer_id
training_set = fe.create_training_set(
    df=labels_df,
    feature_lookups=feature_lookups,
    label="churned",
    exclude_columns=["customer_id"]
)

# 4. Load as pandas and prepare for sklearn
train_pdf = training_set.load_df().toPandas()

X = train_pdf.drop(columns=["churned"])
y = train_pdf["churned"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
print(f"Features:     {list(X_train.columns)}")
print(f"Churn rate:   {y.mean():.1%}")

# COMMAND ----------

# DBTITLE 1,Evaluation Metrics Intro
# MAGIC %md
# MAGIC ## B. Train & Compare Models
# MAGIC
# MAGIC Churn is **imbalanced**, so plain accuracy is misleading (predicting "nobody churns" scores ~91%). We evaluate on metrics that focus on the positive (churn) class:
# MAGIC
# MAGIC | Metric | What it measures |
# MAGIC | --- | --- |
# MAGIC | `pr_auc` | Precision-recall AUC — primary, threshold-free summary for a rare class |
# MAGIC | `recall_pos` | Of customers who actually churn, how many we flag |
# MAGIC | `precision_pos` | Of customers we flag, how many really churn |
# MAGIC | `test_f1` | Macro-F1 — kept as the tracked number the deployment gate (08) reads |
# MAGIC
# MAGIC Each model is logged as its **own MLflow run** (`RandomForest`, `XGBoost`) so you can compare them side by side in the Experiment UI.

# COMMAND ----------

# DBTITLE 1,Set the MLflow Experiment
experiment_path = DA.experiment_path
if mlflow.get_experiment_by_name(experiment_path) is None:
    mlflow.create_experiment(experiment_path)
    print(f"Created experiment: {experiment_path}")
else:
    print(f"Experiment already exists: {experiment_path}")
mlflow.set_experiment(experiment_path)

# COMMAND ----------

# DBTITLE 1,Train, Evaluate & Log Both Models
from xgboost import XGBClassifier

def evaluate(model):
    """Compute churn-focused metrics on the held-out test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "pr_auc": float(average_precision_score(y_test, y_proba)),           # primary
        "recall_pos": float(recall_score(y_test, y_pred, pos_label=1)),
        "precision_pos": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, average="macro")),         # tracked / gate metric
    }


def train_and_log(model, model_type, params, run_name):
    """Fit a model, log it as its own MLflow run, and return its metrics + model_uri."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(DA.tags)
        mlflow.set_tag("model_family", model_type)      # easy filter/group in the compare view
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("training_method", "FeatureLookup")
        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        metrics = evaluate(model)
        mlflow.log_metrics(metrics)

        info = fe.log_model(
            model=model,
            artifact_path="customer_churn_model",
            flavor=mlflow.sklearn,
            training_set=training_set,
        )
        print(f"[{run_name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return {"run_name": run_name, "model_type": model_type,
                "run_id": run.info.run_id, "model_uri": info.model_uri, **metrics}


# scale_pos_weight balances XGBoost for the imbalanced target (neg / pos).
neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
scale_pos_weight = round(neg / max(pos, 1), 3)
print(f"Class balance — negatives: {neg}, positives: {pos}, scale_pos_weight: {scale_pos_weight}\n")

results = []

# Model 1: Random Forest (class_weight balances the imbalanced target)
results.append(train_and_log(
    RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced", n_jobs=-1),
    model_type="RandomForest",
    params={"n_estimators": 200, "max_depth": 8, "class_weight": "balanced"},
    run_name="RandomForest",
))

# Model 2: XGBoost (scale_pos_weight + aucpr eval metric for the rare class)
results.append(train_and_log(
    XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.9,
                  colsample_bytree=0.9, scale_pos_weight=scale_pos_weight,
                  eval_metric="aucpr", random_state=42, n_jobs=-1),
    model_type="XGBoost",
    params={"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1, "scale_pos_weight": scale_pos_weight},
    run_name="XGBoost",
))

# COMMAND ----------

# DBTITLE 1,Compare Runs & Pick the Best (by PR-AUC)
comparison = pd.DataFrame(results)[
    ["run_name", "model_type", "pr_auc", "recall_pos", "precision_pos", "test_f1"]
].sort_values("pr_auc", ascending=False)
display(spark.createDataFrame(comparison))

best = max(results, key=lambda r: r["pr_auc"])
model_uri = best["model_uri"]
print(f"\nBest model by PR-AUC: {best['run_name']} "
      f"(pr_auc={best['pr_auc']:.4f}, recall_pos={best['recall_pos']:.4f}, "
      f"precision_pos={best['precision_pos']:.4f}, test_f1={best['test_f1']:.4f})")
print(f"Best model URI: {model_uri}")

# COMMAND ----------

# DBTITLE 1,Tag the MLflow Experiment
_client = MlflowClient()
_exp = mlflow.get_experiment_by_name(DA.experiment_path)
if _exp:
    for k, v in DA.tags.items():
        _client.set_experiment_tag(_exp.experiment_id, k, v)
    print(f"Tagged experiment {DA.experiment_path} with {DA.tags}")

# COMMAND ----------

# DBTITLE 1,Register Model
# MAGIC %md
# MAGIC ## C. Register the Best Model in Unity Catalog
# MAGIC
# MAGIC Registering models in Unity Catalog provides:
# MAGIC - **Centralized governance** and discoverability
# MAGIC - **Access control** and auditing
# MAGIC - **Versioned** model artifacts
# MAGIC - A clear path to deployment via **Model Serving**
# MAGIC
# MAGIC We register the **PR-AUC winner** and mark it `dev`.

# COMMAND ----------

# DBTITLE 1,Register in UC
# Register the best model (by PR-AUC) in Unity Catalog
mlflow.set_registry_uri("databricks-uc")
model_name = DA.model_name

registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Registered: {registered_model.name} (version {registered_model.version}) — {best['run_name']}")

# COMMAND ----------

# DBTITLE 1,Set Model Alias and Tags
# Set a 'dev' alias for this model version
client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(
    name=registered_model.name,
    alias="dev",
    version=registered_model.version
)
print(f"Alias 'dev' set for {registered_model.name} version {registered_model.version}")

# Record which algorithm won, then tag the registered model for cleanup.
client.set_registered_model_tag(name=registered_model.name, key="best_model", value=best["model_type"])
for k, v in DA.tags.items():
    client.set_registered_model_tag(name=registered_model.name, key=k, value=v)
print(f"Tagged model {registered_model.name} with best_model={best['model_type']} and {DA.tags}")

# COMMAND ----------

# DBTITLE 1,Verify in UI
# MAGIC %md
# MAGIC ## D. Verify in the Databricks UI
# MAGIC
# MAGIC 1. Open the **Experiment** (`bank-churn-training`) — you'll see two runs, **RandomForest** and **XGBoost**; sort by `pr_auc` to compare.
# MAGIC 2. In **Catalog Explorer**, open your catalog/schema → **Models** tab → the registered model shows the `dev` alias and a `best_model` tag.

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## E. Conclusion
# MAGIC
# MAGIC In this notebook, we:
# MAGIC - Loaded the **feature table** from the Feature Store
# MAGIC - Trained **Random Forest** and **XGBoost** using **`FeatureLookup`** for consistent train/inference features
# MAGIC - Logged each as its own **MLflow run** with churn-focused metrics (`pr_auc`, `recall_pos`, `precision_pos`, `test_f1`)
# MAGIC - **Registered the best model by PR-AUC** in Unity Catalog with a `dev` alias
# MAGIC
# MAGIC The model is now ready for deployment. Next: `05-Batch-Inference` for batch scoring, `06-Real-Time-Inference` for serving endpoints, or **Notebook 08 (Continuous Deployment)** to set up the deployment job.