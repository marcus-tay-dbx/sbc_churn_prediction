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
# MAGIC %md-sandbox
# MAGIC # 03 - Feature Engineering on Databricks
# MAGIC
# MAGIC Machine learning uses existing data to build a model to predict future outcomes. In almost all cases, the raw data requires preprocessing and transformation before it can be used to build a model. This process is called **feature engineering**, and the outputs of this process are called **features** - the building blocks of the model.
# MAGIC
# MAGIC In this notebook we turn the `customer_churn` customer-360 table into a governed **Feature Store** table:
# MAGIC
# MAGIC <div style="display: flex; gap: 1.5rem; align-items: flex-start; margin-top: 16px;">
# MAGIC   <div style="flex: 1; background: #F9F7F4; border-top: 3px solid #FF5F46; border-radius: 8px; padding: 1.25rem;">
# MAGIC     <h4>What we build</h4>
# MAGIC     <ul>
# MAGIC       <li>Select the model's <strong>feature variables</strong> and primary key (<code>customer_id</code>)</li>
# MAGIC       <li>Apply <strong>business logic</strong> to engineer a new feature (<code>balanceCategory</code>)</li>
# MAGIC       <li>Publish a <strong>Feature Store</strong> table in Unity Catalog</li>
# MAGIC       <li>Preview a <strong>synced table</strong> (Lakebase) for low-latency online lookups</li>
# MAGIC     </ul>
# MAGIC   </div>
# MAGIC   <div style="flex: 1; background: #EEEDE9; border-top: 3px solid #1E4651; border-radius: 8px; padding: 1.25rem;">
# MAGIC     <h4>Why the Feature Store</h4>
# MAGIC     <ul>
# MAGIC       <li><strong>Consistency</strong> — the same feature logic is used in training and inference</li>
# MAGIC       <li><strong>Reuse</strong> — features are discoverable and shareable across teams</li>
# MAGIC       <li><strong>Lineage</strong> — Unity Catalog tracks how features feed models</li>
# MAGIC     </ul>
# MAGIC   </div>
# MAGIC </div>
# MAGIC
# MAGIC **Prerequisites**: Run `00-Setup` first.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Load Data
# `df` is loaded by 00-Setup; re-point it here for clarity in this notebook.
df = spark.table("customer_churn")
print(f"Loaded {df.count()} rows from customer_churn")

# COMMAND ----------

# DBTITLE 1,Feature Selection & Leakage Note
# MAGIC %md-sandbox
# MAGIC ## A. Feature Selection
# MAGIC
# MAGIC We predict `churned` from **demographic** and **behavioral** signals. Two columns in
# MAGIC `customer_churn` — `attrition_risk_score` and `balance_outflow_30d_usd` — were used to
# MAGIC *derive* the label, so including them as features would be **target leakage**. We
# MAGIC deliberately **exclude** them from the feature set and let the model learn from
# MAGIC genuine predictors instead.
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 14px 18px; border-radius: 4px; margin: 12px 0;">
# MAGIC <strong style="color: #c62828;">Avoid target leakage</strong>
# MAGIC <p style="margin: 6px 0 0 0; color: #333;">Never train on a feature that encodes the label. Here, <code>attrition_risk_score</code> and <code>balance_outflow_30d_usd</code> are dropped — the model must learn churn from tenure, balances, product mix, and recent transaction behavior.</p>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Define Feature Variables
feature_variables = ['tenure_years',
                     'total_balance_usd',
                     'num_products',
                     'num_deposit_products',
                     'has_maturing_cd',
                     'txn_count_60d',
                     'total_outflow_60d_usd',
                     'withdrawal_count_60d',
                     'tier_rank',
                     'churned']
prediction_variable = 'churned'
primary_key = ['customer_id']

# COMMAND ----------

# DBTITLE 1,Select Features
feature_df = df.select(primary_key + feature_variables)
display(feature_df)

# COMMAND ----------

# DBTITLE 1,Business Logic Intro
# MAGIC %md
# MAGIC ## B. Apply Business Logic — Engineer `balanceCategory`
# MAGIC
# MAGIC Total deposit balance is the single biggest driver of *revenue at risk* when a customer churns. We discretize it into three bands using the 25th and 75th percentiles, mirroring how a relationship manager thinks about "small / typical / high-value" customers:
# MAGIC
# MAGIC | Encoding | Meaning |
# MAGIC | --- | --- |
# MAGIC | `0.0` | Low balance (≤ Q1) |
# MAGIC | `1.0` | Average balance (Q1 < balance < Q3) |
# MAGIC | `2.0` | High balance (≥ Q3) |

# COMMAND ----------

# DBTITLE 1,Apply Business Logic
quantiles = feature_df.approxQuantile("total_balance_usd", [0.25, 0.75], 0.0)

Q1, Q3 = quantiles

feature_df2 = feature_df.withColumn(
    "balanceCategory",
    when(col("total_balance_usd") <= Q1, 0.0)
    .when((col("total_balance_usd") > Q1) & (col("total_balance_usd") < Q3), 1.0)
    .otherwise(2.0)
)
# Encoding: 0.0 = Low balance, 1.0 = Average balance, 2.0 = High balance

print(f"Balance quartiles — Q1: {Q1:,.0f}   Q3: {Q3:,.0f}")
display(feature_df2)

# COMMAND ----------

# DBTITLE 1,Feature Store Intro
# MAGIC %md
# MAGIC ## C. Publish to the Feature Store
# MAGIC
# MAGIC The **`FeatureEngineeringClient`** writes our engineered features to a Unity Catalog feature table keyed by `customer_id`. Downstream notebooks (`04-Model-Training`, `05-Batch-Inference`) use **`FeatureLookup`** against this table so the exact same features are used everywhere.

# COMMAND ----------

# DBTITLE 1,Instantiate Feature Engineering Client
fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Create Feature Table in Unity Catalog
feature_table_name = DA.feature_table_name

print(f"Feature table: {feature_table_name}")

try:
    fe.create_table(
        name=feature_table_name,
        primary_keys=primary_key,
        df=feature_df2,
        description="Meridian Bank customer churn features",
        tags={"sbc": "true", "churn-prediction": "true"}
    )
    print("Created new feature table")
except Exception as e:
    if "already exists" in str(e):
        fe.write_table(
            name=feature_table_name,
            df=feature_df2,
            mode="overwrite"
        )
        print("Overwrote existing feature table")
    else:
        raise e

# COMMAND ----------

# DBTITLE 1,Tag Feature Table for Cleanup
# Ensure the cleanup tags are present even if the table already existed.
tag_table(feature_table_name)

# COMMAND ----------

# DBTITLE 1,Synced Table (Lakebase) Overview
# MAGIC %md
# MAGIC ## D. Synced Table for Online Lookups (Lakebase)
# MAGIC
# MAGIC For **real-time** inference, features must be served with millisecond latency. A **synced table** publishes a continuously-updated replica of the feature table into **Lakebase** (serverless Postgres), so a serving endpoint can look features up by `customer_id` at request time.
# MAGIC
# MAGIC Create one from the Catalog UI (no code required):
# MAGIC
# MAGIC 1. Open the feature table `customer_churn_features` in **Catalog Explorer**
# MAGIC 2. Click **Create** → **Synced table**
# MAGIC 3. Configure:
# MAGIC     - **Name**: `customer_churn_feature_sync`
# MAGIC     - **Database**: `databricks_postgres`
# MAGIC     - **Database instance**: Lakebase Serverless (Autoscaling)
# MAGIC     - **Primary key**: `customer_id`
# MAGIC     - **Sync mode**: *Snapshot* (one-time / scheduled) or *Triggered* / *Continuous* (requires Change Data Feed)
# MAGIC 4. Click **Create**
# MAGIC
# MAGIC > **Change Data Feed (CDF)**: Triggered and Continuous sync modes require CDF on the source Delta table so only changed rows are propagated. Enable it with `ALTER TABLE ... SET TBLPROPERTIES (delta.enableChangeDataFeed = true)`.

# COMMAND ----------

# DBTITLE 1,(Optional) Enable Change Data Feed for Triggered/Continuous Sync
# Enabling CDF lets Lakebase sync incrementally (Triggered / Continuous modes).
try:
    spark.sql(f"ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"Change Data Feed enabled on {feature_table_name}")
except Exception as e:
    print(f"Could not enable CDF: {e}")

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## E. Conclusion
# MAGIC
# MAGIC In this notebook, we:
# MAGIC - Selected feature variables and **excluded leakage columns**
# MAGIC - Engineered a new categorical feature **`balanceCategory`** with business logic
# MAGIC - Published a **Feature Store** table in Unity Catalog (tagged for cleanup)
# MAGIC - Previewed a **synced table (Lakebase)** for low-latency online serving
# MAGIC
# MAGIC Next: Proceed to **04-Model-Training** to train a model using `FeatureLookup`.