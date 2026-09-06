# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # 09 - Compute Resource Selection
# MAGIC
# MAGIC Databricks offers different compute for different workloads. This guide covers the main compute
# MAGIC **SKUs**, maps each stage of this churn workshop to a **recommended compute type**, and closes
# MAGIC with **serverless vs classic**.

# COMMAND ----------

# DBTITLE 1,Compute Types (SKUs)
# MAGIC %md
# MAGIC ## Compute types (SKUs)
# MAGIC
# MAGIC - **All-purpose compute** — interactive clusters for notebooks, EDA, and ad-hoc work. Stay up until stopped (or auto-terminate); highest DBU rate. Use for **development**, not scheduled jobs.
# MAGIC - **Jobs compute** — ephemeral clusters spun up per job run and torn down after. **Lower DBU rate** than all-purpose. Use for **scheduled / automated** batch and pipelines.
# MAGIC - **SQL warehouses** — compute for SQL analytics, dashboards, and BI (Databricks SQL).
# MAGIC - **Lakeflow Declarative Pipelines compute** — managed compute for declarative ETL pipelines (feature pipelines, Lakeflow Designer output).
# MAGIC - **Model Serving** — managed compute for real-time model endpoints; billed by workload size + concurrency, with scale-to-zero.
# MAGIC - **Lakebase** — managed **PostgreSQL** (OLTP + online feature store); billed by **CU-hours + storage**. Not Spark compute — a low-latency serving/transactional store.

# COMMAND ----------

# DBTITLE 1,Recommended Compute by Workload
# MAGIC %md
# MAGIC ## Recommended compute by workload
# MAGIC
# MAGIC | Workload | Example | Recommended compute | Why |
# MAGIC | --- | --- | --- | --- |
# MAGIC | **Interactive execution in a notebook** | EDA, dev — this workshop | **All purpose / interactive notebook compute** | Instant start, autoscale, scale-to-zero; ML env ships feature-engineering / MLflow / xgboost |
# MAGIC | **Batch or scheduled processing** | batch inference `05`, retrain job | **Jobs compute** | Ephemeral per-run, cheaper than all-purpose; right-sizes to the data |
# MAGIC | **Feature Store update** | `03`, feature pipelines | **Jobs compute** or **Lakeflow Declarative Pipelines** | Spark transformations writing Delta feature tables; incremental + scheduled |
# MAGIC | **Online feature store** | low-latency lookups for real-time | **Lakebase compute** | Serving-layer store, **not** cluster compute; millisecond key lookups for the endpoint |
# MAGIC | **Model serving** | real-time endpoint `06` | **Model Serving compute** (serverless) | Managed, autoscaling REST endpoint — no cluster to manage |

# COMMAND ----------

# DBTITLE 1,Serverless vs Classic
# MAGIC %md
# MAGIC ## Serverless vs Classic
# MAGIC
# MAGIC Most SKUs (notebooks, jobs, pipelines, SQL warehouses, model serving) come in two flavors:
# MAGIC
# MAGIC - **Classic compute** — runs in **your cloud account / VPC**. You choose node types, autoscaling, and cluster libraries; startup takes minutes. Best when you need **custom instance types (e.g., specific GPUs)**, **VPC / network controls**, or long-running clusters.
# MAGIC - **Serverless compute** — runs in **Databricks-managed infrastructure**. **Instant start**, autoscaling, **scale-to-zero**, no cluster/infra to manage. Best for **interactive and bursty** workloads and simplicity.
# MAGIC
# MAGIC **Rule of thumb:** default to **serverless** (this workshop uses serverless throughout) for fast start, no management, and scale-to-zero savings. Choose **classic** when you need custom node types / GPUs, cluster-scoped libraries, or specific networking.

# COMMAND ----------

# DBTITLE 1,How This Maps to the Workshop
# MAGIC %md
# MAGIC ### How this maps to the workshop
# MAGIC - **`00`–`07` interactive runs** → Serverless notebook compute (ML v5).
# MAGIC - **`05` batch inference & the retrain job (`07` E)** → Serverless jobs compute.
# MAGIC - **`03` feature table** → a Spark job/pipeline; publish an **online** copy (Lakebase / Online Table) for real-time lookups.
# MAGIC - **`06` endpoint** → Model Serving, backed by the online store for low-latency `customer_id` lookups.
# MAGIC
# MAGIC