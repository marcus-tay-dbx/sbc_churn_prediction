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
# DBTITLE 1,SBC Workshop Overview
# MAGIC %md-sandbox
# MAGIC # SBC Workshop — Bank Churn Prediction on Databricks
# MAGIC
# MAGIC Welcome to the **SBC Workshop**. This notebook is the starting point for the workshop. Run the setup cell below before proceeding.
# MAGIC
# MAGIC In this workshop, we work an end-to-end machine learning use case for **Meridian Bank**: predicting **customer attrition (churn)**. A competitor's savings-rate promotion has pushed some of the bank's most valuable, longest-tenured customers toward the exit — our job is to build, deploy, and monitor a model that flags at-risk customers before they leave.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - **Serverless Compute, ML Environment Version 5 (ML v5)** — select the environment version in the right sidebar before running
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Workshop Agenda
# MAGIC
# MAGIC <div style="display: grid; grid-template-columns: 60px 1fr 1fr; gap: 0; margin-top: 16px; font-size: 14px;">
# MAGIC
# MAGIC   <div style="background: #1E4651; color: white; padding: 10px 12px; font-weight: 700; border-radius: 8px 0 0 0;">No.</div>
# MAGIC   <div style="background: #1E4651; color: white; padding: 10px 12px; font-weight: 700;">Notebook</div>
# MAGIC   <div style="background: #1E4651; color: white; padding: 10px 12px; font-weight: 700; border-radius: 0 8px 0 0;">Topics</div>
# MAGIC
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;"><strong>00</strong></div>
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Overview and Set-Up <em>(this notebook)</em></div>
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Catalog, schema, raw-data generation, and the customer-360 table</div>
# MAGIC
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;"><strong>01</strong></div>
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Notebook Exploration</div>
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">UI features, basic EDA, Genie Code prompts</div>
# MAGIC
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;"><strong>02</strong></div>
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">EDA with Genie Code</div>
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Agent Mode overview, EDA prompts, visualization prompts (hands-on exercise)</div>
# MAGIC
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;"><strong>03</strong></div>
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Feature Engineering</div>
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Feature variables, business logic (balanceCategory), Feature Store, synced tables (Lakebase)</div>
# MAGIC
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;"><strong>04</strong></div>
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Model Training</div>
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">FeatureLookup, Random Forest, MLflow tracking, Unity Catalog model registry</div>
# MAGIC
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;"><strong>05</strong></div>
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Batch Inference</div>
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">fe.score_batch(), direct model loading, scoring pipelines</div>
# MAGIC
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;"><strong>06</strong></div>
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Real-Time Inference</div>
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Model Serving endpoints, REST API queries, Serving UI</div>
# MAGIC
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;"><strong>07</strong></div>
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Observability &amp; Continuous Retrain</div>
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">MLflow tracking, feature importance, SHAP, confusion matrix, Lakehouse Monitoring</div>
# MAGIC
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;"><strong>08</strong></div>
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Continuous Deployment</div>
# MAGIC   <div style="background: #F9F7F4; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Deployment job creation, model-to-job linking, Evaluate → Approve → Deploy pipeline</div>
# MAGIC
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF; border-radius: 0 0 0 8px;"><strong>09</strong></div>
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF;">Compute Resource Selection</div>
# MAGIC   <div style="background: #FFFFFF; padding: 10px 12px; border-bottom: 1px solid #E5E3DF; border-radius: 0 0 8px 0;">Recommended compute per workload: interactive, batch/scheduled, feature store, online store, serving</div>
# MAGIC
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Run Classroom Setup
# MAGIC
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,Notebook Exploration
# MAGIC
# MAGIC %md
# MAGIC ## A. Notebook Exploration

# COMMAND ----------

# DBTITLE 1,Notebook UI Features
# MAGIC
# MAGIC %md
# MAGIC ### Databricks Notebook UI Features
# MAGIC
# MAGIC ### Multi-Language Support
# MAGIC Databricks notebooks support **Python, SQL, Scala, R, and Markdown** in a single notebook. Use magic commands (`%python`, `%sql`, `%md`) to switch languages per cell.
# MAGIC
# MAGIC ### Live Collaboration
# MAGIC - Multiple users can edit the same notebook simultaneously
# MAGIC - Real-time presence indicators show who is viewing/editing
# MAGIC - Comments can be added to specific cells for review
# MAGIC
# MAGIC ### Version History
# MAGIC - Every change is automatically versioned
# MAGIC - Access via **File > Revision history** to compare or restore previous versions
# MAGIC - Git integration available for source control
# MAGIC
# MAGIC ### Sharing
# MAGIC - Share notebooks with workspace users via **Share** button
# MAGIC - Set permissions: Can View, Can Run, Can Edit, Can Manage
# MAGIC - Export as HTML, DBC, or Jupyter (`.ipynb`) formats

# COMMAND ----------

# DBTITLE 1,Load the Dataset

df = spark.table("customer_churn")
display(df)

# COMMAND ----------

# DBTITLE 1,Multi-Language Demo - SQL
# MAGIC
# MAGIC %sql
# MAGIC SELECT tier,
# MAGIC        COUNT(*) as customers,
# MAGIC        ROUND(AVG(churned), 3) as churn_rate,
# MAGIC        ROUND(AVG(total_balance_usd), 0) as avg_balance,
# MAGIC        ROUND(AVG(tenure_years), 1) as avg_tenure
# MAGIC FROM customer_churn
# MAGIC GROUP BY tier
# MAGIC ORDER BY churn_rate DESC

# COMMAND ----------

# DBTITLE 1,Introduction to Genie Code
# MAGIC
# MAGIC %md
# MAGIC ## B. Introduction to Genie Code
# MAGIC
# MAGIC In this section, we introduce **Databricks Genie Code** — a conversational AI agent that helps you build data science workflows through natural language prompts.

# COMMAND ----------

# DBTITLE 1,What is Genie Code
# MAGIC
# MAGIC %md
# MAGIC ### What is Genie Code?
# MAGIC
# MAGIC **Genie Code** is a conversational AI agent embedded in the Databricks UI that helps you build data science workflows by:
# MAGIC
# MAGIC - Understanding your dataset context (tables, schemas, metadata)
# MAGIC - Generating executable notebook code
# MAGIC - Running cells with your approval
# MAGIC - Reading outputs and iteratively fixing errors
# MAGIC - Logging experiments and results
# MAGIC
# MAGIC The old way → “AutoML runs a pipeline for me”
# MAGIC
# MAGIC The new way → “Genie Code writes and runs the pipeline with me - transparently.”

# COMMAND ----------

# DBTITLE 1,Best Practices for Agent Mode Prompts
# MAGIC
# MAGIC %md
# MAGIC ### Best Practices for  Prompts
# MAGIC
# MAGIC To get reliable and repeatable results, write prompts that clearly specify:
# MAGIC
# MAGIC - **Dataset name**
# MAGIC - **Target column**
# MAGIC - **Task type** (classification or regression)
# MAGIC - **Columns to exclude** (such as identifiers)
# MAGIC - **Train/test split instructions**
# MAGIC - **Evaluation metrics**
# MAGIC - **Where to log results** (MLflow experiment name)
# MAGIC
# MAGIC **Example:**
# MAGIC > "Train a classification model using `customer_churn`. The label is `churned`.
# MAGIC > Exclude `customer_id`, `attrition_risk_score`, and `balance_outflow_30d_usd`. Split train/test with seed 42.
# MAGIC > Evaluate using F1-score and ROC-AUC. Log runs to MLflow."

# COMMAND ----------

# DBTITLE 1,Important Considerations
# MAGIC
# MAGIC %md-sandbox
# MAGIC ### Important Considerations
# MAGIC
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Always Review Generated Code</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Agent Mode can accelerate development, but you remain responsible for reviewing the generated code and verifying correctness before running it.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <div style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Troubleshooting the Genie Code Panel</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">If you do not see the Genie Code panel, try the following:</p>
# MAGIC <ul style="margin: 8px 0 0 18px; color: #333;">
# MAGIC <li>Confirm you are using a supported Databricks Runtime version</li>
# MAGIC <li>Ensure Databricks Genie Code is enabled in your workspace</li>
# MAGIC <li>Refresh your browser or reopen the notebook</li>
# MAGIC </ul>
# MAGIC <p style="margin: 10px 0 0 0; color: #333;"><strong>Note:</strong> Your instructor may have already configured the required workspace settings for this course.</p>
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Know Your Data (EDA)
# MAGIC
# MAGIC %md
# MAGIC ## C. Know Your Data (EDA)
# MAGIC
# MAGIC ### Read and Inspect the Dataset
# MAGIC
# MAGIC In this section, we explore the **Meridian Bank** customer-360 dataset. Each row is one customer, with demographic attributes (tier, tenure, income band), holdings aggregates (product count, total balance, maturing CDs), and recent transaction behavior. A data scientist uses this data to predict the `churned` label — whether a customer is at risk of leaving the bank.
# MAGIC
# MAGIC The classroom setup created one table: `customer_churn`.
# MAGIC
# MAGIC ### Inspect Statistics: Numerical Values and Visuals
# MAGIC Here we will exhibit different ways in which you can display and visualize descriptive statistics.
# MAGIC
# MAGIC describe(<spark_or_pandas_dataframe>) - This method will only return a table with the necessary information.
# MAGIC display(<spark_or_pandas_dataframe>) - This will return the table. From this, we can build a visual to inspect the feature variables.
# MAGIC Custom code - We can use the pandas DataFrame along with other Python libraries to build custom visualizations.

# COMMAND ----------

# DBTITLE 1,Genie Code Prompt
# MAGIC
# MAGIC %md-sandbox
# MAGIC <div style="border-left: 4px solid #f44336; background: #ffebee; padding: 16px 20px; border-radius: 4px; margin: 16px 0;">
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #c62828; font-size: 1.1em;">Open Genie Code</strong>
# MAGIC <p style="margin: 8px 0 0 0; color: #333;">Click the Genie icon at the top right, next to your workspace name </p>
# MAGIC
# MAGIC </div>
# MAGIC </div>
# MAGIC </div>
# MAGIC
# MAGIC <div id="prompt-box" style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0; position: relative;">
# MAGIC
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Copy Prompt to Genie Code </strong>
# MAGIC
# MAGIC <button onclick="var t=document.getElementById('prompt-body-1').innerText;var a=document.createElement('textarea');a.value=t;a.style.position='fixed';a.style.opacity='0';document.body.appendChild(a);a.select();document.execCommand('copy');document.body.removeChild(a);this.textContent='Copied!';setTimeout(()=>{this.textContent='Copy'},2000)" style="position: absolute; top: 12px; right: 12px; background: #1976d2; color: white; border: none; border-radius: 4px; padding: 6px 14px; cursor: pointer; font-size: 0.85em; font-weight: 500;">Copy</button>
# MAGIC
# MAGIC <div id="prompt-body-1">
# MAGIC
# MAGIC Using the `customer_churn` table in the current catalog and schema, generate 4 code cells placed as follows:
# MAGIC
# MAGIC 1. After the markdown cell titled **"Genie Code Prompt"**: Read `customer_churn` into a Spark DataFrame called `df` and a pandas DataFrame called `pdf`; Then display `df`.
# MAGIC 2. Display summary statistics for `df`.
# MAGIC 3. Print summary statistics for `pdf`.
# MAGIC 4. Group `df` by `tier` and show the `min`, `Q1` (25th percentile), `median`, `Q3` (75th percentile), and `max` of the `total_balance_usd` column.
# MAGIC
# MAGIC
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Bubble Chart Instructions
# MAGIC
# MAGIC %md
# MAGIC ### Bubble Chart Using GUI Visualization Editor
# MAGIC
# MAGIC We can now use the **Visualization Editor** in the Databricks UI to build a bubble chart using our grouped summary statistics.
# MAGIC
# MAGIC 1. Create the grouped DataFrame in the following cell.
# MAGIC 2. In the output result cell:
# MAGIC     - Click the **+** dropdown next to Table (top-right of the table display).
# MAGIC     - Select **Visualization**.
# MAGIC 3. In the **Visualization Editor**:
# MAGIC     - Select **Bubble** as the visualization type.
# MAGIC     - Under **X column**, select `tier`.
# MAGIC     - Under **Y columns**, select `churn_rate`.
# MAGIC     - Under **Group by**, select `customers`.
# MAGIC     - Under **Bubble size column**, select `customers`.
# MAGIC     - Under **Bubble size coefficient**, check if it's `1`.
# MAGIC     - Leave **Bubble size proportional to** as `Diameter`.
# MAGIC 4. Click **Save** to render the chart.
# MAGIC
# MAGIC This creates a bubble chart that shows:
# MAGIC
# MAGIC - Customer **tier** on the x-axis.
# MAGIC - **Churn rate** on the y-axis.
# MAGIC - **Bubble size** proportional to the number of customers.

# COMMAND ----------

# DBTITLE 1,Genie Code Prompt 2
# MAGIC
# MAGIC %md-sandbox
# MAGIC <div id="prompt-box" style="border-left: 4px solid #1976d2; background: #e3f2fd; padding: 16px 20px; border-radius: 4px; margin: 16px 0; position: relative;">
# MAGIC
# MAGIC <div style="display: flex; align-items: flex-start; gap: 12px;">
# MAGIC <div>
# MAGIC <strong style="color: #0d47a1; font-size: 1.1em;">Copy Prompt to Genie Code </strong>
# MAGIC
# MAGIC <button onclick="var t=document.getElementById('prompt-body-2').innerText;var a=document.createElement('textarea');a.value=t;a.style.position='fixed';a.style.opacity='0';document.body.appendChild(a);a.select();document.execCommand('copy');document.body.removeChild(a);this.textContent='Copied!';setTimeout(()=>{this.textContent='Copy'},2000)" style="position: absolute; top: 12px; right: 12px; background: #1976d2; color: white; border: none; border-radius: 4px; padding: 6px 14px; cursor: pointer; font-size: 0.85em; font-weight: 500;">Copy</button>
# MAGIC
# MAGIC <div id="prompt-body-2">
# MAGIC
# MAGIC After the markdown cell titled **"Genie Code Prompt 2"**: Group `df` by `tier` and compute the churn rate and customer count.
# MAGIC
# MAGIC </div>
# MAGIC

# COMMAND ----------

# DBTITLE 1,Churn rate and count by tier

tier_stats = df.groupBy("tier").agg(
    F.round(F.avg("churned"), 3).alias("churn_rate"),
    F.count("*").alias("customers")
).orderBy("churn_rate", ascending=False)

display(tier_stats)

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC
# MAGIC %md
# MAGIC ## D. Conclusion
# MAGIC
# MAGIC In this notebook, we explored:
# MAGIC - **Databricks notebook UI features**: multi-language cells, collaboration, versioning, sharing
# MAGIC - **Basic EDA** using both Spark DataFrames and pandas
# MAGIC - **Built-in visualization editor** for interactive charts
# MAGIC - **SQL and Python** in the same notebook for flexible exploration
# MAGIC
# MAGIC Next: Proceed to `02-EDA-with-Genie-Code` to explore the data using Genie Code.