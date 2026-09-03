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
# MAGIC # 02 - EDA with Genie Code
# MAGIC
# MAGIC In this notebook, we use **Databricks Genie Code ** to perform comprehensive EDA and feature engineering through natural language prompts. This is an interactive, hands-on exercise.
# MAGIC
# MAGIC **Prerequisites**: Run `00-Setup` first.

# COMMAND ----------

# DBTITLE 1,Setup
# MAGIC %run "./00-Setup"

# COMMAND ----------

# DBTITLE 1,How to Use Genie Code
# MAGIC %md
# MAGIC ## A. How to Use Genie Code for EDA
# MAGIC
# MAGIC Genie Code can perform sophisticated EDA tasks through natural language instructions.
# MAGIC
# MAGIC **How to access Agent Mode:**
# MAGIC 1. Open the **Genie Code** panel on the right side of the notebook
# MAGIC 2. Type your prompt in the input box
# MAGIC 3. Genie Code will generate code cells, which you can review and approve
# MAGIC
# MAGIC **Best practices:**
# MAGIC - Be specific about dataset name and target column
# MAGIC - Specify what analysis you want (distributions, correlations, outliers)
# MAGIC - Reference specific cells to control where code is placed

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## B. Exercise C1: Statistical Analysis and Data Profiling
# MAGIC
# MAGIC Use Genie Code to generate comprehensive statistical summaries and identify data quality issues.
# MAGIC
# MAGIC 1. Select this cell to ensure any code cells Genie Code generates will be placed below
# MAGIC 2. Copy the prompt below and paste it into Genie Code's input box
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
# MAGIC Perform comprehensive EDA on @customer_churn. I want to understand column statistics, data distributions, missing values, and potential data quality issues. Think like a data scientist and provide insights about what drives the `churned` label.
# MAGIC
# MAGIC Place any generated cells immediately below the cell called C1. Statistical Analysis and Data Profiling.
# MAGIC
# MAGIC
# MAGIC </div>
# MAGIC

# COMMAND ----------

# DBTITLE 1,Dataset Overview
pdf = df.toPandas()
print(f"Shape: {pdf.shape[0]} rows x {pdf.shape[1]} columns")
print(f"\nColumn types:\n{pdf.dtypes.value_counts().to_string()}")
print(f"\nMemory usage: {pdf.memory_usage(deep=True).sum() / 1024:.1f} KB")
display(df.limit(5))

# COMMAND ----------

# DBTITLE 1,Descriptive Statistics
display(df.summary())

# COMMAND ----------

# DBTITLE 1,Missing Values and Duplicates
missing_counts = df.select(
    [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
)
print("Missing values per column:")
display(missing_counts)

total = df.count()
distinct = df.distinct().count()
print(f"\nTotal rows: {total}")
print(f"Distinct rows: {distinct}")
print(f"Duplicate rows: {total - distinct}")

# COMMAND ----------

# DBTITLE 1,Target Variable Distribution
churn_counts = pdf['churned'].value_counts().sort_index()
labels_map = {0: 'Retained', 1: 'Churned'}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar([labels_map[i] for i in churn_counts.index], churn_counts.values,
            color=['#1E4651', '#FF5F46'], edgecolor='white')
axes[0].set_xlabel('Outcome')
axes[0].set_ylabel('Count')
axes[0].set_title('Churn Distribution')
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 100, str(v), ha='center', fontsize=9)

axes[1].pie(churn_counts.values, labels=[labels_map[i] for i in churn_counts.index],
            autopct='%1.1f%%', colors=['#1E4651', '#FF5F46'])
axes[1].set_title('Churn Distribution (%)')

plt.tight_layout()
plt.show()

churn_rate = churn_counts.get(1, 0) / total * 100
print(f"\nClass imbalance: churned customers account for {churn_rate:.1f}% of the book — a realistic, imbalanced target.")

# COMMAND ----------

# DBTITLE 1,Correlation Analysis
# Numeric columns only (drop identifiers and string categoricals).
drop_cols = ['customer_id', 'tier', 'home_metro', 'state', 'annual_income_band']
numeric_cols = [c for c in pdf.columns if c not in drop_cols]
corr = pdf[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.show()

churn_corr = corr['churned'].drop('churned').abs().sort_values(ascending=False)
print("\nTop features correlated with churn (absolute):")
for feat, val in churn_corr.items():
    direction = 'positive' if corr.loc[feat, 'churned'] > 0 else 'negative'
    print(f"  {feat:.<30} {corr.loc[feat, 'churned']:+.3f} ({direction})")

# COMMAND ----------

# DBTITLE 1,Outlier Detection (IQR)
# Use `c` as the loop variable (not `col`) to avoid shadowing the imported
feature_cols_eda = [c for c in numeric_cols if c not in ['churned', 'tier_rank', 'has_maturing_cd']]

outlier_summary = []
for c in feature_cols_eda:
    Q1 = pdf[c].quantile(0.25)
    Q3 = pdf[c].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((pdf[c] < lower) | (pdf[c] > upper)).sum()
    outlier_summary.append({
        'Feature': c, 'Q1': round(Q1, 3), 'Q3': round(Q3, 3),
        'IQR': round(IQR, 3), 'Lower': round(lower, 3), 'Upper': round(upper, 3),
        'Outliers': int(n_outliers), 'Outlier %': round(n_outliers / len(pdf) * 100, 1)
    })

outlier_df = pd.DataFrame(outlier_summary).sort_values('Outliers', ascending=False)
display(spark.createDataFrame(outlier_df))

print(f"\nKey findings:")
top = outlier_df.iloc[0]
print(f"  Most outliers: {top['Feature']} ({top['Outliers']} outliers, {top['Outlier %']}%)")
print(f"  Features with >5% outliers: {', '.join(outlier_df[outlier_df['Outlier %'] > 5]['Feature'].tolist()) or 'None'}")
print("  Note: high-balance and large-outflow outliers are exactly the at-risk, rate-shopping cohort — signal, not noise.")

# COMMAND ----------

# DBTITLE 1,C2 - Visualization Prompt
# MAGIC %md-sandbox
# MAGIC ## C. Exercise C2: Data Visualization and Pattern Discovery
# MAGIC
# MAGIC Leverage Genie Code's visualization capabilities to uncover patterns and relationships.
# MAGIC
# MAGIC 1. Select this cell
# MAGIC 2. Copy the prompt below and paste it into Genie Code's input box
# MAGIC
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
# MAGIC Generate correlation analysis and heatmaps to identify relationships between numeric columns and the churned label.
# MAGIC
# MAGIC Place any generated cells immediately below the cell called C2. Data Visualization and Pattern Discovery.
# MAGIC <div>

# COMMAND ----------

# DBTITLE 1,Correlation Heatmap (Full Matrix)
drop_cols = ['customer_id', 'tier', 'home_metro', 'state', 'annual_income_band']
numeric_cols = [c for c in pdf.columns if c not in drop_cols]
corr = pdf[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'label': 'Pearson r'}, ax=ax)
ax.set_title('Full Feature Correlation Matrix', fontsize=15, pad=12)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Top Feature Pairs by Correlation Strength
pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        pairs.append({
            'Feature 1': corr.columns[i],
            'Feature 2': corr.columns[j],
            'Correlation': round(corr.iloc[i, j], 3),
            'Abs Correlation': round(abs(corr.iloc[i, j]), 3),
            'Direction': 'Positive' if corr.iloc[i, j] > 0 else 'Negative'
        })

pairs_df = pd.DataFrame(pairs).sort_values('Abs Correlation', ascending=False)

print("Top 10 strongest feature correlations:")
display(spark.createDataFrame(pairs_df.head(10).drop(columns=['Abs Correlation'])))

print("\nNotable relationships:")
for _, row in pairs_df.head(5).iterrows():
    print(f"  {row['Feature 1']} <-> {row['Feature 2']}: r={row['Correlation']:+.3f} ({row['Direction'].lower()})")

# COMMAND ----------

# DBTITLE 1,Churn Rate by Top Features (Binned)
top_features = ['total_balance_usd', 'tenure_years', 'total_outflow_60d_usd', 'tier_rank']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, feat in enumerate(top_features):
    ax = axes[idx // 2][idx % 2]
    tmp = pdf[[feat, 'churned']].copy()
    try:
        tmp['bin'] = pd.qcut(tmp[feat], q=5, duplicates='drop')
    except Exception:
        tmp['bin'] = pd.cut(tmp[feat], bins=5)
    grp = tmp.groupby('bin', observed=True)['churned'].mean()
    ax.bar([str(b) for b in grp.index], grp.values, color='#1E4651')
    ax.set_xlabel(feat)
    ax.set_ylabel('churn rate')
    ax.set_title(f'Churn rate by {feat}')
    ax.tick_params(axis='x', rotation=30)

plt.suptitle('Churn Rate Across Feature Bins', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Feature Distributions by Churn Outcome
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
plot_features = ['total_balance_usd', 'tenure_years', 'num_products',
                 'total_outflow_60d_usd', 'withdrawal_count_60d', 'tier_rank']

for idx, feat in enumerate(plot_features):
    ax = axes[idx // 3][idx % 3]
    pdf.boxplot(column=feat, by='churned', ax=ax,
                boxprops=dict(color='#1E4651'), medianprops=dict(color='#FF5F46', linewidth=2),
                whiskerprops=dict(color='#1E4651'), capprops=dict(color='#1E4651'))
    ax.set_title(f'{feat}', fontsize=11)
    ax.set_xlabel('churned (0 = retained, 1 = churned)')
    ax.set_ylabel(feat)

plt.suptitle('Feature Distributions by Churn Outcome', fontsize=14)
plt.tight_layout()
plt.show()

print("Observations:")
print("  - Churned customers skew to higher balances and more recent outflow — the rate-shopping signal")
print("  - Higher tiers (affluent/private) show elevated churn — they chase competitor rates")
print("  - Long tenure alone does not protect: the most valuable, longest-tenured customers are at risk")

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC ## D. Conclusion
# MAGIC
# MAGIC In this exercise, you used **Genie Code (Agent Mode)** to:
# MAGIC - Generate comprehensive **statistical analysis** and data profiling code
# MAGIC - Create **visualizations** for pattern discovery
# MAGIC - Identify key relationships between features and the `churned` target variable
# MAGIC
# MAGIC Genie Code accelerates EDA by generating production-quality analysis code from natural language prompts, while keeping you in full control of the workflow.
# MAGIC
# MAGIC Next: Proceed to `03-Feature-Engineering` to prepare features for model training.
