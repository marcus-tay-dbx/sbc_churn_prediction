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
# DBTITLE 1,Install Dependencies
# MAGIC %pip install databricks-feature-engineering --quiet

# COMMAND ----------

# DBTITLE 1,Step 1: Set Catalog and Schema
# Derive catalog name from the current user's username (e.g. "first.last" → "first_last")
username = spark.sql("SELECT current_user()").collect()[0][0]
default_catalog = username.split("@")[0].replace(".", "_").replace("-", "_")

# Configurable via notebook widgets — override these to point at a different catalog/schema
dbutils.widgets.text("catalog_name", default_catalog, "Catalog Name")
dbutils.widgets.text("schema_name", "sbc_churn_prediction", "Schema Name")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

# Create catalog if the user has permission; if not, verify it was pre-created by an admin
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
except Exception as e:
    err = str(e)
    if "PERMISSION_DENIED" in err or "UNAUTHORIZED" in err:
        existing = [r.catalog for r in spark.sql("SHOW CATALOGS").collect()]
        if catalog_name in existing:
            print(f"ℹ No CREATE CATALOG permission, but '{catalog_name}' already exists — proceeding")
        else:
            raise RuntimeError(
                f"Cannot create catalog '{catalog_name}' and it does not exist.\n"
                f"Ask a metastore admin to run:\n"
                f"  CREATE CATALOG IF NOT EXISTS {catalog_name};\n"
                f"  GRANT USE CATALOG, CREATE SCHEMA ON CATALOG {catalog_name} TO `{username}`;\n"
                f"Or override the 'catalog_name' widget to an existing catalog."
            ) from e
    else:
        raise

spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

print(f"✓ Catalog: {catalog_name}")
print(f"✓ Schema:  {schema_name}")

# COMMAND ----------

# DBTITLE 1,Step 2: Shared Imports (used across notebooks 01-08)
# Single import block for the whole workshop. Every notebook runs `%run "./00-Setup"`,
# so these names are available downstream — no per-notebook import cells needed.
import os, re, json, pickle, warnings, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from types import SimpleNamespace
from datetime import datetime, timedelta

from pyspark.sql import DataFrame, functions as F, Window
from pyspark.sql.functions import col, when, expr

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking.client import MlflowClient
from mlflow.deployments import get_deploy_client

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    average_precision_score, precision_score, recall_score,
)
# xgboost and shap are NOT imported here — they are not pre-installed on serverless
# compute and are installed/imported locally in 04-Model-Training and 07-Observability.

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    Task, NotebookTask, Source, TaskDependency, JobParameterDefinition
)

warnings.filterwarnings("ignore")
np.set_printoptions(precision=2)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
workshop_dir = os.path.dirname(notebook_path)
course_dir = os.path.dirname(workshop_dir)

table_name = "customer_churn"
raw_root = f"/Volumes/{catalog_name}/{schema_name}/raw_data"

# COMMAND ----------

# DBTITLE 1,Raw Data Generator (merged from generate_data)
# ── Meridian Bank — synthetic raw data generator ─────────────────────────────
# Produces the six raw datasets (customers, products, holdings, transactions,
# risk_snapshots, retention_campaigns) as parquet in the UC Volume `raw_data`,
# using pure Spark idioms (spark.range + F.when + broadcast joins + Window +
# F.element_at against literal arrays — no driver loops, no .collect() on big
# tables). The load-bearing anomaly: a competitor savings-rate promo ~3 weeks ago
# pushed the bank's most valuable, longest-tenured customers holding maturing CDs
# into elevated attrition risk, with balance starting to flow out. Hero at-risk
# account: CUST-0000214 (12-year affluent, large CD maturing in ~9 days).
def generate_raw_data(CATALOG, SCHEMA):
    RAW_VOL = "raw_data"

    # ── Story timeline ────────────────────────────────────────────────────────
    # NOW is the single source of truth. Default is ROLLING (datetime.now()).
    # Set MERIDIAN_PIN_TIME=1 to freeze for recorded demos / baked-in IDs.
    STORY_PINNED_NOW = datetime(2026, 8, 1)
    NOW = STORY_PINNED_NOW if os.environ.get("MERIDIAN_PIN_TIME") == "1" else datetime.now()

    HIST_START = NOW - timedelta(days=18 * 30)        # 18-month txn + campaign history
    HIST_END = NOW - timedelta(days=1)
    HIST_SPAN_DAYS = (HIST_END - HIST_START).days
    PROMO_ONSET = NOW - timedelta(days=21)            # competitor promo begins ~3 weeks ago
    RISK_RAMP = NOW - timedelta(days=18)              # affected customers' risk scores climb
    SNAPSHOT_DATE = NOW - timedelta(days=1)           # the "current" customer-360 snapshot
    RISK_WINDOW_START = NOW - timedelta(days=14)      # daily risk snapshots for the last ~14 days

    # ── Deterministic story anchors ───────────────────────────────────────────
    N_CUSTOMERS = 40_000
    N_AFFECTED = 220                                  # high-value customers pushed into HIGH risk
    N_MODERATE = 120                                  # secondary cohort at MODERATE risk / smaller balances
    NIM = 0.025                                       # net interest margin (revenue-at-risk factor)

    HERO_CUST = "CUST-0000214"                        # 12-year affluent — the demo's spotlight
    HERO_CD = "PROD-DEP-2001"                         # the maturing 18-month CD the hero holds

    AFFECTED_PRODUCTS = ["PROD-DEP-2001", "PROD-DEP-2002", "PROD-DEP-2003"]
    COMPETITOR_RATE = 0.0385                          # the promo rate customers are chasing

    CATALOG_PRODUCTS = [
        ("PROD-DEP-2001", "18-Month Certificate of Deposit", "CD", "deposit", 0.0325, 1000.0,
         "18-month term CD for savers locking in a fixed rate; penalty on early withdrawal. For rate-focused deposit customers."),
        ("PROD-DEP-2002", "High-Yield Savings", "Savings", "deposit", 0.0290, 0.0,
         "Liquid high-yield savings account, tiered rate on higher balances. For customers holding cash who want yield with access."),
        ("PROD-DEP-2003", "12-Month Certificate of Deposit", "CD", "deposit", 0.0300, 1000.0,
         "12-month term CD, shorter lock for rate-focused savers. Alternative to the 18-month CD."),
        ("PROD-INV-3001", "Wealth Advisory Account", "Advisory", "investment", None, 100000.0,
         "Managed wealth advisory account with a dedicated advisor. For affluent and private-tier customers with investable assets; a cross-sell for high-balance depositors."),
        ("PROD-CRD-4001", "Premier Rewards Credit Card", "Card", "lending", None, 0.0,
         "Premium rewards credit card, travel + cashback perks. For mass-affluent and above with strong relationship tenure; a cross-sell for depositors without a card."),
        ("PROD-LN-5001", "Home Equity Line of Credit", "HELOC", "lending", 0.0725, 0.0,
         "Revolving home-equity line of credit. For homeowners with equity; a lending cross-sell for established relationship customers."),
        ("PROD-DEP-2010", "Everyday Checking", "Checking", "deposit", 0.0010, 0.0,
         "No-frills everyday checking account with direct deposit and bill pay. The core relationship anchor product."),
        ("PROD-DEP-2011", "Money Market Account", "Savings", "deposit", 0.0250, 2500.0,
         "Money market account, tiered yield with limited monthly transactions. For savers wanting a blend of yield and access."),
        ("PROD-LN-5002", "30-Year Fixed Mortgage", "Mortgage", "lending", 0.0665, 0.0,
         "30-year fixed-rate home mortgage. For homebuyers; a long-tenure relationship product."),
        ("PROD-LN-5003", "Auto Loan", "Auto", "lending", 0.0620, 0.0,
         "Fixed-rate auto loan for new and used vehicles. Broad-eligibility lending product."),
        ("PROD-INV-3002", "Self-Directed Brokerage", "Brokerage", "investment", None, 0.0,
         "Self-directed online brokerage account. For customers who want to invest without an advisor; a cross-sell for affluent depositors."),
    ]
    CROSS_SELL_TARGETS = ["PROD-INV-3001", "PROD-CRD-4001", "PROD-LN-5001"]

    print(f"NOW: {NOW.date()} ({'pinned' if os.environ.get('MERIDIAN_PIN_TIME') == '1' else 'rolling'})")
    print(f"PROMO_ONSET: {PROMO_ONSET.date()}  SNAPSHOT_DATE: {SNAPSHOT_DATE.date()}")
    print(f"Hero: {HERO_CUST} at risk on {HERO_CD}; competitor rate {COMPETITOR_RATE:.2%}")

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{RAW_VOL}")
    RAW_VOL_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{RAW_VOL}"

    def _raw_path(table):
        return f"{RAW_VOL_ROOT}/{table[4:] if table.startswith('raw_') else table}"

    def _save(df, table):
        path = _raw_path(table)
        df.write.mode("overwrite").parquet(path)
        n = spark.read.parquet(path).count()
        print(f"  ✓ {table:26s} rows={n:>10,}  → {path}")

    # ── 1. Customers ───────────────────────────────────────────────────────────
    print("\n[1/6] Generating customers...")
    _METROS = [
        ("New York", "NY", 40.71, -74.01), ("Boston", "MA", 42.36, -71.06),
        ("Chicago", "IL", 41.88, -87.63), ("San Francisco", "CA", 37.77, -122.42),
        ("Dallas", "TX", 32.78, -96.80), ("Atlanta", "GA", 33.75, -84.39),
        ("Denver", "CO", 39.74, -104.99), ("Seattle", "WA", 47.61, -122.33),
        ("Miami", "FL", 25.76, -80.19), ("Philadelphia", "PA", 39.95, -75.16),
        ("Charlotte", "NC", 35.23, -80.84), ("Minneapolis", "MN", 44.98, -93.27),
    ]
    _TIERS = ["mass", "mass_affluent", "affluent", "private"]
    _INCOME_BANDS = ["<50k", "50-100k", "100-200k", "200-500k", "500k+"]

    metro_arr = F.array(*[F.lit(m[0]) for m in _METROS])
    state_arr = F.array(*[F.lit(m[1]) for m in _METROS])
    lat_arr = F.array(*[F.lit(float(m[2])) for m in _METROS])
    lng_arr = F.array(*[F.lit(float(m[3])) for m in _METROS])
    income_arr = F.array(*[F.lit(b) for b in _INCOME_BANDS])

    AFFECTED_IDX = [213] + [i for i in range(400, 400 + (N_AFFECTED - 1) * 37, 37)][: N_AFFECTED - 1]
    affected_idx_arr = F.array(*[F.lit(int(i)) for i in AFFECTED_IDX])
    MODERATE_IDX = [i for i in range(20000, 20000 + N_MODERATE * 53, 53)][:N_MODERATE]
    moderate_idx_arr = F.array(*[F.lit(int(i)) for i in MODERATE_IDX])

    customers_df = (
        spark.range(0, N_CUSTOMERS)
        .withColumn("customer_id", F.concat(F.lit("CUST-"), F.lpad((F.col("id") + 1).cast("string"), 7, "0")))
        .withColumn("_mi", (F.rand(1) * len(_METROS)).cast("int"))
        .withColumn("is_affected", F.array_contains(affected_idx_arr, F.col("id").cast("int")))
        .withColumn("is_moderate", F.array_contains(moderate_idx_arr, F.col("id").cast("int")))
        .withColumn("_tier_pick", (F.rand(2) * 4).cast("int"))
        .withColumn(
            "tier",
            F.when(F.col("customer_id") == HERO_CUST, F.lit("affluent"))
            .when(F.col("is_affected") & (F.rand(3) < 0.7), F.lit("affluent"))
            .when(F.col("is_affected"), F.lit("private"))
            .when(F.col("is_moderate"), F.lit("mass_affluent"))
            .when(F.col("_tier_pick") == 3, F.lit("private"))
            .when(F.col("_tier_pick") == 2, F.lit("affluent"))
            .when(F.col("_tier_pick") == 1, F.lit("mass_affluent"))
            .otherwise(F.lit("mass")),
        )
        .withColumn(
            "tenure_years",
            F.when(F.col("customer_id") == HERO_CUST, F.lit(12))
            .when(F.col("is_affected"), (8 + F.rand(4) * 12).cast("int"))
            .when(F.col("is_moderate"), (3 + F.rand(41) * 8).cast("int"))
            .otherwise((1 + F.rand(5) * 15).cast("int")),
        )
        .withColumn("home_metro", F.element_at(metro_arr, F.col("_mi") + 1))
        .withColumn("state", F.element_at(state_arr, F.col("_mi") + 1))
        .withColumn("customer_lat", F.round(F.element_at(lat_arr, F.col("_mi") + 1) + (F.rand(6) - 0.5) * 0.1, 2))
        .withColumn("customer_lng", F.round(F.element_at(lng_arr, F.col("_mi") + 1) + (F.rand(7) - 0.5) * 0.1, 2))
        .withColumn("annual_income_band", F.element_at(income_arr, (F.rand(8) * len(_INCOME_BANDS) + 1).cast("int")))
        .withColumn("home_branch_id", F.concat(F.lit("BR-"), F.lpad(((F.rand(9) * 180 + 1).cast("int")).cast("string"), 4, "0")))
        .withColumn("join_date", F.date_sub(F.lit(NOW.date().isoformat()).cast("date"), (F.col("tenure_years") * 365 + (F.rand(10) * 200).cast("int"))))
        .withColumn("customer_display_name", F.concat(F.lit("Customer "), F.substring(F.col("customer_id"), 6, 7)))
        .withColumn(
            "profile_summary",
            F.concat_ws(
                " ",
                F.col("tier"), F.lit("tier relationship,"),
                F.col("tenure_years").cast("string"), F.lit("year tenure,"),
                F.lit("home metro"), F.col("home_metro"), F.lit("."),
                F.when(F.col("is_affected") | F.col("is_moderate"), F.lit("Holds a maturing certificate of deposit; rate-sensitive, comparing competitor savings rates."))
                .otherwise(F.lit("Stable deposit relationship, routine servicing, no active concerns.")),
            ),
        )
        .withColumn("is_active", F.lit(True))
        .select(
            "customer_id", "customer_display_name", "tier", "tenure_years", "home_branch_id",
            "home_metro", "state", "customer_lat", "customer_lng", "annual_income_band",
            "join_date", "profile_summary", "is_active",
        )
    )
    _save(customers_df, "raw_customers")

    AFFECTED_CUSTS = [f"CUST-{i + 1:07d}" for i in AFFECTED_IDX]
    MODERATE_CUSTS = [f"CUST-{i + 1:07d}" for i in MODERATE_IDX]
    ATRISK_CUSTS = AFFECTED_CUSTS + MODERATE_CUSTS

    # ── 2. Products ────────────────────────────────────────────────────────────
    print("\n[2/6] Generating products...")
    products_df = (
        spark.createDataFrame(
            [(p[0], p[1], p[2], p[3], p[4], p[5], p[6]) for p in CATALOG_PRODUCTS],
            "product_id string, product_name string, product_type string, segment string, "
            "rate_apy double, min_balance_usd double, description string",
        )
        .withColumn("is_active", F.lit(True))
    )
    _save(products_df, "raw_products")

    # ── 3. Holdings ────────────────────────────────────────────────────────────
    print("\n[3/6] Generating holdings...")
    affected_cust_arr = F.array(*[F.lit(c) for c in AFFECTED_CUSTS])
    moderate_cust_arr = F.array(*[F.lit(c) for c in MODERATE_CUSTS])
    _deposit_prods = [p[0] for p in CATALOG_PRODUCTS if p[3] == "deposit"]
    _other_prods = [p[0] for p in CATALOG_PRODUCTS if p[3] != "deposit"]
    dep_arr = F.array(*[F.lit(p) for p in _deposit_prods])
    other_arr = F.array(*[F.lit(p) for p in _other_prods])
    prod_rate = {p[0]: (p[4] if p[4] is not None else 0.0) for p in CATALOG_PRODUCTS}
    rate_map = F.create_map(*[x for pid, r in prod_rate.items() for x in (F.lit(pid), F.lit(float(r)))])

    cust_base = (
        customers_df.select("customer_id", "tier")
        .withColumn("is_affected", F.array_contains(affected_cust_arr, F.col("customer_id")))
        .withColumn("is_moderate", F.array_contains(moderate_cust_arr, F.col("customer_id")))
    )

    checking = (
        cust_base
        .withColumn("product_id", F.lit("PROD-DEP-2010"))
        .withColumn("balance_usd", F.round(500 + F.rand(11) * 8000, 2))
        .withColumn("maturity_date", F.lit(None).cast("date"))
    )

    def _sampled_slot(seed_a, seed_b, seed_c):
        return (
            cust_base
            .withColumn("_use_dep", F.rand(seed_a) < 0.6)
            .withColumn(
                "product_id",
                F.when(F.col("_use_dep"), F.element_at(dep_arr, (F.rand(seed_b) * len(_deposit_prods) + 1).cast("int")))
                .otherwise(F.element_at(other_arr, (F.rand(seed_b) * len(_other_prods) + 1).cast("int"))),
            )
            .withColumn(
                "balance_usd",
                F.when(F.col("product_id").isin(_deposit_prods), F.round(2000 + F.rand(seed_c) * 60000, 2))
                .otherwise(F.round(F.rand(seed_c) * 40000, 2)),
            )
            .withColumn("maturity_date", F.lit(None).cast("date"))
            .drop("_use_dep")
        )

    slot1 = _sampled_slot(12, 13, 14)
    slot2 = _sampled_slot(15, 16, 17)

    _aff_prod_arr = F.array(*[F.lit(p) for p in AFFECTED_PRODUCTS])
    affected_holding = (
        cust_base.filter(F.col("is_affected") | F.col("is_moderate"))
        .withColumn(
            "product_id",
            F.when(F.col("customer_id") == HERO_CUST, F.lit(HERO_CD))
            .otherwise(F.element_at(_aff_prod_arr, (F.rand(18) * len(AFFECTED_PRODUCTS) + 1).cast("int"))),
        )
        .withColumn(
            "balance_usd",
            F.when(F.col("customer_id") == HERO_CUST, F.lit(650000.0))
            .when(F.col("is_moderate"), F.round(30000 + F.rand(19) * 90000, 2))
            .otherwise(F.round(200000 + F.rand(19) * 1000000, 2)),
        )
        .withColumn(
            "maturity_date",
            F.when(F.col("customer_id") == HERO_CUST, F.lit((NOW + timedelta(days=9)).date().isoformat()).cast("date"))
            .otherwise(F.date_add(F.lit(SNAPSHOT_DATE.date().isoformat()).cast("date"), (2 + F.rand(20) * 38).cast("int"))),
        )
    )

    holdings_all = (
        checking.unionByName(slot1).unionByName(slot2).unionByName(affected_holding)
        .withColumn("rate_apy", F.coalesce(F.element_at(rate_map, F.col("product_id")), F.lit(0.0)))
        .withColumn("open_date", F.date_sub(F.lit(NOW.date().isoformat()).cast("date"), (F.rand(21) * 3000 + 200).cast("int")))
        .withColumn(
            "status",
            F.when(F.col("maturity_date").isNotNull() & (F.col("maturity_date") < F.lit(SNAPSHOT_DATE.date().isoformat()).cast("date")), F.lit("matured"))
            .otherwise(F.lit("active")),
        )
        .withColumn("account_id", F.concat(F.lit("ACCT-"), F.lpad((F.monotonically_increasing_id() % 90000000 + 1).cast("string"), 8, "0")))
        .select("account_id", "customer_id", "product_id", "balance_usd", "open_date", "maturity_date", "rate_apy", "status")
    )
    _save(holdings_all, "raw_holdings")

    # ── 4. Transactions ────────────────────────────────────────────────────────
    print("\n[4/6] Generating transactions...")
    ramp_off = (SNAPSHOT_DATE - RISK_RAMP).days
    affected_txn = (
        spark.createDataFrame([(c,) for c in AFFECTED_CUSTS], "customer_id string")
        .crossJoin(spark.range(0, 60).withColumnRenamed("id", "day_offset"))
        .withColumn("txn_date", F.date_sub(F.lit(SNAPSHOT_DATE.date().isoformat()).cast("date"), F.col("day_offset").cast("int")))
        .withColumn("_ramped", F.col("day_offset") <= F.lit(ramp_off))
        .filter(F.col("_ramped") & (F.rand(31) < 0.5))
        .withColumn("txn_type", F.when(F.rand(32) < 0.6, F.lit("transfer_out")).otherwise(F.lit("withdrawal")))
        .withColumn("amount_usd", F.round(-(5000 + F.rand(33) * 80000), 2))
        .withColumn("channel", F.element_at(F.array(F.lit("online"), F.lit("mobile"), F.lit("branch")), (F.rand(34) * 3 + 1).cast("int")))
        .withColumn("account_id", F.lit("ACCT-OUTFLOW"))
        .select("customer_id", "account_id", "txn_date", "amount_usd", "txn_type", "channel")
    )

    N_BASELINE = 3_500_000
    cust_all_arr = F.array(*[F.lit(f"CUST-{i + 1:07d}") for i in range(3000)])
    _n_cust_subset = 3000
    baseline_txn = (
        spark.range(0, N_BASELINE)
        .withColumn("customer_id", F.element_at(cust_all_arr, (F.rand(41) * _n_cust_subset + 1).cast("int")))
        .withColumn("txn_date", F.date_sub(F.lit(HIST_END.date().isoformat()).cast("date"), (F.rand(42) * HIST_SPAN_DAYS).cast("int")))
        .withColumn("_r", F.rand(43))
        .withColumn(
            "txn_type",
            F.when(F.col("_r") < 0.45, F.lit("deposit")).when(F.col("_r") < 0.8, F.lit("withdrawal"))
            .when(F.col("_r") < 0.9, F.lit("fee")).otherwise(F.lit("interest")),
        )
        .withColumn(
            "amount_usd",
            F.when(F.col("txn_type") == "deposit", F.round(100 + F.rand(44) * 5000, 2))
            .when(F.col("txn_type") == "withdrawal", F.round(-(50 + F.rand(45) * 2000), 2))
            .when(F.col("txn_type") == "fee", F.round(-(5 + F.rand(46) * 35), 2))
            .otherwise(F.round(1 + F.rand(47) * 120, 2)),
        )
        .withColumn("channel", F.element_at(F.array(F.lit("online"), F.lit("mobile"), F.lit("branch"), F.lit("atm")), (F.rand(48) * 4 + 1).cast("int")))
        .withColumn("account_id", F.lit("ACCT-BASELINE"))
        .select("customer_id", "account_id", "txn_date", "amount_usd", "txn_type", "channel")
    )

    txn_df = (
        affected_txn.unionByName(baseline_txn)
        .withColumn("txn_id", F.concat(F.lit("TXN-"), F.lpad((F.monotonically_increasing_id() % 90000000 + 1).cast("string"), 8, "0")))
        .select("txn_id", "customer_id", "account_id", "txn_date", "amount_usd", "txn_type", "channel")
    )
    _save(txn_df, "raw_transactions")

    # ── 5. Risk snapshots ──────────────────────────────────────────────────────
    print("\n[5/6] Generating risk snapshots...")
    _CHURN_NOTES = [
        "asked about competitor CD rates", "mentioned moving funds at maturity",
        "rate shopping, called twice this week", "large transfer out pending", "unhappy with renewal rate",
    ]
    _HEALTHY_NOTES = ["routine service call", "satisfied, no concerns", None, None]
    churn_arr = F.array(*[F.lit(x) for x in _CHURN_NOTES])
    healthy_arr = F.array(*[(F.lit(x) if x is not None else F.lit(None).cast("string")) for x in _HEALTHY_NOTES])

    n_snap_days = (SNAPSHOT_DATE - RISK_WINDOW_START).days + 1

    affected_risk = (
        spark.createDataFrame([(c,) for c in AFFECTED_CUSTS], "customer_id string")
        .crossJoin(spark.range(0, n_snap_days).withColumnRenamed("id", "d"))
        .withColumn("snapshot_date", F.date_sub(F.lit(SNAPSHOT_DATE.date().isoformat()).cast("date"), F.col("d").cast("int")))
        .withColumn("_progress", (F.lit(n_snap_days - 1) - F.col("d")) / F.lit(float(max(n_snap_days - 1, 1))))
        .withColumn(
            "attrition_risk_score",
            F.when(
                F.col("customer_id") == HERO_CUST,
                F.round(F.least(F.lit(0.9), 0.25 + F.col("_progress") * 0.61), 3),
            ).otherwise(F.round(F.least(F.lit(0.95), 0.2 + F.col("_progress") * (0.6 + F.rand(51) * 0.2)), 3)),
        )
        .withColumn("balance_outflow_30d_usd", F.round(F.col("_progress") * (20000 + F.rand(52) * 120000), 2))
        .withColumn(
            "servicing_note_text",
            F.when(F.rand(53) < 0.85, F.element_at(churn_arr, (F.rand(54) * len(_CHURN_NOTES) + 1).cast("int")))
            .when(F.rand(55) < 0.3, F.element_at(healthy_arr, (F.rand(56) * len(_HEALTHY_NOTES) + 1).cast("int")))
            .otherwise(F.lit(None).cast("string")),
        )
        .select("customer_id", "snapshot_date", "attrition_risk_score", "balance_outflow_30d_usd", "servicing_note_text")
    )

    moderate_risk = (
        spark.createDataFrame([(c,) for c in MODERATE_CUSTS], "customer_id string")
        .withColumn("snapshot_date", F.lit(SNAPSHOT_DATE.date().isoformat()).cast("date"))
        .withColumn("attrition_risk_score", F.round(0.42 + F.rand(57) * 0.21, 3))
        .withColumn("balance_outflow_30d_usd", F.round(2000 + F.rand(58) * 15000, 2))
        .withColumn(
            "servicing_note_text",
            F.when(F.rand(59) < 0.6, F.element_at(churn_arr, (F.rand(60) * len(_CHURN_NOTES) + 1).cast("int")))
            .otherwise(F.element_at(healthy_arr, (F.rand(66) * len(_HEALTHY_NOTES) + 1).cast("int"))),
        )
        .select("customer_id", "snapshot_date", "attrition_risk_score", "balance_outflow_30d_usd", "servicing_note_text")
    )

    atrisk_cust_arr = F.array(*[F.lit(c) for c in ATRISK_CUSTS])
    everyday_risk = (
        spark.range(0, N_CUSTOMERS)
        .withColumn("customer_id", F.concat(F.lit("CUST-"), F.lpad((F.col("id") + 1).cast("string"), 7, "0")))
        .withColumn("is_atrisk", F.array_contains(atrisk_cust_arr, F.col("customer_id")))
        .filter(~F.col("is_atrisk"))
        .withColumn("snapshot_date", F.lit(SNAPSHOT_DATE.date().isoformat()).cast("date"))
        .withColumn("attrition_risk_score", F.round(0.05 + F.rand(61) * 0.2, 3))
        .withColumn("balance_outflow_30d_usd", F.round(F.rand(62) * 2000, 2))
        .withColumn("servicing_note_text", F.element_at(healthy_arr, (F.rand(63) * len(_HEALTHY_NOTES) + 1).cast("int")))
        .select("customer_id", "snapshot_date", "attrition_risk_score", "balance_outflow_30d_usd", "servicing_note_text")
    )

    risk_df = affected_risk.unionByName(moderate_risk).unionByName(everyday_risk)
    _save(risk_df, "raw_risk_snapshots")

    # ── 6. Retention campaigns ─────────────────────────────────────────────────
    print("\n[6/6] Generating retention campaigns...")
    cust_pop_arr = F.array(*[F.lit(f"CUST-{i + 1:07d}") for i in range(8000)])
    aff_prod_arr2 = F.array(*[F.lit(p) for p in AFFECTED_PRODUCTS])
    xsell_arr = F.array(*[F.lit(p) for p in CROSS_SELL_TARGETS])

    campaigns_df = (
        spark.range(0, 35_000)
        .withColumn("campaign_id", F.concat(F.lit("CMP-"), F.lpad((F.col("id") + 1).cast("string"), 8, "0")))
        .withColumn("customer_id", F.element_at(cust_pop_arr, (F.rand(71) * 8000 + 1).cast("int")))
        .withColumn("action_type", F.element_at(F.array(F.lit("retention_offer"), F.lit("retention_offer"), F.lit("cross_sell"), F.lit("rm_outreach")), (F.rand(72) * 4 + 1).cast("int")))
        .withColumn("product_id", F.element_at(aff_prod_arr2, (F.rand(73) * len(AFFECTED_PRODUCTS) + 1).cast("int")))
        .withColumn("offered_product_id", F.when(F.col("action_type") == "cross_sell", F.element_at(xsell_arr, (F.rand(74) * len(CROSS_SELL_TARGETS) + 1).cast("int"))).otherwise(F.lit(None).cast("string")))
        .withColumn("balance_at_risk_usd", F.round(20000 + F.rand(75) * 900000, 2))
        .withColumn("attrition_risk_at_action", F.round(0.3 + F.rand(76) * 0.65, 3))
        .withColumn("initiated_date", F.date_sub(F.lit(HIST_END.date().isoformat()).cast("date"), (F.rand(77) * HIST_SPAN_DAYS).cast("int")))
        .withColumn("days_to_resolve", F.when(F.col("action_type") == "retention_offer", (3 + F.rand(78) * 10).cast("int")).when(F.col("action_type") == "cross_sell", (5 + F.rand(79) * 20).cast("int")).otherwise((1 + F.rand(80) * 5).cast("int")))
        .withColumn(
            "_p_retain",
            F.when(F.col("action_type") == "retention_offer", F.least(F.lit(0.9), 0.45 + F.col("attrition_risk_at_action") * 0.4))
            .when(F.col("action_type") == "cross_sell", F.greatest(F.lit(0.1), 0.6 - F.col("attrition_risk_at_action") * 0.5))
            .otherwise(F.greatest(F.lit(0.05), 0.4 - F.col("attrition_risk_at_action") * 0.35)),
        )
        .withColumn("retained", (F.rand(81) < F.col("_p_retain")))
        .withColumn(
            "retained_revenue_usd",
            F.when(F.col("retained"), F.round(F.col("balance_at_risk_usd") * F.lit(NIM) * 3 * F.col("_p_retain"), 2)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "cost_usd",
            F.when(F.col("action_type") == "retention_offer", F.round(F.col("balance_at_risk_usd") * 0.006, 2))
            .when(F.col("action_type") == "cross_sell", F.lit(50.0))
            .otherwise(F.lit(40.0)),
        )
        .withColumn(
            "margin_impact_usd",
            F.when((F.col("action_type") == "cross_sell") & (~F.col("retained")), F.round(F.rand(82) * 200, 2)).otherwise(F.lit(0.0)),
        )
        .select(
            "campaign_id", "customer_id", "product_id", "action_type", "offered_product_id",
            "balance_at_risk_usd", "attrition_risk_at_action", "initiated_date", "days_to_resolve",
            "retained", "retained_revenue_usd", "margin_impact_usd", "cost_usd",
        )
    )
    _save(campaigns_df, "raw_retention_campaigns")

    print(f"\n✅ Meridian raw data generated in {CATALOG}.{SCHEMA} (volume {RAW_VOL}).")

# COMMAND ----------

# DBTITLE 1,Provision Raw Data (generate if missing)
# The raw Meridian Bank datasets (6 parquet datasets in the `raw_data` volume) are
# generated inline by the function above. If they aren't present yet, generate them.
# This is the equivalent of the wine workshop's CSV load step.
def _raw_data_exists():
    try:
        spark.read.parquet(f"{raw_root}/customers").limit(1).count()
        return True
    except Exception:
        return False

if _raw_data_exists():
    print(f"Raw data already present in {raw_root}")
else:
    print(f"Raw data not found — generating into {catalog_name}.{schema_name} ...")
    generate_raw_data(catalog_name, schema_name)

# COMMAND ----------

# DBTITLE 1,Build the customer_churn Analytic Table
# Flatten the 6 raw datasets into ONE customer-360 table (one row per customer),
# mirroring the wine workshop's single wine_quality_table. Each customer gets
# demographic, holdings, and recent-transaction features, plus a binary `churned`
# label derived from the latest attrition risk snapshot.
if spark.catalog.tableExists(f"{catalog_name}.{schema_name}.{table_name}"):
    print(f"Using existing {table_name} in {catalog_name}.{schema_name}")
else:
    print(f"Creating {table_name} in {catalog_name}.{schema_name}")

    customers    = spark.read.parquet(f"{raw_root}/customers")
    holdings     = spark.read.parquet(f"{raw_root}/holdings")
    transactions = spark.read.parquet(f"{raw_root}/transactions")
    risk         = spark.read.parquet(f"{raw_root}/risk_snapshots")

    # Latest risk snapshot per customer (the label source).
    w_risk = Window.partitionBy("customer_id").orderBy(F.col("snapshot_date").desc())
    latest_risk = (
        risk.withColumn("_rn", F.row_number().over(w_risk))
        .filter(F.col("_rn") == 1)
        .select("customer_id", "attrition_risk_score", "balance_outflow_30d_usd")
    )

    # Holdings aggregates: product breadth, total balance, deposit count, maturing CD.
    hold_agg = holdings.groupBy("customer_id").agg(
        F.count("*").alias("num_products"),
        F.round(F.sum("balance_usd"), 2).alias("total_balance_usd"),
        F.sum(F.when(F.col("product_id").startswith("PROD-DEP"), 1).otherwise(0)).alias("num_deposit_products"),
        F.max(F.when(F.col("maturity_date").isNotNull() & (F.col("status") == "active"), 1).otherwise(0)).alias("has_maturing_cd"),
    )

    # Recent transaction behavior — last 60 days relative to the latest snapshot.
    ref_date = risk.agg(F.max("snapshot_date")).first()[0]
    cutoff = F.date_sub(F.lit(ref_date.isoformat()).cast("date"), 60)
    txn_agg = (
        transactions.filter(F.col("txn_date") >= cutoff)
        .groupBy("customer_id").agg(
            F.count("*").alias("txn_count_60d"),
            F.round(F.sum(F.when(F.col("amount_usd") < 0, F.col("amount_usd")).otherwise(0.0)), 2).alias("total_outflow_60d_usd"),
            F.sum(F.when(F.col("txn_type").isin("withdrawal", "transfer_out"), 1).otherwise(0)).alias("withdrawal_count_60d"),
        )
    )

    tier_rank = (
        F.when(F.col("tier") == "private", 3)
        .when(F.col("tier") == "affluent", 2)
        .when(F.col("tier") == "mass_affluent", 1)
        .otherwise(0)
    )

    base = (
        customers.select("customer_id", "tier", "tenure_years", "home_metro", "state", "annual_income_band")
        .join(hold_agg, "customer_id", "left")
        .join(txn_agg, "customer_id", "left")
        .join(latest_risk, "customer_id", "left")
        .withColumn("tier_rank", tier_rank)
        .fillna({
            "num_products": 0, "total_balance_usd": 0.0, "num_deposit_products": 0, "has_maturing_cd": 0,
            "txn_count_60d": 0, "total_outflow_60d_usd": 0.0, "withdrawal_count_60d": 0,
            "attrition_risk_score": 0.0, "balance_outflow_30d_usd": 0.0,
        })
    )

    # ── Churn label ──────────────────────────────────────────────────────────
    # At-risk customers (latest attrition_risk_score >= 0.42) mostly churn. For
    # everyone else, a feature-driven propensity (recent outflow, high tier,
    # maturing CD) plus a small base rate — seeded for reproducibility. This lands
    # the overall churn rate in a learnable ~8-12% band while keeping real signal
    # in the demographic/behavioral features (see 03 for the leakage note: the
    # risk score itself is NOT used as a training feature).
    churn_p = (
        F.when(F.col("attrition_risk_score") >= 0.42, F.lit(0.85))
        .otherwise(
            F.lit(0.06)
            + F.when(F.col("total_outflow_60d_usd") < 0, 0.06).otherwise(0.0)
            + F.when(F.col("tier_rank") >= 2, 0.03).otherwise(0.0)
            + F.when(F.col("has_maturing_cd") == 1, 0.03).otherwise(0.0)
        )
    )
    churn_df = (
        base.withColumn("_p", churn_p)
        .withColumn("churned", (F.rand(42) < F.col("_p")).cast("int"))
        .drop("_p")
        .select(
            "customer_id", "tier", "tier_rank", "tenure_years", "home_metro", "state",
            "annual_income_band", "num_products", "total_balance_usd", "num_deposit_products",
            "has_maturing_cd", "txn_count_60d", "total_outflow_60d_usd", "withdrawal_count_60d",
            "attrition_risk_score", "balance_outflow_30d_usd", "churned",
        )
    )

    churn_df.write.mode("overwrite").saveAsTable(table_name)
    n = spark.table(table_name).count()
    rate = spark.table(table_name).agg(F.avg("churned")).first()[0]
    print(f"Created {table_name} with {n} rows  (churn rate {rate:.1%})")

# COMMAND ----------

# DBTITLE 1,Load Dataset
# Expose the customer-360 DataFrame so every notebook can use `df` right after
# running `%run "./00-Setup"` (no separate load cell needed downstream).
df = spark.table(table_name)
print(f"Loaded {table_name}: {df.count()} rows")

# COMMAND ----------

# DBTITLE 1,Tag Assets for Cleanup (sbc, churn-prediction)
# Tag every object this workshop creates with `sbc` and `churn-prediction` so all
# assets can be found and dropped together later.
WORKSHOP_TAGS = {"sbc": "true", "churn-prediction": "true"}
_tag_sql = ", ".join([f"'{k}' = '{v}'" for k, v in WORKSHOP_TAGS.items()])

def tag_table(fqname):
    try:
        spark.sql(f"ALTER TABLE {fqname} SET TAGS ({_tag_sql})")
        print(f"Tagged table {fqname}")
    except Exception as e:
        print(f"Could not tag {fqname}: {e}")

tag_table(f"{catalog_name}.{schema_name}.{table_name}")
try:
    spark.sql(f"ALTER VOLUME {catalog_name}.{schema_name}.raw_data SET TAGS ({_tag_sql})")
    print("Tagged volume raw_data")
except Exception as e:
    print(f"Could not tag volume raw_data: {e}")

# COMMAND ----------

# DBTITLE 1,Build the DA Config Object
experiments_dir = f"/Workspace{workshop_dir}/experiments"
os.makedirs(experiments_dir, exist_ok=True)

model_name = f"{catalog_name}.{schema_name}.bank_churn_model"
feature_table_name = f"{catalog_name}.{schema_name}.customer_churn_features"
base_table_name = f"{catalog_name}.{schema_name}.customer_churn"
experiment_path = f"/Workspace{workshop_dir}/experiments/bank-churn-training"
model_uri = f"models:/{model_name}@dev"
endpoint_name = "sbc-bank-churn-" + re.sub(r'[^a-zA-Z0-9-]', '-', username)
# Offline holdout-evaluation log written by 07 Section C (predictions + true labels).
# Distinct from the LIVE serving table monitored in 07 Section D
# (customer_churn_inference_unpacked). Named _eval_log to avoid confusion.
inference_log = f"{catalog_name}.{schema_name}.customer_churn_eval_log"

# The numeric feature columns used by FeatureLookup across notebooks 04/05/07/08.
feature_columns = [
    "tenure_years", "total_balance_usd", "num_products", "num_deposit_products",
    "has_maturing_cd", "txn_count_60d", "total_outflow_60d_usd", "withdrawal_count_60d",
    "tier_rank", "balanceCategory",
]

DA = SimpleNamespace(
    username=username,
    catalog_name=catalog_name,
    schema_name=schema_name,
    workshop_dir=workshop_dir,
    model_name=model_name,
    feature_table_name=feature_table_name,
    base_table_name=base_table_name,
    experiment_path=experiment_path,
    model_uri=model_uri,
    endpoint_name=endpoint_name,
    inference_log=inference_log,
    feature_columns=feature_columns,
    tags=WORKSHOP_TAGS,
    tag_sql=_tag_sql,
)

del model_name, feature_table_name, base_table_name, experiment_path, model_uri, endpoint_name, inference_log, feature_columns
print("Setup complete!")

# COMMAND ----------

# DBTITLE 1,Verify Setup
print(f"Username:       {DA.username}")
print(f"Catalog:        {DA.catalog_name}")
print(f"Schema:         {DA.schema_name}")
print(f"Base table:     {DA.base_table_name}")
print(f"Feature table:  {DA.feature_table_name}")
print(f"Model:          {DA.model_name}")
print(f"Endpoint:       {DA.endpoint_name}")

# COMMAND ----------

# DBTITLE 1,07-Observability — Load Dev Model & Held-out Split
# 00-Setup runs INLINE via `%run`, so `notebook_path` (set near the top) resolves to the
# CALLING notebook. We only do the heavier dev-model download + held-out split when 00 is
# run from 07-Observability; every other notebook that `%run`s 00-Setup skips this entirely.
if notebook_path.endswith("07-Observability"):
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    model_name = DA.model_name
    model_uri = DA.model_uri
    sk_model = None
    try:
        model_version = client.get_model_version_by_alias(model_name, "dev")
        run_id = model_version.run_id
        artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="bank_churn_model")
        for _root, _dirs, _files in os.walk(artifact_path):
            for _f in _files:
                if _f.endswith(".pkl"):
                    with open(os.path.join(_root, _f), "rb") as _fh:
                        sk_model = pickle.load(_fh)
                    break
            if sk_model is not None:
                break

        # Held-out split from the offline feature table (same split settings as 04-Model-Training).
        _feat_pdf = spark.table("customer_churn_features").toPandas()
        feature_cols = DA.feature_columns
        X = _feat_pdf[feature_cols]
        y = _feat_pdf["churned"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Print the loaded model only if the artifact was actually found.
        if sk_model is not None:
            print(f"Loaded model: {model_name} (version {model_version.version})")
            print(f"Model type:   {type(sk_model).__name__}")
        else:
            print("⚠️  Model artifact (.pkl) not found — SHAP/explainability (Section B) will be skipped.")
        print(f"Held-out test set: {X_test.shape[0]} samples × {X_test.shape[1]} features")
    except Exception as e:
        print(f"⚠️  Dev model/data not loaded (run 04-Model-Training first): {e}")
print("Setup complete!")