# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Shared config
# MAGIC Run via `%run ./00_config` from every process notebook. Widgets make each
# MAGIC notebook a parameterised Databricks Job task (and callable from Dataiku).

# COMMAND ----------
import sys, os
sys.path.append(os.path.abspath(".."))          # repo root (Databricks Repos / Git folder)

dbutils.widgets.text("catalog", "aml")
dbutils.widgets.text("schema", "gold")
dbutils.widgets.text("run_label", "manual")
CATALOG, SCHEMA = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
RUN_LABEL = dbutils.widgets.get("run_label")

def tbl(name: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{name}"

def load_cust_month(months: int = 6):
    """Customer-month aggregates (built upstream in the Silver->Gold DLT pipeline).
    At full scale, fit on a stratified sample; score with Spark afterwards."""
    return (spark.table(tbl("cust_month"))
                 .where(f"month >= add_months(current_date(), -{months})")
                 .toPandas())

def write_delta(pdf, name: str, mode: str = "append"):
    from pyspark.sql import functions as F
    (spark.createDataFrame(pdf)
          .withColumn("run_label", F.lit(RUN_LABEL))
          .withColumn("created_at", F.current_timestamp())
          .write.mode(mode).option("mergeSchema", "true").saveAsTable(tbl(name)))
