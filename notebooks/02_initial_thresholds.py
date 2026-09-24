# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Initial thresholds (percentile rules)
# MAGIC Per (scenario, segment) P95 of the scenario metric, floored by any policy
# MAGIC minimum. The starting point when no SAR labels exist for a new scenario.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
from src.tm import thresholds

cm = load_cust_month().merge(spark.table(tbl("customer_segment")).toPandas()[["customer_id", "segment"]],
                             on="customer_id")
dbutils.widgets.text("init_pct", "95")
init = thresholds.initial_thresholds(cm, pct=float(dbutils.widgets.get("init_pct")))
display(init)
write_delta(init.assign(version=f"init-{RUN_LABEL}", status="PROPOSED"), "threshold_versions")
