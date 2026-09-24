# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · SAR-driven threshold tuning (ATL) + BTL sampling plan
# MAGIC Highest threshold per cell that keeps SAR capture >= the recall floor.
# MAGIC Cells with too few SARs keep P95 and are flagged. Output is **PROPOSED** —
# MAGIC a threshold only becomes ACTIVE after compliance sign-off (Dataiku Govern).

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import mlflow
from src.tm import thresholds

dbutils.widgets.text("recall_floor", "0.95")
floor = float(dbutils.widgets.get("recall_floor"))
cm = load_cust_month().merge(spark.table(tbl("customer_segment")).toPandas()[["customer_id", "segment"]],
                             on="customer_id")

tuned = thresholds.tune(cm, recall_floor=floor)
btl = thresholds.btl_sample_plan(cm, tuned)
display(tuned); display(btl)

# COMMAND ----------
# MAGIC %md Sweep curves for one cell — the chart that goes in the tuning memo.

# COMMAND ----------
sc = thresholds.SCENARIO_REGISTRY[0]
display(thresholds.sweep(cm, sc, sorted(cm.segment.unique())[0]))

# COMMAND ----------
with mlflow.start_run(run_name="threshold_tuning"):
    mlflow.log_param("recall_floor", floor)
    mlflow.log_metric("cells_tuned", int((tuned.status == "TUNED").sum()))
    mlflow.log_metric("cells_insufficient_sars", int((tuned.status != "TUNED").sum()))
    mlflow.log_table(tuned, "tuned_thresholds.json")
    mlflow.log_table(btl, "btl_plan.json")

write_delta(tuned.assign(version=f"tuned-{RUN_LABEL}", status="PROPOSED"), "threshold_versions")
write_delta(btl.assign(version=f"tuned-{RUN_LABEL}"), "btl_sample_plan")
