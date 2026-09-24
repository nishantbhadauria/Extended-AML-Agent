# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Alert Risk Scoring with an Explainable Boosting Machine
# MAGIC Glassbox model (InterpretML EBM), time-based validation, auto-close cutoff
# MAGIC chosen by SAR-leakage tolerance, hard overrides for CFT/CPF/HIGH CRR/
# MAGIC sanctions/PEP. Registered to Unity Catalog with the cutoff and the
# MAGIC global explanation as artefacts.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
from src.tm import ars, thresholds, registry, explain

dbutils.widgets.text("miss_tolerance", "0.01")
tol = float(dbutils.widgets.get("miss_tolerance"))
cm = load_cust_month().merge(spark.table(tbl("customer_segment")).toPandas()[["customer_id", "segment"]],
                             on="customer_id")
active = spark.table(tbl("threshold_versions")).where("status = 'ACTIVE'").toPandas()
alerts = ars.build_ars_frame(thresholds.generate_alerts(cm, active))
model = ars.ARSModel(miss_tolerance=tol).fit(alerts)
print(model.metrics)

# COMMAND ----------
display(explain.global_importance(model.ebm))
display(model.calibration_table(alerts))
display(explain.shape_function(model.ebm, "exceedance"))

# COMMAND ----------
# MAGIC %md Interactive EBM dashboard (validation review)

# COMMAND ----------
from interpret import show
show(model.ebm.explain_global())

# COMMAND ----------
run_id = registry.log_to_mlflow(model, extra_artifacts={"thresholds_version": active.version.unique().tolist()})
print("run", run_id)
