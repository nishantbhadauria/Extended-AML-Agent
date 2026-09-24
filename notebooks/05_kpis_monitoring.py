# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · KPIs & ongoing monitoring
# MAGIC Scenario productivity, SAR coverage, auto-close rate and leakage, plus PSI
# MAGIC drift of the ARS score and key features. Schedule monthly; feeds the MI
# MAGIC pack and the MRM ongoing-monitoring report.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
from src.tm import ars, kpis, thresholds, registry

cm = load_cust_month().merge(spark.table(tbl("customer_segment")).toPandas()[["customer_id", "segment"]],
                             on="customer_id")
active = spark.table(tbl("threshold_versions")).where("status = 'ACTIVE'").toPandas()
alerts = ars.build_ars_frame(thresholds.generate_alerts(cm, active))
dbutils.widgets.text("ars_cutoff", "")          # the governed cutoff from notebook 04's run
model = registry.load_from_mlflow(f"models:/{registry.UC_MODEL_NAME}@champion",
                                  cutoff=float(dbutils.widgets.get("ars_cutoff")))
scored = model.score(alerts)

sk = kpis.scenario_kpis(alerts); pk = kpis.programme_kpis(cm, alerts, scored)
display(sk); print(pk)

months = sorted(scored.month.unique())
drift = kpis.drift_report(scored[scored.month < months[-1]], scored[scored.month == months[-1]],
                          ["ars_score", "exceedance", "cross_border_ratio", "rapid_in_out_ratio"])
display(drift)

import pandas as pd
write_delta(sk, "kpi_scenario_history")
write_delta(pd.DataFrame([pk]), "kpi_programme_history")
write_delta(drift, "drift_history")
