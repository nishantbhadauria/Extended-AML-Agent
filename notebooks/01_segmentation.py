# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Customer segmentation
# MAGIC Entity-type split, then KMeans with silhouette-selected k on behavioural
# MAGIC features. Output: segment assignments, a plain-number profile for sign-off,
# MAGIC and stability vs the latest month.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import mlflow
from src.tm import segmentation

cm = load_cust_month()
months = sorted(cm.month.unique())
feats_ref = segmentation.customer_features(cm[cm.month < months[-1]])
seg = segmentation.SegmentModel().fit(feats_ref)
feats_ref["segment"] = seg.assign(feats_ref)
feats_cur = segmentation.customer_features(cm[cm.month == months[-1]])
feats_cur["segment"] = seg.assign(feats_cur)
profile = segmentation.segment_profile(feats_ref)
display(profile)

# COMMAND ----------
display(segmentation.migration_matrix(feats_ref, feats_cur).reset_index())
stab = segmentation.stability(feats_ref, feats_cur)

with mlflow.start_run(run_name="segmentation"):
    mlflow.log_params(seg.k_by_entity)
    mlflow.log_metrics({f"silhouette_{k}": v for k, v in seg.silhouettes.items()})
    mlflow.log_metric("segment_stability", stab)
    import joblib; joblib.dump(seg, "/tmp/segment_model.joblib")
    mlflow.log_artifact("/tmp/segment_model.joblib")
    mlflow.log_table(profile, "segment_profile.json")

write_delta(feats_ref[["customer_id", "entity_type", "segment"]], "customer_segment", "overwrite")
write_delta(profile, "segment_profile_history")
