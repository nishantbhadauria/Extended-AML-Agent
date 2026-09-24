"""Persistence for the governed artefacts: the ARS model and threshold sets.

Production: MLflow (Unity Catalog model registry) for the EBM — it's
sklearn-compatible, so `mlflow.sklearn` logs it natively — with the cutoff,
miss tolerance, validation metrics and the global importance table logged
alongside, so the validation pack is reproducible from the run id.
Thresholds are versioned rows in a Delta table (see notebooks/03).

Local/dev: a joblib file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

from src.tm.ars import ARSModel

UC_MODEL_NAME = os.environ.get("AML_ARS_UC_MODEL", "aml.models.alert_risk_score_ebm")


def save_local(model: ARSModel, path: str | Path) -> None:
    joblib.dump(model, path)


def load_local(path: str | Path) -> ARSModel:
    return joblib.load(path)


def log_to_mlflow(model: ARSModel, extra_artifacts: dict | None = None,
                  register: bool = True) -> str:
    """Log the EBM + cutoff + metrics; optionally register in Unity Catalog."""
    import mlflow
    import mlflow.sklearn

    from src.tm.explain import global_importance

    mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run(run_name="ars_ebm") as run:
        mlflow.log_params({"miss_tolerance": model.miss_tolerance,
                           **{f"ebm_{k}": v for k, v in model.ebm_params.items()}})
        mlflow.log_metrics({k: v for k, v in model.metrics.items()
                            if isinstance(v, (int, float)) and v is not None})
        mlflow.log_dict({"cutoff": model.cutoff}, "ars_cutoff.json")
        mlflow.log_table(global_importance(model.ebm), "global_importance.json")
        for name, obj in (extra_artifacts or {}).items():
            mlflow.log_text(json.dumps(obj, default=str, indent=2), f"{name}.json")
        mlflow.sklearn.log_model(
            model.ebm, "model",
            registered_model_name=UC_MODEL_NAME if register else None)
        return run.info.run_id


def load_from_mlflow(model_uri: str, cutoff: float, miss_tolerance: float = 0.01) -> ARSModel:
    """Rehydrate an ARSModel from a registered EBM + its governed cutoff."""
    import mlflow.sklearn

    m = ARSModel(miss_tolerance=miss_tolerance)
    m.ebm, m.cutoff = mlflow.sklearn.load_model(model_uri), cutoff
    return m
