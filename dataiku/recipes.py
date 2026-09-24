"""Dataiku integration scaffold.

Where Dataiku fits vs Databricks in this project:

  * Databricks  = compute + canonical data + heavy ML + Unity Catalog lineage.
                  The pattern job (spark_apply_patterns) and the LLM calls run
                  here; the gold Delta tables live here.
  * Dataiku     = the analyst-and-governance surface. Compliance analysts tune
                  scenarios visually, orchestrate the batch runs (Scenarios),
                  and drive the human-in-the-loop approvals (Govern node) for
                  both the SAR drafts and the policy-loop change proposals.

Two integration paths:

  1. Dataiku reads/writes the SAME Delta tables via a Databricks connection, so
     both tools operate on one source of truth (recommended).
  2. A Dataiku Python recipe calls this project's code directly (below), useful
     for the pattern-scoring recipe and for wrapping the policy mapper as a
     reviewable step.

The snippets are illustrative — inside Dataiku you'd use dataiku.Dataset(...)
handles rather than the plain functions here.
"""
from __future__ import annotations


# --------------------------------------------------------------------------- #
# Recipe 1 — score transactions with the typology flags (Spark recipe).
# In Dataiku: a PySpark recipe, input = gold fact_transaction (Databricks
# connection), output = fact_transaction_scored.
# --------------------------------------------------------------------------- #
def recipe_score_patterns():
    """Pseudo-body for a Dataiku PySpark recipe."""
    # import dataiku
    # from dataiku import spark as dkuspark
    # from pyspark.sql import SparkSession
    from src.patterns.detection_patterns import apply_all_patterns

    # sqlContext / SparkSession provided by the Dataiku Spark runtime
    # inp = dkuspark.get_dataframe(sqlContext, dataiku.Dataset("fact_transaction"))
    # out = apply_all_patterns(inp)
    # dkuspark.write_with_schema(dataiku.Dataset("fact_transaction_scored"), out)
    _ = apply_all_patterns  # reference so linters see the intended call
    raise NotImplementedError("Paste into a Dataiku PySpark recipe and uncomment.")


# --------------------------------------------------------------------------- #
# Recipe 2 — policy mapping as a governed, reviewable step.
# Input dataset: guidance excerpts. Output dataset: proposed control changes,
# routed into a Dataiku sign-off workflow before anything touches a live rule.
# --------------------------------------------------------------------------- #
def recipe_policy_proposals():
    """Pseudo-body for a Dataiku Python recipe."""
    # import dataiku, pandas as pd
    from src.agents.agents import PolicyMapper
    from src.llm.gateway import configure_dspy

    configure_dspy(use_gateway=True)
    mapper = PolicyMapper()

    # excerpts = dataiku.Dataset("regulatory_excerpts").get_dataframe()
    # rows = []
    # for r in excerpts.itertuples():
    #     m = mapper.forward(r.excerpt, r.source, control_catalogue=CATALOGUE)
    #     rows.append({"source": r.source, **m, "status": "PENDING_REVIEW"})
    # dataiku.Dataset("proposed_control_changes").write_with_schema(pd.DataFrame(rows))
    _ = mapper
    raise NotImplementedError("Paste into a Dataiku Python recipe and uncomment.")


# --------------------------------------------------------------------------- #
# Scenario — schedule scoring + refresh, with a data-quality check gate.
# Build this in the Dataiku Scenario UI:
#   step 1: run recipe_score_patterns (Spark)
#   step 2: metrics/checks on fact_transaction_scored (row count, null rate)
#   step 3: on pass, refresh the API's cache / notify; on fail, alert + stop
# Keeping the schedule and the checks in Dataiku gives compliance an auditable,
# owned orchestration layer rather than a cron buried in a notebook.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Recipe 3 — threshold approval (the sign-off gate between tuning and ACTIVE).
# Input: threshold_versions rows with status = PROPOSED (written by notebook 03).
# Dataiku Govern holds the sign-off; on approval this recipe flips status to
# ACTIVE and supersedes the previous ACTIVE version for the same cell. Nothing
# in the Databricks notebooks ever writes ACTIVE — that's the control.
# --------------------------------------------------------------------------- #
def recipe_activate_thresholds(approved_version: str, approver: str):
    """Pseudo-body for a Dataiku SQL/Python recipe on the Databricks connection."""
    sql = f"""
    MERGE INTO aml.gold.threshold_versions t
    USING (SELECT scenario_id, segment FROM aml.gold.threshold_versions
           WHERE version = '{approved_version}') n
    ON t.scenario_id = n.scenario_id AND t.segment = n.segment AND t.status = 'ACTIVE'
    WHEN MATCHED THEN UPDATE SET t.status = 'SUPERSEDED';
    UPDATE aml.gold.threshold_versions
       SET status = 'ACTIVE', approved_by = '{approver}', approved_at = current_timestamp()
     WHERE version = '{approved_version}';
    """
    # Execute with dataiku.SQLExecutor2(connection="databricks_aml").query_to_df(...)
    # and attach the tuning memo (sweep curves + BTL plan) to the Govern item.
    return sql


# --------------------------------------------------------------------------- #
# Suggested Dataiku Scenario for the monthly TM cycle (Databricks Jobs do the
# compute; Dataiku owns sequencing, checks and approvals):
#   1. Trigger Databricks job: 01_segmentation -> 03_sar_threshold_tuning
#   2. Check: every cell has status TUNED or INSUFFICIENT_SARS_KEPT_P95
#   3. Create Govern sign-off item with tuned table + BTL plan attached
#   4. (after approval) recipe_activate_thresholds
#   5. Trigger 04_ars_ebm (retrain on ACTIVE thresholds) -> MRM review item
#   6. Trigger 05_kpis_monitoring; alert if any PSI status == INVESTIGATE
# --------------------------------------------------------------------------- #
