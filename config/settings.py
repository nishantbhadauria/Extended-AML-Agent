"""Central configuration.

Everything environment-specific lives here so the agent code stays portable.
At FAB, wire the values below to the Databricks secret scope / Dataiku
project variables rather than committing real endpoints or credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass
class LLMConfig:
    """Two-model setup mirroring the cash agent (main code-gen model +
    lighter summarizer). Defaults target Azure OpenAI on Databricks, but the
    provider is pluggable via a LiteLLM-style model string."""

    # Primary reasoning / code-generation model (was wmt-gpt-4o on the cash agent).
    investigator_model: str = _env("AML_INVESTIGATOR_MODEL", "azure/gpt-4o")
    # Cheaper model for narrative summarization (was gemini-2.0-flash).
    summarizer_model: str = _env("AML_SUMMARIZER_MODEL", "azure/gpt-4o-mini")

    # Gateway / endpoint. For a corporate LLM gateway, point this at the proxy
    # and let the custom provider inject the OAuth/JWT header (see llm/gateway.py).
    api_base: str = _env("AML_LLM_API_BASE", "")
    api_version: str = _env("AML_LLM_API_VERSION", "2024-08-01-preview")

    temperature: float = 0.0          # deterministic — this is compliance, not chat
    max_tokens: int = 4096
    request_timeout_s: int = 90


@dataclass
class LakehouseConfig:
    """Databricks connection. Use the SQL connector from a service/host, or a
    live SparkSession when running inside a Databricks notebook/job."""

    server_hostname: str = _env("DATABRICKS_HOST", "")
    http_path: str = _env("DATABRICKS_HTTP_PATH", "")
    access_token: str = _env("DATABRICKS_TOKEN", "")

    catalog: str = _env("AML_CATALOG", "aml")          # Unity Catalog
    gold_schema: str = _env("AML_GOLD_SCHEMA", "gold")

    # Canonical gold tables the agent reads from.
    tx_table: str = "fact_transaction"                  # conformed transactions
    party_table: str = "dim_party"                      # entity-resolved customers
    edges_table: str = "party_edges"                    # counterparty graph edges
    alerts_table: str = "alert"                         # generated alerts
    feedback_table: str = "agent_feedback"              # thumbs up/down + rationale


@dataclass
class RiskConfig:
    """Thresholds surfaced to the agent as context. These are *defaults* — in
    production they are owned by the policy loop and versioned, never hardcoded
    in a scenario. Values are illustrative."""

    structuring_window_days: int = 7
    structuring_min_txns: int = 3
    # UAE STR reporting has no monetary threshold; this is only a *structuring*
    # heuristic for amounts kept just below a round reporting-style figure.
    just_below_amount: float = 55_000.0     # e.g. DPMS cash threshold (AED)
    rapid_movement_hours: int = 48
    dormancy_days: int = 180
    high_velocity_txn_count: int = 20


@dataclass
class Settings:
    """Top-level settings bundle."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    lakehouse: LakehouseConfig = field(default_factory=LakehouseConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Sanctions / watchlist sources screened in real time. In production these
    # are versioned; store the version id alongside every screening decision.
    watchlists: tuple[str, ...] = (
        "UN_SECURITY_COUNCIL",
        "UAE_LOCAL_TERRORIST_LIST",
        "CBUAE_SPECIFIED",
        "OFAC_SDN",
        "EU_CONSOLIDATED",
        "UK_HMT",
    )


SETTINGS = Settings()
