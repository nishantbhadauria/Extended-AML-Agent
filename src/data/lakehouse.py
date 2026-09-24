"""Databricks lakehouse access.

Replaces the cash agent's BigQuery layer. Two modes:

  * `SqlClient` — for the FastAPI service running *outside* Databricks: uses the
    databricks-sql-connector to pull result sets into pandas for the agent's
    execution namespace. Keep pulls scoped (a party, a date window) — never
    `SELECT *` the whole fact table into the driver.
  * `spark_apply_patterns` — for the *inside-Databricks* batch job that stamps
    the flag_ columns onto the gold table (see patterns/detection_patterns.py).

Feedback (thumbs up/down + rationale) is written back to a Delta table, the
Databricks equivalent of the cash agent's BigQuery feedback log — later used to
compile the DSPy program with an optimizer once enough labels accrue.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config.settings import SETTINGS

LH = SETTINGS.lakehouse


def _fqn(table: str) -> str:
    return f"{LH.catalog}.{LH.gold_schema}.{table}"


class SqlClient:
    """Thin wrapper over databricks-sql-connector returning pandas frames."""

    def __init__(self) -> None:
        # Imported lazily so the module loads in environments without the driver.
        from databricks import sql  # type: ignore

        self._conn = sql.connect(
            server_hostname=LH.server_hostname,
            http_path=LH.http_path,
            access_token=LH.access_token,
        )

    def query(self, sql_text: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run parameterised SQL and return a pandas DataFrame."""
        with self._conn.cursor() as cur:
            cur.execute(sql_text, params or {})
            cols = [c[0] for c in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)

    # --- Scoped pulls that populate the agent's execution namespace ---------- #
    def transactions_for_party(self, party_id: str, days: int = 180) -> pd.DataFrame:
        """Transactions for one party over the last `days` days."""
        return self.query(
            f"""
            SELECT * FROM {_fqn(LH.tx_table)}
            WHERE party_id = :pid
              AND event_ts >= current_timestamp() - INTERVAL {int(days)} DAYS
            """,
            {"pid": party_id},
        )

    def party(self, party_id: str) -> pd.DataFrame:
        """The entity-resolved party record."""
        return self.query(
            f"SELECT * FROM {_fqn(LH.party_table)} WHERE party_id = :pid",
            {"pid": party_id},
        )

    def edges_for_party(self, party_id: str) -> pd.DataFrame:
        """Counterparty graph edges touching the party."""
        return self.query(
            f"""
            SELECT * FROM {_fqn(LH.edges_table)}
            WHERE src_party_id = :pid OR dst_party_id = :pid
            """,
            {"pid": party_id},
        )

    def open_alerts(self, limit: int = 500) -> pd.DataFrame:
        """Currently open alerts, capped at `limit`."""
        return self.query(
            f"SELECT * FROM {_fqn(LH.alerts_table)} WHERE status = 'OPEN' LIMIT {int(limit)}"
        )

    def load_case_frames(self, party_id: str) -> dict[str, pd.DataFrame]:
        """The four DataFrames bound into the sandbox namespace for a case."""
        return {
            "tx": self.transactions_for_party(party_id),
            "party": self.party(party_id),
            "edges": self.edges_for_party(party_id),
            "alerts": self.open_alerts(),
        }

    def log_feedback(self, record: dict[str, Any]) -> None:
        """Append thumbs up/down + rationale to the Delta feedback table."""
        record = {"logged_at": datetime.now(timezone.utc).isoformat(), **record}
        cols = ", ".join(record.keys())
        placeholders = ", ".join(f":{k}" for k in record)
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {_fqn(LH.feedback_table)} ({cols}) VALUES ({placeholders})",
                record,
            )

    def schema_string(self) -> str:
        """Compact schema text for the agent's `table_schemas` input."""
        parts = []
        for label, tbl in [("tx", LH.tx_table), ("party", LH.party_table),
                           ("edges", LH.edges_table), ("alerts", LH.alerts_table)]:
            info = self.query(f"DESCRIBE {_fqn(tbl)}")
            cols = ", ".join(f"{r.col_name}:{r.data_type}" for r in info.itertuples())
            parts.append(f"{label} ({tbl}): {cols}")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Inside-Databricks batch entry point (call from a job / Dataiku recipe).
# --------------------------------------------------------------------------- #
def spark_apply_patterns() -> None:
    """Read gold transactions, stamp flag_ columns, write back a scored table."""
    from pyspark.sql import SparkSession
    from src.patterns.detection_patterns import apply_all_patterns

    spark = SparkSession.builder.getOrCreate()
    src = spark.read.table(_fqn(LH.tx_table))
    scored = apply_all_patterns(src)
    (scored.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(_fqn(f"{LH.tx_table}_scored")))
