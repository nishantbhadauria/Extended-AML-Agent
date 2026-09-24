# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Regulatory corpus ingestion (RAG)
# MAGIC Article-aware chunks of the UAE corpus -> Delta table -> Databricks Vector
# MAGIC Search delta-sync index. Re-run when CBUAE/Cabinet/DFSA issue new versions;
# MAGIC bump `version` so old and new text coexist for audit.

# COMMAND ----------
# MAGIC %run ./00_config

# COMMAND ----------
import pandas as pd
from src.rag.ingest import UAE_CORPUS_MANIFEST, chunk_document, load_pdf_text

dbutils.widgets.text("docs_volume", "/Volumes/aml/raw/regulatory_docs")
dbutils.widgets.text("version", "v1")
root, version = dbutils.widgets.get("docs_volume"), dbutils.widgets.get("version")

rows = []
for m in UAE_CORPUS_MANIFEST:
    path = f"{root}/{m['doc_id']}.pdf"
    try:
        text = load_pdf_text(path)
    except FileNotFoundError:
        print("missing", path); continue
    rows += [{**c.meta(), "text": c.text} for c in
             chunk_document(text, m["doc_id"], m["title"], version, m["effective"], m["jurisdiction"])]
chunks = pd.DataFrame(rows)
display(chunks.groupby("doc_id").size().reset_index(name="chunks"))
(spark.createDataFrame(chunks).write.mode("append").option("mergeSchema", "true")
      .option("delta.enableChangeDataFeed", "true").saveAsTable(tbl("regulatory_chunks")))

# COMMAND ----------
# One-time: create the delta-sync index (managed embeddings via the endpoint).
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()
# vsc.create_delta_sync_index(
#     endpoint_name="aml-vs", index_name=tbl("regulatory_chunks_idx"),
#     source_table_name=tbl("regulatory_chunks"), primary_key="chunk_id",
#     pipeline_type="TRIGGERED", embedding_source_column="text",
#     embedding_model_endpoint_name="databricks-gte-large-en")
