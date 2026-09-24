"""Streamlit front end.

Mirrors the cash agent's four-page layout, extended for AML/CFT/CPF:

  * Overview        — portfolio-level risk dashboard (alert volumes, top flags)
  * Investigation   — NL chatbot: ask a question about a case, see the generated
                      code + evidence, then summarize into a DRAFT SAR narrative
  * Patterns        — explore the pre-computed typology flags
  * Case deep-dive  — one party: profile, counterparty graph, flagged activity
  * Policy panel    — paste new guidance, get a PROPOSED control change to review
  * Compliance Q&A  — grounded, cited answers over the UAE regulatory corpus
                      (multi-agent RAG with citation/confidence/consistency gates)
  * ARS explain     — score alerts with the EBM and show each alert's drivers

The UI is a thin client over the FastAPI service — no agent logic here.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.io as pio
import requests
import streamlit as st

API = os.environ.get("AML_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="AML/CFT/CPF Agent", layout="wide")


def _post(path: str, payload: dict) -> dict:
    r = requests.post(f"{API}{path}", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


page = st.sidebar.radio(
    "Section",
    ["Overview", "Investigation", "Patterns", "Case deep-dive", "Policy panel",
     "Compliance Q&A", "ARS explain"],
)
st.sidebar.caption("Decision-support only. Nothing here files an STR/SAR — every "
                   "narrative is a draft for human review.")

# ------------------------------- Overview ---------------------------------- #
if page == "Overview":
    st.title("Portfolio risk overview")
    st.info("Wire this to a scheduled query over the scored table — alert volume "
            "by day, open cases by risk tier, and the most-fired typology flags.")
    st.caption("Placeholder charts; connect to fact_transaction_scored.")

# ----------------------------- Investigation ------------------------------- #
elif page == "Investigation":
    st.title("Investigation chatbot")
    party_id = st.text_input("Party ID under review", value="")
    question = st.text_area("Ask a question about this case",
                            placeholder="e.g. Show all structuring-flagged activity "
                                        "in the last 90 days with the driving amounts.")
    if st.button("Investigate", type="primary") and party_id and question:
        with st.spinner("Generating and running analysis..."):
            res = _post("/investigate", {"question": question, "party_id": party_id})
        with st.expander("Generated code", expanded=False):
            st.code(res["python_code"], language="python")
        st.caption(res["explanation"])
        if not res["ok"]:
            st.error(res["payload"])
        elif res["kind"] == "dataframe":
            df = pd.DataFrame(res["payload"])
            st.dataframe(df, use_container_width=True)
            st.session_state["last_result_preview"] = df.head(20).to_string()
        elif res["kind"] == "figure":
            st.plotly_chart(pio.from_json(res["payload"]), use_container_width=True)
            st.session_state["last_result_preview"] = "Chart produced."
        st.session_state["last_code"] = res["python_code"]
        st.session_state["last_question"] = question

    if st.session_state.get("last_result_preview"):
        st.divider()
        if st.button("Summarize into a DRAFT narrative"):
            with st.spinner("Summarizing..."):
                s = _post("/summarize", {
                    "question": st.session_state["last_question"],
                    "code": st.session_state["last_code"],
                    "result_preview": st.session_state["last_result_preview"],
                })
            st.subheader("Insight");            st.write(s["insight"])
            st.subheader("DRAFT SAR narrative"); st.warning(s["sar_narrative_draft"])
            st.subheader("Recommended action");  st.write(s["recommended_action"])
            st.subheader("Data-quality caveats"); st.write(s["data_quality_caveats"])
        c1, c2 = st.columns(2)
        if c1.button("👍 Useful"):
            _post("/feedback", {"session_id": "ui", "question": st.session_state["last_question"],
                               "code": st.session_state["last_code"], "verdict": "up"})
            st.toast("Logged.")
        if c2.button("👎 Not useful"):
            _post("/feedback", {"session_id": "ui", "question": st.session_state["last_question"],
                               "code": st.session_state["last_code"], "verdict": "down"})
            st.toast("Logged.")

# ------------------------------- Patterns ---------------------------------- #
elif page == "Patterns":
    st.title("Typology / pattern catalogue")
    data = requests.get(f"{API}/patterns", timeout=30).json()
    st.metric("Pre-computed flags", data["count"])
    st.text(data["catalogue"])

# ----------------------------- Case deep-dive ------------------------------ #
elif page == "Case deep-dive":
    st.title("Case deep-dive")
    st.text_input("Party ID")
    st.info("Render the party profile, the counterparty subgraph (edges), and the "
            "flagged-activity timeline here. Graph via networkx/pyvis over the "
            "edges frame returned by the API.")

# ------------------------------ Policy panel ------------------------------- #
elif page == "Policy panel":
    st.title("Regulatory policy panel")
    st.caption("Paste an excerpt of new guidance. The agent PROPOSES which "
               "controls it affects — a compliance officer approves before any "
               "live rule changes.")
    excerpt = st.text_area("Guidance excerpt")
    source = st.text_input("Source (document id + version)", value="CBUAE-2026-04-16 v1")
    catalogue = st.text_area("Control catalogue (ids + descriptions)",
                             value="See patterns page for current flags.")
    if st.button("Propose mapping") and excerpt:
        with st.spinner("Mapping guidance to controls..."):
            m = _post("/policy/map", {"guidance_excerpt": excerpt,
                                      "guidance_source": source,
                                      "control_catalogue": catalogue})
        st.subheader("Obligation");        st.write(m["obligation_summary"])
        st.subheader("Affected controls"); st.write(m["affected_controls"])
        st.subheader("Proposed change");   st.warning(m["proposed_change"])
        st.subheader("Rationale");         st.caption(m["rationale"])
        st.button("Send to compliance officer for approval (stub)")

# ----------------------------- Compliance Q&A ------------------------------ #
elif page == "Compliance Q&A":
    st.title("Compliance Q&A (UAE regulatory corpus)")
    q = st.text_area("Question", placeholder="What CDD must be refreshed on a trigger event "
                                              "under Cabinet Decision 134 of 2025?")
    c1, c2 = st.columns(2)
    multi = c1.toggle("Multi-agent validation", value=True)
    juris = c2.selectbox("Jurisdiction filter", ["All", "UAE", "UAE-DIFC", "INTL"])
    if st.button("Ask", type="primary") and q:
        with st.spinner("Retrieving and validating..."):
            r = _post("/rag/query", {"question": q, "multi_agent": multi,
                                     "jurisdiction": None if juris == "All" else juris})
        if r["status"] == "GROUNDED":
            st.success("Grounded answer")
        else:
            st.error("Insufficient grounding — read the sources below before relying on this.")
        st.write(r["answer"])
        g = r["gates"]
        cols = st.columns(3)
        cols[0].metric("Citation gate", "pass" if g.get("citation") else "fail")
        cols[1].metric("Confidence", f"{r['confidence']:.2f}" if r.get("confidence") is not None else "–")
        cols[2].metric("Consistency", "pass" if g.get("consistency") else "–" if "consistency" not in g else "fail")
        if r.get("consistency_issues") and r["consistency_issues"].lower() != "none":
            st.warning(r["consistency_issues"])
        with st.expander(f"Sources ({len(r['sources'])})"):
            st.dataframe(pd.DataFrame(r["sources"]), use_container_width=True)

# ------------------------------- ARS explain ------------------------------- #
elif page == "ARS explain":
    st.title("Alert risk score — explained")
    st.caption("EBM (InterpretML) glassbox model. The contributions below ARE the model, "
               "not an approximation of it.")
    g = requests.get(f"{API}/ars/global", timeout=60).json()
    imp = pd.DataFrame(g["importance"])
    st.subheader("What drives the model overall")
    st.bar_chart(imp.set_index("term").head(12))

    st.subheader("Score alerts")
    up = st.file_uploader("CSV of alerts (ARS feature columns + typology)", type="csv")
    if up is not None:
        alerts = pd.read_csv(up)
        res = _post("/ars/score", {"alerts": alerts.to_dict("records")})
        out = alerts.assign(
            ars_score=[x["ars_score"] for x in res["results"]],
            route=[x["route"] for x in res["results"]],
            hard_override=[x["hard_override"] for x in res["results"]])
        st.caption(f"Auto-close cutoff: {res['cutoff']:.4g}")
        st.dataframe(out, use_container_width=True)
        i = st.number_input("Explain row", 0, len(out) - 1, 0)
        st.code(res["results"][int(i)]["explanation"])
