from pathlib import Path
import json
from typing import List, Dict, Any

import streamlit as st

LOG_PATH = Path("logs/requests.jsonl")

st.set_page_config(page_title="RAG Monitoring", layout="wide")
st.title("RAG Monitoring Dashboard")

st.write("Ez az oldal a FastAPI backend által generált 'logs/requests.json1' fájlt olvassa.")

if not LOG_PATH.exists():
    st.warning(f"Még nem létezik a log file: {LOG_PATH}")
    st.stop()

records: List[Dict[str, Any]] = []
with LOG_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            records.append(rec)
        except json.JSONDecodeError:
            continue

if not records:
    st.warning("Nincsenek logolt kérések.")
    st.stop()

total_requests = len(records)
avg_latency = sum(r.get("total_latency_sec", 0.0) for r in records) / total_requests

latencies = [r.get("first_token_latency_sec") for r in records if r.get("first_token_latency_sec") is not None]
if latencies:
    avg_first_token = sum(latencies) / len(latencies)
else:
    avg_first_token = 0.0

total_cost = sum(r.get("cost_estimate", 0.0) for r in records)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Összes kérés", total_requests)
with col2:
    st.metric("Átlag latency (s)", f"{avg_latency:.3f}")
with col3:
    st.metric(
        "Átlag first-token latency (s)", 
        '-' if not latencies else f"{avg_first_token:.3f}",
    )
with col4:
    st.metric("Becsült összköltség ($)", f"{total_cost:.4f}")

st.markdown("---")

st.subheader("Utolsó 50 kérés")
records_sorted = sorted(records, key=lambda r: r.get("ts", 0), reverse=True)[:50]

table_rows = []
for r in records_sorted:
    table_rows.append(
        {
            "időbélyeg": r.get("ts"),
            "endpoint": r.get("endpoint"),
            "session": r.get("session_id"),
            "latency (s)": round(r.get("total_latency_sec", 0.0), 3),
            "input_tokens": r.get("input_tokens_est"),
            "output_tokens": r.get("output_tokens_est"),
            "cost ($)": round(r.get("cost_estimate", 0.0), 6),
            "kérés": r.get("question")[:60] + "..." if r.get("question") else "",
        }
    )

st.dataframe(table_rows)