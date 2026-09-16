"""
dashboard.py — run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="LLM Eval Dashboard", layout="wide")
st.title("🧪 LLM Evaluation Dashboard")
st.caption("Medical QA · LLaMA-3 vs Mixtral vs Gemma-7B — all via Groq")

@st.cache_data
def load():
    return pd.read_csv("eval_results.csv", encoding="latin1")

try:
    df = load()
except FileNotFoundError:
    st.error("eval_results.csv not found. Run pipeline.py first.")
    st.stop()

st.sidebar.header("Filters")
models = st.sidebar.multiselect("Models", df["model"].unique(), default=list(df["model"].unique()))
df = df[df["model"].isin(models)]

avg = df.groupby("model")[
    ["faithfulness", "relevancy", "correctness", "latency_sec", "tokens_used"]
].mean().round(3).reset_index()

st.subheader("Model Averages")
st.dataframe(avg, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(avg, x="model", y="faithfulness", color="model",
                 title="Faithfulness (higher = better)", text_auto=".3f")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(avg, x="model", y="correctness", color="model",
                 title="Correctness (higher = better)", text_auto=".3f")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig = px.bar(avg, x="model", y="relevancy", color="model",
                 title="Relevancy (higher = better)", text_auto=".3f")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.bar(avg, x="model", y="latency_sec", color="model",
                 title="Avg Latency in seconds (lower = better)", text_auto=".3f")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Overall Model Profile (Radar)")
metrics = ["faithfulness", "relevancy", "correctness"]
fig_radar = go.Figure()
colors = ["#636EFA", "#EF553B", "#00CC96"]
for i, row in avg.iterrows():
    values = [row[m] for m in metrics] + [row[metrics[0]]]
    fig_radar.add_trace(go.Scatterpolar(
        r=values, theta=metrics + [metrics[0]],
        fill="toself", name=row["model"], line_color=colors[i % len(colors)]
    ))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
st.plotly_chart(fig_radar, use_container_width=True)

st.subheader("Latency vs Correctness")
fig_scatter = px.scatter(avg, x="latency_sec", y="correctness", text="model",
                         size="tokens_used", title="Latency vs Correctness (bubble = tokens used)")
fig_scatter.update_traces(textposition="top center")
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Per-Question Breakdown")
selected_model = st.selectbox("Select model", df["model"].unique())
cols = ["question", "answer", "faithfulness", "relevancy", "correctness", "latency_sec"]
st.dataframe(df[df["model"] == selected_model][cols], use_container_width=True, hide_index=True)
