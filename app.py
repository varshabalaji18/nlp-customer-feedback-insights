"""Streamlit application for interactive customer-feedback intelligence."""

from __future__ import annotations

import html
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import FeedbackInsightsPipeline
from src.preprocessing import load_feedback


st.set_page_config(page_title="Feedback Intelligence", page_icon="🧭", layout="wide")
st.title("🧭 Customer Feedback Intelligence")
st.caption("Transformer sentiment, entity extraction, and actionable topic signals for unstructured feedback.")


@st.cache_resource(show_spinner="Loading NLP models…")
def get_pipeline() -> FeedbackInsightsPipeline:
    return FeedbackInsightsPipeline()


def highlight_entities(text: str, entities: list[dict]) -> str:
    """Render safe HTML with entity spans highlighted by label."""
    pieces: list[str] = []
    cursor = 0
    for entity in sorted(entities, key=lambda item: item["start"]):
        start, end = int(entity["start"]), int(entity["end"])
        if start < cursor:
            continue
        pieces.append(html.escape(text[cursor:start]))
        label = html.escape(entity["label"])
        value = html.escape(text[start:end])
        pieces.append(f'<mark class="entity entity-{label.lower()}">{value}<small>{label}</small></mark>')
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def parse_upload(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower().rsplit(".", 1)[-1]
    temp_path = f"{uploaded_file.name}"
    with open(temp_path, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return load_feedback(temp_path)


def render_batch() -> None:
    st.subheader("Batch Processor")
    st.write("Upload a CSV, TSV, JSON, or TXT file. A `text` column is preferred; common aliases are detected automatically.")
    uploaded = st.file_uploader("Feedback file", type=["csv", "tsv", "json", "txt", "text"], key="batch_upload")
    if uploaded is None:
        st.info("Use the included sample file to test the workflow: data/sample/customer_feedback.csv")
        return
    try:
        frame = parse_upload(uploaded)
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        return
    st.write(f"Loaded **{len(frame):,}** feedback records.")
    if st.button("Run batch analysis", type="primary"):
        with st.spinner("Running batched transformer inference and extraction…"):
            enriched = get_pipeline().analyze_texts(frame["text"].tolist())
        st.session_state["batch_results"] = enriched
    results = st.session_state.get("batch_results")
    if results is None:
        return
    left, middle, right = st.columns(3)
    left.metric("Records analyzed", f"{len(results):,}")
    middle.metric("Positive", f"{(results.sentiment == 'positive').sum():,}")
    right.metric("Negative", f"{(results.sentiment == 'negative').sum():,}")
    chart_col, topics_col = st.columns(2)
    with chart_col:
        counts = results["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
        fig = px.pie(counts, names="sentiment", values="count", hole=0.55, title="Sentiment distribution", color="sentiment", color_discrete_map={"positive": "#22c55e", "negative": "#ef4444"})
        st.plotly_chart(fig, use_container_width=True)
    with topics_col:
        keywords = Counter(keyword for values in results["keywords"] for keyword in values).most_common(12)
        topic_frame = pd.DataFrame(keywords, columns=["topic", "mentions"])
        fig = px.bar(topic_frame.sort_values("mentions"), x="mentions", y="topic", orientation="h", title="Top extracted topics", color="mentions", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    entities = Counter(entity for values in results["entity_text"] for entity in values.split(", ") if entity)
    if entities:
        st.plotly_chart(px.bar(pd.DataFrame(entities.most_common(12), columns=["entity", "mentions"]), x="entity", y="mentions", title="Top entities"), use_container_width=True)
    st.dataframe(results[["text", "sentiment", "sentiment_confidence", "entity_text", "keywords"]], use_container_width=True, hide_index=True)
    st.download_button("Download enriched CSV", results.to_csv(index=False), "enriched_feedback.csv", "text/csv")


def render_playground() -> None:
    st.subheader("Live Playground")
    text = st.text_area("Paste a customer review or support ticket", "The mobile app crashes when I try to pay, but support resolved my issue quickly.", height=140)
    if st.button("Analyze snippet", type="primary") and text.strip():
        with st.spinner("Analyzing snippet…"):
            result = get_pipeline().analyze_texts([text]).iloc[0]
        sentiment_col, confidence_col = st.columns(2)
        sentiment_col.metric("Sentiment", str(result["sentiment"]).title())
        confidence_col.metric("Confidence", f"{result['sentiment_confidence']:.1%}")
        st.markdown("**Entity highlights**")
        st.markdown(f'<div class="highlight-box">{highlight_entities(text, result["entities"])}</div>', unsafe_allow_html=True)
        st.caption("Entity labels are shown inside the highlighted spans.")
        st.json({"sentiment_probabilities": result["sentiment_probabilities"], "keywords": result["keywords"], "entities": result["entities"]})


st.markdown("""
<style>
.highlight-box { padding: 1rem; border-radius: .6rem; background: #f8fafc; line-height: 2.2; border: 1px solid #e2e8f0; }
mark.entity { padding: .15rem .35rem; border-radius: .3rem; background: #bfdbfe; }
mark.entity small { margin-left: .35rem; font-size: .65rem; color: #1e3a8a; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

batch_tab, playground_tab = st.tabs(["📊 Batch Processor", "🧪 Live Playground"])
with batch_tab:
    render_batch()
with playground_tab:
    render_playground()

