"""Executive-facing Streamlit application for customer feedback intelligence."""

from __future__ import annotations

import html
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import FeedbackInsightsPipeline
from src.preprocessing import load_feedback


st.set_page_config(page_title="SignalDesk | Customer Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")


def inject_style() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color:#0b1324; letter-spacing:-.025em; }
    .block-container { padding:2.5rem 4rem 4rem; max-width:1500px; }
    [data-testid="stSidebar"] { background:#0b1324; border-right:0; }
    [data-testid="stSidebar"] * { color:#dbeafe !important; }
    .brand-mark { font-family:'Space Grotesk'; color:#fff; font-size:1.3rem; font-weight:700; letter-spacing:-.04em; }
    .brand-sub { color:#94a3b8; font-size:.72rem; margin-top:.15rem; }
    .hero { background:linear-gradient(115deg,#0b1324 0%,#162b52 68%,#1d4ed8 100%); border-radius:22px; padding:2.2rem 2.5rem; color:white; position:relative; overflow:hidden; margin-bottom:1.8rem; }
    .hero:after { content:''; position:absolute; width:260px; height:260px; right:-80px; top:-120px; border:35px solid rgba(147,197,253,.13); border-radius:50%; }
    .eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.68rem; font-weight:700; color:#93c5fd; }
    .hero h1 { color:#fff !important; font-size:2.25rem; margin:.45rem 0 .5rem; }
    .hero p { color:#cbd5e1; margin:0; max-width:680px; }
    .hero-badge { display:inline-block; margin-top:1.1rem; padding:.35rem .7rem; border:1px solid rgba(191,219,254,.3); border-radius:999px; color:#dbeafe; font-size:.75rem; }
    .section-label { color:#64748b; text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:700; margin:1.5rem 0 .7rem; }
    .kpi { background:#fff; border:1px solid #e2e8f0; border-radius:16px; padding:1.1rem 1.2rem; min-height:112px; box-shadow:0 8px 24px rgba(15,23,42,.045); }
    .kpi-label { color:#64748b; font-size:.75rem; font-weight:600; }
    .kpi-value { color:#0b1324; font-family:'Space Grotesk'; font-size:1.8rem; font-weight:700; margin:.25rem 0; }
    .kpi-note { color:#94a3b8; font-size:.72rem; }
    .insight { background:#f8fafc; border-left:4px solid #2563eb; border-radius:10px; padding:1rem 1.1rem; margin:.4rem 0; color:#334155; font-size:.85rem; }
    .insight strong { color:#0f172a; }
    .highlight-box { padding:1.25rem; border-radius:14px; background:#f8fafc; line-height:2.4; border:1px solid #e2e8f0; font-size:1.02rem; }
    mark.entity { padding:.2rem .42rem; border-radius:.4rem; background:#dbeafe; color:#1e3a8a; }
    mark.entity small { margin-left:.35rem; font-size:.63rem; color:#1d4ed8; font-weight:700; text-transform:uppercase; }
    .legend { color:#64748b; font-size:.75rem; }
    </style>
    """, unsafe_allow_html=True)


inject_style()


@st.cache_resource(show_spinner="Loading NLP models…")
def get_pipeline() -> FeedbackInsightsPipeline:
    return FeedbackInsightsPipeline()


def highlight_entities(text: str, entities: list[dict]) -> str:
    pieces: list[str] = []
    cursor = 0
    for entity in sorted(entities, key=lambda item: item["start"]):
        start, end = int(entity["start"]), int(entity["end"])
        if start < cursor:
            continue
        pieces.extend([html.escape(text[cursor:start]), f'<mark class="entity">{html.escape(text[start:end])}<small>{html.escape(entity["label"])}</small></mark>'])
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def parse_upload(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        path = handle.name
    return load_feedback(path)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="brand-mark">◈ SignalDesk</div><div class="brand-sub">Customer intelligence workspace</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("**Decision support**")
        st.caption("Convert unstructured feedback into prioritized signals for product, support, and operations teams.")
        st.markdown("**Model stack**")
        st.caption("DistilBERT sentiment · spaCy NER · explainable keyword baseline")
        st.divider()
        st.markdown("**Product note**")
        st.caption("Use the scan to move from recurring customer friction to a prioritized conversation with the owning team.")
        st.markdown("<div style='color:#64748b;font-size:.7rem;margin-top:2rem'>Portfolio project · v1.0</div>", unsafe_allow_html=True)


def build_insights(results: pd.DataFrame) -> list[str]:
    negative = int((results.sentiment == "negative").sum())
    total = len(results)
    keywords = Counter(keyword for values in results["keywords"] for keyword in values).most_common(3)
    insights = [f"<strong>{negative / total:.0%} of analyzed feedback is negative.</strong> Prioritize the largest recurring friction signals before optimizing delight drivers."]
    if keywords:
        insights.append(f"<strong>{keywords[0][0].title()} is the leading signal.</strong> Review representative examples and route this theme to the relevant product or operations owner.")
    high_conf = int((results["sentiment_confidence"] >= .9).sum())
    insights.append(f"<strong>{high_conf:,} records have high-confidence predictions.</strong> Use lower-confidence cases as a human-review queue before automating escalation.")
    return insights


def render_batch() -> None:
    st.markdown('<div class="section-label">Portfolio workspace / Batch intelligence</div>', unsafe_allow_html=True)
    st.subheader("Turn feedback volume into a decision surface")
    st.caption("Upload a customer feedback export, then review sentiment, recurring friction, and extractable business signals in one view.")
    uploaded = st.file_uploader("Drop a CSV, TSV, JSON, or TXT file", type=["csv", "tsv", "json", "txt", "text"], label_visibility="collapsed")
    sample = st.button("Load sample & analyze")
    if uploaded is None and not sample:
        st.info("Start with the included sample dataset or upload your own feedback export.")
        return
    try:
        frame = parse_upload(uploaded) if uploaded is not None else load_feedback(Path(__file__).parent / "data/sample/customer_feedback.csv")
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")
        return
    source_label = uploaded.name if uploaded is not None else "customer_feedback.csv"
    st.caption(f"Source: **{source_label}** · {len(frame):,} records ready")
    run_scan = st.button("Run intelligence scan", type="primary", use_container_width=True)
    if sample or run_scan:
        with st.spinner("Running transformer inference and extraction…"):
            st.session_state["batch_results"] = get_pipeline().analyze_texts(frame["text"].tolist())
    results = st.session_state.get("batch_results")
    if results is None:
        return
    negative = int((results.sentiment == "negative").sum())
    positive = int((results.sentiment == "positive").sum())
    high_conf = int((results.sentiment_confidence >= .9).sum())
    st.markdown('<div class="section-label">Executive readout</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, note in [(k1,"Feedback analyzed",f"{len(results):,}","records in this scan"),(k2,"Negative signal",f"{negative / len(results):.0%}",f"{negative:,} records"),(k3,"Positive signal",f"{positive / len(results):.0%}",f"{positive:,} records"),(k4,"High-confidence",f"{high_conf / len(results):.0%}","confidence ≥ 90%")]:
        col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">What the model is telling us</div>', unsafe_allow_html=True)
    for message in build_insights(results):
        st.markdown(f'<div class="insight">{message}</div>', unsafe_allow_html=True)
    chart_col, topic_col = st.columns([1, 1.25])
    with chart_col:
        counts = results["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
        fig = px.pie(counts, names="sentiment", values="count", hole=.68, color="sentiment", color_discrete_map={"positive":"#0f766e","negative":"#dc2626"})
        fig.update_layout(title="Sentiment mix", margin=dict(t=55,b=10,l=10,r=10), legend=dict(orientation="h",y=-.05), paper_bgcolor="rgba(0,0,0,0)")
        fig.update_traces(textinfo="percent", textfont_size=14)
        st.plotly_chart(fig, use_container_width=True)
    with topic_col:
        keywords = Counter(keyword for values in results["keywords"] for keyword in values).most_common(10)
        topic_frame = pd.DataFrame(keywords, columns=["topic", "mentions"]).sort_values("mentions")
        fig = px.bar(topic_frame, x="mentions", y="topic", orientation="h", color="mentions", color_continuous_scale=[[0,"#bfdbfe"],[1,"#2563eb"]])
        fig.update_layout(title="Recurring language signals", margin=dict(t=55,b=10,l=10,r=10), coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    entities = Counter(entity for values in results["entity_text"] for entity in values.split(", ") if entity)
    if entities:
        st.markdown('<div class="section-label">Entity landscape</div>', unsafe_allow_html=True)
        fig = px.bar(pd.DataFrame(entities.most_common(10), columns=["entity","mentions"]), x="entity", y="mentions", color="mentions", color_continuous_scale="Blues")
        fig.update_layout(margin=dict(t=25,b=10,l=10,r=10), coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-label">Enriched feedback table</div>', unsafe_allow_html=True)
    st.dataframe(results[["text","sentiment","sentiment_confidence","entity_text","keywords"]], use_container_width=True, hide_index=True, column_config={"sentiment_confidence":st.column_config.ProgressColumn("Confidence",min_value=0,max_value=1,format="%.0%%")})
    st.download_button("Download enriched CSV", results.to_csv(index=False), "signaldesk_enriched_feedback.csv", "text/csv")


def render_playground() -> None:
    st.markdown('<div class="section-label">Portfolio workspace / Live analysis</div>', unsafe_allow_html=True)
    st.subheader("Inspect one customer voice at a time")
    st.caption("Use this surface to demonstrate how a raw support message becomes a structured, explainable prediction.")
    text = st.text_area("Customer feedback", "The mobile app crashes when I try to pay, but support resolved my issue quickly.", height=145, label_visibility="collapsed")
    if st.button("Analyze customer voice", type="primary") and text.strip():
        with st.spinner("Scoring sentiment and extracting entities…"):
            result = get_pipeline().analyze_texts([text]).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted sentiment", str(result["sentiment"]).title())
        c2.metric("Model confidence", f"{result['sentiment_confidence']:.1%}")
        c3.metric("Extracted entities", len(result["entities"]))
        left, right = st.columns([1.25, .75])
        with left:
            st.markdown('<div class="section-label">Explainable text view</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="highlight-box">{highlight_entities(text, result["entities"])}</div>', unsafe_allow_html=True)
            st.markdown('<div class="legend">Blue tags identify extracted entities and their labels.</div>', unsafe_allow_html=True)
        with right:
            probs = result["sentiment_probabilities"]
            fig = go.Figure(go.Bar(x=list(probs.values()), y=[key.title() for key in probs], orientation="h", marker_color=["#dc2626" if key == "negative" else "#0f766e" for key in probs]))
            fig.update_layout(title="Confidence profile", xaxis_title="Probability", xaxis_range=[0,1], margin=dict(t=55,b=15,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="section-label">Structured extraction</div>', unsafe_allow_html=True)
        st.json({"sentiment_probabilities": result["sentiment_probabilities"], "keywords": result["keywords"], "entities": result["entities"]})


render_sidebar()
st.markdown('<div class="hero"><div class="eyebrow">Unstructured data → business signal</div><h1>Customer feedback, made actionable.</h1><p>SignalDesk converts reviews and support tickets into sentiment, entities, recurring themes, and an evidence trail for the next product decision.</p><span class="hero-badge">● Live NLP workspace · Hugging Face + spaCy + Plotly</span></div>', unsafe_allow_html=True)
batch_tab, playground_tab = st.tabs(["◈ Executive dashboard", "⌁ Live playground"])
with batch_tab:
    render_batch()
with playground_tab:
    render_playground()

