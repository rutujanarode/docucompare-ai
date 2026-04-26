"""
DocuCompare AI — Compare ML, DL & Transformer document understanding
"""

import streamlit as st
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from io import BytesIO

from utils.document_processor import extract_text_from_pdf, chunk_text
from models.ml_model import TFIDFModel
from models.dl_model import Word2VecModel
from models.transformer_model import BERTModel
from utils.evaluator import evaluate_models, generate_comparison_report

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuCompare AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0a0a0f; color: #e8e8f0; }

.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%); }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.model-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    transition: all 0.3s ease;
}
.model-card:hover { border-color: rgba(99,179,237,0.4); background: rgba(99,179,237,0.05); }

.ml-badge   { background: linear-gradient(135deg, #f6546a, #c0392b); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-family: 'Space Mono', monospace; font-weight: 700; }
.dl-badge   { background: linear-gradient(135deg, #f7971e, #f3ca20); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-family: 'Space Mono', monospace; font-weight: 700; color: #000; }
.bert-badge { background: linear-gradient(135deg, #11998e, #38ef7d); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-family: 'Space Mono', monospace; font-weight: 700; color: #000; }

.answer-box {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 14px;
    line-height: 1.7;
}
.answer-ml   { border-color: #f6546a; }
.answer-dl   { border-color: #f7971e; }
.answer-bert { border-color: #38ef7d; }

.metric-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    margin: 4px;
}

.score-high { background: rgba(56,239,125,0.15); border: 1px solid #38ef7d; color: #38ef7d; }
.score-mid  { background: rgba(247,151,30,0.15);  border: 1px solid #f7971e; color: #f7971e; }
.score-low  { background: rgba(246,84,106,0.15);  border: 1px solid #f6546a; color: #f6546a; }

.header-glow {
    text-align: center;
    padding: 40px 0 20px;
}
.header-glow h1 {
    font-size: 2.8rem !important;
    background: linear-gradient(135deg, #63b3ed, #b794f4, #38ef7d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
}
.subtitle { color: #888; font-size: 1rem; margin-top: -10px; }

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important; }

.chunk-count { 
    color: #38ef7d; 
    font-family: 'Space Mono', monospace; 
    font-size: 13px;
    background: rgba(56,239,125,0.08);
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid rgba(56,239,125,0.2);
}

div[data-testid="stExpander"] { border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────────
for key in ["chunks", "ml_model", "dl_model", "bert_model", "models_ready", "doc_text"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "history" not in st.session_state:
    st.session_state.history = []


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-glow">
    <h1>🧠 DocuCompare AI</h1>
    <p class="subtitle">Scientifically comparing ML · Deep Learning · Transformers on your documents</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    chunk_size   = st.slider("Chunk Size (words)",    50, 500, 150, 50)
    chunk_overlap = st.slider("Chunk Overlap (words)", 0,  100,  30, 10)
    top_k        = st.slider("Top-K Chunks Retrieved", 1, 10, 3)

    st.markdown("---")
    st.markdown("### 🧪 Models")
    use_ml   = st.checkbox("TF-IDF (ML)",         value=True)
    use_dl   = st.checkbox("Word2Vec (DL)",        value=True)
    use_bert = st.checkbox("BERT (Transformer)",   value=True)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **DocuCompare AI** runs the same question through 3 AI brains and shows you *exactly* how differently they think.
    
    - 🔴 **TF-IDF** — keyword frequency
    - 🟡 **Word2Vec** — semantic similarity  
    - 🟢 **BERT** — deep contextual understanding
    """)


# ─── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload & Index", "❓ Ask & Compare", "📊 Analytics", "📋 Report"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Document Upload
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload Your Document")

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Drop a PDF or TXT file here",
            type=["pdf", "txt"],
            help="Max ~20 MB"
        )

    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📚 Use Sample Document"):
            sample_text = """
Machine Learning is a subset of artificial intelligence that enables systems to learn 
from data and improve performance without being explicitly programmed. It uses statistical 
techniques to give computers the ability to learn from experience.

Deep Learning is a branch of machine learning based on artificial neural networks with 
multiple layers. These deep neural networks can automatically learn representations of data 
with multiple levels of abstraction. Deep learning has revolutionized computer vision, 
natural language processing, and speech recognition.

Transformers are a type of deep learning model introduced in the paper "Attention Is All 
You Need" by Vaswani et al. in 2017. They rely on self-attention mechanisms rather than 
recurrence, making them highly parallelizable and effective for sequential data.

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language 
model developed by Google. It is trained on large text corpora using masked language 
modeling and next sentence prediction. BERT captures bidirectional context — it looks at 
the full sentence simultaneously, unlike earlier models that processed text left-to-right.

Natural Language Processing (NLP) is the field of AI concerned with the interaction 
between computers and human language. Tasks include text classification, named entity 
recognition, machine translation, question answering, and text summarization.

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used in 
information retrieval. It reflects how important a word is to a document in a collection. 
It is calculated by multiplying the term frequency (how often a word appears in a document) 
by the inverse document frequency (how rare the word is across all documents).

Word2Vec is a neural network-based technique that learns word embeddings from large text 
corpora. Developed by Google researchers in 2013, it represents words as dense vectors in 
a high-dimensional space, where semantically similar words are close to each other.

Overfitting occurs when a machine learning model learns the training data too well, 
including its noise and outliers, resulting in poor generalization to new data. 
Regularization techniques like dropout, L1/L2 regularization, and early stopping help 
prevent overfitting.

Gradient descent is an optimization algorithm used to minimize the loss function in 
machine learning models. It iteratively adjusts model parameters in the direction of the 
negative gradient. Variants include batch gradient descent, stochastic gradient descent 
(SGD), and mini-batch gradient descent.

Convolutional Neural Networks (CNNs) are deep learning architectures primarily used for 
image recognition tasks. They use convolutional layers to automatically learn spatial 
hierarchies of features from input images. CNNs have achieved remarkable success in object 
detection, face recognition, and medical image analysis.
            """
            st.session_state.doc_text = sample_text.strip()
            st.success("✅ Sample document loaded!")

    # ── Process document ──────────────────────────────────────────────────────
    raw_text = None
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            raw_text = extract_text_from_pdf(uploaded_file)
        else:
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.session_state.doc_text = raw_text

    if st.session_state.doc_text:
        with st.expander("👁️ Preview Document Text", expanded=False):
            st.text_area("", st.session_state.doc_text[:3000] + ("..." if len(st.session_state.doc_text) > 3000 else ""), height=200)

        st.markdown("---")
        st.markdown("### Index Document into All 3 Models")

        if st.button("🚀 Build Index (All Models)", use_container_width=True):
            chunks = chunk_text(st.session_state.doc_text, chunk_size=chunk_size, overlap=chunk_overlap)
            st.session_state.chunks = chunks

            prog = st.progress(0)
            status = st.empty()

            if use_ml:
                status.markdown("⚙️ Training **TF-IDF** model...")
                ml = TFIDFModel()
                ml.fit(chunks)
                st.session_state.ml_model = ml
                prog.progress(33)

            if use_dl:
                status.markdown("⚙️ Training **Word2Vec** model...")
                dl = Word2VecModel()
                dl.fit(chunks)
                st.session_state.dl_model = dl
                prog.progress(66)

            if use_bert:
                status.markdown("⚙️ Loading **BERT** embeddings (this may take ~30s on first run)...")
                bert = BERTModel()
                bert.fit(chunks)
                st.session_state.bert_model = bert
                prog.progress(100)

            st.session_state.models_ready = True
            status.empty()
            prog.empty()

            st.success(f"✅ Indexed **{len(chunks)} chunks** across all models. Head to the **Ask & Compare** tab!")
            st.markdown(f'<div class="chunk-count">📦 {len(chunks)} text chunks indexed · chunk_size={chunk_size} · overlap={chunk_overlap}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Ask & Compare
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.models_ready:
        st.info("👆 Please upload a document and build the index in the **Upload & Index** tab first.")
    else:
        st.markdown("### Ask a Question")

        # Suggested questions
        st.markdown("**💡 Try these:**")
        q_cols = st.columns(3)
        suggestions = [
            "What is machine learning?",
            "Explain deep learning",
            "How does BERT work?",
            "What is TF-IDF?",
            "Explain Word2Vec embeddings",
            "What causes overfitting?",
        ]
        for i, sug in enumerate(suggestions):
            if q_cols[i % 3].button(sug, key=f"sug_{i}"):
                st.session_state["prefill_q"] = sug

        question = st.text_input(
            "Your question:",
            value=st.session_state.get("prefill_q", ""),
            placeholder="e.g. What is machine learning?",
        )

        if st.button("🔍 Run Comparison", use_container_width=True) and question.strip():
            results = {}
            times   = {}

            with st.spinner("Running through all models..."):
                if use_ml and st.session_state.ml_model:
                    t0 = time.time()
                    results["ML (TF-IDF)"] = st.session_state.ml_model.retrieve_and_answer(question, top_k=top_k)
                    times["ML (TF-IDF)"]   = round((time.time() - t0) * 1000, 1)

                if use_dl and st.session_state.dl_model:
                    t0 = time.time()
                    results["DL (Word2Vec)"] = st.session_state.dl_model.retrieve_and_answer(question, top_k=top_k)
                    times["DL (Word2Vec)"]   = round((time.time() - t0) * 1000, 1)

                if use_bert and st.session_state.bert_model:
                    t0 = time.time()
                    results["Transformer (BERT)"] = st.session_state.bert_model.retrieve_and_answer(question, top_k=top_k)
                    times["Transformer (BERT)"]   = round((time.time() - t0) * 1000, 1)

            # Save to history
            st.session_state.history.append({
                "question": question,
                "results": results,
                "times": times,
            })

            # ── Display answers ────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🤖 Model Answers")

            model_styles = {
                "ML (TF-IDF)":         ("ml",   "🔴", "#f6546a"),
                "DL (Word2Vec)":        ("dl",   "🟡", "#f7971e"),
                "Transformer (BERT)":   ("bert", "🟢", "#38ef7d"),
            }

            for model_name, data in results.items():
                style, emoji, color = model_styles[model_name]
                with st.container():
                    st.markdown(f"""
                    <div class="model-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <span class="{style}-badge">{emoji} {model_name}</span>
                            <span style="color:#888; font-family:'Space Mono',monospace; font-size:12px;">
                                ⏱ {times[model_name]}ms &nbsp;|&nbsp; 🎯 Score: {data['top_score']:.3f}
                            </span>
                        </div>
                        <div class="answer-box answer-{style}">
                            {data['answer']}
                        </div>
                        <details style="margin-top:8px;">
                            <summary style="color:#888; font-size:12px; cursor:pointer;">📎 Retrieved chunks</summary>
                            <div style="margin-top:8px; font-size:12px; color:#aaa; font-family:'Space Mono',monospace;">
                                {'<hr style="border-color:rgba(255,255,255,0.05)">'.join(
                                    f'<b>Chunk {i+1}</b> (score={s:.3f})<br>{c[:200]}...'
                                    for i, (c, s) in enumerate(zip(data["chunks"], data["scores"]))
                                )}
                            </div>
                        </details>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Side-by-side metrics ───────────────────────────────────────
            from utils.evaluator import compute_metrics
            st.markdown("### Quick Metrics")
            st.caption("Relevance Score = fraction of your query keywords found in the answer (model-independent, fair comparison). Raw Score = cosine similarity in each model's own vector space (not comparable across models).")
            m_cols = st.columns(len(results))
            for col, (model_name, data) in zip(m_cols, results.items()):
                m = compute_metrics(question, data)
                rel   = m["relevance_score"]
                raw   = m["raw_retrieval_score"]
                qual  = m["quality_score"]
                cls   = "score-high" if rel >= 0.6 else ("score-mid" if rel >= 0.3 else "score-low")
                col.markdown(f"""
                <div style="text-align:center; padding:16px; background:rgba(255,255,255,0.03); border-radius:10px;">
                    <div style="font-size:12px; color:#888; margin-bottom:10px; font-weight:600;">{model_name}</div>
                    <div style="font-size:11px; color:#aaa; margin-bottom:2px;">Relevance Score</div>
                    <div class="metric-pill {cls}" style="margin-bottom:6px;">{ rel:.2f}</div><br>
                    <div style="font-size:11px; color:#aaa; margin-bottom:2px;">Quality Score</div>
                    <div class="metric-pill {cls}" style="margin-bottom:6px;">{qual:.2f}</div><br>
                    <div style="font-size:11px; color:#aaa; margin-bottom:2px;">Raw Retrieval</div>
                    <div class="metric-pill" style="background:rgba(150,150,150,0.1); border:1px solid #666; color:#aaa; margin-bottom:6px;">{raw:.3f}</div><br>
                    <div class="metric-pill" style="background:rgba(183,148,244,0.1); border:1px solid #b794f4; color:#b794f4;">
                        {times[model_name]}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.history:
        st.info("Ask some questions first — analytics will appear here.")
    else:
        from utils.evaluator import compute_metrics
        st.markdown("### Comparative Analytics")
        st.caption("All charts use Relevance Score (model-independent) for fair comparison. Raw retrieval scores are shown separately as they operate in different vector spaces and cannot be compared across models.")

        history = st.session_state.history

        # Build dataframe with proper metrics
        rows = []
        for h in history:
            for model, data in h["results"].items():
                m = compute_metrics(h["question"], data)
                rows.append({
                    "Question":        h["question"][:35] + "...",
                    "Model":           model,
                    "Relevance":       m["relevance_score"],
                    "Quality":         m["quality_score"],
                    "Raw Score":       m["raw_retrieval_score"],
                    "Time (ms)":       h["times"][model],
                })
        df = pd.DataFrame(rows)

        model_colors = {
            "ML (TF-IDF)":        "#e07b6a",
            "DL (Word2Vec)":       "#d4a84b",
            "Transformer (BERT)":  "#5a9e7a",
        }

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Relevance Score by Model")
            st.caption("Fraction of query keywords found in the answer — fair across all models")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#faf9f7")
            ax.set_facecolor("#faf9f7")
            avg_rel = df.groupby("Model")["Relevance"].mean()
            bars = ax.bar(
                [m.split("(")[0].strip() for m in avg_rel.index],
                avg_rel.values,
                color=[model_colors.get(m, "#888") for m in avg_rel.index],
                edgecolor="none", alpha=0.85, width=0.45
            )
            ax.set_ylabel("Avg Relevance Score", color="#555", fontsize=11)
            ax.set_ylim(0, 1.1)
            ax.tick_params(colors="#555")
            ax.spines[:].set_visible(False)
            ax.yaxis.grid(True, color="#ddd", linestyle="--", alpha=0.7)
            ax.set_axisbelow(True)
            for bar, val in zip(bars, avg_rel.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{val:.2f}", ha="center", color="#333", fontsize=10, fontweight="bold")
            st.pyplot(fig)

        with col_b:
            st.markdown("#### Response Time")
            st.caption("Lower is better — shows computational cost of each approach")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor("#faf9f7")
            ax2.set_facecolor("#faf9f7")
            avg_times = df.groupby("Model")["Time (ms)"].mean()
            ax2.barh(
                [m.split("(")[0].strip() for m in avg_times.index],
                avg_times.values,
                color=[model_colors.get(m, "#888") for m in avg_times.index],
                edgecolor="none", alpha=0.85, height=0.4
            )
            ax2.set_xlabel("Avg Response Time (ms)", color="#555", fontsize=11)
            ax2.tick_params(colors="#555")
            ax2.spines[:].set_visible(False)
            ax2.xaxis.grid(True, color="#ddd", linestyle="--", alpha=0.7)
            ax2.set_axisbelow(True)
            for i, val in enumerate(avg_times.values):
                ax2.text(val + 0.5, i, f"{val:.1f}ms", va="center", color="#333", fontsize=10)
            st.pyplot(fig2)

        # Quality score bar chart
        st.markdown("#### Overall Quality Score per Model")
        st.caption("Quality = 60% Relevance + 30% Raw Retrieval + 10% Answer Precision")
        fig4, ax4 = plt.subplots(figsize=(10, 3.5))
        fig4.patch.set_facecolor("#faf9f7")
        ax4.set_facecolor("#faf9f7")
        avg_qual = df.groupby("Model")["Quality"].mean()
        bars4 = ax4.bar(
            [m for m in avg_qual.index],
            avg_qual.values,
            color=[model_colors.get(m, "#888") for m in avg_qual.index],
            edgecolor="none", alpha=0.85, width=0.35
        )
        ax4.set_ylabel("Avg Quality Score", color="#555", fontsize=11)
        ax4.set_ylim(0, 1.1)
        ax4.tick_params(colors="#555")
        ax4.spines[:].set_visible(False)
        ax4.yaxis.grid(True, color="#ddd", linestyle="--", alpha=0.7)
        ax4.set_axisbelow(True)
        for bar, val in zip(bars4, avg_qual.values):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", color="#333", fontsize=10, fontweight="bold")
        st.pyplot(fig4)

        # Relevance score table — fair comparison
        st.markdown("#### Per-Question Relevance Score Table")
        st.caption("Green = high relevance, Red = low relevance")
        pivot = df.pivot_table(index="Question", columns="Model", values="Relevance", aggfunc="first").round(2)
        st.dataframe(pivot.style.background_gradient(cmap="RdYlGn", axis=None, vmin=0, vmax=1), use_container_width=True)

        # Speed vs Relevance scatter
        st.markdown("#### Speed vs Relevance Trade-off")
        st.caption("Ideal model: top-right (fast + relevant). Shows accuracy-cost trade-off.")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        fig3.patch.set_facecolor("#faf9f7")
        ax3.set_facecolor("#faf9f7")
        for model in df["Model"].unique():
            sub = df[df["Model"] == model]
            ax3.scatter(sub["Time (ms)"], sub["Relevance"],
                        color=model_colors.get(model, "#888"),
                        label=model, s=90, alpha=0.85, edgecolors="white", linewidths=0.8)
        ax3.set_xlabel("Response Time (ms)", color="#555")
        ax3.set_ylabel("Relevance Score", color="#555")
        ax3.set_ylim(-0.05, 1.1)
        ax3.tick_params(colors="#555")
        ax3.spines[:].set_visible(False)
        ax3.legend(facecolor="#fff", labelcolor="#333", framealpha=0.9, edgecolor="#ddd")
        ax3.grid(color="#ddd", linestyle="--", alpha=0.6)
        st.pyplot(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Report
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📋 Comparison Report")

    if not st.session_state.history:
        st.info("Ask some questions first — a report will be generated here.")
    else:
        report = generate_comparison_report(st.session_state.history)
        st.markdown(report)

        st.download_button(
            "⬇️ Download Report (Markdown)",
            data=report,
            file_name="docucompare_report.md",
            mime="text/markdown",
        )