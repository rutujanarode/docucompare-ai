# 🧠 DocuCompare AI

> **Scientifically compare how ML, Deep Learning & Transformers understand documents differently.**

![Python](https://img.shields.io/badge/Python-3.10+-blue) 
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Models](https://img.shields.io/badge/Models-TF--IDF%20%7C%20Word2Vec%20%7C%20BERT-green)

---

## 🔷 What This Does

Upload any PDF or text document → ask questions → see **3 AI brains** answer simultaneously:

| Brain | Technology | Type |
|-------|-----------|------|
| 🔴 Brain 1 | TF-IDF + Cosine Similarity | Classical ML |
| 🟡 Brain 2 | Word2Vec Embeddings | Deep Learning |
| 🟢 Brain 3 | BERT (MiniLM) Sentence Transformers | Transformer |

Then compare: accuracy · speed · semantic understanding · retrieved evidence.

---

## ⚡ Quickstart (3 options)

### Option 1 — Google Colab (Recommended, FREE GPU)
1. Open `DocuCompare_Colab.ipynb` in Google Colab
2. Run cells top-to-bottom
3. Click the ngrok URL that appears

### Option 2 — Run Locally
```bash
# 1. Navigate to project folder
cd docucompare

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### Option 3 — VS Code / PyCharm
Open the folder, open terminal, run `pip install -r requirements.txt` then `streamlit run app.py`.

---

## 📁 Project Structure

```
docucompare/
├── app.py                        ← Main Streamlit UI
├── requirements.txt              ← All dependencies
├── DocuCompare_Colab.ipynb       ← Colab launcher notebook
│
├── models/
│   ├── ml_model.py               ← TF-IDF (Classical ML)
│   ├── dl_model.py               ← Word2Vec (Deep Learning)
│   └── transformer_model.py      ← BERT (Transformer)
│
└── utils/
    ├── document_processor.py     ← PDF extraction + chunking
    └── evaluator.py              ← Metrics + report generation
```

---

## 🧠 Architecture

```
        📄 Document (PDF / TXT)
               ↓
        Text Extraction (PyMuPDF)
               ↓
        Chunking (overlapping windows)
               ↓
    ┌──────────┼──────────┐
    ↓          ↓          ↓
 TF-IDF    Word2Vec     BERT
(sklearn)  (gensim)  (sentence-transformers)
    ↓          ↓          ↓
Sparse     Dense      Contextual
vectors    vectors    embeddings
    ↓          ↓          ↓
    └──────────┼──────────┘
               ↓
        Cosine Similarity Ranking
               ↓
        Top-K Chunk Retrieval
               ↓
        Answer + Comparison Dashboard
```


---

## 🔬 Key Concepts to Explain to Evaluators

### TF-IDF (ML)
- Converts text to sparse vectors based on word frequency
- Matches by exact keywords — "car" ≠ "automobile"
- O(n) lookup, extremely fast, low memory

### Word2Vec (DL)
- Neural network trained on skip-gram objective
- "car" and "automobile" land near each other in 100D space
- Loses word order when averaging; no cross-word attention

### BERT (Transformer)
- 12-layer Transformer with self-attention
- Reads the ENTIRE sentence bidirectionally before embedding it
- "I didn't like it" and "I loved it" get very different vectors
- MiniLM variant: 33M parameters, 5x faster than BERT-base

---

## 🎓 Academic Contribution

This project is a **comparative empirical study** showing that:
1. Semantic similarity (Word2Vec) outperforms keyword matching (TF-IDF) on paraphrase queries
2. Contextual embeddings (BERT) outperform semantic averaging (Word2Vec) on complex/nuanced questions
3. There is a clear accuracy-vs-speed trade-off across the three paradigms

---

## 🛠️ Dependencies

| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `scikit-learn` | TF-IDF vectorizer + cosine similarity |
| `gensim` | Word2Vec training |
| `sentence-transformers` | BERT sentence encoding |
| `PyMuPDF` | PDF text extraction |
| `pandas` / `matplotlib` | Analytics & charts |

---

*Built with ❤️ for academic evaluation & resume-level demonstration.*
