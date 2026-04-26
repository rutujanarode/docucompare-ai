"""
models/ml_model.py
TF-IDF based retrieval + answer generation.
Represents the Classical ML approach.
"""

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TFIDFModel:
    """
    Classical ML approach: TF-IDF vectorization + cosine similarity retrieval.
    
    How it works:
    1. Each chunk is converted to a sparse TF-IDF vector
    2. The query is vectorized in the same space
    3. Cosine similarity ranks chunks
    4. Top-k chunks are concatenated as the "answer"
    
    Limitation: Purely keyword-based. "automobile" ≠ "car".
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),       # unigrams + bigrams
            max_features=10_000,
            sublinear_tf=True,        # log(1+tf) dampens common words
        )
        self.chunk_vectors = None
        self.chunks: List[str] = []

    def fit(self, chunks: List[str]) -> None:
        """Index all chunks."""
        self.chunks = chunks
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[tuple]:
        """Return (chunk, score) pairs sorted by relevance."""
        if self.chunk_vectors is None:
            return []
        q_vec = self.vectorizer.transform([query])
        sims  = cosine_similarity(q_vec, self.chunk_vectors).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_indices]

    def retrieve_and_answer(self, query: str, top_k: int = 3) -> Dict:
        """Retrieve + format result dict."""
        results = self.retrieve(query, top_k)
        if not results:
            return {"answer": "No relevant content found.", "chunks": [], "scores": [], "top_score": 0.0}

        chunks, scores = zip(*results)
        # Simple answer: concatenate top chunks
        answer = self._build_answer(query, list(chunks), list(scores))

        return {
            "answer":    answer,
            "chunks":    list(chunks),
            "scores":    list(scores),
            "top_score": scores[0],
            "model":     "TF-IDF",
        }

    def _build_answer(self, query: str, chunks: List[str], scores: List[float]) -> str:
        """
        Extract only the most relevant sentences from top chunks.
        Scores each sentence individually by query term overlap.
        """
        import re

        # Extended stopwords for cleaner query terms
        stopwords = {
            "what", "is", "how", "does", "the", "a", "an", "of", "in", "to",
            "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "did", "will", "would", "could", "should", "may", "might",
            "and", "or", "but", "for", "with", "this", "that", "these", "those",
            "it", "its", "by", "on", "at", "from", "as", "into", "about"
        }
        q_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())) - stopwords

        # Collect all sentences from top 2 chunks only
        all_sentences = []
        for chunk in chunks[:2]:
            for sent in re.split(r'(?<=[.!?])\s+', chunk):
                sent = sent.strip()
                # Skip very short or header-like sentences
                if len(sent) < 30 or re.match(r'^[A-Z][a-zA-Z\s]{0,30}$', sent):
                    continue
                all_sentences.append(sent)

        if not all_sentences:
            return chunks[0][:400]

        # Score each sentence by how many query terms it contains
        scored = []
        for sent in all_sentences:
            s_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower()))
            overlap = len(q_terms & s_terms)
            # Penalize very long sentences (likely noise)
            length_penalty = min(1.0, 80 / max(len(sent.split()), 1))
            scored.append((overlap + length_penalty * 0.1, sent))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Pick top sentences that actually contain query terms
        selected = []
        for score, sent in scored:
            if score > 0 and sent not in selected:
                selected.append(sent)
            if len(selected) >= 3:
                break

        # Fallback: just return best chunk's first 2 sentences
        if not selected:
            fallback = re.split(r'(?<=[.!?])\s+', chunks[0])
            selected = [s.strip() for s in fallback[:2] if len(s.strip()) > 30]

        return " ".join(selected)
