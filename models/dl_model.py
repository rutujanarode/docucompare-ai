"""
models/dl_model.py
Word2Vec based retrieval — Deep Learning approach.

Root cause of 0.998 bug: plain average of word vectors on a small
corpus makes all chunks nearly identical in direction (cosine ~ 1).

Fix: TF-IDF weighted word vectors — rare/important words in a chunk
get higher weight, so chunks become meaningfully different from each other.
"""

from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class Word2VecModel:

    def __init__(self, vector_size: int = 150, window: int = 5, min_count: int = 1):
        self.vector_size  = vector_size
        self.window       = window
        self.min_count    = min_count
        self.model        = None
        self.tfidf        = None          # used to weight word vectors
        self.vocab_idf    = {}            # word -> idf weight
        self.chunks: List[str] = []
        self.chunk_vecs: np.ndarray = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

    def _tfidf_weighted_embed(self, text: str) -> np.ndarray:
        """
        Embed text as a TF-IDF weighted sum of word vectors.
        Words with higher IDF (more unique/rare) contribute more to the vector.
        This makes chunk vectors meaningfully different from each other.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.vector_size)

        # Count term frequencies in this text
        from collections import Counter
        tf = Counter(tokens)
        total = sum(tf.values())

        weighted_vecs = []
        weights = []
        for tok, count in tf.items():
            if tok not in self.model.wv:
                continue
            term_freq = count / total
            idf = self.vocab_idf.get(tok, 1.0)
            weight = term_freq * idf
            weighted_vecs.append(self.model.wv[tok] * weight)
            weights.append(weight)

        if not weighted_vecs:
            return np.zeros(self.vector_size)

        # Weighted average
        vec = np.sum(weighted_vecs, axis=0) / (sum(weights) + 1e-8)
        return vec

    def fit(self, chunks: List[str]) -> None:
        from gensim.models import Word2Vec

        self.chunks   = chunks
        sentences     = [self._tokenize(c) for c in chunks]

        # Train Word2Vec with more epochs for better embeddings on small corpus
        self.model = Word2Vec(
            sentences   = sentences,
            vector_size = self.vector_size,
            window      = self.window,
            min_count   = self.min_count,
            workers     = 2,
            sg          = 1,
            epochs      = 40,
            negative    = 10,
        )

        # Fit TF-IDF separately to get IDF weights for each word
        self.tfidf = TfidfVectorizer(min_df=1, max_df=0.95)
        self.tfidf.fit(chunks)
        feature_names = self.tfidf.get_feature_names_out()
        idf_values    = self.tfidf.idf_
        self.vocab_idf = dict(zip(feature_names, idf_values))

        # Pre-compute weighted chunk vectors
        self.chunk_vecs = np.vstack([
            self._tfidf_weighted_embed(c) for c in chunks
        ])

    def retrieve(self, query: str, top_k: int = 3) -> List[tuple]:
        if self.chunk_vecs is None:
            return []

        q_vec = self._tfidf_weighted_embed(query)
        if np.linalg.norm(q_vec) < 1e-8:
            return []

        q_vec = q_vec.reshape(1, -1)
        chunk_norms = np.linalg.norm(self.chunk_vecs, axis=1)
        valid_mask  = chunk_norms > 1e-8

        sims = np.zeros(len(self.chunks))
        if valid_mask.any():
            valid_sims = cosine_similarity(q_vec, self.chunk_vecs[valid_mask]).flatten()
            sims[valid_mask] = valid_sims

        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def retrieve_and_answer(self, query: str, top_k: int = 3) -> Dict:
        results = self.retrieve(query, top_k)
        if not results:
            return {"answer": "No relevant content found.", "chunks": [], "scores": [], "top_score": 0.0}

        chunks, scores = zip(*results)
        answer = self._extract_answer(query, list(chunks))

        return {
            "answer":    answer,
            "chunks":    list(chunks),
            "scores":    list(scores),
            "top_score": scores[0],
            "model":     "Word2Vec",
        }

    def _extract_answer(self, query: str, chunks: List[str]) -> str:
        stopwords = {
            "what", "is", "how", "does", "the", "a", "an", "of", "in", "to",
            "are", "was", "were", "be", "been", "have", "has", "do", "did",
            "and", "or", "but", "for", "with", "this", "that", "it", "its",
            "used", "use", "using", "can", "which", "such", "also", "more"
        }
        q_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())) - stopwords

        sentences = []
        for chunk in chunks[:2]:
            for sent in re.split(r'(?<=[.!?])\s+', chunk):
                sent = sent.strip()
                if len(sent) < 35 or re.match(r'^[A-Z][a-zA-Z\s]{0,25}$', sent):
                    continue
                sentences.append(sent)

        if not sentences:
            return chunks[0][:400]

        scored = []
        for sent in sentences:
            s_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower()))
            overlap = len(q_terms & s_terms)
            scored.append((overlap, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [s for sc, s in scored if sc > 0][:3]

        if not selected:
            parts    = re.split(r'(?<=[.!?])\s+', chunks[0])
            selected = [p.strip() for p in parts[:2] if len(p.strip()) > 35]

        return " ".join(selected)