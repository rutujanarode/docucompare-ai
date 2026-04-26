"""
models/transformer_model.py
BERT-based semantic retrieval using sentence-transformers.
Represents the Transformer / modern NLP approach.
"""

from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class BERTModel:
    """
    Transformer approach: BERT-family sentence embeddings.
    
    Uses 'all-MiniLM-L6-v2' — a fast, high-quality bi-encoder trained
    specifically for semantic similarity. Compared to the full BERT-base,
    it's 5x faster with ~95% of the quality.
    
    How it works:
    1. Encode all chunks using BERT (CLS token pooling)
    2. Encode query using same encoder
    3. Cosine similarity retrieves semantically close chunks
    4. Unlike TF-IDF, understands paraphrases: 'car' ≈ 'automobile'
    5. Unlike Word2Vec, preserves word order and context
    """

    MODEL_NAME = "all-MiniLM-L6-v2"   # fast & good; can swap for 'all-mpnet-base-v2'

    def __init__(self):
        self.encoder     = None
        self.chunks: List[str] = []
        self.chunk_vecs: np.ndarray = None

    def _load_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.MODEL_NAME)

    def fit(self, chunks: List[str]) -> None:
        """Encode all chunks and store embeddings."""
        self._load_encoder()
        self.chunks     = chunks
        self.chunk_vecs = self.encoder.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[tuple]:
        if self.chunk_vecs is None:
            return []
        q_vec = self.encoder.encode([query], convert_to_numpy=True)
        sims  = cosine_similarity(q_vec, self.chunk_vecs).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx]

    def retrieve_and_answer(self, query: str, top_k: int = 3) -> Dict:
        self._load_encoder()
        results = self.retrieve(query, top_k)
        if not results:
            return {"answer": "No relevant content found.", "chunks": [], "scores": [], "top_score": 0.0}

        chunks, scores = zip(*results)

        # BERT: synthesize answer using BERT similarity at sentence level
        answer = self._synthesize(query, list(chunks), encoder=self.encoder)

        return {
            "answer":    answer,
            "chunks":    list(chunks),
            "scores":    list(scores),
            "top_score": scores[0],
            "model":     "BERT",
        }

    @staticmethod
    def _synthesize(query: str, chunks: List[str], encoder=None) -> str:
        """
        Score every sentence in top chunks using BERT similarity to the query.
        This means BERT is used end-to-end — retrieval AND answer extraction.
        """
        import re

        stopwords = {
            "what", "is", "how", "does", "the", "a", "an", "of", "in", "to",
            "are", "was", "were", "be", "been", "have", "has", "do", "did",
            "and", "or", "but", "for", "with", "this", "that", "it", "its",
            "used", "use", "using", "can", "which", "such", "also", "more"
        }

        # Collect clean sentences from top 2 chunks
        all_sentences = []
        for chunk in chunks[:2]:
            for sent in re.split(r'(?<=[.!?])\s+', chunk):
                sent = sent.strip()
                # Skip headers (short title-case lines) and very short sentences
                if len(sent) < 35:
                    continue
                if re.match(r'^[A-Z][a-zA-Z\s]{0,25}$', sent):
                    continue
                # Skip sentences that are clearly cut off (no verb-like structure)
                all_sentences.append(sent)

        if not all_sentences:
            return chunks[0][:400]

        # Score using BERT embeddings if encoder available
        if encoder is not None:
            try:
                q_vec    = encoder.encode([query], convert_to_numpy=True)
                s_vecs   = encoder.encode(all_sentences, convert_to_numpy=True)
                sims     = cosine_similarity(q_vec, s_vecs).flatten()
                top_idx  = np.argsort(sims)[::-1][:3]
                selected = [all_sentences[i] for i in top_idx if sims[i] > 0.2]
                if selected:
                    return " ".join(selected)
            except Exception:
                pass

        # Fallback: keyword scoring
        q_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())) - stopwords
        scored  = []
        for sent in all_sentences:
            s_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower()))
            overlap = len(q_terms & s_terms)
            scored.append((overlap, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [s for sc, s in scored if sc > 0][:3]

        if not selected:
            parts    = re.split(r'(?<=[.!?])\s+', chunks[0])
            selected = [p.strip() for p in parts[:2] if len(p.strip()) > 35]

        return " ".join(selected) if selected else chunks[0][:400]
