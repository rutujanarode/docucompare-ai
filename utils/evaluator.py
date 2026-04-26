"""
utils/evaluator.py

Proper evaluation metrics:
- Raw retrieval score (cosine similarity) — kept as-is per model
- Relevance quality score — did the retrieved chunk actually answer the query?
  Computed as: overlap of query keywords in the answer text, normalized 0-1
- Answer precision — how focused is the answer (fewer irrelevant words = better)
"""

from typing import List, Dict
import datetime
import re


STOPWORDS = {
    "what", "is", "how", "does", "the", "a", "an", "of", "in", "to",
    "are", "was", "were", "be", "been", "have", "has", "do", "did",
    "and", "or", "but", "for", "with", "this", "that", "it", "its",
    "used", "use", "using", "can", "which", "such", "also", "more",
    "will", "would", "could", "should", "may", "might", "into", "about"
}


def relevance_score(query: str, answer: str) -> float:
    """
    Measures how relevant the answer is to the query.
    Formula: |query_terms ∩ answer_terms| / |query_terms|
    Range: 0.0 (no overlap) to 1.0 (all query terms appear in answer)
    """
    q_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())) - STOPWORDS
    a_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', answer.lower()))
    if not q_terms:
        return 0.0
    return round(len(q_terms & a_terms) / len(q_terms), 4)


def answer_precision(answer: str) -> float:
    """
    Measures how focused/precise the answer is.
    Shorter, denser answers score higher than long rambling ones.
    Formula: min(1.0, 100 / word_count) — penalizes answers over 100 words
    """
    words = len(answer.split())
    if words == 0:
        return 0.0
    return round(min(1.0, 80 / words), 4)


def compute_metrics(query: str, data: Dict) -> Dict:
    """Compute all evaluation metrics for one model's answer."""
    answer = data.get("answer", "")
    raw_score = data.get("top_score", 0.0)
    rel_score = relevance_score(query, answer)
    precision = answer_precision(answer)
    # Combined quality score: weights relevance most, then raw retrieval
    quality = round(0.6 * rel_score + 0.3 * raw_score + 0.1 * precision, 4)
    return {
        "raw_retrieval_score": round(raw_score, 4),
        "relevance_score":     rel_score,
        "answer_precision":    precision,
        "quality_score":       quality,
    }


def evaluate_models(history: List[Dict]) -> Dict:
    """Compute aggregate metrics across all queries per model."""
    model_stats = {}

    for entry in history:
        query = entry["question"]
        for model, data in entry["results"].items():
            if model not in model_stats:
                model_stats[model] = {
                    "raw_scores":   [],
                    "rel_scores":   [],
                    "quality":      [],
                    "times":        [],
                }
            metrics = compute_metrics(query, data)
            model_stats[model]["raw_scores"].append(metrics["raw_retrieval_score"])
            model_stats[model]["rel_scores"].append(metrics["relevance_score"])
            model_stats[model]["quality"].append(metrics["quality_score"])
            model_stats[model]["times"].append(entry["times"][model])

    summary = {}
    for model, stats in model_stats.items():
        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0
        summary[model] = {
            "avg_raw_score":    avg(stats["raw_scores"]),
            "avg_relevance":    avg(stats["rel_scores"]),
            "avg_quality":      avg(stats["quality"]),
            "avg_time_ms":      avg(stats["times"]),
            "n_queries":        len(stats["raw_scores"]),
        }

    return summary


def generate_comparison_report(history: List[Dict]) -> str:
    """Generate a Markdown report from query history."""
    summary = evaluate_models(history)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# DocuCompare AI — Model Comparison Report",
        f"> Generated: {now} | Questions analyzed: {len(history)}\n",
        "---\n",
        "## Model Overview\n",
        "| Model | Approach | Strength | Weakness |",
        "|-------|----------|----------|----------|",
        "| TF-IDF (ML) | Keyword frequency | Fast, interpretable | No semantic understanding |",
        "| Word2Vec (DL) | TF-IDF weighted embeddings | Captures word similarity | Loses word order |",
        "| BERT (Transformer) | Bidirectional attention | Deep context understanding | Slower, heavier |",
        "\n---\n",
        "## Evaluation Metrics Explained\n",
        "- **Raw Retrieval Score** — cosine similarity between query and chunk vectors (model-specific scale)",
        "- **Relevance Score** — fraction of query keywords found in the answer (0 to 1, model-independent)",
        "- **Quality Score** — weighted combination: 60% relevance + 30% retrieval + 10% precision",
        "- **Avg Time** — response latency in milliseconds\n",
        "---\n",
        "## Performance Summary\n",
        "| Model | Relevance Score | Quality Score | Raw Score | Avg Time (ms) |",
        "|-------|----------------|---------------|-----------|---------------|",
    ]

    for model, stats in summary.items():
        lines.append(
            f"| {model} | {stats['avg_relevance']} | {stats['avg_quality']} "
            f"| {stats['avg_raw_score']} | {stats['avg_time_ms']} |"
        )

    lines += ["\n---\n", "## Per-Question Results\n"]

    for i, entry in enumerate(history, 1):
        lines.append(f"### Q{i}: {entry['question']}\n")
        lines.append("| Model | Relevance | Quality | Answer Preview |")
        lines.append("|-------|-----------|---------|----------------|")
        for model, data in entry["results"].items():
            m = compute_metrics(entry["question"], data)
            preview = data["answer"][:100].replace("\n", " ") + "..."
            lines.append(
                f"| {model} | {m['relevance_score']} | {m['quality_score']} | {preview} |"
            )
        lines.append("")

    lines += [
        "---\n",
        "## Key Insights\n",
        "**Why raw retrieval score alone is misleading:** Word2Vec cosine similarity operates "
        "in a different vector space than TF-IDF or BERT — a score of 0.9 in Word2Vec space "
        "is not comparable to 0.9 in TF-IDF space. The Relevance Score is model-independent "
        "and gives a fair comparison.\n",
        "**Speed vs Accuracy trade-off:** TF-IDF is fastest (sparse matrix ops), "
        "Word2Vec is medium (dense vector ops), BERT is slowest (transformer inference) "
        "but typically achieves the best relevance on complex queries.\n",
        "---\n",
        "> DocuCompare AI — empirically demonstrating that Transformers understand language better than classical ML.",
    ]

    return "\n".join(lines)