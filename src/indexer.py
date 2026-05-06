import json
import logging
import math
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Only matches letters and apostrophes e.g keeps words like "it's" together
_TOKEN_RE = re.compile(r"[a-zA-Z']+")

# some words filtered out before indexing to reduce noise
STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "it", "its", "this", "that", "these", "those", "i", "you", "he",
        "she", "we", "they", "me", "him", "her", "us", "them", "my",
        "your", "his", "our", "their", "what", "which", "who", "not",
        "if", "as", "so", "up", "out", "no", "about", "into", "than",
        "then", "when", "where", "how", "all", "each", "more", "also",
    }
)


# Tokenisation helpers
def tokenise(text: str) -> list[str]:
    """Lowercase and tokenise text, filtering out stop words and single characters"""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t.strip("'") for t in tokens if t.strip("'") not in STOP_WORDS and len(t.strip("'")) > 1]



# InvertedIndex class
class InvertedIndex:
    """Inverted index with TF-IDF scoring
    Structure: { term -> { url -> { count, tf_idf } } }
    A dict-of-dicts gives O(1) average lookup for both term and document
    which is much faster than scanning a list for every query
    """
    def __init__(self) -> None:
        self._index: dict[str, dict[str, dict[str, Any]]] = {}
        self._doc_count: int = 0

    # Building
    def add_document(self, url: str, text: str) -> None:
        """Tokenise text and store term frequencies for url
        TF-IDF scores are left at 0.0 until compute_tf_idf() is called
        """
        tokens = tokenise(text)
        if not tokens:
            logger.debug("No tokens extracted from %s — skipping.", url)
            return

        self._doc_count += 1
        total_terms = len(tokens)
        term_freq: dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        # Store count and a placeholder TF
        for term, count in term_freq.items():
            if term not in self._index:
                self._index[term] = {}
            tf = count / total_terms
            self._index[term][url] = {"count": count, "tf": tf, "tf_idf": 0.0}
        logger.debug("Indexed %s (%d unique tokens).", url, len(term_freq))

    def compute_tf_idf(self) -> None:
        """Compute TF-IDF scores across all documents
        Must be called after all documents are added
        Complexity: O(T * D) where T = vocab size, D = docs per term
        TF  = count / total_terms  (normalises for document length)
        IDF = log((N+1) / (df+1)) + 1  (smoothed to avoid division by zero)
        """
        N = self._doc_count
        if N == 0:
            return

        for term, docs in self._index.items():
            df = len(docs)
            idf = math.log((N + 1) / (df + 1)) + 1 #avoids division-by-zero if df == N
            for url, stats in docs.items():
                stats["tf_idf"] = round(stats["tf"] * idf, 6)
                stats.pop("tf", None) # remove raw tf as no longer needed

        logger.info("TF-IDF scores computed for %d terms.", len(self._index))


    # Querying
    def search(
        self,
        query: str,
        operator: str = "OR",
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Search the index and return results ranked by TF-IDF score
        operator="OR"  returns pages containing any query term
        operator="AND" returns only pages containing all query terms
        """
        query = query.strip()
        if not query:
            return []

        terms = tokenise(query)
        if not terms:
            return []

        operator = operator.upper()
        if operator not in ("AND", "OR"):
            raise ValueError(f"operator must be 'AND' or 'OR', got {operator!r}")

        # Build set of matching docs for each term
        term_doc_sets: list[set[str]] = []
        for term in terms:
            if term in self._index:
                term_doc_sets.append(set(self._index[term].keys()))
            else:
                term_doc_sets.append(set())

        # Determine which URLs to score
        if operator == "AND":
            if not term_doc_sets:
                return []
            candidates = set.intersection(*term_doc_sets)
        else:  # OR
            if not term_doc_sets:
                return []
            candidates = set.union(*term_doc_sets)

        results: list[dict[str, Any]] = []
        for url in candidates:
            score = 0.0
            term_counts: dict[str, int] = {}
            for term in terms:
                if term in self._index and url in self._index[term]:
                    entry = self._index[term][url]
                    score += entry["tf_idf"]
                    term_counts[term] = entry["count"]
            results.append({"url": url, "score": round(score, 6), "term_counts": term_counts})

        # sorted by descending TF-IDF score
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_n]

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about the index"""

        total_terms = len(self._index)
        total_docs = self._doc_count
        avg_docs = (
            sum(len(v) for v in self._index.values()) / total_terms
            if total_terms
            else 0
        )
        return {
            "total_terms": total_terms,
            "total_documents": total_docs,
            "average_docs_per_term": round(avg_docs, 2),
        }


    # Persistence
    def save(self, path: str | Path) -> None:
        """Save the index to a JSON file
        JSON chosen as its human-readable and portable across Python versions
        """
        path = Path(path)
        payload = {
            "meta": {
                "doc_count": self._doc_count,
                "term_count": len(self._index),
            },
            "index": self._index,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.info("Index saved to %s (%d terms).", path, len(self._index))

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        """Load an index from a JSON file and return a new InvertedIndex instance
        gives FileNotFoundError if the path doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        instance = cls()
        instance._doc_count = payload["meta"]["doc_count"]
        instance._index = payload["index"]
        logger.info(
            "Index loaded from %s (%d terms, %d docs).",
            path,
            len(instance._index),
            instance._doc_count,
        )
        return instance



    # Display
    def print_index(self, max_terms: int = 50) -> None:
        """Print a summary of the index to stdout, capped at max_terms entries"""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"  INVERTED INDEX SUMMARY")
        print(f"{'='*60}")
        print(f"  Total unique terms : {stats['total_terms']:,}")
        print(f"  Total documents    : {stats['total_documents']:,}")
        print(f"  Avg docs per term  : {stats['average_docs_per_term']}")
        print(f"{'='*60}\n")

        terms = sorted(self._index.keys())[:max_terms]
        for term in terms:
            docs = self._index[term]
            top_url = max(docs, key=lambda u: docs[u]["tf_idf"])
            top_score = docs[top_url]["tf_idf"]
            print(
                f"  {term:<25} | {len(docs):>3} doc(s) | "
                f"top score: {top_score:.4f} | best: ...{top_url[-40:]}"
            )

        if len(self._index) > max_terms:
            print(f"\n  ... and {len(self._index) - max_terms} more terms.")
        print()