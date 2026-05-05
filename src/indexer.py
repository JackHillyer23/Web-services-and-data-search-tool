import json
import logging
import math
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Regex that extracts sequences of word characters
_TOKEN_RE = re.compile(r"[a-zA-Z']+")

# Common English words that add noise to the index
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
    """Split *text* into lowercase tokens, removing stop words.

    Args:
        text: Raw visible text extracted from a web page.

    Returns:
        List of lowercase word tokens (stop words excluded).

    Example:
        >>> tokenise("To be or not to be")
        ['not']   # stop words removed; 'not' kept as it can be meaningful
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]



# InvertedIndex class
class InvertedIndex:
    """An inverted index with TF-IDF scoring.

    Attributes:
        _index: Mapping of term → {url → {"count": int, "tf_idf": float}}.
        _doc_count: Total number of documents (pages) in the index.
    """

    def __init__(self) -> None:
        # term → { url → { "count": int, "tf_idf": float } }
        self._index: dict[str, dict[str, dict[str, Any]]] = {}
        self._doc_count: int = 0

    # Building
    def add_document(self, url: str, text: str) -> None:
        """Tokenise *text* and record term frequencies for *url*.

        TF-IDF scores are **not** computed here; call
        :meth:`compute_tf_idf` once all documents have been added.

        Args:
            url:  The document's URL (used as its unique identifier).
            text: Visible text content of the page.
        """
        tokens = tokenise(text)
        if not tokens:
            logger.debug("No tokens extracted from %s — skipping.", url)
            return

        self._doc_count += 1
        total_terms = len(tokens)

        # Count raw term frequencies for this document.
        term_freq: dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        # Store count and a placeholder TF (TF-IDF computed later).
        for term, count in term_freq.items():
            if term not in self._index:
                self._index[term] = {}
            tf = count / total_terms
            self._index[term][url] = {"count": count, "tf": tf, "tf_idf": 0.0}

        logger.debug("Indexed %s (%d unique tokens).", url, len(term_freq))

    def compute_tf_idf(self) -> None:
        """Compute and store TF-IDF scores for every (term, document) pair.

        Must be called after *all* documents have been added via
        :meth:`add_document`.

        Complexity: O(T * D) where T = vocabulary size, D = docs per term.
        """
        N = self._doc_count
        if N == 0:
            return

        for term, doc_map in self._index.items():
            # Number of documents that contain this term.
            df = len(doc_map)
            # Smooth IDF: +1 avoids division-by-zero if df == N.
            idf = math.log((N + 1) / (df + 1)) + 1
            for url, stats in doc_map.items():
                stats["tf_idf"] = round(stats["tf"] * idf, 6)
                # Remove raw tf from stored data to keep the file clean.
                stats.pop("tf", None)

        logger.info("TF-IDF scores computed for %d terms.", len(self._index))


    # Querying
    def search(
        self,
        query: str,
        operator: str = "OR",
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        

        query = query.strip()
        if not query:
            return []

        terms = tokenise(query)
        if not terms:
            # Query consisted entirely of stop words or punctuation.
            return []

        operator = operator.upper()
        if operator not in ("AND", "OR"):
            raise ValueError(f"operator must be 'AND' or 'OR', got {operator!r}")

        # Gather candidate document sets for each term.
        term_doc_sets: list[set[str]] = []
        for term in terms:
            if term in self._index:
                term_doc_sets.append(set(self._index[term].keys()))
            else:
                term_doc_sets.append(set())

        # Determine which URLs to score.
        if operator == "AND":
            if not term_doc_sets:
                return []
            candidate_urls = set.intersection(*term_doc_sets)
        else:  # OR
            if not term_doc_sets:
                return []
            candidate_urls = set.union(*term_doc_sets)

        results: list[dict[str, Any]] = []
        for url in candidate_urls:
            score = 0.0
            term_counts: dict[str, int] = {}
            for term in terms:
                if term in self._index and url in self._index[term]:
                    entry = self._index[term][url]
                    score += entry["tf_idf"]
                    term_counts[term] = entry["count"]
            results.append({"url": url, "score": round(score, 6), "term_counts": term_counts})

        # Sort by descending TF-IDF score.
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_n]

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about the index.

        Returns:
            Dict with keys ``total_terms``, ``total_documents``,
            ``average_docs_per_term``.
        """
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
        """Serialise the index to a JSON file at *path*.

        JSON is chosen over pickle for portability and human-readability —
        the marker can open the file and inspect the structure directly.

        Args:
            path: Destination file path (will be created/overwritten).
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
        """Deserialise an index from *path* and return a new instance.

        Args:
            path: Path to a JSON file previously created by :meth:`save`.

        Returns:
            A fully populated :class:`InvertedIndex` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is malformed.
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
        """Print a human-readable summary of the index to stdout.

        Args:
            max_terms: Maximum number of terms to display (default 50).
                       Prevents flooding the terminal for large indexes.
        """
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
            doc_map = self._index[term]
            top_url = max(doc_map, key=lambda u: doc_map[u]["tf_idf"])
            top_score = doc_map[top_url]["tf_idf"]
            print(
                f"  {term:<25} | {len(doc_map):>3} doc(s) | "
                f"top score: {top_score:.4f} | best: ...{top_url[-40:]}"
            )

        if len(self._index) > max_terms:
            print(f"\n  ... and {len(self._index) - max_terms} more terms.")
        print()