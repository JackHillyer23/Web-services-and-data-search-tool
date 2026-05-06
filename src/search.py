import logging
from typing import Any
from indexer import InvertedIndex
import re

logger = logging.getLogger(__name__)

def parse_query(raw_query: str) -> tuple[str, str]:
    """Detect AND/OR boolean operator in the query string
    Returns a (cleaned_query, operator) tuple. Defaults to OR if no operator is found.
    """
    if " AND " in raw_query.upper():
        operator = "AND"
        cleaned = re.sub(r"\bAND\b", "", raw_query, flags=re.IGNORECASE).strip()
        # Collapse extra spaces left by removal.
        cleaned = " ".join(cleaned.split())
    elif " OR " in raw_query.upper():
        operator = "OR"
        cleaned = re.sub(r"\bOR\b", "", raw_query, flags=re.IGNORECASE).strip()
        cleaned = " ".join(cleaned.split())
    else:
        operator = "OR"
        cleaned = raw_query.strip()

    return cleaned, operator


def format_results(results: list[dict[str, Any]], query: str, operator: str) -> None:
    """Print ranked search results to stdout"""
    print(f"\n{'='*60}")
    print(f"  Results for: '{query}'  [{operator}]")
    print(f"{'='*60}")

    if not results:
        print("  No results found.\n")
        return

    print(f"  Found {len(results)} result(s), ranked by TF-IDF score:\n")
    for rank, result in enumerate(results, start=1):
        url = result["url"]
        score = result["score"]
        term_counts = result["term_counts"]
        counts = ", ".join(
            f"'{t}' ×{c}" for t, c in sorted(term_counts.items())
        )
        print(f"  {rank:>2}. {url}")
        print(f"      Score : {score:.4f}")
        print(f"      Terms : {counts}")
        print()


def find(index: InvertedIndex, raw_query: str, top_n: int = 10) -> None:
    """Search the index for raw_query and print ranked results
    Handles empty queries, stop-word-only queries, and no results
    """
    if not raw_query.strip():
        print("\n  Please enter a search query.\n")
        return

    cleaned_query, operator = parse_query(raw_query)
    logger.debug("Parsed query: %r  operator: %s", cleaned_query, operator)
    results = index.search(cleaned_query, operator=operator, top_n=top_n)
    format_results(results, cleaned_query, operator)