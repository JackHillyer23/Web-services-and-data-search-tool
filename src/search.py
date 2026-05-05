import logging
from typing import Any

from indexer import InvertedIndex

logger = logging.getLogger(__name__)


def parse_query(raw_query: str) -> tuple[str, str]:
    """Detect an explicit Boolean operator in *raw_query*.

    Supports ``AND`` and ``OR`` as case-insensitive keywords between terms.
    If no operator is found, defaults to ``OR``.

    Args:
        raw_query: Raw string typed by the user (e.g. ``"love AND truth"``).

    Returns:
        A ``(cleaned_query, operator)`` tuple where *cleaned_query* has the
        operator keyword removed and *operator* is ``"AND"`` or ``"OR"``.

    Examples:
        >>> parse_query("love AND truth")
        ("love truth", "AND")
        >>> parse_query("love OR truth")
        ("love truth", "OR")
        >>> parse_query("love truth")
        ("love truth", "OR")
    """
    upper = raw_query.upper()
    if " AND " in upper:
        operator = "AND"
        # Remove the AND keyword (case-insensitive) from the query.
        import re
        cleaned = re.sub(r"\bAND\b", "", raw_query, flags=re.IGNORECASE).strip()
        # Collapse extra spaces left by removal.
        cleaned = " ".join(cleaned.split())
    elif " OR " in upper:
        operator = "OR"
        import re
        cleaned = re.sub(r"\bOR\b", "", raw_query, flags=re.IGNORECASE).strip()
        cleaned = " ".join(cleaned.split())
    else:
        operator = "OR"
        cleaned = raw_query.strip()

    return cleaned, operator


def format_results(results: list[dict[str, Any]], query: str, operator: str) -> None:
    """Pretty-print search results to stdout.

    Args:
        results:  List of result dicts from :meth:`InvertedIndex.search`.
        query:    The cleaned query string (for display).
        operator: The Boolean operator used (``"AND"`` / ``"OR"``).
    """
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
        counts_str = ", ".join(
            f"'{t}' ×{c}" for t, c in sorted(term_counts.items())
        )
        print(f"  {rank:>2}. {url}")
        print(f"      Score : {score:.4f}")
        print(f"      Terms : {counts_str}")
        print()


def find(index: InvertedIndex, raw_query: str, top_n: int = 10) -> None:
    """Parse *raw_query*, search *index*, and print ranked results.

    This is the top-level function called by main.py for the ``find``
    command.

    Args:
        index:     A loaded :class:`InvertedIndex` instance.
        raw_query: The user's raw search string.
        top_n:     Maximum results to show (default 10).

    Edge cases handled:
        * Empty string → informs user, exits cleanly.
        * Stop-word-only query → informs user, exits cleanly.
        * No matching documents → "No results found." message.
        * AND query with zero intersection → clear no-results message.
    """
    if not raw_query.strip():
        print("\n  Please enter a search query.\n")
        return

    cleaned_query, operator = parse_query(raw_query)
    logger.debug("Parsed query: %r  operator: %s", cleaned_query, operator)

    results = index.search(cleaned_query, operator=operator, top_n=top_n)
    format_results(results, cleaned_query, operator)