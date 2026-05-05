import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running from the project root.
sys.path.insert(0, str(Path(__file__).parent))

from crawler import crawl
from indexer import InvertedIndex
from search import find

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Default values
DEFAULT_URL = "https://quotes.toscrape.com"
DEFAULT_INDEX_PATH = Path(__file__).parent.parent / "data" / "index.json"


# Command handlers
def cmd_build(args: argparse.Namespace) -> None:
    """Crawl *args.url* and write the inverted index to *args.output*.

    Steps:
      1. Crawl the site (BFS, politeness-limited).
      2. Add each page to the InvertedIndex.
      3. Compute TF-IDF scores across all documents.
      4. Save the index to disk as JSON.
    """
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[build] Starting crawl of {args.url}")
    print(f"[build] Index will be saved to: {output_path}\n")

    index = InvertedIndex()
    pages_indexed = 0
    for page in crawl(args.url, max_pages=args.max_pages):
        index.add_document(page["url"], page["text"])
        pages_indexed += 1
        print(f"  [{pages_indexed}] Indexed: {page['url']}")

    if pages_indexed == 0:
        print("\n[build] No pages were crawled. Check the URL and your connection.")
        sys.exit(1)

    print(f"\n[build] Computing TF-IDF scores across {pages_indexed} page(s)...")
    index.compute_tf_idf()

    index.save(output_path)
    stats = index.get_stats()
    print(f"\n[build] Done!")
    print(f"  Pages indexed : {stats['total_documents']}")
    print(f"  Unique terms  : {stats['total_terms']:,}")
    print(f"  Index file    : {output_path}\n")


def cmd_load(args: argparse.Namespace) -> None:
    """Load the index from *args.index* and print confirmation.

    This command is primarily useful for verifying the index file is valid
    and not corrupted, without running a full search.
    """
    index_path = Path(args.index)
    try:
        index = InvertedIndex.load(index_path)
    except FileNotFoundError:
        print(f"\n[load] Error: Index file not found at {index_path}")
        print("[load] Run 'python main.py build' first to create the index.\n")
        sys.exit(1)

    stats = index.get_stats()
    print(f"\n[load] Index loaded successfully from {index_path}")
    print(f"  Total terms     : {stats['total_terms']:,}")
    print(f"  Total documents : {stats['total_documents']}")
    print(f"  Avg docs/term   : {stats['average_docs_per_term']}\n")


def cmd_print(args: argparse.Namespace) -> None:
    """Load and pretty-print the index contents."""
    index_path = Path(args.index)
    try:
        index = InvertedIndex.load(index_path)
    except FileNotFoundError:
        print(f"\n[print] Error: Index file not found at {index_path}")
        print("[print] Run 'python main.py build' first.\n")
        sys.exit(1)

    index.print_index(max_terms=args.max_terms)


def cmd_find(args: argparse.Namespace) -> None:
    """Load the index and search for *args.query*."""
    index_path = Path(args.index)
    try:
        index = InvertedIndex.load(index_path)
    except FileNotFoundError:
        print(f"\n[find] Error: Index file not found at {index_path}")
        print("[find] Run 'python main.py build' first.\n")
        sys.exit(1)

    query = " ".join(args.query)  # Rejoin multi-word queries.
    find(index, query, top_n=args.top)


# Argument parser
def build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="search-tool",
        description="A TF-IDF powered web search tool.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = subparsers.add_parser("build", help="Crawl a site and build the index.")
    p_build.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Seed URL for the crawler (default: {DEFAULT_URL})",
    )
    p_build.add_argument(
        "--output",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to save the index JSON file.",
    )
    p_build.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Limit crawl to N pages (useful for testing).",
    )

    # load
    p_load = subparsers.add_parser("load", help="Load and validate the index.")
    p_load.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to the index JSON file.",
    )

    # print
    p_print = subparsers.add_parser("print", help="Display index contents.")
    p_print.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to the index JSON file.",
    )
    p_print.add_argument(
        "--max-terms",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of terms to display (default: 50).",
    )

    # find
    p_find = subparsers.add_parser("find", help="Search the index.")
    p_find.add_argument(
        "query",
        nargs="+",
        help='Search query. Use AND/OR between terms: "love AND truth"',
    )
    p_find.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help="Path to the index JSON file.",
    )
    p_find.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Number of results to return (default: 10).",
    )

    return parser


# Entry 
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "build": cmd_build,
        "load":  cmd_load,
        "print": cmd_print,
        "find":  cmd_find,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()