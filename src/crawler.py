import time
import logging
from collections import deque
from urllib.parse import urljoin, urlparse
from typing import Generator
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
POLITENESS_DELAY: float = 6.0

# default request timeout so the crawler never hangs forever
REQUEST_TIMEOUT: int = 10

# A minimal browser-like User-Agent so the server knows who we are.
HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; UniSearchBot/1.0; "
        "+https://github.com/student/search-tool)"
    )
}


def _is_same_domain(base_url: str, candidate_url: str) -> bool:
    """Return True if *candidate_url* belongs to the same domain as *base_url*.

    This prevents the crawler from wandering off to external sites.

    Args:
        base_url: The seed / starting URL.
        candidate_url: A URL extracted from an anchor tag.

    Returns:
        True when both URLs share the same netloc (e.g. quotes.toscrape.com).
    """
    return urlparse(base_url).netloc == urlparse(candidate_url).netloc


def _extract_links(soup: BeautifulSoup, current_url: str, base_url: str) -> list[str]:
    """Extract all internal, absolute URLs from *soup*.

    Args:
        soup: Parsed HTML for the current page.
        current_url: The URL of the page being parsed (used to resolve
            relative hrefs).
        base_url: The crawler's seed URL (used for domain filtering).

    Returns:
        A deduplicated list of absolute internal URLs found on the page.
    """
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]
        absolute = urljoin(current_url, href)
        # Strip fragments — #section links resolve to the same page.
        absolute = absolute.split("#")[0]
        if absolute and _is_same_domain(base_url, absolute):
            links.append(absolute)
    # Deduplicate while preserving order.
    return list(dict.fromkeys(links))


def _extract_text(soup: BeautifulSoup) -> str:
    """Extract visible text content from *soup*, stripping HTML boilerplate.

    Removes <script> and <style> elements before extracting text so that
    JavaScript source code and CSS rules are not indexed.

    Args:
        soup: Parsed HTML for a page.

    Returns:
        A single whitespace-normalised string of visible text.
    """
    # Remove non-visible elements in place.
    for element in soup(["script", "style", "meta", "head"]):
        element.decompose()

    text = soup.get_text(separator=" ")
    # Collapse runs of whitespace into a single space.
    return " ".join(text.split())


def crawl(
    start_url: str,
    delay: float = POLITENESS_DELAY,
    max_pages: int | None = None,
) -> Generator[dict[str, str], None, None]:
    """Crawl *start_url* using BFS and yield page data for each visited URL.

    Complexity:
        Time:  O(P * L) where P = number of pages, L = links per page.
        Space: O(P) for the visited set and queue.

    Args:
        start_url: The seed URL where crawling begins.
        delay: Seconds to wait between HTTP requests (default 6 s).
        max_pages: Optional cap on the number of pages to visit; useful
            during testing so we don't crawl the entire site.

    Yields:
        A dict with keys:
            ``url``   – the page's URL (str)
            ``title`` – the page's <title> text, or empty string (str)
            ``text``  – visible text content of the page (str)

    Example:
        >>> for page in crawl("https://quotes.toscrape.com"):
        ...     print(page["url"], page["title"])
    """
    visited: set[str] = set()
    queue: deque[str] = deque([start_url])
    pages_crawled: int = 0

    logger.info("Starting crawl from %s", start_url)

    while queue:
        url = queue.popleft()

        # Skip if already processed.
        if url in visited:
            continue

        # Respect optional page cap.
        if max_pages is not None and pages_crawled >= max_pages:
            logger.info("Reached max_pages limit (%d). Stopping.", max_pages)
            break

        try:
            logger.debug("Fetching %s", url)
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Raises for 4xx / 5xx responses.
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error for %s: %s", url, exc)
            visited.add(url)
            continue
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Connection error for %s: %s", url, exc)
            visited.add(url)
            continue
        except requests.exceptions.Timeout:
            logger.warning("Timeout fetching %s", url)
            visited.add(url)
            continue
        except requests.exceptions.RequestException as exc:
            logger.warning("Unexpected request error for %s: %s", url, exc)
            visited.add(url)
            continue

        visited.add(url)
        pages_crawled += 1

        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        text = _extract_text(soup)

        yield {"url": url, "title": title, "text": text}

        # Enqueue undiscovered internal links.
        for link in _extract_links(soup, url, start_url):
            if link not in visited:
                queue.append(link)

        logger.debug(
            "Crawled %d page(s). Queue length: %d", pages_crawled, len(queue)
        )

        #skip after the final page to avoid unnecessary wait
        if queue:
            time.sleep(delay)

    logger.info("Crawl complete. Total pages crawled: %d", pages_crawled)