import time
import logging
from collections import deque
from urllib.parse import urljoin, urlparse
from typing import Generator
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
POLITENESS_DELAY: float = 6.0
REQUEST_TIMEOUT: int = 10


HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; UniSearchBot/1.0; "
        "+https://github.com/JackHillyer/search-tool)"
    )
}

def _normalise_url(url: str) -> str:
    """ Normalise trailing slashes so /page and /page/ are treated as the same URL"""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return parsed._replace(path=path).geturl()


def _is_same_domain(base_url: str, candidate_url: str) -> bool:
    """Check both URLs share the same domain to keep the crawler on-site"""
    return urlparse(base_url).netloc == urlparse(candidate_url).netloc


def _extract_links(soup: BeautifulSoup, current_url: str, base_url: str) -> list[str]:
    """Return a deduplicated list of internal absolute URLs found on the page"""
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href: str = tag["href"]
        absolute = urljoin(current_url, href)
        absolute = absolute.split("#")[0] #drop fragments 
        if absolute and _is_same_domain(base_url, absolute):
            links.append(absolute)

    return list(dict.fromkeys(links)) # Deduplicate while preserving order.


def _extract_text(soup: BeautifulSoup) -> str:
    """strips scripts, styles and boilerplate then return normalised visible text."""
    for item in soup(["script", "style", "meta", "head"]):
        item.decompose()

    text = soup.get_text(separator=" ")
    # Collapse runs of whitespace into a single space.
    return " ".join(text.split())


def crawl(
    start_url: str,
    delay: float = POLITENESS_DELAY,
    max_pages: int | None = None,
) -> Generator[dict[str, str], None, None]:
    """Crawl start_url using BFS, yielding page data as each URL is visited
    Uses a deque for O(1) pops and a visited set for O(1) duplicate checks
    Waits `delay` seconds between requests to stay polite to the server
    Yields dicts with keys: url, title, text
    """
    visited: set[str] = set()
    queue: deque[str] = deque([_normalise_url(start_url)])
    crawled: int = 0
    logger.info("Starting crawl from %s", start_url)
    while queue:
        url = queue.popleft()
        # Skip if already processed.
        if url in visited:
            continue

        if max_pages is not None and crawled >= max_pages:
            logger.info("Reached max_pages limit (%d). Stopping.", max_pages)
            break

        try:
            logger.debug("Fetching %s", url)
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
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
        crawled += 1

        soup = BeautifulSoup(response.text, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        text = _extract_text(soup)
        yield {"url": url, "title": title, "text": text}

        #enqueue undiscovered internal links.
        for link in _extract_links(soup, url, start_url):
            link = _normalise_url(link)
            if link not in visited:
                queue.append(link)

        logger.debug(
            "Crawled %d page(s). Queue length: %d", crawled, len(queue)
        )


        if queue:
            time.sleep(delay) # skip delay after the last page

    logger.info("Crawl complete. Total pages crawled: %d", crawled)