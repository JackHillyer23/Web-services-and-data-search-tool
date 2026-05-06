import sys
import requests
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler import _extract_links, _extract_text, _is_same_domain, crawl
from bs4 import BeautifulSoup


def _make_response(html: str, status_code: int = 200) -> MagicMock:
    """Return a mock Response mimicking requests.get()."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = html
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# _is_same_domain 

class TestIsSameDomain:
    def test_same_domain_returns_true(self):
        assert _is_same_domain(
            "https://quotes.toscrape.com",
            "https://quotes.toscrape.com/page/2/",
        )

    def test_different_domain_returns_false(self):
        assert not _is_same_domain(
            "https://quotes.toscrape.com",
            "https://example.com/page",
        )

    def test_subdomain_treated_as_different(self):
        assert not _is_same_domain(
            "https://quotes.toscrape.com",
            "https://other.toscrape.com/page",
        )


# _extract_links

class TestExtractLinks:
    BASE = "https://quotes.toscrape.com"

    def _soup(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def test_internal_links_extracted(self):
        html = '<a href="/page/2/">next</a><a href="/author/Einstein/">einstein</a>'
        soup = self._soup(html)
        links = _extract_links(soup, self.BASE, self.BASE)
        assert "https://quotes.toscrape.com/page/2/" in links
        assert "https://quotes.toscrape.com/author/Einstein/" in links

    def test_external_links_excluded(self):
        html = '<a href="https://external.com/about">ext</a>'
        soup = self._soup(html)
        links = _extract_links(soup, self.BASE, self.BASE)
        assert links == []

    def test_fragment_stripped_from_links(self):
        html = '<a href="/page/2/#section">next</a>'
        soup = self._soup(html)
        links = _extract_links(soup, self.BASE, self.BASE)
        assert "https://quotes.toscrape.com/page/2/" in links
        assert all("#" not in link for link in links)

    def test_duplicate_links_deduplicated(self):
        html = '<a href="/page/2/">a</a><a href="/page/2/">b</a>'
        soup = self._soup(html)
        links = _extract_links(soup, self.BASE, self.BASE)
        assert links.count("https://quotes.toscrape.com/page/2/") == 1

    def test_no_links_returns_empty_list(self):
        soup = self._soup("<p>No links here</p>")
        assert _extract_links(soup, self.BASE, self.BASE) == []


# _extract_text

class TestExtractText:
    def test_removes_script_tags(self):
        html = "<html><body><script>var x=1;</script><p>Hello world</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        text = _extract_text(soup)
        assert "var x" not in text
        assert "Hello world" in text

    def test_removes_style_tags(self):
        html = "<html><body><style>body{color:red}</style><p>Visible</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        text = _extract_text(soup)
        assert "color:red" not in text
        assert "Visible" in text

    def test_whitespace_normalised(self):
        html = "<p>  Too   many    spaces  </p>"
        soup = BeautifulSoup(html, "html.parser")
        text = _extract_text(soup)
        assert "  " not in text

    def test_empty_page_returns_empty_string(self):
        soup = BeautifulSoup("", "html.parser")
        assert _extract_text(soup) == ""


# crawl()

class TestCrawl:
    BASE = "https://quotes.toscrape.com"

    HOME_HTML = """
    <html>
      <head><title>Quotes Home</title></head>
      <body>
        <p>Welcome to quotes!</p>
        <a href="/page/2/">Next</a>
      </body>
    </html>
    """

    PAGE2_HTML = """
    <html>
      <head><title>Page 2</title></head>
      <body>
        <p>More quotes here.</p>
      </body>
    </html>
    """

    @patch("crawler.time.sleep")
    @patch("crawler.requests.get")
    def test_yields_correct_number_of_pages(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _make_response(self.HOME_HTML),
            _make_response(self.PAGE2_HTML),
        ]
        pages = list(crawl(self.BASE, delay=0))
        assert len(pages) == 2

    @patch("crawler.time.sleep")
    @patch("crawler.requests.get")
    def test_page_data_structure(self, mock_get, mock_sleep):
        mock_get.return_value = _make_response(self.HOME_HTML)
        pages = list(crawl(self.BASE, delay=0, max_pages=1))
        assert len(pages) == 1
        page = pages[0]
        assert "url" in page
        assert "title" in page
        assert "text" in page
        assert page["title"] == "Quotes Home"
        assert "Welcome" in page["text"]

    @patch("crawler.time.sleep")
    @patch("crawler.requests.get")
    def test_max_pages_respected(self, mock_get, mock_sleep):
        mock_get.return_value = _make_response(self.HOME_HTML)
        pages = list(crawl(self.BASE, delay=0, max_pages=1))
        assert len(pages) == 1

    @patch("crawler.time.sleep")
    @patch("crawler.requests.get")
    def test_http_error_skipped_gracefully(self, mock_get, mock_sleep):
        error_resp = MagicMock()
        error_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = error_resp
        pages = list(crawl(self.BASE, delay=0, max_pages=1))
        assert pages == []

    @patch("crawler.time.sleep")
    @patch("crawler.requests.get")
    def test_connection_error_skipped_gracefully(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.exceptions.ConnectionError("refused")
        pages = list(crawl(self.BASE, delay=0, max_pages=1))
        assert pages == []

    @patch("crawler.time.sleep")
    @patch("crawler.requests.get")
    def test_timeout_skipped_gracefully(self, mock_get, mock_sleep):
        mock_get.side_effect = requests.exceptions.Timeout()
        pages = list(crawl(self.BASE, delay=0, max_pages=1))
        assert pages == []

    @patch("crawler.time.sleep")
    @patch("crawler.requests.get")
    def test_no_duplicate_pages(self, mock_get, mock_sleep):
        self_link_html = """
        <html><body>
          <a href="/">Home</a>
          <a href="/">Home again</a>
        </body></html>
        """
        mock_get.return_value = _make_response(self_link_html)
        pages = list(crawl(self.BASE, delay=0, max_pages=5))
        urls = [p["url"] for p in pages]
        assert len(urls) == len(set(urls))

    @patch("crawler.time.sleep")
    @patch("crawler.requests.get")
    def test_politeness_sleep_called(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _make_response(self.HOME_HTML),
            _make_response(self.PAGE2_HTML),
        ]
        list(crawl(self.BASE, delay=6))
        mock_sleep.assert_called_once_with(6)