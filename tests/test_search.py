import sys
from pathlib import Path
from search import parse_query, find, format_results
from indexer import InvertedIndex
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# parse_query
class TestParseQuery:
    def test_and_operator_detected(self):
        cleaned, op = parse_query("love AND truth")
        assert op == "AND"
        assert "AND" not in cleaned.upper()

    def test_or_operator_detected(self):
        cleaned, op = parse_query("love OR truth")
        assert op == "OR"
        assert "OR" not in cleaned.upper()

    def test_no_operator_defaults_to_or(self):
        cleaned, op = parse_query("love truth")
        assert op == "OR"

    def test_case_insensitive_operator(self):
        _, op = parse_query("love and truth")
        assert op == "AND"

    def test_cleaned_query_has_no_operator_keyword(self):
        cleaned, _ = parse_query("love AND truth")
        assert "AND" not in cleaned.upper()
        assert "love" in cleaned
        assert "truth" in cleaned

    def test_empty_query(self):
        cleaned, op = parse_query("")
        assert cleaned == ""
        assert op == "OR"

    def test_single_word(self):
        cleaned, op = parse_query("love")
        assert cleaned == "love"
        assert op == "OR"


# find
class TestFind:
    def _build_index(self) -> InvertedIndex:
        idx = InvertedIndex()
        idx.add_document("http://page1.com", "love is beautiful truth")
        idx.add_document("http://page2.com", "truth matters wisdom")
        idx.add_document("http://page3.com", "wisdom love friendship")
        idx.compute_tf_idf()
        return idx

    def test_find_prints_results(self, capsys):
        idx = self._build_index()
        find(idx, "love")
        captured = capsys.readouterr()
        assert "page1.com" in captured.out or "page3.com" in captured.out

    def test_find_empty_query_prints_message(self, capsys):
        idx = self._build_index()
        find(idx, "")
        captured = capsys.readouterr()
        assert "enter a search query" in captured.out.lower()

    def test_find_no_results_prints_message(self, capsys):
        idx = self._build_index()
        find(idx, "xyznonexistent")
        captured = capsys.readouterr()
        assert "no results" in captured.out.lower()

    def test_find_and_operator(self, capsys):
        idx = self._build_index()
        find(idx, "love AND truth")
        captured = capsys.readouterr()
        assert "AND" in captured.out

    def test_find_stop_words_only(self, capsys):
        idx = self._build_index()
        find(idx, "the and or")
        captured = capsys.readouterr()
        assert "no results" in captured.out.lower()


# format_results
class TestFormatResults:
    def test_empty_results_shows_no_results(self, capsys):
        format_results([], "love", "OR")
        captured = capsys.readouterr()
        assert "no results" in captured.out.lower()

    def test_results_show_url_and_score(self, capsys):
        results = [
            {"url": "http://example.com", "score": 0.42, "term_counts": {"love": 2}}
        ]
        format_results(results, "love", "OR")
        captured = capsys.readouterr()
        assert "example.com" in captured.out
        assert "0.4200" in captured.out

    def test_rank_numbers_shown(self, capsys):
        results = [
            {"url": "http://a.com", "score": 0.9, "term_counts": {"love": 1}},
            {"url": "http://b.com", "score": 0.5, "term_counts": {"love": 1}},
        ]
        format_results(results, "love", "OR")
        captured = capsys.readouterr()
        assert "1." in captured.out
        assert "2." in captured.out