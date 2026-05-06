import json
import sys
from pathlib import Path
from indexer import InvertedIndex, tokenise, STOP_WORDS
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# tokenise
class TestTokenise:
    def test_basic_tokenisation(self):
        tokens = tokenise("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercased(self):
        tokens = tokenise("HELLO World")
        assert all(t == t.lower() for t in tokens)

    def test_stop_words_removed(self):
        for stop in list(STOP_WORDS)[:10]:
            tokens = tokenise(stop)
            assert stop not in tokens

    def test_short_tokens_removed(self):
        tokens = tokenise("a b c hello")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "hello" in tokens

    def test_empty_string(self):
        assert tokenise("") == []

    def test_only_stop_words(self):
        assert tokenise("the and or but") == []

    def test_punctuation_stripped(self):
        tokens = tokenise("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens

    def test_numbers_excluded(self):
        # regex only matches [a-zA-Z'] so pure numbers are dropped
        tokens = tokenise("12345 hello 999")
        assert "12345" not in tokens
        assert "hello" in tokens


# add_document
class TestAddDocument:
    def test_terms_added_to_index(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love is eternal")
        assert "love" in idx._index
        assert "eternal" in idx._index

    def test_doc_count_incremented(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love")
        idx.add_document("http://b.com", "truth")
        assert idx._doc_count == 2

    def test_same_url_multiple_calls(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love truth")
        assert "love" in idx._index
        assert "http://a.com" in idx._index["love"]

    def test_empty_text_skipped(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "")
        assert idx._doc_count == 0

    def test_stop_words_only_skipped(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "the and or")
        assert idx._doc_count == 0

    def test_count_stored_correctly(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love love love truth")
        assert idx._index["love"]["http://a.com"]["count"] == 3
        assert idx._index["truth"]["http://a.com"]["count"] == 1


# compute_tf_idf

class TestComputeTfIdf:
    def test_tf_idf_scores_populated(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love truth")
        idx.add_document("http://b.com", "love friendship")
        idx.compute_tf_idf()
        for term, doc_map in idx._index.items():
            for url, stats in doc_map.items():
                assert "tf_idf" in stats
                assert isinstance(stats["tf_idf"], float)

    def test_rare_term_scores_higher(self):
        """A word in 1/3 docs should score higher than one in all 3."""
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love truth unique")
        idx.add_document("http://b.com", "love truth")
        idx.add_document("http://c.com", "love truth")
        idx.compute_tf_idf()
        score_unique = idx._index["unique"]["http://a.com"]["tf_idf"]
        score_love   = idx._index["love"]["http://a.com"]["tf_idf"]
        assert score_unique > score_love

    def test_empty_index_no_crash(self):
        idx = InvertedIndex()
        idx.compute_tf_idf()

    def test_raw_tf_removed_after_compute(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love truth")
        idx.compute_tf_idf()
        for term, doc_map in idx._index.items():
            for url, stats in doc_map.items():
                assert "tf" not in stats


# search
class TestSearch:
    def _build_index(self) -> InvertedIndex:
        idx = InvertedIndex()
        idx.add_document("http://page1.com", "love is beautiful truth")
        idx.add_document("http://page2.com", "truth matters wisdom")
        idx.add_document("http://page3.com", "wisdom love friendship")
        idx.compute_tf_idf()
        return idx

    def test_or_search_returns_all_matching(self):
        idx = self._build_index()
        results = idx.search("love truth", operator="OR")
        assert len(results) == 3

    def test_and_search_requires_all_terms(self):
        idx = self._build_index()
        results = idx.search("love truth", operator="AND")
        urls = [r["url"] for r in results]
        assert urls == ["http://page1.com"]

    def test_and_search_no_intersection_returns_empty(self):
        idx = self._build_index()
        results = idx.search("friendship truth", operator="AND")
        assert results == []

    def test_results_sorted_by_score_descending(self):
        idx = self._build_index()
        results = idx.search("love", operator="OR")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self):
        idx = self._build_index()
        assert idx.search("") == []

    def test_stop_words_only_query_returns_empty(self):
        idx = self._build_index()
        assert idx.search("the and or") == []

    def test_unknown_word_returns_empty(self):
        idx = self._build_index()
        assert idx.search("nonexistentword12345") == []

    def test_top_n_limits_results(self):
        idx = self._build_index()
        results = idx.search("love truth wisdom", operator="OR", top_n=2)
        assert len(results) <= 2

    def test_invalid_operator_raises(self):
        idx = self._build_index()
        with pytest.raises(ValueError):
            idx.search("love", operator="XOR")

    def test_term_counts_in_results(self):
        idx = self._build_index()
        results = idx.search("love", operator="OR")
        for r in results:
            assert "term_counts" in r
            assert "love" in r["term_counts"]


# persistence
class TestPersistence:
    def test_save_creates_file(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love truth")
        idx.compute_tf_idf()
        path = tmp_path / "index.json"
        idx.save(path)
        assert path.exists()

    def test_save_load_round_trip(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love truth wisdom")
        idx.add_document("http://b.com", "friendship love")
        idx.compute_tf_idf()
        path = tmp_path / "index.json"
        idx.save(path)
        loaded = InvertedIndex.load(path)
        assert loaded._doc_count == idx._doc_count
        assert set(loaded._index.keys()) == set(idx._index.keys())

    def test_load_preserves_search_results(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love is eternal")
        idx.add_document("http://b.com", "truth and wisdom")
        idx.compute_tf_idf()
        path = tmp_path / "index.json"
        idx.save(path)
        loaded = InvertedIndex.load(path)
        results = loaded.search("love")
        assert len(results) > 0
        assert results[0]["url"] == "http://a.com"

    def test_load_nonexistent_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            InvertedIndex.load(tmp_path / "does_not_exist.json")

    def test_saved_file_is_valid_json(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "hello world")
        idx.compute_tf_idf()
        path = tmp_path / "index.json"
        idx.save(path)
        with path.open() as fh:
            data = json.load(fh)
        assert "meta" in data
        assert "index" in data


# stats and print
class TestStats:
    def test_get_stats_empty_index(self):
        idx = InvertedIndex()
        stats = idx.get_stats()
        assert stats["total_terms"] == 0
        assert stats["total_documents"] == 0
        assert stats["average_docs_per_term"] == 0

    def test_get_stats_populated(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love truth")
        idx.add_document("http://b.com", "love friendship")
        idx.compute_tf_idf()
        stats = idx.get_stats()
        assert stats["total_documents"] == 2
        assert stats["total_terms"] >= 3

    def test_print_index_smoke(self, capsys):
        """print_index should not raise and should produce output."""
        idx = InvertedIndex()
        idx.add_document("http://a.com", "love truth wisdom")
        idx.compute_tf_idf()
        idx.print_index()
        captured = capsys.readouterr()
        assert "INVERTED INDEX" in captured.out
        assert "love" in captured.out or "truth" in captured.out