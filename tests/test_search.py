"""Tests for the keyword index, fusion, filters, the reranker and the three setups (no network)."""

import pytest

from askmyrepo import keyword_index, reranker, search, vector_index
from askmyrepo.chunker import Chunk
from askmyrepo.vector_index import Hit


def chunk(path, text, version="2.0", latest=True, repo="shoplite", start=1):
    """A chunk with just enough to search."""
    return Chunk(repo, path, "python", "", start, start + 9, "abc1234", text, version, latest)


def hit(name, score=0.0):
    """A hit whose id is easy to read."""
    return Hit(score, "shoplite", f"{name}.py", "python", "", 1, 9, "abc1234", name, "2.0")


@pytest.fixture
def keywords(tmp_path):
    """A keyword index over four chunks: two versions of checkout, stock, and the warehouse."""
    chunks = [
        chunk("shoplite/services/stock.py", "def reserve_for_order(): ReservationEntry "
              "inv_reservation_ledger hold stock"),
        chunk("shoplite/services/checkout.py", "def checkout(): take_stock after payment",
              version="1.1", latest=False),
        chunk("shoplite/services/checkout.py", "def checkout(): reserve_for_order then charge"),
        chunk("warehouse/store.py", "def set_level(): stock_levels sync_log", repo="warehouse"),
    ]
    keyword_index.save_chunks(chunks, "syntax", tmp_path)
    return keyword_index.KeywordIndex("syntax", tmp_path)


def test_an_exact_name_is_found_by_keyword(keywords):
    hits = keywords.search("Where is inv_reservation_ledger written?")
    assert [h.path for h in hits] == ["shoplite/services/stock.py"]


def test_error_codes_and_names_stay_one_word():
    assert keyword_index.tokenize(["Where is SL-ORD-409 raised in inv_reservation_ledger?"])[0] == [
        "where", "sl-ord-409", "raised", "inv_reservation_ledger"]


def test_keyword_filters_match_the_vector_store_filters(keywords):
    def paths(**filters):
        """Paths found for "checkout" with these filters."""
        found = keywords.search("checkout stock stock_levels", filters=filters)
        return [(h.path, h.version) for h in found]

    assert ("shoplite/services/checkout.py", "1.1") in paths()
    assert ("shoplite/services/checkout.py", "1.1") not in paths(latest=True)
    assert paths(version="1.1") == [("shoplite/services/checkout.py", "1.1")]
    assert paths(repo="warehouse") == [("warehouse/store.py", "2.0")]
    assert all(p.startswith("shoplite/services/") for p, _ in paths(folder="shoplite/services"))


def test_a_missing_keyword_index_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="run `index` first"):
        keyword_index.KeywordIndex("syntax", tmp_path)


def test_fusion_rewards_a_passage_both_searches_found():
    by_meaning = [hit("a"), hit("b"), hit("c")]
    by_keyword = [hit("c"), hit("d")]
    fused = search.fuse_by_rank(by_meaning, by_keyword)
    assert [h.text for h in fused] == ["c", "a", "b", "d"]  # c: 1/63 + 1/61 beats a: 1/61
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)


def test_fusion_ignores_the_raw_scores():
    fused = search.fuse_by_rank([hit("a", 0.9), hit("b", 0.1)], [hit("b", 900.0), hit("a", 1.0)])
    assert fused[0].score == pytest.approx(fused[1].score)  # same ranks, same fused score


def test_filters_become_pinecone_filters():
    assert search.pinecone_filter({}) is None
    assert search.pinecone_filter({"repo": "shoplite", "folder": "docs", "latest": True}) == {
        "repo": {"$eq": "shoplite"}, "folders": {"$in": ["docs"]}, "latest": {"$eq": True}}


class FakeReranker:
    """Scores a passage by how many times it says "stock"."""

    def predict(self, pairs, batch_size):
        """One score per pair."""
        return [text.count("stock") for _, text in pairs]


def test_the_reranker_keeps_the_best_passages_with_new_scores():
    hits = [hit("no"), hit("stock stock"), hit("stock")]
    kept = reranker.rerank_top("q", hits, keep=2, model=FakeReranker())
    assert [(h.text, h.score) for h in kept] == [("stock stock", 2.0), ("stock", 1.0)]
    assert reranker.rerank_top("q", [], model=FakeReranker()) == []


def test_each_setup_uses_the_right_stages(monkeypatch, keywords):
    calls = []

    def fake_meaning(index, question, namespace, top_k, metadata_filter):
        """Record the call and return two hits."""
        calls.append(("meaning", top_k, metadata_filter))
        return [hit("meaning-1"), hit("meaning-2")]

    monkeypatch.setattr(vector_index, "search_by_meaning", fake_meaning)
    monkeypatch.setattr(reranker, "load_model", lambda: FakeReranker())
    searcher = search.Searcher(None, keywords, "syntax")

    assert len(searcher.search("hold stock", "meaning", {"latest": True})) == 2
    assert calls[-1] == ("meaning", 5, {"latest": {"$eq": True}})
    fused = searcher.search("hold stock", "fused", {"latest": True})
    assert calls[-1][1] == search.CANDIDATES  # each search hands on 20
    assert {h.path for h in fused} >= {"shoplite/services/stock.py", "meaning-1.py"}
    reranked = searcher.search("hold stock", "reranked", {"latest": True}, top_k=1)
    assert reranked[0].path == "shoplite/services/stock.py"  # the passage with "stock" in it


def test_the_candidate_list_size_can_be_changed(monkeypatch, keywords):
    seen = []
    monkeypatch.setattr(vector_index, "search_by_meaning",
                        lambda index, q, ns, top_k, f: seen.append(top_k) or [])
    monkeypatch.setattr(reranker, "load_model", lambda: FakeReranker())
    search.Searcher(None, keywords, "syntax").search("stock", "reranked", candidates=300)
    assert seen == [300]
