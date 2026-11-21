"""Tests for the embedder, the index, the evaluation and the answer, with fakes (no network)."""

from types import SimpleNamespace

import numpy as np
import pytest

from askmyrepo import answer, embedder, evaluate, vector_index
from askmyrepo.chunker import Chunk
from askmyrepo.vector_index import Hit

CODE = "def order_number(): ..."


class FakeModel:
    """Stands in for the embedding model and records how it was called."""

    def __init__(self):
        """Nothing called yet."""
        self.calls = []

    def encode_document(self, texts, **kwargs):
        """One vector per text."""
        self.calls.append(("document", texts, kwargs))
        return np.ones((len(texts), 3))

    def encode_query(self, texts, **kwargs):
        """One vector per question."""
        self.calls.append(("query", texts, kwargs))
        return np.zeros((len(texts), 3))


class FakeIndex:
    """Stands in for a Pinecone index."""

    def __init__(self):
        """An empty store."""
        self.stored = {}

    def upsert(self, vectors, namespace):
        """Keep the records."""
        for record in vectors:
            self.stored[(namespace, record["id"])] = record

    def query(self, vector, top_k, namespace, include_metadata, filter=None):
        """Return the stored records, with line numbers as floats as Pinecone does."""
        self.last_filter = filter
        matches = []
        for (ns, _), record in self.stored.items():
            if ns == namespace:
                metadata = dict(record["metadata"])
                metadata["start_line"] = float(metadata["start_line"])
                metadata["end_line"] = float(metadata["end_line"])
                matches.append(SimpleNamespace(score=0.9, metadata=metadata))
        return SimpleNamespace(matches=matches[:top_k])


def make_chunk(path="shoplite/utils/ids.py", start=1, end=20):
    """A whole-file chunk."""
    return Chunk("shoplite", path, "python", "", start, end, "abc1234", CODE)


def make_hit(path="shoplite/utils/ids.py", repo="shoplite", text=CODE, symbol="order_number"):
    """A search hit."""
    return Hit(0.5, repo, path, "python", symbol, 18, 20, "abc1234", text)


def test_passages_get_no_prompt_and_questions_get_the_query_prompt():
    model = FakeModel()
    assert embedder.embed_documents(["a", "b"], model) == [[1.0, 1.0, 1.0]] * 2
    assert embedder.embed_question("q", model) == [0.0, 0.0, 0.0]
    assert [call[0] for call in model.calls] == ["document", "query"]
    assert all(call[2]["normalize_embeddings"] for call in model.calls)


def test_upload_then_search_returns_the_chunk_with_integer_lines(monkeypatch):
    monkeypatch.setattr(vector_index, "embed_question", lambda q: [0.0, 0.0, 0.0])
    index = FakeIndex()
    vector_index.upload(index, [make_chunk()], [[0.1, 0.2, 0.3]], "files")

    hits = vector_index.search_by_meaning(index, "order number format?", "files")

    assert hits[0].location == "shoplite/shoplite/utils/ids.py:1-20"
    assert isinstance(hits[0].start_line, int)
    assert vector_index.search_by_meaning(index, "q", "other") == []  # other namespace is empty


def test_upload_refuses_mismatched_chunks_and_vectors():
    with pytest.raises(ValueError):
        vector_index.upload(FakeIndex(), [make_chunk()], [], "files")


def test_waiting_gives_up_with_a_clear_error(monkeypatch):
    monkeypatch.setattr(vector_index, "READY_TIMEOUT_S", 0)
    index = SimpleNamespace(describe_index_stats=lambda: SimpleNamespace(namespaces={}))
    with pytest.raises(TimeoutError, match="only 0 of 5 vectors"):
        vector_index.wait_until_searchable(index, "files", 5, sleep=lambda s: None)


def test_connect_without_a_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PINECONE_API_KEY is not set"):
        vector_index.connect()


def test_a_hit_is_any_passage_from_an_answering_file():
    sources = [{"repo": "shoplite", "path": "shoplite/utils/ids.py", "symbol": "order_number"}]
    assert evaluate.is_hit([make_hit("docs/api.md"), make_hit()], sources)
    assert not evaluate.is_hit([make_hit(repo="shoplite-warehouse")], sources)
    assert not evaluate.is_hit([], sources)


def test_recall_counts_hits_over_questions():
    questions = [{"id": "Q1", "question": "a", "sources": [{"repo": "shoplite",
                                                             "path": "shoplite/utils/ids.py"}]},
                 {"id": "Q2", "question": "b", "sources": [{"repo": "shoplite", "path": "x.py"}]}]
    result = evaluate.measure_recall(questions, lambda q: [make_hit()] * 7)
    assert (result["hits"], result["questions"], result["recall"]) == (1, 2, 0.5)
    assert len(result["rows"][0]["returned"]) == 5  # only the top 5 count


def test_save_run_writes_json(tmp_path):
    path = evaluate.save_run({"recall": 1.0}, "recall-files", tmp_path)
    assert path.name.endswith("-recall-files.json")
    assert '"recall": 1.0' in path.read_text()


def test_an_answer_hit_needs_the_answering_function_in_the_passage():
    sources = [{"repo": "shoplite", "path": "shoplite/utils/ids.py", "symbol": "order_number"},
               {"repo": "shoplite", "path": "docs/api.md"}]
    module_part = make_hit(text="import hashlib\n\nTOKEN_BYTES = 32")
    assert evaluate.is_hit([module_part], sources)              # the right file...
    assert not evaluate.is_answer_hit([module_part], sources)   # ...but not the answer
    assert evaluate.is_answer_hit([make_hit()], sources)        # def order_number is in it
    assert evaluate.is_answer_hit([make_hit("docs/api.md", text="# API")], sources)  # no symbol


def test_a_method_counts_by_its_own_name():
    assert evaluate.defines("class A:\n    def run(self):\n        pass", "A.run")
    assert not evaluate.defines("run = 1", "A.run")


def test_recall_reports_answer_hits_and_characters_sent():
    questions = [{"id": "Q1", "question": "a",
                  "sources": [{"repo": "shoplite", "path": "shoplite/utils/ids.py",
                               "symbol": "order_number"}]}]
    result = evaluate.measure_recall(questions, lambda q: [make_hit(), make_hit(text="x")])
    assert (result["hits"], result["answer_hits"]) == (1, 1)
    assert result["characters_mean"] == len(CODE) + 1


ANSWER = ('{"answered": true, "claims": [{"statement": "Order numbers look like '
          'SL-20260613-000042.", "passage": 1}], "refusal": ""}')
REFUSAL = '{"answered": false, "claims": [], "refusal": "Nothing about wishlists."}'


def test_the_prompt_numbers_the_passages_and_ends_with_the_question(fake_client):
    client = fake_client([ANSWER])
    result, usage = answer.answer_question(client, "What format do order numbers use?",
                                           [make_hit()])
    sent = client.histories[0][0].text
    assert sent.startswith("Passages:\n\n[1] shoplite/shoplite/utils/ids.py:18-20\n")
    assert sent.endswith("Question: What format do order numbers use?")
    assert result.answered and result.claims[0].passage == 1
    assert usage.input_tokens == 1000


def test_each_claim_is_rendered_with_its_file_and_lines():
    result = answer.parse_answer(ANSWER, 1)
    assert answer.render(result, [make_hit()]) == (
        "- Order numbers look like SL-20260613-000042.\n"
        "    ../shoplite/shoplite/utils/ids.py:18  (lines 18-20, order_number)")


def test_a_claim_from_an_old_version_says_so():
    hit = make_hit()
    hit.version, hit.latest = "1.1", False
    assert "(version 1.1, not on disk)" in answer.render(answer.parse_answer(ANSWER, 1), [hit])


def test_a_refusal_is_rendered_as_a_refusal():
    result = answer.parse_answer(REFUSAL, 5)
    assert answer.render(result, []) == "Not in the code I searched: Nothing about wishlists."


@pytest.mark.parametrize("reply, problem", [
    ('{"answered": true, "claims": [], "refusal": ""}', "answered, but with no claims"),
    ('{"answered": true, "claims": [{"statement": "x", "passage": 6}], "refusal": ""}',
     "cites passage 6, but there are 5"),
    ('{"answered": false, "claims": [], "refusal": "  "}', "must have no claims and say"),
    ('{"answered": false, "claims": [{"statement": "x", "passage": 1}], "refusal": "no"}',
     "must have no claims"),
    ('Shipping is free.', "not an Answer"),
    ('{"answered": true}', "not an Answer"),
])
def test_replies_that_break_the_contract_are_rejected(reply, problem):
    with pytest.raises(answer.ContractBroken, match=problem):
        answer.parse_answer(reply, 5)


def test_the_cache_asks_the_model_once_per_identical_request(fake_client, tmp_path):
    client = evaluate.CachedClient(fake_client([ANSWER, REFUSAL]), tmp_path)
    first, _ = answer.answer_question(client, "q", [make_hit()])
    second, usage = answer.answer_question(client, "q", [make_hit()])
    third, _ = answer.answer_question(client, "another q", [make_hit()])
    assert client.calls == 2
    assert first == second and usage.input_tokens == 1000  # the saved usage comes back
    assert not third.answered


def test_answer_and_refusal_rates_are_counted_separately():
    questions = [{"id": "Q1", "question": "a", "answerable": True},
                 {"id": "Q2", "question": "b", "answerable": True},
                 {"id": "Q3", "question": "c", "answerable": False}]
    replies = {"a": ANSWER, "b": REFUSAL, "c": "not json"}

    def fake_answer(question, hits):
        """Parse a scripted reply."""
        return answer.parse_answer(replies[question], len(hits)), evaluate.Usage(10, 2)

    result = evaluate.measure_answers(
        questions, evaluate.pipeline_responder(lambda q: [make_hit()], fake_answer))

    assert (result["answer_rate"], result["refusal_rate"], result["contract_broken"]) == (
        "1/2", "0/1", 1)
    assert result["rows"][0]["claims"][0]["where"] == "shoplite/shoplite/utils/ids.py:18-20"
    assert result["rows"][0]["sources"][0]["text"] == CODE  # kept for the judge
    assert result["usage"]["input_tokens"] == 20


def test_code_names_in_a_question_are_found():
    names = answer.names_in
    assert names("How does reserve_stock_for_cart() handle it?") == ["reserve_stock_for_cart"]
    assert names("Where is inv_reservation_ledger written to?") == ["inv_reservation_ledger"]
    assert answer.names_in("What does GatewayError mean?") == ["GatewayError"]
    assert answer.names_in("How much does shipping cost?") == []


def test_an_answer_about_a_name_no_source_mentions_becomes_a_refusal(fake_client):
    client = fake_client([ANSWER])  # the model answers anyway
    result, _ = answer.answer_question(client, "What does cancel_order() return?", [make_hit()])
    assert not result.answered
    assert result.refusal == "The code I searched has no cancel_order."


def test_a_name_the_sources_do_mention_is_left_alone(fake_client):
    client = fake_client([ANSWER])
    result, _ = answer.answer_question(client, "What does order_number() return?", [make_hit()])
    assert result.answered
