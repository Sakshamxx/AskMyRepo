"""Tests for the SQL tool, the agent's tools and the agent answering with numbered sources."""

import sqlite3

import pytest

from askmyrepo import agent_tools
from askmyrepo.agent import Limits, run_agent
from askmyrepo.answer import Answer, ContractBroken
from askmyrepo.database import connect_read_only, run_select
from askmyrepo.tools import ToolError, run_tool
from askmyrepo.vector_index import Hit


@pytest.fixture
def db(tmp_path):
    """A small database with orders, opened read-only the way the tool opens it."""
    path = tmp_path / "shop.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE orders (id INTEGER, status TEXT)")
        conn.executemany("INSERT INTO orders VALUES (?, ?)",
                         [(i, "paid" if i % 2 else "shipped") for i in range(1, 121)])
    return connect_read_only(path)


def test_a_select_returns_columns_and_rows(db):
    result = run_select(db, "SELECT status, COUNT(*) AS n FROM orders GROUP BY status;")
    assert result == {"columns": ["status", "n"], "rows": [["paid", 60], ["shipped", 60]],
                      "truncated": False}


def test_rows_are_capped_and_the_cap_is_reported(db):
    result = run_select(db, "SELECT id FROM orders", max_rows=50)
    assert len(result["rows"]) == 50 and result["truncated"]


@pytest.mark.parametrize("sql, problem", [
    ("DELETE FROM orders", "only a single SELECT"),
    ("SELECT 1; DROP TABLE orders", "one statement only"),
    ("WITH x AS (SELECT 1) DELETE FROM orders", "not authorized"),
    ("PRAGMA table_info(orders)", "only a single SELECT"),
    ("SELECT * FROM missing", "no such table"),
])
def test_anything_but_reading_is_refused(db, sql, problem):
    with pytest.raises(ToolError, match=problem):
        run_select(db, sql)


def test_a_slow_query_is_stopped(db):
    slow = "SELECT COUNT(*) FROM orders a, orders b, orders c, orders d"
    with pytest.raises(ToolError, match="took longer than"):
        run_select(db, slow, timeout_s=0.1)
    assert run_select(db, "SELECT COUNT(*) FROM orders")["rows"] == [[120]]  # still usable


def test_a_missing_database_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="run ShopLite's scripts/seed.py"):
        connect_read_only(tmp_path / "nowhere.db")


def make_hit(name: str) -> Hit:
    """A passage from a file named after `name`."""
    return Hit(0.5, "shoplite", f"shoplite/{name}.py", "python", name, 1, 9, "abc1234",
               f"def {name}(): ...", "2.0")


class FakeSearcher:
    """Returns one passage per search and records the filters."""

    def __init__(self):
        """No searches yet."""
        self.calls = []

    def search(self, question, setup="reranked", filters=None, top_k=5):
        """One hit named after how many searches there have been."""
        self.calls.append(filters)
        return [make_hit(f"found{len(self.calls)}")]


def test_sources_are_numbered_across_both_tools(db):
    sources = agent_tools.Sources()
    searcher = FakeSearcher()
    tools = {t.name: t for t in agent_tools.make_tools(searcher, db, sources, {"latest": True})}

    first = run_tool(tools, "search_code", {"query": "orders table", "folder": "shoplite/models/"})
    query = run_tool(tools, "query_database", {"sql": "SELECT COUNT(*) FROM orders"})

    assert first["result"][0]["source"] == 1 and query["result"]["source"] == 2
    assert searcher.calls == [{"latest": True, "folder": "shoplite/models"}]
    assert sources.describe(1) == "../shoplite/shoplite/found1.py:1  (lines 1-9, found1)"
    assert sources.describe(2) == "SQL: SELECT COUNT(*) FROM orders"


def test_the_search_budget_is_enforced(db):
    tools = {t.name: t for t in agent_tools.make_tools(FakeSearcher(), db, agent_tools.Sources(),
                                                        max_searches=2)}
    results = [run_tool(tools, "search_code", {"query": "x"}) for _ in range(3)]
    assert "result" in results[1]
    assert "search budget of 2 spent" in results[2]["error"]


def test_a_sql_error_goes_back_to_the_model_as_an_error(db):
    tools = {t.name: t for t in agent_tools.make_tools(FakeSearcher(), db, agent_tools.Sources())}
    assert "only a single SELECT" in run_tool(tools, "query_database",
                                              {"sql": "DROP TABLE orders"})["error"]


def test_the_agent_searches_queries_and_answers_with_sources(db, fake_client):
    sources = agent_tools.Sources()
    tools = agent_tools.make_tools(FakeSearcher(), db, sources)
    final = ('{"answered": true, "claims": [{"statement": "60 orders are paid.", "passage": 2}, '
             '{"statement": "Statuses live in orders.status.", "passage": 1}], "refusal": ""}')
    client = fake_client([[("search_code", {"query": "order status column"})],
                          [("query_database", {"sql": "SELECT COUNT(*) FROM orders "
                                                      "WHERE status = 'paid'"})],
                          final])

    run = run_agent(client, "How many orders are paid?", tools, Answer, Limits(max_steps=5),
                    log=lambda line: None)

    assert run.stop_reason == "done" and run.tool_calls == 2
    agent_tools.check_agent_answer(run.answer, sources)
    assert agent_tools.render_agent_answer(run.answer, sources) == (
        "- 60 orders are paid.\n    SQL: SELECT COUNT(*) FROM orders WHERE status = 'paid'\n"
        "- Statuses live in orders.status.\n"
        "    ../shoplite/shoplite/found1.py:1  (lines 1-9, found1)")


def test_a_claim_citing_a_source_never_shown_breaks_the_contract():
    answer = Answer(answered=True, claims=[{"statement": "x", "passage": 3}], refusal="")
    with pytest.raises(ContractBroken, match="cites passage 3, but there are 0"):
        agent_tools.check_agent_answer(answer, agent_tools.Sources())


def test_the_agent_is_told_not_to_answer_about_something_similar():
    from askmyrepo.agent import SYSTEM_PROMPT
    assert "don't answer about something similar" in " ".join(SYSTEM_PROMPT.split())


def test_weak_passages_are_dropped_but_the_best_always_stays():
    def scored(name, score):
        hit = make_hit(name)
        hit.score = score
        return hit

    hits = [scored("a", 9.0), scored("b", 6.0), scored("c", 4.0)]
    assert [h.symbol for h in agent_tools.drop_weak(hits, gap=4.0)] == ["a", "b"]
    assert [h.symbol for h in agent_tools.drop_weak([scored("z", -3.0)], gap=1.0)] == ["z"]
    assert agent_tools.drop_weak([]) == []


def test_a_passage_already_shown_is_not_numbered_again(db):
    class SameHit:
        def search(self, query, setup="reranked", filters=None, top_k=5, candidates=20):
            return [make_hit("same")]

    sources = agent_tools.Sources()
    tools = {t.name: t for t in agent_tools.make_tools(SameHit(), db, sources)}
    first = run_tool(tools, "search_code", {"query": "x"})
    second = run_tool(tools, "search_code", {"query": "y"})
    assert first["result"][0]["source"] == 1
    assert "already shown" in second["result"][0]["note"] and len(sources.items) == 1
