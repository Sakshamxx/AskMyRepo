"""The agent's two tools, search_code and query_database, and the numbered sources they return."""

import sqlite3

from askmyrepo.agent import Limits, run_agent
from askmyrepo.answer import Answer, ContractBroken, check_contract, guard_names, link
from askmyrepo.database import run_select
from askmyrepo.search import Searcher
from askmyrepo.tools import Tool, ToolError
from askmyrepo.tracing import Tracer
from askmyrepo.vector_index import Hit

MAX_SEARCHES = 3  # the search budget for one question (was 4; fewer searches, fewer passages shown)
PASSAGES_PER_SEARCH = 3  # was 5: context precision counts every passage the agent was shown
SCORE_GAP = 8.0  # drop passages this far below the best in a search; scores run about -4 to 6,
# and a gap of 4 dropped a passage that held the answer, so this only removes the clearly unrelated


# The numbered list of everything shown to the agent, so claims can cite [n].
class Sources:
    """Everything the agent was shown in one run, numbered [1], [2]... in the order it saw them."""

    # Starts empty for each question.
    def __init__(self):
        """No sources yet."""
        self.items: list[Hit | dict] = []  # a Hit for a passage, a dict for a query result

    # Numbers a new passage or query result in the order it was shown.
    def add(self, item) -> int:
        """Number a new source and return its number."""
        self.items.append(item)
        return len(self.items)

    # Lets search_code skip passages the agent has already seen.
    def has(self, hit: Hit) -> bool:
        """True if this exact passage (same file, same lines, same version) was already shown."""
        return any(isinstance(item, Hit) and (item.repo, item.path, item.start_line, item.end_line,
                                              item.version) == (hit.repo, hit.path, hit.start_line,
                                                                hit.end_line, hit.version)
                   for item in self.items)

    # Turns a source number into the file:lines (or SQL) shown to the user.
    def describe(self, number: int) -> str:
        """Where source [number] came from: a file and lines, or the SQL that produced it."""
        item = self.items[number - 1]
        if isinstance(item, Hit):
            symbol = f", {item.symbol}" if item.symbol else ""
            return f"{link(item)}  (lines {item.start_line}-{item.end_line}{symbol})"
        return f"SQL: {item['sql']}"


# Filter on the reranker score so clearly unrelated passages aren't shown.
def drop_weak(hits: list[Hit], gap: float = SCORE_GAP) -> list[Hit]:
    """Keep the best passage and any within `gap` of its (reranker) score; hits are best first."""
    if not hits:
        return hits
    return [hit for hit in hits if hit.score >= hits[0].score - gap]


# Creates the two tools for one question; they share one source list and search budget.
def make_tools(searcher: Searcher, conn: sqlite3.Connection, sources: Sources,
               filters: dict | None = None, max_searches: int = MAX_SEARCHES) -> list[Tool]:
    """Build the two tools for one question; they share its sources and its search budget."""
    searches = {"done": 0}

    # Tool 1: search the indexed code, schema and docs.
    def search_code(query: str, repo: str = "", folder: str = "") -> list[dict]:
        """Search the code, schema and docs: up to 3 numbered passages, best first, none repeated.
        Optional repo ("shoplite", "shoplite-warehouse") and folder ("shoplite/services") filter."""
        if searches["done"] >= max_searches:
            raise ToolError(f"search budget of {max_searches} spent: answer with what you have")
        searches["done"] += 1
        narrowed = dict(filters or {})
        if repo:
            narrowed["repo"] = repo
        if folder:
            narrowed["folder"] = folder.strip("/")
        hits = searcher.search(query, filters=narrowed, top_k=PASSAGES_PER_SEARCH)
        hits = drop_weak(hits)
        fresh = [hit for hit in hits if not sources.has(hit)]
        if hits and not fresh:
            return [{"note": "every passage this search found was already shown: search with "
                             "other words, or answer with what you have"}]
        return [{"source": sources.add(hit), "location": hit.location, "version": hit.version,
                 "text": hit.text} for hit in fresh]

    # Tool 2: run one read-only SELECT on ShopLite's database.
    def query_database(sql: str) -> dict:
        """Run one read-only SELECT on ShopLite's SQLite database (at most 50 rows, 2 seconds).
        Use real table and column names from the schema. Returns a numbered result."""
        result = run_select(conn, sql)
        return {"source": sources.add({"sql": sql, **result}), **result}

    return [Tool.from_function(search_code), Tool.from_function(query_database)]


# Formats the final answer for the terminal.
def render_agent_answer(answer: Answer, sources: Sources) -> str:
    """Each claim with its file and lines, or the SQL behind it; or the refusal."""
    if not answer.answered:
        return f"Not in the code or data I searched: {answer.refusal}"
    return "\n".join(f"- {claim.statement}\n    {sources.describe(claim.passage)}"
                     for claim in answer.claims)


# Rejects any claim that cites a source number the agent never saw.
def check_agent_answer(answer: Answer, sources: Sources) -> None:
    """The same contract as the pipeline: every claim cites a source the agent was shown."""
    check_contract(answer, len(sources.items))


# Public entry point: run the agent, check the contract, apply the name guard.
def ask_agent(client, searcher: Searcher, conn: sqlite3.Connection, question: str,
              filters: dict | None = None, limits: Limits | None = None, log=print,
              tracer: Tracer | None = None) -> dict:
    """Run the agent on one question; return its checked answer, usage, sources and steps."""
    sources = Sources()
    tools = make_tools(searcher, conn, sources, filters)
    tracer = tracer or Tracer(enabled=False)
    with tracer.run("askmyrepo ask", question) as record:
        run = run_agent(client, question, tools, Answer, limits or Limits(max_steps=8), log=log,
                        tracer=tracer)
        record(output=run.answer.model_dump() if run.answer else run.stop_reason,
               metadata={"stop_reason": run.stop_reason, "steps": run.steps,
                         "cost_usd": run.cost_usd})
    if run.answer is None:
        raise ContractBroken(f"no answer: {run.stop_reason}")
    check_agent_answer(run.answer, sources)
    texts = [item.text if isinstance(item, Hit) else str(item) for item in sources.items]
    final = guard_names(run.answer, question, texts)
    described = [{"location": item.location, "version": item.version, "text": item.text}
                 if isinstance(item, Hit) else {"sql": item["sql"], "text": str(item["rows"])}
                 for item in sources.items]
    return {"answer": final, "usage": run.usage, "sources": described, "steps": run.steps,
            "run": run, "source_log": sources}
