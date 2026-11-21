"""Run the labelled questions: recall at 5 (the right file, and the answer in a passage), and
the answer and refusal rates."""

import hashlib
import json
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from askmyrepo.answer import ContractBroken
from askmyrepo.llm import Message, ModelReply, Usage
from askmyrepo.vector_index import Hit


# Reads the labelled questions file.
def load_questions(path: Path) -> list[dict]:
    """Read questions.jsonl: one {"id", "question", "sources"} per line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# Recall check: did a returned passage come from an answering file?
def is_hit(hits: list[Hit], sources: list[dict]) -> bool:
    """True if any returned passage comes from one of the files that answer the question."""
    wanted = {(source["repo"], source["path"]) for source in sources}
    return any((hit.repo, hit.path) in wanted for hit in hits)


# Checks whether a passage actually defines the named function or class.
def defines(text: str, symbol: str) -> bool:
    """True if the text defines the function or class (for "Class.method", the method)."""
    name = re.escape(symbol.split(".")[-1])
    return re.search(rf"^\s*(async\s+)?(def|class)\s+{name}\b", text, re.MULTILINE) is not None


# Stricter check: the answer itself must be inside a returned passage.
def is_answer_hit(hits: list[Hit], sources: list[dict]) -> bool:
    """True if a returned passage holds what answers: the named function or class of a source,
    or, for a source with no symbol (docs, SQL), any passage of that file."""
    for source in sources:
        for hit in hits:
            if (hit.repo, hit.path) == (source["repo"], source["path"]):
                if "symbol" not in source or defines(hit.text, source["symbol"]):
                    return True
    return False


# Runs every question through search and counts the hits.
def measure_recall(questions: list[dict], search, top_k: int = 5) -> dict:
    """Search every question; search(question) returns hits, best first."""
    rows = []
    for q in questions:
        started = time.perf_counter()
        hits = search(q["question"])[:top_k]
        rows.append({"id": q["id"], "question": q["question"], "hit": is_hit(hits, q["sources"]),
                     "answer_hit": is_answer_hit(hits, q["sources"]),
                     "characters": sum(len(h.text) for h in hits),
                     "seconds": round(time.perf_counter() - started, 3),
                     "returned": [{"location": h.location, "symbol": h.symbol,
                                   "score": round(h.score, 4)} for h in hits]})
    hits_count = sum(row["hit"] for row in rows)
    answer_hits = sum(row["answer_hit"] for row in rows)
    characters = [row["characters"] for row in rows]
    return {"top_k": top_k, "questions": len(rows), "hits": hits_count, "answer_hits": answer_hits,
            "recall": round(hits_count / len(rows), 3) if rows else 0.0,
            "characters_mean": round(sum(characters) / len(characters)) if rows else 0,
            "rows": rows}


# Writes a run to runs/ so every reported number has a file behind it.
def save_run(result: dict, name: str, runs_dir: Path = Path("runs")) -> Path:
    """Save a run as runs/<date-time>-<name>.json and return the path."""
    runs_dir.mkdir(exist_ok=True)
    path = runs_dir / f"{datetime.now():%Y-%m-%d-%H%M%S}-{name}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# Saves model replies to disk so repeated runs cost nothing.
class CachedClient:
    """Wraps a ModelClient so an identical request is answered from disk, not asked twice."""

    # Wraps the real client and sets the cache folder.
    def __init__(self, client, cache_dir: Path = Path("runs/cache")):
        """Keep the real client and where the saved replies go."""
        self.client = client
        self.model = client.model
        self.cache_dir = cache_dir
        self.calls = 0  # real calls made, for staying inside the free quota

    # Hashes the whole request so identical requests share one saved reply.
    def key(self, system: str, history: list, output_schema) -> str:
        """A fingerprint of everything that decides the reply."""
        parts = [self.model, system, output_schema.__name__ if output_schema else ""]
        parts += [f"{m.role}:{m.text}" for m in history]
        return hashlib.sha256("\n\x00".join(parts).encode()).hexdigest()[:32]

    # Returns the saved reply, or asks the model and saves it.
    def reply(self, system, history, tools, output_schema=None) -> ModelReply:
        """The saved reply if there is one; otherwise ask, save and return it."""
        if tools:  # tool turns aren't cached
            return self.client.reply(system, history, tools, output_schema)
        path = self.cache_dir / f"{self.key(system, history, output_schema)}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            text = saved["text"]  # the usage is the original call's, so costs stay comparable
            return ModelReply(text, [], Usage(**saved["usage"]), Message(role="model", text=text))
        reply = self.client.reply(system, history, tools, output_schema)
        self.calls += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.model, "text": reply.text,
                                    "usage": asdict(reply.usage)}), encoding="utf-8")
        return reply

    # Passes straight through; token counts are not cached.
    def count_tokens(self, history) -> int:
        """Not cached."""
        return self.client.count_tokens(history)


# Adapts the one-search pipeline to the evaluator's respond() shape.
def pipeline_responder(search, answer_fn):
    """respond(question) for the pipeline: search, then answer from the passages."""
    # Answers one question and reports the sources used.
    def respond(question: str) -> dict:
        """Answer one question and describe its sources."""
        hits = search(question)
        answer, usage = answer_fn(question, hits)
        return {"answer": answer, "usage": usage,
                "sources": [{"location": h.location, "version": h.version, "text": h.text}
                            for h in hits]}
    return respond


# Computes the answer rate and the refusal rate over the labelled questions.
def measure_answers(questions: list[dict], respond) -> dict:
    """Answer every question; count answers on answerable ones and refusals on the rest.
    respond(question) returns {"answer", "usage", "sources"}, or raises ContractBroken."""
    rows, usage = [], Usage()
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "answerable": q["answerable"]}
        try:
            response = respond(q["question"])
            answer, sources = response["answer"], response["sources"]
            usage.add(response["usage"])
            row.update(answered=answer.answered, refusal=answer.refusal, contract="kept",
                       claims=[{"statement": c.statement, "source": c.passage,
                                "where": sources[c.passage - 1].get("location")
                                or sources[c.passage - 1].get("sql")}
                               for c in answer.claims],
                       sources=sources, steps=response.get("steps"))
        except ContractBroken as exc:
            row.update(answered=None, contract=f"broken: {exc}", claims=[], sources=[])
        rows.append(row)
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    return {
        "answer_rate": f"{sum(r['answered'] is True for r in answerable)}/{len(answerable)}",
        "refusal_rate": f"{sum(r['answered'] is False for r in unanswerable)}/{len(unanswerable)}",
        "contract_broken": sum(r["contract"] != "kept" for r in rows),
        "usage": asdict(usage), "rows": rows}
