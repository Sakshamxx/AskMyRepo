"""Our own judge: four measures of an answer, scored by a model; supporting quotes required."""

import re
from typing import Literal

from pydantic import BaseModel, Field

from askmyrepo.llm import Message, ModelClient, Usage

JUDGE_PROMPT = """You check answers given by a question-answering system about a codebase.
You get the question, the reference answer, the numbered sources the system
was shown, and the system's claims, each citing one source. Judge strictly:
- For each claim: is it supported by the source it cites? Supported means the source says it,
  not that it sounds plausible next to the source. If supported, copy the exact words from the
  source that say it into quote. If not, leave quote empty.
- answers_question: does the system's answer address the question asked (yes, partly or no)?
  A refusal answers a question only if the reference says there is no answer.
- relevant_sources: the numbers of the sources that help answer the question.
- reference_found: do the sources, taken together, contain the facts in the reference answer
  (all, some or none)?"""


# The judge's yes/no on one claim, with the quote it relied on.
class ClaimVerdict(BaseModel):
    """The judge's verdict on one claim."""

    claim: int = Field(description="The claim's number, counting from 1.")
    supported: bool
    quote: str = Field(description="The exact words in the cited source that support it, or "
                                   "empty.")


# Everything the judge returns for one answer.
class Judgement(BaseModel):
    """The judge's reply for one answer."""

    claims: list[ClaimVerdict]
    answers_question: Literal["yes", "partly", "no"]
    relevant_sources: list[int]
    reference_found: Literal["all", "some", "none"]


# Makes quotes and sources comparable (case, markdown, spacing).
def normalise(text: str) -> str:
    """Lower-case, without Markdown's ` and * marks, with runs of spaces made single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[`*]", "", text)).strip().lower()


# Code check that the judge's quote really appears in the source.
def quote_is_in(quote: str, source_text: str) -> bool:
    """True if the quote (at least a few words) really appears in the source."""
    quote = normalise(quote)
    return len(quote) >= 8 and quote in normalise(source_text)


# Extracts the numbers and names a claim asserts.
def specifics(text: str) -> set[str]:
    """The numbers and code-like names in a claim (with _ or a digit, CamelCase, or in `...`)."""
    found = set()
    for groups in re.findall(r"`([^`]+)`|\b([\w.]*[_\d][\w.]*|[A-Z][a-z]+[A-Z]\w*)\b", text):
        for part in groups:
            for piece in part.split("."):
                if piece and (re.search(r"[_\d]", piece) or re.match(r"[A-Z][a-z]+[A-Z]", piece)):
                    found.add(piece.lower())
    return found


# Code check that every number and name in a claim is in its source.
def names_are_in_source(claim: str, source: dict) -> bool:
    """True if every number and name in the claim appears in the source it cites."""
    where = source.get("location") or source.get("sql", "")
    text = normalise(f"{where} {source['text']}")
    return all(name in text for name in specifics(claim))


# Assembles question, answer, sources and reference for the judge.
def build_judge_prompt(row: dict, reference: str) -> str:
    """Everything the judge needs for one answer."""
    sources = "\n\n".join(f"[{n}] {s.get('location') or 'SQL: ' + s.get('sql', '')}\n{s['text']}"
                          for n, s in enumerate(row["sources"], start=1))
    if row["answered"]:
        claims = "\n".join(f"{n}. {c['statement']} (cites source [{c['source']}])"
                           for n, c in enumerate(row["claims"], start=1))
    else:
        claims = f"(no claims: the system refused: {row.get('refusal', '')})"
    return (f"Question: {row['question']}\n\nReference answer: {reference}\n\n"
            f"Sources:\n\n{sources or '(none)'}\n\nThe system's claims:\n{claims}")


# Turns a judgement into faithfulness, relevance, precision and recall.
def score_row(row: dict, judgement: Judgement, require_quote: bool = True) -> dict:
    """Turn the judge's reply into the four measures for one answer.
    With require_quote, a claim needs its quote, numbers and names to be in its source."""
    verdicts = {v.claim: v for v in judgement.claims}
    supported = []
    for n, claim in enumerate(row["claims"], start=1):
        verdict = verdicts.get(n)
        ok = bool(verdict and verdict.supported)
        if ok and require_quote:
            source = row["sources"][claim["source"] - 1]
            ok = (quote_is_in(verdict.quote, source["text"])
                  and names_are_in_source(claim["statement"], source))
        supported.append(ok)
    shown = len(row["sources"])
    cited = {claim["source"] for claim in row["claims"]}
    relevant = {n for n in judgement.relevant_sources if 1 <= n <= shown}
    # With no answer to find, "relevant sources" and "reference found" mean nothing: not scored.
    answerable = row.get("answerable", True)
    return {
        "claims_supported": supported,
        "faithfulness": sum(supported) / len(supported) if supported else None,
        "answer_relevance": {"yes": 1.0, "partly": 0.5, "no": 0.0}[judgement.answers_question],
        "context_precision": len(relevant) / shown if shown and answerable else None,
        # Of the sources the answer actually cites, how many did the judge call relevant?
        "cited_precision": (len(relevant & cited) / len(cited)
                            if cited and answerable else None),
        "context_recall": ({"all": 1.0, "some": 0.5, "none": 0.0}[judgement.reference_found]
                           if answerable else None),
    }


# One judge call plus scoring for a single answer.
def judge_row(client: ModelClient, row: dict, reference: str,
              require_quote: bool = True) -> tuple[dict, Judgement, Usage]:
    """Ask the judge about one answer and score it."""
    prompt = build_judge_prompt(row, reference)
    reply = client.reply(JUDGE_PROMPT, [Message(role="user", text=prompt)], tools=[],
                         output_schema=Judgement)
    judgement = Judgement.model_validate_json(reply.text)
    return score_row(row, judgement, require_quote), judgement, reply.usage


# Average that ignores questions where a measure doesn't apply.
def mean(values: list) -> float | None:
    """The average of the values that aren't None."""
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 3) if present else None


# Averages the four measures over a whole run.
def summarise(scores: list[dict]) -> dict:
    """The four measures averaged over a run."""
    return {measure: mean([s[measure] for s in scores])
            for measure in ("faithfulness", "answer_relevance", "context_precision",
                            "context_recall")}


# How far the judge is from the hand labels (the check on the judge).
def disagreement(judge_labels: dict[str, bool], hand_labels: dict[str, bool]) -> dict:
    """How often the judge and the hand labels disagree on "is this claim supported?"."""
    shared = sorted(set(judge_labels) & set(hand_labels))
    differ = [key for key in shared if judge_labels[key] != hand_labels[key]]
    return {"checked": len(shared), "disagreed": len(differ),
            "rate": round(len(differ) / len(shared), 3) if shared else None, "which": differ}
