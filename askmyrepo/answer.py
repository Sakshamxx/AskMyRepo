"""The answer contract: every claim carries its source, or the answer is a refusal."""

import re

from pydantic import BaseModel, Field, ValidationError

from askmyrepo.llm import Message, ModelClient, Usage
from askmyrepo.vector_index import Hit

SYSTEM_PROMPT = """You answer questions about a codebase (ShopLite and its warehouse service).
Use only the numbered passages you are given; never use what you know from elsewhere.
- If the passages answer the question: set answered to true and give one claim per fact, each
  with the number of the passage that supports it. Keep claims short and specific.
- If they don't: set answered to false, give no claims, and say in refusal what is missing.
  Don't guess, and don't describe something similar instead."""


# One statement plus the number of the source that supports it.
class Claim(BaseModel):
    """One fact in the answer, and the passage it comes from."""

    statement: str = Field(description="One fact, in one sentence.")
    passage: int = Field(description="The number of the passage that supports the fact.")


# The only shape an answer may take: sourced claims, or a refusal.
class Answer(BaseModel):
    """The fixed shape every answer must have."""

    answered: bool = Field(description="True only if the passages answer the question.")
    claims: list[Claim] = Field(
        description="The facts, each with its passage. Empty if not answered.")
    refusal: str = Field(description="If not answered: what the passages don't cover. Else empty.")


# Raised when the model's reply breaks the rules above.
class ContractBroken(Exception):
    """The model's reply doesn't keep the answer contract."""


# Numbers passages [1], [2]... so the model can cite them.
def label_passages(hits: list[Hit]) -> str:
    """Number each passage and say where it comes from, so a claim can point at it."""
    return "\n\n".join(
        f"[{n}] {hit.location}" + (f" (version {hit.version})" if hit.version else "")
        + f"\n{hit.text}" for n, hit in enumerate(hits, start=1))


# Puts the passages before the question in the user message.
def build_prompt(question: str, hits: list[Hit]) -> str:
    """The passages first, then the question."""
    return f"Passages:\n\n{label_passages(hits)}\n\nQuestion: {question}"


# Enforces the contract: every claim cites a real source, or it is a refusal.
def check_contract(answer: Answer, passage_count: int) -> None:
    """Raise ContractBroken unless the answer is either sourced claims or a refusal."""
    if answer.answered:
        if not answer.claims:
            raise ContractBroken("answered, but with no claims")
        for claim in answer.claims:
            if not 1 <= claim.passage <= passage_count:
                raise ContractBroken(f"claim cites passage {claim.passage}, "
                                     f"but there are {passage_count}")
    elif answer.claims or not answer.refusal.strip():
        raise ContractBroken("a refusal must have no claims and say what is missing")


CODE_NAME = re.compile(r"\b([A-Za-z_]\w*)\(\)|\b([a-z]\w*_\w+)\b|\b([A-Z][a-z]+[A-Z]\w*)\b")


# Finds function/class-like names in a question (e.g. get_stock()).
def names_in(question: str) -> list[str]:
    """Code names a question asks about: name(), snake_case or CamelCase words."""
    return [next(g for g in match if g) for match in CODE_NAME.findall(question)]


# Names asked about that appear in none of the sources.
def missing_names(question: str, source_texts: list[str]) -> list[str]:
    """The code names in the question that no source contains."""
    everything = " ".join(source_texts).lower()
    return [name for name in names_in(question) if name.lower() not in everything]


# Turns an answer about a non-existent name into a refusal.
def guard_names(answer: Answer, question: str, source_texts: list[str]) -> Answer:
    """An answer can't be about a name its sources never mention: turn it into a refusal."""
    missing = missing_names(question, source_texts)
    if answer.answered and missing:
        return Answer(answered=False, claims=[],
                      refusal=f"The code I searched has no {', '.join(missing)}.")
    return answer


# Parses the model's JSON and checks it against the contract.
def parse_answer(text: str, passage_count: int) -> Answer:
    """Read the model's JSON into an Answer and check the contract."""
    try:
        answer = Answer.model_validate_json(text)
    except ValidationError as exc:
        raise ContractBroken(f"not an Answer: {exc.errors()[0]['msg']}") from exc
    check_contract(answer, passage_count)
    return answer


# Pipeline mode: one model call over the passages already found.
def answer_question(client: ModelClient, question: str, hits: list[Hit]) -> tuple[Answer, Usage]:
    """Ask the model for an Answer from these passages; returns it and the tokens used."""
    reply = client.reply(SYSTEM_PROMPT, [Message(role="user", text=build_prompt(question, hits))],
                         tools=[], output_schema=Answer)
    answer = parse_answer(reply.text, len(hits))
    return guard_names(answer, question, [hit.text for hit in hits]), reply.usage


# Builds the clickable path:line shown under each claim.
def link(hit: Hit) -> str:
    """A path and line an editor can open, e.g. ../shoplite/shoplite/services/stock.py:43."""
    if not hit.latest:
        return f"{hit.repo}/{hit.path}:{hit.start_line} (version {hit.version}, not on disk)"
    return f"../{hit.repo}/{hit.path}:{hit.start_line}"


# Formats a pipeline answer for the terminal.
def render(answer: Answer, hits: list[Hit]) -> str:
    """Each claim with the file and lines under it, or the refusal."""
    if not answer.answered:
        return f"Not in the code I searched: {answer.refusal}"
    lines = []
    for claim in answer.claims:
        hit = hits[claim.passage - 1]
        symbol = f", {hit.symbol}" if hit.symbol else ""
        lines.append(f"- {claim.statement}\n    {link(hit)}  (lines {hit.start_line}-"
                     f"{hit.end_line}{symbol})")
    return "\n".join(lines)
