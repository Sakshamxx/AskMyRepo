"""Baseline: paste the whole repository into one prompt and ask.
It is the approach retrieval replaces, measured first so later steps have a number to beat."""

import time

from askmyrepo.llm import Message, ModelClient, Usage
from askmyrepo.manifest import CorpusFile

SYSTEM_PROMPT = "Answer the question using only the repository below. Name the file you used."


# Concatenates every corpus file into one big prompt.
def repository_text(files: list[CorpusFile]) -> str:
    """Every file in the manifest, one after another, each under its repository and path."""
    return "\n\n".join(f"=== {f.repo}/{f.path} ===\n{f.text}" for f in files)


# Asks the provider for the exact token count of that prompt.
def count_repository_tokens(client: ModelClient, files: list[CorpusFile]) -> int:
    """Ask the provider how many input tokens the whole repository is (its own counting call)."""
    return client.count_tokens([Message(role="user", text=repository_text(files))])


# Runs the baseline question and measures answer, tokens, cost and time.
def ask_whole_repository(client: ModelClient, files: list[CorpusFile],
                         question: str) -> tuple[str, Usage, float]:
    """Paste the whole repository and ask; return the answer, the usage and the seconds."""
    started = time.perf_counter()
    prompt = f"{repository_text(files)}\n\nQuestion: {question}"
    reply = client.reply(SYSTEM_PROMPT, [Message(role="user", text=prompt)], tools=[])
    return reply.text.strip(), reply.usage, time.perf_counter() - started
