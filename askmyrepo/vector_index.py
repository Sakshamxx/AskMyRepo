"""The Pinecone index: create it, upload passages, and search it by meaning."""

import os
import time
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

from askmyrepo.chunker import Chunk
from askmyrepo.embedder import DIMENSIONS, embed_question

INDEX_NAME = "askmyrepo"
METRIC = "cosine"           # the metric Qwen3-Embedding was trained for
CLOUD, REGION = "aws", "us-east-1"  # the only region on Pinecone's free Starter plan
UPLOAD_BATCH = 100
READY_TIMEOUT_S = 60


# One search result: a passage, its location and its score.
@dataclass
class Hit:
    """One passage that came back from a search, with its score."""

    score: float
    repo: str
    path: str
    language: str
    symbol: str
    start_line: int
    end_line: int
    commit: str
    text: str
    version: str = ""
    latest: bool = True

    # path:lines string shown to users.
    @property
    def location(self) -> str:
        """Where the passage is, e.g. shoplite/shoplite/services/stock.py:43-53."""
        return f"{self.repo}/{self.path}:{self.start_line}-{self.end_line}"

    # Same id as the chunk it came from (used to dedupe and fuse).
    @property
    def id(self) -> str:
        """The same id as the chunk it came from."""
        return f"{self.repo}@{self.version}:{self.path}:{self.start_line}-{self.end_line}"


# Creates the Pinecone client from PINECONE_API_KEY.
def connect() -> Pinecone:
    """A Pinecone client, using PINECONE_API_KEY."""
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set (copy .env.example to .env)")
    return Pinecone(api_key=api_key)


# Opens the 'askmyrepo' index, creating it on first use.
def open_index(pc: Pinecone, name: str = INDEX_NAME, metric: str = METRIC):
    """Open the index, creating it first if it doesn't exist."""
    if not pc.has_index(name):
        pc.create_index(name=name, dimension=DIMENSIONS, metric=metric,
                        spec=ServerlessSpec(cloud=CLOUD, region=REGION))
    return pc.Index(name)


# Empties a namespace so a re-index starts clean.
def clear_namespace(index, namespace: str) -> None:
    """Delete every vector in one namespace, so a reindex starts from nothing."""
    try:
        index.delete(delete_all=True, namespace=namespace)
    except NotFoundException:
        pass  # the namespace doesn't exist yet


# Stores each chunk's vector with its metadata and text.
def upload(index, chunks: list[Chunk], vectors: list[list[float]], namespace: str) -> None:
    """Store each chunk's vector with its metadata (where it came from, and its text)."""
    records = [{"id": chunk.id, "values": vector, "metadata": chunk.metadata()}
               for chunk, vector in zip(chunks, vectors, strict=True)]
    for start in range(0, len(records), UPLOAD_BATCH):
        index.upsert(vectors=records[start:start + UPLOAD_BATCH], namespace=namespace)


# Pinecone writes are eventually consistent; wait until all are visible.
def wait_until_searchable(index, namespace: str, expected: int, sleep=time.sleep) -> None:
    """Wait until Pinecone reports every uploaded vector (new writes take a moment to appear)."""
    deadline = time.monotonic() + READY_TIMEOUT_S
    while True:
        stats = index.describe_index_stats()
        count = stats.namespaces[namespace].vector_count if namespace in stats.namespaces else 0
        if count >= expected:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"only {count} of {expected} vectors visible after "
                               f"{READY_TIMEOUT_S} s")
        sleep(2)


# Nearest-neighbour search by embedding, optionally filtered.
def search_by_meaning(index, question: str, namespace: str, top_k: int = 5,
                      metadata_filter: dict | None = None) -> list[Hit]:
    """The top_k passages closest in meaning to the question, best first (optionally filtered)."""
    result = index.query(vector=embed_question(question), top_k=top_k, namespace=namespace,
                         filter=metadata_filter, include_metadata=True)
    return [to_hit(match.score, match.metadata) for match in result.matches]


# Rebuilds a Hit from Pinecone's stored metadata.
def to_hit(score: float, metadata: dict) -> Hit:
    """Build a Hit; Pinecone returns stored numbers as floats, so line numbers become ints again."""
    fields = {key: value for key, value in metadata.items() if key != "folders"}
    fields["start_line"] = int(fields["start_line"])
    fields["end_line"] = int(fields["end_line"])
    return Hit(score=score, **fields)
