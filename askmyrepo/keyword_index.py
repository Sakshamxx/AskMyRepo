"""The BM25 keyword index over the same chunks: somewhere for an exact name to match."""

import json
from pathlib import Path

import bm25s

from askmyrepo.chunker import Chunk
from askmyrepo.vector_index import Hit

INDEX_DIR = Path("index")  # local files, rebuilt by `index`; not in Git
# Words of two or more characters, keeping _ and - inside them, so names such as
# inv_reservation_ledger and error codes such as SL-ORD-409 stay one word. (bm25s's default
# splits at "-", which turns SL-ORD-409 into "sl", "ord" and "409": words found everywhere.)
TOKEN_PATTERN = r"(?u)\b\w[\w-]*\w\b"


# Keeps names like inv_reservation_ledger and SL-ORD-409 as single words.
def tokenize(texts: list[str]) -> list[list[str]]:
    """Lower-case words; code names and error codes stay whole."""
    return bm25s.tokenize(texts, token_pattern=TOKEN_PATTERN, stopwords="en", return_ids=False,
                          show_progress=False)


# Writes the chunks the keyword index will be built from.
def save_chunks(chunks: list[Chunk], namespace: str, index_dir: Path = INDEX_DIR) -> Path:
    """Save the chunks the keyword index is built from (one JSON object per line)."""
    index_dir.mkdir(exist_ok=True)
    path = index_dir / f"chunks-{namespace}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.metadata(), ensure_ascii=False) + "\n")
    return path


# BM25 search over one namespace's chunks, for exact-name matches.
class KeywordIndex:
    """BM25 scores for every chunk of one namespace."""

    # Loads the saved chunks and builds the BM25 index in memory.
    def __init__(self, namespace: str, index_dir: Path = INDEX_DIR):
        """Load the saved chunks and build the BM25 index from them."""
        path = index_dir / f"chunks-{namespace}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `index` first")
        self.chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokenize([chunk["text"] for chunk in self.chunks]), show_progress=False)

    # Top BM25 matches that pass the repo/folder/version filters.
    def search(self, question: str, top_k: int = 20, filters: dict | None = None) -> list[Hit]:
        """The top_k chunks by BM25 score that pass the filters, best first."""
        scores = self.bm25.get_scores(tokenize([question])[0])
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked:
            if scores[i] <= 0:
                break  # no word of the question is in the rest
            chunk = self.chunks[i]
            if passes(chunk, filters or {}):
                fields = {key: value for key, value in chunk.items() if key != "folders"}
                hits.append(Hit(score=float(scores[i]), **fields))
                if len(hits) == top_k:
                    break
        return hits


# Applies the same filters as Pinecone so both searches agree.
def passes(chunk: dict, filters: dict) -> bool:
    """The same filters as the vector store: repository, folder, version, latest."""
    if "repo" in filters and chunk["repo"] != filters["repo"]:
        return False
    if "folder" in filters and filters["folder"] not in chunk["folders"]:
        return False
    if "version" in filters and chunk["version"] != filters["version"]:
        return False
    return not ("latest" in filters and chunk["latest"] != filters["latest"])
