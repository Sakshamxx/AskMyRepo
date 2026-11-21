"""Both searches, fused by rank, filtered, then reranked."""

from askmyrepo import reranker, vector_index
from askmyrepo.keyword_index import KeywordIndex
from askmyrepo.vector_index import Hit

SETUPS = ("meaning", "fused", "reranked")
RRF_K = 60       # the usual constant in reciprocal rank fusion
CANDIDATES = 20  # how many each search hands on, and how many the reranker reads


# Reciprocal-rank fusion: merges meaning and keyword rankings.
def fuse_by_rank(*ranked_lists: list[Hit], k: int = RRF_K) -> list[Hit]:
    """Merge ranked lists: each passage scores 1/(k + rank) in every list it is in, summed.
    Only ranks are used, so a cosine score and a BM25 score never have to be compared."""
    fused: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            fused[hit.id] = fused.get(hit.id, 0.0) + 1 / (k + rank)
            first_seen.setdefault(hit.id, hit)
    order = sorted(fused, key=lambda hit_id: fused[hit_id], reverse=True)
    for hit_id in order:
        first_seen[hit_id].score = fused[hit_id]
    return [first_seen[hit_id] for hit_id in order]


# Converts our filter dict into Pinecone's filter syntax.
def pinecone_filter(filters: dict) -> dict | None:
    """Our filters in Pinecone's filter language."""
    clauses = {}
    if "repo" in filters:
        clauses["repo"] = {"$eq": filters["repo"]}
    if "folder" in filters:
        clauses["folders"] = {"$in": [filters["folder"]]}  # matches if the list contains it
    if "version" in filters:
        clauses["version"] = {"$eq": filters["version"]}
    if "latest" in filters:
        clauses["latest"] = {"$eq": filters["latest"]}
    return clauses or None


# The search pipeline: meaning + keyword, fused, filtered, reranked.
class Searcher:
    """Runs one of the three setups: meaning only, fused, or fused then reranked."""

    # Keeps the handles it needs for every search.
    def __init__(self, index, keywords: KeywordIndex, namespace: str):
        """Keep the vector index, the keyword index and the namespace."""
        self.index = index
        self.keywords = keywords
        self.namespace = namespace

    # Runs the chosen setup and returns the top passages.
    def search(self, question: str, setup: str = "reranked", filters: dict | None = None,
               top_k: int = 5, candidates: int = CANDIDATES) -> list[Hit]:
        """The top_k passages for a question, using the chosen setup. `candidates` is how many
        each search hands on, and how many the reranker reads (a short list, on purpose)."""
        filters = filters or {}
        if setup == "meaning":
            return vector_index.search_by_meaning(self.index, question, self.namespace, top_k,
                                                  pinecone_filter(filters))
        by_meaning = vector_index.search_by_meaning(self.index, question, self.namespace,
                                                    candidates, pinecone_filter(filters))
        by_keyword = self.keywords.search(question, candidates, filters)
        fused = fuse_by_rank(by_meaning, by_keyword)
        if setup == "fused":
            return fused[:top_k]
        return reranker.rerank_top(question, fused[:candidates], keep=top_k)
