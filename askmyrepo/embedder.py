"""Turn text into vectors with the local Qwen3-Embedding-0.6B model (no key, no per-call cost)."""

from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DIMENSIONS = 1024     # every vector has this many numbers; the Pinecone index must match
MAX_TOKENS = 8192     # longer texts are cut; this bounds memory on a laptop

# With no GPU (a Windows laptop, for instance) the model runs on the processor, where its default
# number format (bfloat16) is very slow: measured on an M1 Pro's CPU, 24 passages took 227 s in
# bfloat16 and 19 s in float32 with batches of 4. On a GPU the default format is kept.
HAS_GPU = torch.cuda.is_available() or torch.backends.mps.is_available()
BATCH_SIZE = 8 if HAS_GPU else 4


# Loads the local Qwen3 embedding model once and reuses it.
@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """Load the model once (the first call downloads it, about 1.2 GB)."""
    options = {} if HAS_GPU else {"model_kwargs": {"dtype": torch.float32}}
    model = SentenceTransformer(MODEL_NAME, **options)
    model.max_seq_length = MAX_TOKENS
    return model


# Embeds passages for storage (no query instruction).
def embed_documents(texts: list[str], model=None) -> list[list[float]]:
    """Vectors for passages to store. Passages get no instruction prompt."""
    model = model or load_model()
    vectors = model.encode_document(texts, batch_size=BATCH_SIZE, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


# Embeds a question with the model's 'query' prompt.
def embed_question(question: str, model=None) -> list[float]:
    """The vector for a question. Questions get the model's "query" instruction prompt."""
    model = model or load_model()
    return model.encode_query([question], normalize_embeddings=True)[0].tolist()
