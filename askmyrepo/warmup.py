"""Load the embedding and reranking models in a background thread, once per process.
Started by serve.py so the first visitor doesn't wait for them (the load takes about a minute)."""

import threading
import time

_started = False  # guards against starting the thread twice


# Loads both local models (cached for the whole process), then logs how long it took.
def _load() -> None:
    """Load the embedder and the reranker so later calls find them already in memory."""
    from askmyrepo import embedder, reranker

    began = time.perf_counter()
    embedder.load_model()
    reranker.load_model()
    print(f"models ready in {time.perf_counter() - began:.0f} s", flush=True)


# Starts the background load once; safe to call more than once.
def start() -> None:
    """Begin loading the models without blocking the caller."""
    global _started
    if not _started:
        _started = True
        threading.Thread(target=_load, name="model-warmup", daemon=True).start()
