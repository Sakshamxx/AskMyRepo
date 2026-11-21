# AskMyRepo

Ask a codebase **and its database** a question in plain English. Every claim in the answer comes with the file and lines (or the SQL) it was based on, and if the sources don't hold the answer, AskMyRepo says so instead of guessing.

It reads **ShopLite** (a demo online shop: a Flask JSON API) and **shoplite-warehouse** (its stock-sync service). Built across Sessions 7–12 of the applied-AI bootcamp.

```
You ask ──▶ agent ──▶ search_code  (code, schema, docs)  ──┐
                 └──▶ query_database (read-only SELECT)  ──┤
                                                           ▼
                          answer = claims, each citing [source n]  (or a refusal)
```

---

## 1. How it works

**Indexing** (`index` command, run once)

1. `manifest.py` reads `corpus.toml`: which repositories, which versions (ShopLite 2.0 and 1.1, warehouse 1.0), which files, and what to skip. Files are read from **Git**, so virtual environments and databases never get in.
2. `chunker.py` cuts each file into passages on function, class and heading boundaries (tree-sitter), so a citation points at about a dozen lines, not a whole file.
3. `embedder.py` turns each passage into a vector with the local **Qwen3-Embedding-0.6B** model (no per-call cost).
4. `vector_index.py` uploads the vectors to **Pinecone**; `keyword_index.py` saves the same passages for a local **BM25** index.

**Searching** (`search.py`)

1. A search by meaning (Pinecone) and a keyword search (BM25, good at exact names such as `inv_reservation_ledger` or `SL-ORD-409`).
2. **Rank fusion** merges the two lists.
3. **Filters**: only the newest version by default (so stale 1.1 code doesn't win), optionally by repository or folder.
4. `reranker.py` re-orders the best candidates with the local **Qwen3-Reranker-0.6B**.

**Answering** (`ask`)

- **Agent mode (default)**, `agent.py` + `agent_tools.py`: a hand-written loop (think → act → read the result → repeat). Two tools:
  - `search_code`: the search above, a few numbered passages per call, within a search budget.
  - `query_database`: one read-only `SELECT` on ShopLite's SQLite file, guarded in three layers (read-only connection, SQLite authorizer, row and time limits) in `database.py`.
- **Pipeline mode (`--pipeline`)**, `answer.py`: one search, then one model call. Cheaper, but it can't use the database.
- **The answer contract** (`answer.py`): every claim must cite a source the model was actually shown, or the answer is a refusal. A **name guard** turns an answer about a function that no source mentions into a refusal.

**Measuring** (`evaluate.py`, `judge.py`)

- `evaluate` runs the 40 labelled questions in `questions.jsonl`: recall at 5, and the **answer rate** (answerable questions answered) and **refusal rate** (unanswerable ones refused).
- `judge` scores a saved run on four measures with a model as judge, then checks its quotes, numbers and names in code, and compares it with hand labels:

| Measure | Plain words | Also called |
|---|---|---|
| faithfulness | does the cited source say what the claim says | supported |
| answer_relevance | does the answer address the question | answers the question |
| context_precision | share of shown sources that help | relevant sources |
| context_recall | do the sources hold the reference answer's facts | reference found |

Every measured result is saved as JSON in `runs/` and summarised in `reports/slot2-report.md`.

---

## 2. Project structure

```
repo/
├── app.py                  Streamlit web app (the front end)
├── askmyrepo/              the Python package
│   ├── __main__.py         command line: files, baseline, index, search, ask, evaluate, judge
│   ├── manifest.py         reads corpus.toml, lists the files to index from Git
│   ├── chunker.py          cuts files into passages (whole file / syntax / fixed window)
│   ├── embedder.py         local Qwen3 embedding model
│   ├── vector_index.py     Pinecone: create, upload, search by meaning; the Hit type
│   ├── keyword_index.py    BM25 keyword index over the same passages
│   ├── search.py           fuses both searches, filters, reranks
│   ├── reranker.py         local Qwen3 reranker
│   ├── agent.py            the agent loop (limits, usage, cost, tracing)
│   ├── agent_tools.py      the two tools, the numbered sources, ask_agent
│   ├── database.py         the read-only SQL runner
│   ├── answer.py           the answer contract, pipeline answering, the name guard
│   ├── baseline.py         the "paste the whole repository" baseline
│   ├── evaluate.py         recall, answer and refusal rates, saved runs
│   ├── judge.py            the four-measure judge and the hand-label check
│   ├── llm.py              model interface, Gemini client, errors, price table
│   ├── tools.py            typed functions turned into tools for the model
│   └── tracing.py          optional Langfuse tracing
├── tests/                  123 tests (no keys, no network)
├── scripts/
│   └── compare_with_ragas.py   one-off comparison with ragas (own environment)
├── notebooks/
│   └── embeddings_by_hand.ipynb  Session 7's embeddings notebook
├── reports/
│   └── slot2-report.md     the measured results
├── runs/                   saved JSON for every measured run
├── corpus.toml             which repositories, versions and files are read
├── questions.jsonl         the 40 labelled questions
├── pyproject.toml, uv.lock the libraries (pinned)
├── Dockerfile              local container (models and database mounted)
├── Dockerfile.cloud        Cloud Run image (models and database baked in)
├── docker_setup.md         Docker, step by step
├── GCP_setup.md            every GCP / Cloud Run command
├── .env.example            the settings to copy into .env
└── .gitignore, .dockerignore, .gitleaks.toml, .pre-commit-config.yaml
```

Not in this folder (they are sibling projects): `../shoplite` and `../shoplite-warehouse`, the repositories being asked about.

---

## 3. Set up (macOS)

You need Python 3.12, [uv](https://docs.astral.sh/uv/), a **Gemini** key and a **Pinecone** key. Allow about 8 GB of free disk (PyTorch plus two 1.2 GB models).

```bash
# 1. the project's libraries (includes PyTorch)
cd repo
uv sync

# 2. your keys
cp .env.example .env            # then fill in GEMINI_API_KEY and PINECONE_API_KEY
```

**The repositories to read.** `corpus.toml` expects ShopLite and the warehouse as Git clones next to this folder:

```
work/
├── askmyrepo/            this project (its folder is `repo/` here)
├── shoplite/             git clone of ShopLite, with the v1.1 and v2.0 tags
└── shoplite-warehouse/   git clone of the warehouse service
```

If they are only plain folders (no `.git`), clone them from the reference bundles, or point to other clones with `--corpus` (a copy of `corpus.toml` with different `path =` lines):

```bash
git clone shoplite-reference.bundle ../shoplite
git clone shoplite-warehouse-reference.bundle ../shoplite-warehouse
```

**The database.** Database questions need ShopLite's demo data (20 customers, 50 products, 200 orders):

```bash
cd ../shoplite && uv sync && uv run python scripts/seed.py   # writes instance/shoplite.db
```

The two Qwen models download on first use into `~/.cache/huggingface`. To fetch them now: `uv run hf download Qwen/Qwen3-Embedding-0.6B` and `uv run hf download Qwen/Qwen3-Reranker-0.6B`.

---

## 4. Use it

Activate the environment and load the keys once per terminal:

```bash
source .venv/bin/activate
set -a; source .env; set +a
```

Options go **before** the command name.

```bash
python -m askmyrepo files                      # what corpus.toml will read
python -m askmyrepo index                      # read, cut, embed, upload (once; a few minutes)
python -m askmyrepo search "Where is inv_reservation_ledger written?"
python -m askmyrepo ask "How much does shipping cost, and when is it free?"
python -m askmyrepo ask "How many card payments were declined?"      # uses the database
python -m askmyrepo --pipeline ask "How are passwords stored?"       # one search, no database
python -m askmyrepo evaluate --compare         # recall at 5 for the three search setups
python -m askmyrepo evaluate --answers         # answer and refusal rates (about 140 model calls)
python -m askmyrepo judge runs/<answers run>.json --keys <questions with answers>
python -m askmyrepo --help
```

| Option | What it does |
|---|---|
| `--corpus PATH` | use another `corpus.toml` |
| `--pipeline` | answer with one search instead of the agent |
| `--setup meaning\|fused\|reranked` | which search (default `reranked`) |
| `--all-versions`, `--version 1.1` | stop filtering to the newest version |
| `--repo`, `--folder` | narrow the search |
| `--chunking files\|syntax\|fixed` | how files are cut (`--window`, `--overlap` for `fixed`); each has its own Pinecone namespace |
| `--database PATH` | ShopLite's SQLite file |
| `--candidates N` | how many passages each search hands on to fusion and reranking (20) |

### The web app

```bash
streamlit run app.py          # http://localhost:8501
```

A single chat page: type a question; each claim appears with a chip showing its file and lines (or SQL), plus the cost and time. The sidebar has a repository filter, six example questions and the session's cost. The index must already be built.

### Tests

```bash
pytest -q                     # 123 tests; no keys, no network
ruff check app.py askmyrepo scripts tests
```

---

## 5. Run it in Docker and on Google Cloud

- **[docker_setup.md](docker_setup.md)**: build and run the container locally.
- **[GCP_setup.md](GCP_setup.md)**: every command used to deploy it to Cloud Run (Cloud Build, Secret Manager, public or private access).

---

## 6. Tuning knobs and what they cost

| Setting | Where | Effect |
|---|---|---|
| `MAX_SEARCHES` (3) | `agent_tools.py` | searches the agent may make per question |
| `PASSAGES_PER_SEARCH` (3) | `agent_tools.py` | passages shown per search; fewer raises context precision |
| `SCORE_GAP` (8.0) | `agent_tools.py` | drops passages far below the best reranker score |
| `Limits` (12 steps, 60,000 tokens, 300 s) | `agent.py` | the agent's ceilings |
| `CANDIDATES` (20) | `search.py` | how many passages reach fusion and reranking |

**What we measured when tightening these** (40 labelled questions): showing 3 passages per search and 3 searches cut the sources shown per question from 9.6 to 5.1 and the cost from $0.119 to $0.091, but the answer rate fell from 30/30 to 26/30 (refusal rate stayed 10/10). The earlier settings (5 passages, 4 searches) scored 30/30 and 10/10. Agent runs vary from run to run, so repeat a run before trusting a small change.

---

## 7. Good to know

- **Quotas.** The Gemini free tier allows about 500 requests a day. A 40-question agent run makes about 140 calls and each judge run about 40. When the quota is used up the model returns 429 until it resets.
- **Agent time limit.** One question may take up to 300 s; if the machine is busy, the run can end with "no answer: time limit reached".
- **Read-only database.** The SQL tool refuses anything but one `SELECT`; writes, `ATTACH` and `PRAGMA` are denied by SQLite itself.
- **Faithful is not the same as correct.** The judge can score an answer "supported" when its query was the wrong one (an earlier run answered "Kitchen" for top revenue because it counted failed orders). That is why the reference-found measure and the hand-label check exist.
- **Never commit `.env`.** It is in `.gitignore`; `.gitleaks.toml` and the pre-commit hook scan for secrets.
