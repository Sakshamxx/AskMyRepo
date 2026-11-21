"""Command line for AskMyRepo: files, baseline, index, search, ask, evaluate and judge.
Run `python -m askmyrepo --help`; every command is explained in README.md."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from askmyrepo import (
    agent_tools,
    answer,
    baseline,
    evaluate,
    judge,
    keyword_index,
    vector_index,
)
from askmyrepo.chunker import chunk_files
from askmyrepo.database import connect_read_only
from askmyrepo.embedder import embed_documents
from askmyrepo.llm import GeminiClient, cost_usd
from askmyrepo.manifest import list_corpus_files
from askmyrepo.search import CANDIDATES, SETUPS, Searcher
from askmyrepo.tracing import Tracer

DEFAULT_MODEL = "gemini-3.5-flash-lite"


# Picks the Pinecone namespace so each chunking style keeps its own vectors.
def namespace(args) -> str:
    """One Pinecone namespace per way of cutting: files, syntax, or fixed-<window>-<overlap>."""
    if args.chunking == "fixed":
        return f"fixed-{args.window}-{args.overlap}"
    return args.chunking


# Fails early with a clear message when the Gemini key is missing.
def gemini_client(model: str) -> GeminiClient:
    """A Gemini client from GEMINI_API_KEY, or a clear exit."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set (copy .env.example to .env)")
    return GeminiClient(api_key, model)


# `files` command: shows what corpus.toml will index, before spending anything.
def run_files(args) -> None:
    """List what the manifest reads, and how much of it there is."""
    files = list_corpus_files(Path(args.corpus))
    for f in files:
        print(f"{f.repo}/{f.path}")
    print(f"\n{len(files)} files, {sum(len(f.text) for f in files):,} characters")


# `baseline` command: the paste-everything approach that retrieval is compared with.
def run_baseline(args) -> None:
    """The whole repository in one prompt: tokens, the answer, the cost and the time."""
    client = gemini_client(args.model)
    files = list_corpus_files(Path(args.corpus))
    tokens = baseline.count_repository_tokens(client, files)
    print(f"{len(files)} files, {tokens:,} tokens by the counting call\n")
    text, usage, seconds = baseline.ask_whole_repository(client, files, args.question)
    print(text)
    print(f"\n({usage.input_tokens:,} tokens in, {usage.output_tokens:,} out, {seconds:.1f} s, "
          f"${cost_usd(client.model, usage):.4f})")


# Turns --repo/--folder/--version/--all-versions into the search filter dict.
def filters_from(args) -> dict:
    """The search filters asked for on the command line."""
    filters = {}
    if args.repo:
        filters["repo"] = args.repo
    if args.folder:
        filters["folder"] = args.folder.strip("/")
    if args.version:
        filters["version"] = args.version
    elif not args.all_versions:
        filters["latest"] = True
    return filters


# Connects Pinecone and the keyword index into one Searcher.
def make_searcher(args) -> Searcher:
    """Connect to Pinecone and load the keyword index for the chosen chunking."""
    index = vector_index.open_index(vector_index.connect())
    return Searcher(index, keyword_index.KeywordIndex(namespace(args)), namespace(args))


# `index` command: read, chunk, embed locally, upload, and save the keyword chunks.
def run_index(args) -> None:
    """Read the manifest, chunk, embed locally and upload, replacing what was there."""
    started = time.perf_counter()
    files = list_corpus_files(Path(args.corpus))
    chunks = chunk_files(files, args.chunking, args.window, args.overlap)
    largest = max(chunks, key=lambda chunk: len(chunk.text))
    print(f"{len(files)} files -> {len(chunks)} chunks ({namespace(args)}); largest: "
          f"{largest.repo}/{largest.path} lines {largest.start_line}-{largest.end_line}, "
          f"{len(largest.text):,} characters")
    keyword_index.save_chunks(chunks, namespace(args))
    vectors = embed_documents([chunk.text for chunk in chunks])
    embedded = time.perf_counter()
    print(f"Embedded locally in {embedded - started:.1f} s")
    index = vector_index.open_index(vector_index.connect())
    vector_index.clear_namespace(index, namespace(args))
    vector_index.upload(index, chunks, vectors, namespace(args))
    vector_index.wait_until_searchable(index, namespace(args), len(chunks))
    print(f"Uploaded to Pinecone index '{vector_index.INDEX_NAME}', namespace "
          f"'{namespace(args)}', in {time.perf_counter() - embedded:.1f} s")


# `search` command: shows the best passages and scores, with no model answer.
def run_search(args) -> None:
    """Print the closest passages with their scores."""
    searcher = make_searcher(args)
    started = time.perf_counter()
    hits = searcher.search(args.question, args.setup, filters_from(args), args.top_k,
                           args.candidates)
    seconds = time.perf_counter() - started
    for hit in hits:
        version = f" (v{hit.version})" if hit.version else ""
        symbol = f"  ({hit.symbol})" if hit.symbol else ""
        print(f"{hit.score:8.4f}  {hit.location}{version}{symbol}")
    print(f"\n({args.setup}, {seconds:.1f} s, including loading the models)")


# Opens the SQLite file given by --database, read-only.
def open_database(args):
    """ShopLite's database, read-only."""
    return connect_read_only(Path(args.database))


# `ask` command: agent by default, or the one-search pipeline with --pipeline.
def run_ask(args) -> None:
    """Answer with the agent (or --pipeline): each claim with its source, or a refusal."""
    client = gemini_client(args.model)
    searcher = make_searcher(args)
    try:
        if args.pipeline:
            hits = searcher.search(args.question, args.setup, filters_from(args), args.top_k)
            result, usage = answer.answer_question(client, args.question, hits)
            print(answer.render(result, hits))
        else:
            tracer = Tracer()  # on when the LANGFUSE_* keys are set
            response = agent_tools.ask_agent(client, searcher, open_database(args),
                                             args.question, filters_from(args), tracer=tracer)
            tracer.flush()
            result, usage = response["answer"], response["usage"]
            print("\n" + agent_tools.render_agent_answer(result, response["source_log"]))
    except answer.ContractBroken as exc:
        sys.exit(f"The answer broke the contract: {exc}")
    print(f"\n({usage.input_tokens:,} tokens in, {usage.output_tokens:,} out, "
          f"${cost_usd(client.model, usage):.5f})")


# Runs every labelled question through the answerer and reports the two rates.
def run_answer_evaluation(args, searcher: Searcher, questions: list[dict]) -> None:
    """Answer every question; report the answer rate and the refusal rate together."""
    filters = filters_from(args)
    if args.pipeline:
        client = evaluate.CachedClient(gemini_client(args.model))
        respond = evaluate.pipeline_responder(
            lambda q: searcher.search(q, args.setup, filters, args.top_k),
            lambda q, hits: answer.answer_question(client, q, hits))
    else:
        client = gemini_client(args.model)
        conn = open_database(args)
        tracer = Tracer()  # one trace per question when the LANGFUSE_* keys are set

        # Agent mode: one quiet agent run per question.
        def respond(question: str) -> dict:
            """One agent run, quietly."""
            return agent_tools.ask_agent(client, searcher, conn, question, filters,
                                         log=lambda line: None, tracer=tracer)
    result = evaluate.measure_answers(questions, respond)
    if not args.pipeline:
        tracer.flush()
    result.update(model=args.model, chunking=namespace(args), setup=args.setup, filters=filters,
                  system="pipeline" if args.pipeline else "agent",
                  new_model_calls=getattr(client, "calls", None))
    for row in result["rows"]:
        outcome = {True: "answered", False: "refused "}.get(row["answered"], "BROKEN  ")
        expected = "" if row["answerable"] else "  (no answer exists)"
        print(f"{row['id']}  {outcome}  {row['question']}{expected}")
    print(f"\nAnswer rate (answerable questions answered): {result['answer_rate']}")
    print(f"Refusal rate (no-answer questions refused): {result['refusal_rate']}")
    print(f"Contract broken: {result['contract_broken']}; "
          f"cost ${cost_usd(args.model, evaluate.Usage(**result['usage'])):.4f}")
    print(f"Saved {evaluate.save_run(result, 'answers-' + result['system'])}")


# `evaluate` command: recall at 5, or answer/refusal rates with --answers.
def run_evaluate(args) -> None:
    """Recall at 5 for one setup (or all three with --compare), or the answer and refusal rates."""
    searcher = make_searcher(args)
    questions = evaluate.load_questions(Path(args.questions))
    if args.answers:
        return run_answer_evaluation(args, searcher, questions)
    questions = [q for q in questions if q.get("answerable", True)]  # recall needs an answer
    filters = filters_from(args)
    summary = {}
    for setup in (SETUPS if args.compare else (args.setup,)):
        searcher.search("warm up the models", setup, filters, args.top_k)  # loading isn't timed
        result = evaluate.measure_recall(
            questions, lambda q, s=setup: searcher.search(q, s, filters, args.top_k), args.top_k)
        result.update(chunking=namespace(args), setup=setup, filters=filters)
        for row in result["rows"]:
            if not row["hit"] or not row["answer_hit"]:
                print(f"{setup:9} {row['id']}  file {'hit ' if row['hit'] else 'MISS'}  "
                      f"answer {'hit ' if row['answer_hit'] else 'MISS'}  {row['question']}")
        seconds = [row["seconds"] for row in result["rows"]]
        summary[setup] = (result["hits"], result["answer_hits"], result["questions"],
                          sum(seconds) / len(seconds))
        print(f"Saved {evaluate.save_run(result, f'recall-{namespace(args)}-{setup}')}")
    print(f"\n{'Setup':10} Right file   Answer in a passage   Seconds per question")
    for setup, (hits, answer_hits, total, mean) in summary.items():
        print(f"{setup:10} {hits}/{total:<10} {answer_hits}/{total:<19} {mean:.2f}")


# `judge` command: scores a saved run on four measures and compares with hand labels.
def run_judge(args) -> None:
    """Judge a saved answer run with our own judge and save the four measures."""
    run = json.loads(Path(args.run_file).read_text(encoding="utf-8"))
    keys = {q["id"]: q["answer"] for q in evaluate.load_questions(Path(args.keys))}
    client = evaluate.CachedClient(gemini_client(args.model))
    rows, usage = [], evaluate.Usage()
    for row in run["rows"]:
        if row["contract"] != "kept":
            continue  # a broken answer has nothing to judge
        scores, judgement, used = judge.judge_row(client, row, keys[row["id"]],
                                                  require_quote=not args.no_quote)
        usage.add(used)
        rows.append({"id": row["id"], "answerable": row["answerable"], **scores,
                     "judgement": judgement.model_dump()})
        print(f"{row['id']}  supported {sum(scores['claims_supported'])}/"
              f"{len(scores['claims_supported'])}  answers {judgement.answers_question:6}  "
              f"recall {judgement.reference_found}")
    answerable = [r for r in rows if r["answerable"]]
    result = {"run_file": args.run_file, "judge_model": args.model,
              "quote_required": not args.no_quote, "all_questions": judge.summarise(rows),
              "answerable_only": judge.summarise(answerable),
              "cited_precision": judge.mean([r["cited_precision"] for r in rows]),
              "usage": asdict(usage),
              "new_model_calls": client.calls, "rows": rows}
    labelled = (json.loads(Path(args.hand_labels).read_text(encoding="utf-8"))
                if args.hand_labels else None)
    if labelled and Path(labelled.get("answer_run", "")).name != Path(args.run_file).name:
        print(f"Hand check skipped: the labels are for {labelled.get('answer_run')}, "
              f"not {args.run_file} (they label that run's claims)")
        labelled = None
    if labelled:
        hand = labelled["labels"]
        judged = {f"{r['id']}#{n}": ok for r in rows
                  for n, ok in enumerate(r["claims_supported"], start=1)}
        result["hand_check"] = judge.disagreement(judged, hand)
        check = result["hand_check"]
        print(f"Hand check: judge and hand labels disagree on {check['disagreed']} of "
              f"{check['checked']} claims {check['which']}")
    print(f"\nAll questions: {result['all_questions']}")
    print(f"Answerable only: {result['answerable_only']}")
    print(f"New judge calls: {client.calls}; cost ${cost_usd(args.model, usage):.4f}")
    print(f"Saved {evaluate.save_run(result, 'judged' + ('-no-quote' if args.no_quote else ''))}")


# Entry point: defines every option and command, then dispatches.
def main(argv=None) -> None:
    """Parse the command and run it."""
    parser = argparse.ArgumentParser(prog="askmyrepo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="corpus.toml")
    parser.add_argument("--model", default=os.environ.get("ASKMYREPO_MODEL", DEFAULT_MODEL))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunking", choices=["files", "syntax", "fixed"], default="syntax")
    parser.add_argument("--window", type=int, default=60, help="lines per chunk (fixed)")
    parser.add_argument("--overlap", type=int, default=0, help="lines repeated (fixed)")
    parser.add_argument("--setup", choices=SETUPS, default=SETUPS[-1])  # the best available
    parser.add_argument("--repo", help="only this repository, e.g. shoplite-warehouse")
    parser.add_argument("--folder", help="only this folder, e.g. shoplite/services")
    parser.add_argument("--version", help="only this version, e.g. 1.1")
    parser.add_argument("--all-versions", action="store_true", help="don't filter by version")
    parser.add_argument("--pipeline", action="store_true",
                        help="answer with the fixed pipeline (one search, then answer)")
    parser.add_argument("--database", default=os.environ.get(
        "ASKMYREPO_DATABASE", "../shoplite/instance/shoplite.db"), help="ShopLite's SQLite file")
    parser.add_argument("--candidates", type=int, default=CANDIDATES,
                        help="how many passages each search hands on to fusion and reranking")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("files", help="list the files the manifest reads")
    commands.add_parser("baseline", help="ask with the whole repository in the prompt") \
        .add_argument("question")
    commands.add_parser("index", help="embed the manifest's files and upload them")
    commands.add_parser("search", help="show the closest passages").add_argument("question")
    commands.add_parser("ask", help="answer with the file and line under every claim") \
        .add_argument("question")
    evaluation = commands.add_parser("evaluate", help="measure on the labelled questions")
    evaluation.add_argument("--questions", default="questions.jsonl")
    evaluation.add_argument("--compare", action="store_true", help="recall for all three setups")
    evaluation.add_argument("--answers", action="store_true",
                            help="answer and refusal rates instead of recall (calls the model)")
    judging = commands.add_parser("judge", help="score a saved answer run with our own judge")
    judging.add_argument("run_file", help="a runs/...-answers-*.json file")
    judging.add_argument("--keys", required=True, help="the questions with reference answers")
    judging.add_argument("--hand-labels", help="hand labels, to measure disagreement")
    judging.add_argument("--no-quote", action="store_true",
                         help="trust the judge's verdict without checking its quote")
    judging.add_argument("--model", default=os.environ.get("ASKMYREPO_JUDGE_MODEL", DEFAULT_MODEL))
    args = parser.parse_args(argv)
    {"files": run_files, "baseline": run_baseline, "index": run_index, "search": run_search,
     "ask": run_ask, "evaluate": run_evaluate, "judge": run_judge}[args.command](args)


if __name__ == "__main__":
    main()
