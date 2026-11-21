"""Score a saved answer run with ragas, once, to compare it with our own judge.
Runs in its own throwaway environment (ragas clashes with our packages); see README.md."""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from google import genai
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

MODEL = "gemini-3.5-flash-lite"
PRECISION_QUESTIONS = 10
ATTEMPTS = 3          # Gemini sometimes answers 503 "high demand"; wait and try again
WAIT_SECONDS = 30


# Scores one answer with ragas' metrics.
async def score(row: dict, reference: str, metrics: dict, with_precision: bool) -> dict:
    """ragas scores for one answer."""
    response = " ".join(claim["statement"] for claim in row["claims"])
    contexts = [source["text"] for source in row["sources"]]
    result = {
        "faithfulness": (await metrics["faithfulness"].ascore(
            user_input=row["question"], response=response, retrieved_contexts=contexts)).value,
        "context_recall": (await metrics["recall"].ascore(
            user_input=row["question"], retrieved_contexts=contexts, reference=reference)).value,
    }
    if with_precision:
        result["context_precision"] = (await metrics["precision"].ascore(
            user_input=row["question"], reference=reference, retrieved_contexts=contexts)).value
    return result


# Average that ignores missing scores.
def mean(values: list) -> float | None:
    """The average of the values that aren't None."""
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 3) if present else None


# Scores every answerable question and saves the comparison to runs/.
async def main(answers_path: str, judged_path: str, keys_path: str) -> None:
    """Score every answerable question and save the comparison next to the other runs."""
    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    judged = {r["id"]: r for r in json.loads(Path(judged_path).read_text(encoding="utf-8"))["rows"]}
    keys = {}
    for line in Path(keys_path).read_text(encoding="utf-8").splitlines():
        question = json.loads(line)
        keys[question["id"]] = question["answer"]
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    llm = llm_factory(MODEL, provider="google", client=client)
    metrics = {"faithfulness": Faithfulness(llm=llm), "recall": ContextRecall(llm=llm),
               "precision": ContextPrecision(llm=llm)}
    rows = []
    for n, row in enumerate(r for r in answers["rows"] if r["answerable"] and r["answered"]):
        ragas_scores = None
        for attempt in range(1, ATTEMPTS + 1):
            try:
                ragas_scores = await score(row, keys[row["id"]], metrics, n < PRECISION_QUESTIONS)
                break
            except Exception as exc:  # ragas wraps the model's errors in its own types
                print(f"{row['id']} attempt {attempt} failed: {str(exc)[:120]}", flush=True)
                await asyncio.sleep(WAIT_SECONDS)
        if ragas_scores is None:
            rows.append({"id": row["id"], "skipped": "model unavailable after retries"})
            continue
        ours = judged[row["id"]]
        rows.append({"id": row["id"], "ragas": ragas_scores,
                     "ours": {k: ours[k] for k in ("faithfulness", "context_recall",
                                                    "context_precision")}})
        print(row["id"], "ragas", ragas_scores, "ours", rows[-1]["ours"], flush=True)
    summary = {}
    for measure in ("faithfulness", "context_recall", "context_precision"):
        scored = [r for r in rows if measure in r.get("ragas", {})]
        summary[measure] = {"questions": len(scored),
                            "ragas": mean([r["ragas"][measure] for r in scored]),
                            "ours": mean([r["ours"][measure] for r in scored])}
    result = {"skipped": [r["id"] for r in rows if "skipped" in r], "answers_run": answers_path,
              "judged_run": judged_path, "model": MODEL,
              "ragas_version": "0.4.4", "summary": summary, "rows": rows}
    out = Path("runs") / f"{datetime.now():%Y-%m-%d-%H%M%S}-ragas.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:4]))
