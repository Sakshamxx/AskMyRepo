# AskMyRepo: slot 2 report

Every number here comes from a saved run in `runs/`. Corpus: ShopLite and shoplite-warehouse on `main`, through `corpus.toml` (68 files). Embeddings: local Qwen3-Embedding-0.6B. Index: Pinecone serverless (free Starter plan), cosine.

## S8–S9: recall at 5 on the first ten questions, before and after cutting on boundaries

A hit means at least one of the five passages comes from a file that answers the question.

| Chunking | Chunks | Recall at 5 | Text sent to the answer step (characters, mean of top 5) | Lines per passage (mean) | Run |
|---|---|---|---|---|---|
| Whole files | 68 | 10/10 | 13,105 | 60 | `runs/2026-11-18-185424-recall-files.json` |
| Cut on function, class and heading boundaries | 313 | 10/10 | 3,917 | 12 | `runs/2026-11-18-185436-recall-syntax.json` |

**Honest note.** On this corpus, whole files already reach 10/10, so recall at 5 can't show the gain from cutting on boundaries. The gain shows as a third of the text sent to the model and a citation that points at about a dozen lines instead of a whole file. The first syntax run (`runs/2026-11-18-185321-recall-syntax.json`) scored 9/10: Q09 returned the "Other services" section of `docs/architecture.md`, which does answer it, but the answer key listed only code files. The key was corrected (the docs section added) and both setups re-run.

## Corpus from S10 on

ShopLite 2.0 (`main`) and 1.1 (`v1.1`) and the warehouse 1.0: 119 files, 514 chunks. Every chunk carries its version; searches use the newest version unless asked otherwise (S11).

## S10: the answer contract (pipeline: one search by meaning, top 5, no version filter)

| Measure | Result |
|---|---|
| Answer rate (answerable questions answered) | 22/30 (22/23 on code and schema; the 7 database questions are refused until S12's SQL tool) |
| Refusal rate (no-answer questions refused) | 10/10 |
| Answers that broke the contract | 0 |

Run: `runs/2026-11-18-201337-answers.json` (gemini-3.5-flash-lite). **Stale answer, reproduced:** Q22 was answered from the 1.1 copy of `checkout.py`: "Checkout calls stock.take_stock(item.product_id, item.quantity) to take stock." (`take_stock` no longer exists in 2.0).

## S11: three search setups (30 answerable questions, newest version only)

| Setup | Recall at 5 | Seconds per question | Run |
|---|---|---|---|
| meaning | 23/30 | 0.61 | `runs/2026-11-18-201847-recall-syntax-meaning.json` |
| fused | 23/30 | 0.57 | `runs/2026-11-18-201906-recall-syntax-fused.json` |
| reranked | 27/30 | 3.98 | `runs/2026-11-18-205147-recall-syntax-reranked.json` |

Fusing the keyword search with search by meaning added nothing here; the reranker added four questions and most of the time. Times are on an Apple M1 Pro (16 GB). The reranker (Qwen3-Reranker-0.6B over the top 20, passages cut at 512 tokens, float32, batches of 4) took about 3.0 s per question on the GPU and 6.0 s on the CPU with the same recall (measured separately); in the model's default bfloat16 the CPU took 91 s for 20 passages. **Reranking everything, reproduced:** reranking all 313 current chunks for one question took 39.9 s, with the same top results as the short list.

## S12: the agent, our own judge, and ragas

**Agent** (search_code and query_database, search budget 4, `runs/2026-11-18-211729-answers-agent.json`): answer rate 30/30, refusal rate 10/10, broken contracts 0.

**Our own judge** (`judge.py`, gemini-3.5-flash-lite, `runs/2026-11-18-212433-judged.json`):

| Measure | Quote required | Without the quote check |
|---|---|---|
| faithfulness | 0.972 | 0.989 |
| answer_relevance | 0.975 | 0.975 |
| context_precision | 0.224 | 0.224 |
| context_recall | 0.775 | 0.775 |

**Hand check** (`faculty/hand_labels.json`, the first claim of Q01–Q20, labels drafted by Claude and accepted as they are by the Learning Architect): the judge and the labels disagree on 2 of 20 claims (Q01#1, Q06#1). In both, a real quote supported part of the claim, and the claim added a detail the cited passage doesn't contain (the 15-minute expiry; the table name). Requiring a quote didn't catch them.

**ragas 0.4.4** (run once, in its own environment: `scripts/compare_with_ragas.py`, `runs/2026-11-18-220414-ragas.json`): 21 of 30 answerable questions scored; 9 skipped because Gemini answered 503 "high demand" three times (Q03, Q04, Q07, Q09, Q10, Q14, Q15, Q25, Q27).

| Measure | Questions | ragas | Our judge |
|---|---|---|---|
| faithfulness | 21 | 0.865 | 0.96 |
| context_recall | 21 | 0.869 | 0.81 |
| context_precision | 5 | 0.455 | 0.32 |

**Not run:** the Gemini Batch API (half price) refused both free-tier keys tried (`400 FAILED_PRECONDITION`), so the evaluation ran through the normal API.

## Step 5: the planned failures, tried for real

| Session | Planned failure | Result | Evidence |
|---|---|---|---|
| S7 | the answer in the middle of a pasted repository is missed | did not happen: 3 of 3 middle questions right at 46,581 tokens; cost $0.014 and 3-8 s per question | `runs/2026-11-18-whole-repo-prompt.md` |
| S8 | the wrong distance metric gives quietly wrong results | does not apply: the vectors are normalised, so the metrics give the same order | the embedding model ends in a Normalize layer |
| S8 | an exact name gets noise from search by meaning | not for the table name; **yes for error codes**: `SL-ORD-409` by meaning ranks `utils/ids.py` first and misses the file that raises it; `SL-INV-422` and `SL-CPN-409` return only the docs table; keyword search finds all three | measured this step (search output) |
| S8 | (tried as a replacement) questions embedded without the query prompt | no effect: recall 23/30, the same | `runs/2026-11-18-222849-recall-no-query-prompt.json` |
| S9 | a generated file becomes one huge chunk that crowds out the rest | partly: `docs/openapi.json` (one chunk, 22,853 characters) entered the top 5 for 3 of 30 questions (Q09, Q12, Q27), never first; recall 23/30, unchanged | `runs/2026-11-18-222758-recall-generated-file-meaning.json` |
| S10 | version one answers a missing feature fluently and wrongly | did not happen: version one's prompt declined all 10; so did a naive prompt with no "use only the passages" rule | `runs/2026-11-18-223101-v1-no-answer-questions.json`, `runs/2026-11-18-223245-naive-no-answer-questions.json` |
| S12 | (tried as a fix) a claim's numbers and names must appear in its cited source | flags exactly Q01#1 and Q06#1, the two claims the judge wrongly accepted, and none of the other 53 supported claims | computed from `runs/2026-11-18-212433-judged.json` |

## S9: chunk size and overlap, measured (2026-11-18)

Search by meaning, newest version (ShopLite 2.0 and the warehouse), 30 answerable questions. **Answer in a passage** counts a question when a returned passage contains a code line its answer key relies on (the faculty checker's list); it applies to the 25 questions answered from code. Run: `runs/2026-11-18-234159-size-overlap-answer.json`.

| Splitting | Chunks | Answer in a passage (25) | Right file in the top 5 (30) | Characters sent (mean) |
|---|---|---|---|---|
| syntax | 313 | 22/25 | 23/30 | 4,430 |
| fixed-20-overlap-0 | 233 | 20/25 | 25/30 | 4,001 |
| fixed-40-overlap-0 | 134 | 25/25 | 25/30 | 6,870 |
| fixed-40-overlap-10 | 144 | 24/25 | 25/30 | 7,412 |
| fixed-80-overlap-0 | 85 | 25/25 | 26/30 | 11,057 |
| fixed-80-overlap-20 | 87 | 25/25 | 27/30 | 11,134 |

At about the same size, cutting on syntax beats fixed 20-line windows (22 against 20: the windows cut answers away from their context). Larger windows find every answer but send 55% to 150% more text; overlap added nothing here. File-level recall rewards big chunks, so it can't decide this on its own.

## Further checks (2026-11-18)

- **S7 candidate failure, two versions in one prompt:** ShopLite 1.1 and 2.0 plus the warehouse pasted (119 files, 74,538 tokens, the old copy last): both version-sensitive questions answered correctly from 2.0 ($0.023 and about 7.5 s each). Did not reproduce.
- **Missing functions** (`runs/2026-11-18-231542-missing-function-questions.json`, `runs/2026-11-18-231622-substitution-repeats.json`): version one declined 9 of 10, and for `reserve_stock_for_cart()` described `reserve_for_order` instead, in 3 of 4 tries; version two's contract refused it.
- **Judge, faculty checks** (`runs/2026-11-18-233513-judge-faculty-checks.json`): a second judge model (gemini-3.1-flash-lite) disagreed with the hand labels on 0 of 19 claims. At temperature 0 the same judge asked twice changed claim verdicts 0 times, "answers the question" 0, "reference found" 0 and the relevant-sources list 5 times out of 10 (at the default temperature: 0, 0, 1 and 2): no steadier.
- **A learner's S12 redo:** about 347 Gemini requests (agent 142, judge 40, ragas about 165, worked out from ragas's steps), inside the free tier's 500 a day.
- **Tracing:** an `ask` run arrives in Langfuse as one trace with its model calls and tool calls.
