# S7: the whole repository in the prompt (2026-11-18)

ShopLite 2.0 and the warehouse, the 68 files from `corpus.toml` (newest version), pasted as one prompt: 157,233 characters, **46,581 tokens** by Gemini's counting call. Model: gemini-3.5-flash-lite (free-tier key; costs at the published paid rate). The three questions are answered only in files that sit about half-way through the pasted text (`models/coupon.py`, `faults.py`, `models/inventory.py`).

| Question | Answer | Correct | Tokens in | Seconds | Cost |
|---|---|---|---|---|---|
| What does a NULL value in coupons.max_redemptions mean? | "unlimited total uses allowed across all customers", quoting `models/coupon.py` | yes | 46,618 | 3.3 | $0.0142 |
| How long is the default delay of the slow-query fault switch? | "1500 milliseconds", quoting `DEFAULT_SLOW_QUERY_MS = 1500` in `faults.py` | yes | 46,620 | 8.2 | $0.0141 |
| What does the column comment on the reservation ledger's expires_at say? | "A held entry past this time is released by release_stale_reservations()", quoting `models/inventory.py` | yes | 46,619 | 7.6 | $0.0143 |

**Finding:** the planned S7 failure (an answer in the middle is missed) did not happen at this size. The difference that holds is cost: about $0.014 per question here, against $0.0232 for all 40 pipeline answers (`runs/2026-11-18-201337-answers.json`).
