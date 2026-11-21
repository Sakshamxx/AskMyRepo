"""Tests for our own judge (with a fake model)."""

import json

from askmyrepo import judge

SOURCE = 'FLAT_RATE_PAISE = 4_900  # ₹49\nFREE_SHIPPING_FROM_PAISE = 99_900  # free on ₹999'
ROW = {
    "id": "Q02", "question": "How much does shipping cost?", "answerable": True,
    "answered": True, "refusal": "",
    "claims": [{"statement": "Shipping is ₹49.", "source": 1},
               {"statement": "Shipping is free on Sundays.", "source": 1}],
    "sources": [{"location": "shoplite/shoplite/services/shipping.py:1-13", "text": SOURCE},
                {"location": "shoplite/docs/api.md:1-4", "text": "JSON in, JSON out."}],
}


def judgement(quotes: list[tuple[bool, str]], answers="yes", relevant=(1,), found="all"):
    """A judge reply with these (supported, quote) verdicts."""
    return judge.Judgement(
        claims=[judge.ClaimVerdict(claim=n, supported=ok, quote=q)
                for n, (ok, q) in enumerate(quotes, start=1)],
        answers_question=answers, relevant_sources=list(relevant), reference_found=found)


def test_a_quote_must_really_be_in_the_source():
    assert judge.quote_is_in("FLAT_RATE_PAISE  =\n 4_900", SOURCE)  # spacing doesn't matter
    assert not judge.quote_is_in("free on Sundays", SOURCE)
    assert not judge.quote_is_in("₹49", SOURCE)  # too short to prove anything
    assert judge.quote_is_in("pending_payment → paid", "`pending_payment` → `paid` → `shipped`")


def test_with_quotes_required_an_invented_claim_is_not_supported():
    reply = judgement([(True, "FLAT_RATE_PAISE = 4_900"), (True, "free on Sundays")])
    scores = judge.score_row(ROW, reply, require_quote=True)
    assert scores["claims_supported"] == [True, False]
    assert scores["faithfulness"] == 0.5


def test_without_the_quote_check_the_judge_is_simply_believed():
    reply = judgement([(True, "FLAT_RATE_PAISE = 4_900"), (True, "free on Sundays")])
    assert judge.score_row(ROW, reply, require_quote=False)["faithfulness"] == 1.0


def test_the_other_three_measures():
    reply = judgement([(True, "FLAT_RATE_PAISE = 4_900"), (False, "")], answers="partly",
                      relevant=(1, 7), found="some")  # 7 was never shown: ignored
    scores = judge.score_row(ROW, reply)
    assert (scores["answer_relevance"], scores["context_precision"],
            scores["context_recall"]) == (0.5, 0.5, 0.5)


def test_a_refusal_has_no_faithfulness_score():
    refused = {**ROW, "answered": False, "claims": [], "refusal": "Not covered."}
    scores = judge.score_row(refused, judgement([], answers="no", found="none"))
    assert scores["faithfulness"] is None
    assert "the system refused: Not covered." in judge.build_judge_prompt(refused, "₹49")


def test_judge_row_sends_everything_and_scores_the_reply(fake_client):
    reply = judgement([(True, "FLAT_RATE_PAISE = 4_900"), (False, "")]).model_dump_json()
    client = fake_client([reply])
    scores, parsed, usage = judge.judge_row(client, ROW, "₹49, free from ₹999")
    sent = client.histories[0][0].text
    assert "Reference answer: ₹49, free from ₹999" in sent
    assert "2. Shipping is free on Sundays. (cites source [1])" in sent
    assert scores["faithfulness"] == 0.5 and usage.input_tokens == 1000


def test_summary_averages_only_present_values():
    rows = [{"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 0.4,
             "context_recall": 1.0},
            {"faithfulness": None, "answer_relevance": 0.0, "context_precision": 0.2,
             "context_recall": 0.5}]
    assert judge.summarise(rows) == {"faithfulness": 1.0, "answer_relevance": 0.5,
                                     "context_precision": 0.3, "context_recall": 0.75}


def test_disagreement_counts_only_answers_both_labelled():
    result = judge.disagreement({"Q1#1": True, "Q1#2": True, "Q2#1": False},
                                {"Q1#1": True, "Q1#2": False, "Q3#1": True})
    assert result == {"checked": 2, "disagreed": 1, "rate": 0.5, "which": ["Q1#2"]}
    assert json.dumps(result)  # saved as JSON in runs/


def test_a_claim_naming_something_its_source_lacks_is_not_supported():
    claim = {"statement": "FLAT_RATE_PAISE is 4_900, set in pricing_v2.", "source": 1}
    row = {**ROW, "claims": [claim]}
    reply = judgement([(True, "FLAT_RATE_PAISE = 4_900")])
    assert judge.score_row(row, reply)["claims_supported"] == [False]  # pricing_v2 isn't there
    assert judge.specifics("Holds expire after 15 minutes via `RESERVATION_TTL`.") == {
        "15", "reservation_ttl"}


def test_a_no_answer_question_has_no_precision_or_recall_score():
    refused = {**ROW, "answerable": False, "answered": False, "claims": [], "refusal": "No."}
    scores = judge.score_row(refused, judgement([], answers="yes", found="all"))
    assert scores["context_precision"] is None and scores["context_recall"] is None
    assert scores["answer_relevance"] == 1.0


def test_cited_precision_counts_only_the_sources_the_answer_cites():
    # Both claims cite source 1; source 2 was shown but not cited, so it can't lower this score.
    reply = judgement([(True, "x"), (True, "y")], relevant=(1,))
    scores = judge.score_row(ROW, reply, require_quote=False)
    assert scores["cited_precision"] == 1.0 and scores["context_precision"] == 0.5
    assert judge.score_row(ROW, judgement([(True, "x"), (True, "y")], relevant=(2,)),
                           require_quote=False)["cited_precision"] == 0.0
