"""Tests for the whole-repository baseline (fake model, no network)."""

from askmyrepo import baseline
from askmyrepo.manifest import CorpusFile

FILES = [
    CorpusFile("shoplite", "shoplite/utils/ids.py", "python", "abc1234", "def order_number(): ..."),
    CorpusFile("shoplite", "docs/api.md", "markdown", "abc1234", "# ShopLite API"),
]


def test_every_file_is_pasted_under_its_path():
    text = baseline.repository_text(FILES)
    assert text == ("=== shoplite/shoplite/utils/ids.py ===\ndef order_number(): ...\n\n"
                    "=== shoplite/docs/api.md ===\n# ShopLite API")


def test_the_question_follows_the_whole_repository(fake_client):
    client = fake_client(["One coupon per order."])
    text, usage, seconds = baseline.ask_whole_repository(client, FILES, "How many coupons?")
    sent = client.histories[0][0].text
    assert sent.startswith("=== shoplite/shoplite/utils/ids.py ===")
    assert sent.endswith("Question: How many coupons?")
    assert text == "One coupon per order." and usage.input_tokens == 1000 and seconds >= 0


def test_tokens_come_from_the_providers_counting_call(fake_client):
    assert baseline.count_repository_tokens(fake_client([]), FILES) == 42
