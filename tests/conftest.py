"""Shared fixtures: a fake model client."""

import pytest

from askmyrepo.llm import Message, ModelReply, ToolCall, Usage


class FakeClient:
    """A ModelClient that replays scripted turns and records what it was sent."""

    model = "gemini-3.5-flash-lite"

    def __init__(self, turns: list):
        """turns: each is a list of (tool_name, args) calls, or a final answer string."""
        self.turns = list(turns)
        self.histories = []

    def reply(self, system, history, tools, output_schema=None) -> ModelReply:
        """Return the next scripted turn."""
        self.histories.append(list(history))
        turn = self.turns.pop(0) if self.turns else []  # nothing left: keep asking for nothing
        usage = Usage(input_tokens=1000, output_tokens=100)
        if isinstance(turn, str):
            return ModelReply(turn, [], usage, Message(role="model", text=turn))
        calls = [ToolCall(id=f"c{i}", name=name, args=args) for i, (name, args) in enumerate(turn)]
        return ModelReply("", calls, usage, Message(role="model", tool_calls=calls))

    def count_tokens(self, history) -> int:
        """Pretend count."""
        return 42


@pytest.fixture
def fake_client():
    """Return the FakeClient class so tests can script turns."""
    return FakeClient
