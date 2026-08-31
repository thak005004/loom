"""The one seam both Day 2 LLM consumers (nl_parser, explainer) go
through. `LLMClient` is a structural Protocol — a fake test double just
needs a `.complete(prompt) -> str` method, no inheritance required —
so the parser and explainer's actual logic can be tested deterministically
and offline, without a network call or an API key. `AnthropicLLMClient`
is the real implementation and is not exercised by the test suite.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class AnthropicLLMClient:
    """Thin wrapper around the real Claude API. Requires ANTHROPIC_API_KEY
    in the environment; not covered by tests for that reason."""

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 300) -> None:
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))
