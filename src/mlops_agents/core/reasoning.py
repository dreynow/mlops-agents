"""ReasoningEngine - LLM-backed decision reasoning.

Protocol + implementations for Claude, OpenAI, and Ollama.
Every agent delegates its reasoning to one of these engines.
The engine takes observations and context, returns a structured ReasoningTrace.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import structlog

from mlops_agents.core.decision import ReasoningTrace

logger = structlog.get_logger()

REASONING_SYSTEM_PROMPT = """You are an MLOps decision agent. You analyze observations about ML pipeline stages and produce structured reasoning.

Given a set of observations and context about an ML pipeline stage, you must:
1. Analyze the observations carefully
2. Consider alternatives
3. Produce a clear conclusion with confidence level

Respond with valid JSON matching this schema:
{
    "observations": ["list of key observations you noted"],
    "analysis": "your reasoning about what the observations mean",
    "conclusion": "your final judgment in one sentence",
    "confidence": 0.0-1.0,
    "alternatives_considered": ["other actions you considered and why you rejected them"]
}

Be specific. Reference actual numbers from the observations. Do not hedge unnecessarily - if the data is clear, be decisive."""


@runtime_checkable
class ReasoningEngine(Protocol):
    """Protocol for LLM reasoning backends.

    Implementations: ClaudeReasoner, OpenAIReasoner, OllamaReasoner.
    """

    async def reason(
        self,
        observations: list[str],
        context: dict[str, Any],
        agent_name: str,
        action: str,
    ) -> ReasoningTrace: ...


def _build_reasoning_prompt(
    observations: list[str],
    context: dict[str, Any],
    agent_name: str,
    action: str,
) -> str:
    """Build the user prompt for LLM reasoning. Shared across all engines."""
    obs_text = "\n".join(f"- {o}" for o in observations)
    ctx_text = json.dumps(context, indent=2, default=str)
    return (
        f"Agent: {agent_name}\n"
        f"Action: {action}\n\n"
        f"Observations:\n{obs_text}\n\n"
        f"Context:\n{ctx_text}\n\n"
        f"Analyze these observations and produce your reasoning."
    )


class ClaudeReasoner:
    """Reasoning engine backed by Claude (Anthropic API)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise RuntimeError(
                    "anthropic package required for ClaudeReasoner. "
                    "Install with: pip install mlops-agents[claude]"
                )
        return self._client

    async def reason(
        self,
        observations: list[str],
        context: dict[str, Any],
        agent_name: str,
        action: str,
    ) -> ReasoningTrace:
        client = self._get_client()

        user_prompt = _build_reasoning_prompt(observations, context, agent_name, action)

        logger.info("reasoning.claude.start", agent=agent_name, action=action, model=self.model)

        response = await client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=REASONING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        return _parse_reasoning_json(response.content[0].text, self.model)


class OpenAIReasoner:
    """Reasoning engine backed by OpenAI API."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise RuntimeError(
                    "openai package required for OpenAIReasoner. "
                    "Install with: pip install mlops-agents[openai]"
                )
        return self._client

    async def reason(
        self,
        observations: list[str],
        context: dict[str, Any],
        agent_name: str,
        action: str,
    ) -> ReasoningTrace:
        client = self._get_client()

        user_prompt = _build_reasoning_prompt(observations, context, agent_name, action)

        logger.info("reasoning.openai.start", agent=agent_name, action=action, model=self.model)

        response = await client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        return _parse_reasoning_json(response.choices[0].message.content, self.model)


class OllamaReasoner:
    """Reasoning engine backed by local Ollama instance."""

    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.AsyncClient(host=self.host)
            except ImportError:
                raise RuntimeError(
                    "ollama package required for OllamaReasoner. "
                    "Install with: pip install mlops-agents[ollama]"
                )
        return self._client

    async def reason(
        self,
        observations: list[str],
        context: dict[str, Any],
        agent_name: str,
        action: str,
    ) -> ReasoningTrace:
        client = self._get_client()

        user_prompt = _build_reasoning_prompt(observations, context, agent_name, action)

        logger.info("reasoning.ollama.start", agent=agent_name, action=action, model=self.model)

        response = await client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": REASONING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )

        return _parse_reasoning_json(response["message"]["content"], self.model)


class StaticReasoner:
    """Deterministic reasoner for testing - no LLM calls.

    Returns a pre-configured ReasoningTrace. Useful for unit tests
    and dry-run pipeline validation.
    """

    def __init__(self, default_confidence: float = 0.9, default_approved: bool = True):
        self.default_confidence = default_confidence
        self.default_approved = default_approved

    async def reason(
        self,
        observations: list[str],
        context: dict[str, Any],
        agent_name: str,
        action: str,
    ) -> ReasoningTrace:
        return ReasoningTrace(
            observations=observations,
            analysis=f"Static analysis for {agent_name}/{action}: all observations within bounds.",
            conclusion=f"{'Proceed' if self.default_approved else 'Block'} with {action}.",
            confidence=self.default_confidence,
            alternatives_considered=["No alternatives evaluated (static reasoner)"],
            model_used="static-reasoner",
        )


def _parse_reasoning_json(text: str, model: str) -> ReasoningTrace:
    """Parse LLM JSON response into a ReasoningTrace."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("reasoning.parse_failed", error=str(e), raw=text[:200])
        return ReasoningTrace(
            observations=["Failed to parse LLM response"],
            analysis=text[:500],
            conclusion="Parse error - manual review required",
            confidence=0.0,
            alternatives_considered=[],
            model_used=model,
        )

    return ReasoningTrace(
        observations=data.get("observations", []),
        analysis=data.get("analysis", ""),
        conclusion=data.get("conclusion", ""),
        confidence=float(data.get("confidence", 0.5)),
        alternatives_considered=data.get("alternatives_considered", []),
        model_used=model,
    )
