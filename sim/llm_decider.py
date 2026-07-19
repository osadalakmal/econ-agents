"""LLM decision backend — OpenAI-compatible local endpoint (e.g. LM Studio)."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"buy_more", "buy_less", "hold", "no_change"})
_warned_no_openai = False   # suppress repeated "not installed" noise

# JSON schema passed as structured-output constraint to the endpoint.
# Fail loudly if the endpoint cannot honour it rather than regex-scraping free text.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["buy_more", "buy_less", "hold", "no_change"],
        },
        "quantity_factor": {"type": "number", "minimum": 0.0, "maximum": 10.0},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "quantity_factor"],
    "additionalProperties": False,
}


@dataclass
class LLMDeciderConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = "qwen/qwen3-4b-thinking-2507:2"
    timeout_s: float = 8.0
    max_retries: int = 2
    fallback_action: str = "no_change"
    fallback_quantity_factor: float = 1.0


def _build_prompt(state: dict[str, Any]) -> tuple[str, str]:
    """
    Build (system, user) prompts from raw observation numbers.

    Deliberately avoids persona labels ("you are a hoarder") — the numbers alone
    differentiate agents via their per-agent traits. Mixing persona labels with
    numeric signals causes agents to converge toward a uniform "average persona"
    response that ignores the actual market state.
    """
    system = (
        "You are a market participant deciding how much of a commodity to buy this round. "
        'Respond with ONLY a JSON object in this exact format: {"action": "...", "quantity_factor": 1.0, "reasoning": "..."} '
        "action must be one of: buy_more, buy_less, hold, no_change. "
        "quantity_factor: >1.0 for buy_more (e.g. 1.5 = 50%% more), <1.0 for buy_less (e.g. 0.7 = 30%% less), 1.0 for no_change. "
        "No text outside the JSON object."
    )
    user = (
        "=== Market signals ===\n"
        f"price:              {state.get('price', 0.0):.4f}\n"
        f"price_change_pct:   {state.get('price_change_pct', 0.0):+.1f}%%\n"
        f"stock_ratio:        {state.get('stock_ratio', 1.0):.3f}  "
        "(< 1.0 = shortage, > 1.0 = surplus)\n"
        f"round:              {state.get('round', 0)}\n"
        "\n=== Your situation ===\n"
        f"savings:                  {state.get('agent_savings', 100.0):.1f}\n"
        f"inventory:                {state.get('agent_inventory', 10.0):.1f}\n"
        f"risk_tolerance:           {state.get('risk_tolerance', 0.5):.2f}  "
        "(0 = very cautious, 1 = bold)\n"
        f"planning_horizon_rounds:  {state.get('planning_horizon_rounds', 3)}\n"
        "\nWhat is your buying decision?"
    )
    return system, user


async def decide_llm(
    state: dict[str, Any],
    config: LLMDeciderConfig,
) -> tuple[str, float, str]:
    """
    Call a local LLM to pick a buy/sell decision.

    Returns (action, quantity_factor, reasoning) — same signature as
    BehaviorEngine.decide() so callers need no special-casing.

    Falls back to (fallback_action, fallback_quantity_factor, reason) on timeout
    or server unavailability rather than crashing the round.
    """
    global _warned_no_openai
    try:
        from openai import AsyncOpenAI
    except ImportError:
        if not _warned_no_openai:
            log.warning("openai package not installed — LLM mode unavailable, falling back (install with: pip install openai)")
            _warned_no_openai = True
        return (
            config.fallback_action,
            config.fallback_quantity_factor,
            "llm-fallback: openai not installed",
        )

    client = AsyncOpenAI(base_url=config.base_url, api_key="local")
    system_prompt, user_prompt = _build_prompt(state)
    last_exc: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                    max_tokens=80,
                ),
                timeout=config.timeout_s,
            )
            raw = resp.choices[0].message.content or ""
            # Thinking models prepend <think>...</think> — strip before parsing.
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            if not raw:
                raise ValueError("empty response after stripping think blocks")
            data = json.loads(raw)
            action = data.get("action", config.fallback_action)
            if action not in _VALID_ACTIONS:
                log.warning("LLM returned invalid action %r — using fallback", action)
                action = config.fallback_action
            qf = max(0.0, min(10.0, float(data.get("quantity_factor", config.fallback_quantity_factor))))
            reasoning = data.get("reasoning", "llm")
            return action, qf, f"llm: {reasoning}"

        except asyncio.TimeoutError:
            last_exc = TimeoutError(f"timeout after {config.timeout_s}s")
        except Exception as exc:
            last_exc = exc
        log.debug("LLM attempt %d/%d failed: %s", attempt + 1, config.max_retries + 1, last_exc)

    log.warning(
        "LLM unreachable after %d attempts — falling back to '%s' (%s)",
        config.max_retries + 1,
        config.fallback_action,
        last_exc,
    )
    return (
        config.fallback_action,
        config.fallback_quantity_factor,
        f"llm-fallback: {type(last_exc).__name__}",
    )
