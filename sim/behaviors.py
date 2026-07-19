"""
Behavior engine — interprets YAML-defined agent logic.

Agent modes:
  deterministic  — evaluates ordered rules; first match wins
  stochastic     — picks action from a weighted probability table
  mixed          — tries deterministic rules first; falls back to stochastic
"""
from __future__ import annotations

import ast
import operator as _op
import random
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Safe expression evaluator for rule conditions
# ---------------------------------------------------------------------------

_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Gt: _op.gt,
    ast.GtE: _op.ge,
    ast.Lt: _op.lt,
    ast.LtE: _op.le,
    ast.Eq: _op.eq,
    ast.NotEq: _op.ne,
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
    ast.Not: _op.not_,
    ast.USub: _op.neg,
}


def _eval(node: ast.AST, ctx: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, ctx)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in ctx:
            raise KeyError(f"Unknown variable in condition: '{node.id}'")
        return ctx[node.id]
    if isinstance(node, ast.BinOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval(node.left, ctx), _eval(node.right, ctx))
    if isinstance(node, ast.UnaryOp):
        op = _OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary op: {type(node.op).__name__}")
        return op(_eval(node.operand, ctx))
    if isinstance(node, ast.BoolOp):
        op = _OPS[type(node.op)]
        result = _eval(node.values[0], ctx)
        for val in node.values[1:]:
            result = op(result, _eval(val, ctx))
        return result
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ctx)
        for cmp_op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, ctx)
            if not _OPS[type(cmp_op)](left, right):
                return False
            left = right
        return True
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def eval_condition(expr: str, ctx: dict[str, Any]) -> bool:
    tree = ast.parse(expr, mode="eval")
    return bool(_eval(tree, ctx))


# ---------------------------------------------------------------------------
# Data classes representing parsed behavior config
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    condition: str           # e.g. "price_change_pct > 15"
    action: str              # e.g. "buy_more"
    quantity_factor: float = 1.0
    label: str = ""          # human label for reporting


@dataclass
class WeightedAction:
    action: str
    weight: float
    quantity_factor: float = 1.0


@dataclass
class BehaviorSpec:
    agent_type_id: str
    mode: str                          # deterministic | stochastic | mixed
    proportion: float
    base_quantity: float = 1.0         # baseline units per round
    rules: list[Rule] = field(default_factory=list)
    stochastic_actions: list[WeightedAction] = field(default_factory=list)
    # mixed: try rules first, fall back to stochastic if no rule matches


# ---------------------------------------------------------------------------
# Behavior engine
# ---------------------------------------------------------------------------

class BehaviorEngine:

    def __init__(self, spec: BehaviorSpec, rng: random.Random) -> None:
        self.spec = spec
        self._rng = rng

    def decide(self, observation: dict[str, Any]) -> tuple[str, float, str]:
        """
        Returns (action, quantity_delta, reasoning).
        quantity_delta is signed: positive = buy more, negative = buy less.
        """
        mode = self.spec.mode

        if mode == "deterministic":
            return self._deterministic(observation)
        if mode == "stochastic":
            return self._stochastic()
        if mode == "mixed":
            action, qty, reason = self._deterministic(observation)
            if action != "no_change":
                return action, qty, reason
            return self._stochastic()
        raise ValueError(f"Unknown mode: {mode}")

    def _deterministic(self, obs: dict[str, Any]) -> tuple[str, float, str]:
        for rule in self.spec.rules:
            try:
                if eval_condition(rule.condition, obs):
                    qty = self.spec.base_quantity * rule.quantity_factor
                    return rule.action, _signed(rule.action, qty), rule.label or rule.condition
            except Exception:
                continue
        return "no_change", 0.0, "no rule matched"

    def _stochastic(self) -> tuple[str, float, str]:
        actions = self.spec.stochastic_actions
        weights = [a.weight for a in actions]
        chosen = self._rng.choices(actions, weights=weights, k=1)[0]
        qty = self.spec.base_quantity * chosen.quantity_factor
        return chosen.action, _signed(chosen.action, qty), "stochastic"


def _signed(action: str, qty: float) -> float:
    """Convert action name to signed quantity delta."""
    if action in ("buy_more", "hoard", "stockpile"):
        return abs(qty)
    if action in ("buy_less", "reduce", "abstain"):
        return -abs(qty)
    return 0.0  # hold, wait, no_change, etc.


# ---------------------------------------------------------------------------
# Parse YAML config into BehaviorSpec list
# ---------------------------------------------------------------------------

def parse_behavior_specs(agent_types: list[dict]) -> list[BehaviorSpec]:
    specs = []
    for cfg in agent_types:
        rules = [
            Rule(
                condition=r["condition"],
                action=r["action"],
                quantity_factor=r.get("quantity_factor", 1.0),
                label=r.get("label", ""),
            )
            for r in cfg.get("rules", [])
        ]
        stochastic = [
            WeightedAction(
                action=a["action"],
                weight=a["weight"],
                quantity_factor=a.get("quantity_factor", 1.0),
            )
            for a in cfg.get("stochastic_actions", [])
        ]
        specs.append(BehaviorSpec(
            agent_type_id=cfg["id"],
            mode=cfg["mode"],
            proportion=cfg["proportion"],
            base_quantity=cfg.get("base_quantity", 1.0),
            rules=rules,
            stochastic_actions=stochastic,
        ))
    return specs
