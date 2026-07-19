"""
Behavior engine — interprets YAML-defined consumer logic.

Agent modes:
  deterministic  — evaluates ordered rules; first match wins
  stochastic     — picks action from a weighted probability table
  mixed          — tries deterministic rules first; falls back to stochastic

quantity_demanded semantics (stock model):
  buy_more / hoard  → base_quantity * factor   (factor > 1 → hoarding)
  buy_less / reduce → base_quantity * factor   (factor < 1 → rationing)
  hold / wait       → 0.0                      (drawing down own inventory)
  no_change         → base_quantity            (normal consumption)
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
            raise KeyError(f"Unknown variable: '{node.id}'")
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
# Behavior spec data classes
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    condition: str
    action: str
    quantity_factor: float = 1.0
    label: str = ""


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
    base_quantity: float = 1.0
    rules: list[Rule] = field(default_factory=list)
    stochastic_actions: list[WeightedAction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Behavior engine
# ---------------------------------------------------------------------------

def _action_to_quantity(action: str, base: float, factor: float) -> float:
    """Convert action + factor into a concrete quantity demanded (>= 0)."""
    if action in ("buy_more", "hoard", "stockpile"):
        return base * factor
    if action in ("buy_less", "reduce", "abstain"):
        return base * factor   # factor < 1 encodes the reduction
    if action in ("hold", "wait"):
        return 0.0             # drawing down own inventory this round
    # no_change, maintain: normal baseline consumption
    return base


class BehaviorEngine:

    def __init__(self, spec: BehaviorSpec, rng: random.Random) -> None:
        self.spec = spec
        self._rng = rng

    def decide(self, obs: dict[str, Any]) -> tuple[str, float, str]:
        """Return (action, quantity_demanded, reasoning)."""
        mode = self.spec.mode
        if mode == "deterministic":
            return self._deterministic(obs)
        if mode == "stochastic":
            return self._stochastic()
        if mode == "mixed":
            action, qty, reason = self._deterministic(obs)
            if action != "no_change":
                return action, qty, reason
            return self._stochastic()
        raise ValueError(f"Unknown mode: {mode}")

    def _deterministic(self, obs: dict[str, Any]) -> tuple[str, float, str]:
        for rule in self.spec.rules:
            try:
                if eval_condition(rule.condition, obs):
                    qty = _action_to_quantity(
                        rule.action, self.spec.base_quantity, rule.quantity_factor
                    )
                    return rule.action, qty, rule.label or rule.condition
            except Exception:
                continue
        return "no_change", self.spec.base_quantity, "no rule matched"

    def _stochastic(self) -> tuple[str, float, str]:
        actions = self.spec.stochastic_actions
        weights = [a.weight for a in actions]
        chosen = self._rng.choices(actions, weights=weights, k=1)[0]
        qty = _action_to_quantity(
            chosen.action, self.spec.base_quantity, chosen.quantity_factor
        )
        return chosen.action, qty, "stochastic"


# ---------------------------------------------------------------------------
# Parse YAML config
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
