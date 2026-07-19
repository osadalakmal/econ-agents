# econ-agents

A closed-loop economic agent simulation framework. A configurable population of consumer agents and rational supplier agents interact through a shared commodity market, where price is set endogenously each round by a stock-based algorithm. Everything is driven by a YAML scenario file — no code changes needed to explore new scenarios.

## How it works

Each simulation round runs six phases in order:

```
SHOCK     exogenous event applied to the market (price, stock, or cost)
OBSERVE   world state snapshot distributed to all agents
DECIDE    all consumers and suppliers decide concurrently (asyncio.gather)
HARVEST   production orders placed lag-rounds ago arrive and add to stock
CLEAR     consumer demand settles against stock; price algorithm fires
PLACE     supplier decisions enter the production pipeline
```

### Price-setting algorithm

Uses a stock-based rule:

```
Δprice = −elasticity × ln(current_stock / target_stock)
```

`target_stock = avg_recent_demand × target_stock_days`. When stock falls below target, price rises; when above, it falls. Parameters are configurable per market.

### Consumer agents

Each agent runs as an asyncio coroutine and evaluates a behaviour spec:

| Mode | Description |
|---|---|
| `deterministic` | Evaluates ordered rules top-down; first matching condition wins |
| `stochastic` | Picks from a weighted action table |
| `mixed` | Tries deterministic rules first; falls back to stochastic |

Conditions are safe Python expressions (whitelist AST evaluator — no `eval`). Available variables: `price`, `price_change_pct`, `stock_ratio`, `stock`, `agent_savings`, `agent_inventory`, `round`.

### Supplier agents

Rational producers that observe `price`, `production_cost`, `margin` (price/cost), `stock_ratio`, and `own_pipeline_rounds` (their own committed-but-undelivered output). They evaluate a rule set to adjust their production rate each round, subject to `max_ramp_up` and `max_ramp_down` constraints. New output enters a pipeline and arrives exactly `production_lag` rounds later — the core mechanism for commodity cycle dynamics.

## Quick start

```bash
pip install pyyaml
python main.py                          # runs configs/rice_price.yaml (20 rounds)
python main.py configs/rice_price.yaml --rounds 40
python main.py configs/rice_price.yaml --output-json results.json
```

## Scenario configuration

Everything is specified in YAML. The `configs/rice_price.yaml` scenario simulates a 25% rice price shock, a mid-simulation flood, and a fuel subsidy:

```yaml
seed: 42
rounds: 20

markets:
  rice:
    initial_price: 1.00
    initial_stock: 5000
    production_cost: 0.65
    price_algorithm:
      type: stock_based
      target_stock_days: 10
      elasticity: 0.25
      demand_window: 3
      min_price: 0.20
      max_price: 4.00

suppliers:
  - id: large_producer
    count: 2
    market: rice
    production_cost: 0.60
    initial_production_rate: 300   # units/round each
    production_lag: 5              # rounds before output hits market
    max_ramp_up: 0.10
    max_ramp_down: 0.20
    rules:
      - condition: "margin > 2.0 and stock_ratio < 1.2"
        action: increase_production
        adjustment_factor: 0.10

consumers:
  size: 1000
  agent_types:
    - id: hoarder
      proportion: 0.20
      mode: deterministic
      base_quantity: 1.0
      rules:
        - condition: "price_change_pct > 15 or stock_ratio < 0.5"
          action: buy_more
          quantity_factor: 3.0
    - id: random_actor
      proportion: 0.25
      mode: stochastic
      base_quantity: 1.0
      stochastic_actions:
        - action: buy_more
          weight: 0.28
          quantity_factor: 1.3
        - action: buy_less
          weight: 0.38
          quantity_factor: 0.70

shocks:
  - round: 0
    type: price_shock      # or stock_shock, cost_shock
    market: rice
    delta_pct: 25.0
    description: "Supply chain disruption"
```

### Consumer observation variables

| Variable | Description |
|---|---|
| `price` | Current market price |
| `price_change_pct` | % change from previous round |
| `stock_ratio` | `current_stock / target_stock` — below 1.0 signals shortage |
| `stock` | Absolute stock level |
| `agent_savings` | Agent's wealth |
| `agent_inventory` | Agent's personal inventory |
| `round` | Current round number |

### Supplier observation variables

| Variable | Description |
|---|---|
| `price` | Current market price |
| `production_cost` | This supplier's unit cost |
| `margin` | `price / production_cost` |
| `price_change_pct` | % change from previous round |
| `stock_ratio` | Market stock vs. target |
| `stock` | Absolute market stock |
| `own_pipeline_total` | Units this supplier has committed but not yet delivered |
| `own_pipeline_rounds` | `own_pipeline_total / current_rate` — rounds of output already queued |
| `market_pipeline_total` | All suppliers' combined pending orders |
| `current_production_rate` | This supplier's current rate (units/round) |

### Shock types

| Type | Effect |
|---|---|
| `price_shock` | Multiplies market price by `(1 + delta_pct/100)` |
| `stock_shock` | Multiplies current stock by `(1 + delta_pct/100)` — use negative to destroy stock |
| `cost_shock` | Changes production cost for all suppliers in that market |

## Project layout

```
sim/
  market.py       Market: stock, production pipeline, price algorithm
  supplier.py     SupplierAgent: rational producer with production lag
  agent.py        ConsumerAgent: asyncio coroutine, behaviour engine
  behaviors.py    Rule engine + safe AST expression evaluator
  events.py       Shock and decision data classes
  world.py        World: container for markets; produces snapshots
  simulation.py   Six-phase round loop
  reporting.py    Console output and JSON serialisation
configs/
  rice_price.yaml Example scenario: rice price shock
main.py           CLI entry point
```

## Observable dynamics

The default scenario (20 rounds) demonstrates the **commodity cycle**:

1. **Rounds 0–3**: Price shock triggers heterogeneous consumer response — hoarders panic-buy, government-trusters cut back, budget-constrained are squeezed. Stock depletes.
2. **Rounds 4–8**: Supplier pipeline orders placed at high margins begin arriving. Supply catches up.
3. **Round 6**: Flood destroys 20% of stock — absorbed by incoming supply in this scenario; reduces initial stock in config to observe actual shortage.
4. **Round 12**: Fuel subsidy cuts production costs → margins improve → suppliers ramp harder.
5. **Rounds 15–16**: Stock exceeds target buffer; price begins falling.
6. **Rounds 17+**: Suppliers observe `stock_ratio > 2.0` and cut production; consumers restock on falling prices.

The production lag is the key parameter driving overshoot: long lags mean suppliers commit to high output rates before market conditions change, producing the classic cobweb / bullwhip oscillation.
