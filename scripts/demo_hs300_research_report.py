from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index_platform.reporting.html_report import render_strategy_report
from index_platform.strategy.factors import build_basic_factor_panel, composite_alpha, factor_summary
from index_platform.strategy.optimization import OptimizationConstraints, heuristic_active_weight_optimizer
from index_platform.strategy.risk import performance_summary


def main() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=160)
    stocks = [f"{i:06d}.SZ" for i in range(1, 61)]
    benchmark_weight = pd.Series(rng.dirichlet(np.ones(len(stocks))), index=stocks)

    rows = []
    for i, code in enumerate(stocks):
        quality = rng.normal()
        price = 10 + rng.random() * 20
        for t, date in enumerate(dates):
            drift = 0.0001 + 0.0002 * quality
            price *= 1 + drift + rng.normal(0, 0.015)
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": date,
                    "close": price,
                    "adj_factor": 1.0,
                    "turnover_rate": abs(rng.normal(1.2, 0.25)),
                    "pe_ttm": max(4, 18 - quality * 2 + rng.normal(0, 1.5)),
                    "pb": max(0.3, 2.0 - quality * 0.2 + rng.normal(0, 0.2)),
                    "ps_ttm": max(0.2, 3.0 - quality * 0.2 + rng.normal(0, 0.3)),
                    "dv_ttm": max(0, 1.5 + quality * 0.1 + rng.normal(0, 0.2)),
                    "total_mv": 500 + i * 10 + rng.normal(0, 20),
                    "benchmark_weight": benchmark_weight[code],
                    "roe": 8 + quality * 3 + rng.normal(0, 1),
                    "roa": 4 + quality + rng.normal(0, 0.5),
                    "grossprofit_margin": 25 + quality * 2 + rng.normal(0, 2),
                }
            )

    panel = pd.DataFrame(rows)
    factor_panel = composite_alpha(build_basic_factor_panel(panel))
    factor_stats = factor_summary(factor_panel)

    eligible_dates = pd.Series(dates[60:-25])
    rebalance_dates = pd.DatetimeIndex(
        eligible_dates.groupby(eligible_dates.dt.to_period("M")).max().values
    )
    previous = None
    realized = []
    last_weights = None
    for date in rebalance_dates:
        snapshot = factor_panel[factor_panel["trade_date"].eq(date)].dropna(subset=["alpha_score"])
        opt = heuristic_active_weight_optimizer(
            snapshot,
            previous_weights=previous,
            constraints=OptimizationConstraints(max_active_weight=0.01, max_weight=0.05),
        )
        previous = opt.weights
        last_weights = opt.weights
        next_date = dates[dates.get_loc(date) + 21]
        px0 = panel[panel["trade_date"].eq(date)][["ts_code", "close"]].rename(columns={"close": "px0"})
        px1 = panel[panel["trade_date"].eq(next_date)][["ts_code", "close"]].rename(columns={"close": "px1"})
        period_returns = px0.merge(px1, on="ts_code")
        period_returns["period_return"] = period_returns["px1"] / period_returns["px0"] - 1
        merged = period_returns.merge(opt.weights, on="ts_code")
        realized.append(
            {
                "trade_date": next_date,
                "portfolio_return": (merged["target_weight"] * merged["period_return"]).sum(),
                "benchmark_return": (merged["benchmark_weight"] * merged["period_return"]).sum(),
            }
        )

    returns = pd.DataFrame(realized)
    returns["active_return"] = returns["portfolio_return"] - returns["benchmark_return"]
    perf = performance_summary(returns, periods_per_year=12)
    output = render_strategy_report(
        title="HS300 指数增强策略 Demo 报告",
        performance=perf,
        factor_summary=factor_stats,
        returns=returns,
        weights=last_weights,
        output_path=ROOT / "artifacts" / "reports" / "hs300_demo_report.html",
    )
    factor_stats.to_csv(ROOT / "artifacts" / "reports" / "factor_summary.csv", index=False)
    returns.to_csv(ROOT / "artifacts" / "reports" / "demo_returns.csv", index=False)
    last_weights.to_csv(ROOT / "artifacts" / "reports" / "latest_weights.csv", index=False)
    print(output)


if __name__ == "__main__":
    main()
