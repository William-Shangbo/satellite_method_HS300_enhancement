from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizationConstraints:
    max_active_weight: float = 0.01
    max_weight: float = 0.03
    active_share_budget: float = 0.30
    max_turnover: float | None = 0.40
    long_quantile: float = 0.30
    short_quantile: float = 0.30


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.DataFrame
    diagnostics: dict[str, float]


def heuristic_active_weight_optimizer(
    alpha_snapshot: pd.DataFrame,
    *,
    previous_weights: pd.DataFrame | None = None,
    constraints: OptimizationConstraints = OptimizationConstraints(),
) -> OptimizationResult:
    """Create long-only benchmark-relative weights from alpha scores without QP deps."""
    required = {"ts_code", "benchmark_weight", "alpha_score"}
    missing = required.difference(alpha_snapshot.columns)
    if missing:
        raise ValueError(f"alpha_snapshot missing columns: {sorted(missing)}")

    data = alpha_snapshot[["ts_code", "benchmark_weight", "alpha_score"]].dropna().copy()
    data["benchmark_weight"] = data["benchmark_weight"] / data["benchmark_weight"].sum()
    score = data["alpha_score"] - data["alpha_score"].mean()
    denom = score.abs().sum()
    if denom == 0:
        data["target_weight"] = data["benchmark_weight"]
    else:
        if 0 < constraints.long_quantile < 0.5 and 0 < constraints.short_quantile < 0.5:
            long_cut = data["alpha_score"].quantile(1 - constraints.long_quantile)
            short_cut = data["alpha_score"].quantile(constraints.short_quantile)
            score = score.where((data["alpha_score"] >= long_cut) | (data["alpha_score"] <= short_cut), 0.0)
            denom = score.abs().sum()
        active = score / denom * constraints.active_share_budget if denom else score
        active = active.clip(-constraints.max_active_weight, constraints.max_active_weight)
        data["target_weight"] = data["benchmark_weight"] + active

    data["target_weight"] = data["target_weight"].clip(lower=0, upper=constraints.max_weight)
    data["target_weight"] = data["target_weight"] / data["target_weight"].sum()

    if previous_weights is not None and constraints.max_turnover is not None:
        data = _apply_turnover_limit(data, previous_weights, constraints.max_turnover)

    data["active_weight"] = data["target_weight"] - data["benchmark_weight"]
    diagnostics = {
        "gross_active_weight": float(data["active_weight"].abs().sum()),
        "max_abs_active_weight": float(data["active_weight"].abs().max()),
        "max_weight": float(data["target_weight"].max()),
        "portfolio_stock_count": float((data["target_weight"] > 0).sum()),
        "expected_alpha_score": float((data["target_weight"] * data["alpha_score"]).sum()),
    }
    return OptimizationResult(weights=data.sort_values("target_weight", ascending=False), diagnostics=diagnostics)


def realized_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    weight_col: str = "target_weight",
    return_col: str = "period_return",
) -> pd.DataFrame:
    data = returns.merge(weights[["ts_code", weight_col, "benchmark_weight"]], on="ts_code", how="inner")
    portfolio_return = float((data[weight_col] * data[return_col]).sum())
    benchmark_return = float((data["benchmark_weight"] * data[return_col]).sum())
    return pd.DataFrame(
        [
            {
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "active_return": portfolio_return - benchmark_return,
            }
        ]
    )


def estimate_tracking_error(
    active_weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    periods_per_year: int = 252,
) -> float:
    common = active_weights.index.intersection(covariance.index).intersection(covariance.columns)
    if common.empty:
        return float("nan")
    w = active_weights.loc[common].to_numpy(dtype=float)
    cov = covariance.loc[common, common].to_numpy(dtype=float)
    variance = float(w @ cov @ w.T)
    return float(np.sqrt(max(variance, 0)) * np.sqrt(periods_per_year))


def _apply_turnover_limit(
    target: pd.DataFrame,
    previous_weights: pd.DataFrame,
    max_turnover: float,
) -> pd.DataFrame:
    prev = previous_weights[["ts_code", "target_weight"]].rename(columns={"target_weight": "previous_weight"})
    data = target.merge(prev, on="ts_code", how="left")
    data["previous_weight"] = data["previous_weight"].fillna(0.0)
    turnover = 0.5 * (data["target_weight"] - data["previous_weight"]).abs().sum()
    if turnover <= max_turnover:
        return data.drop(columns=["previous_weight"])

    blend = max_turnover / turnover
    data["target_weight"] = data["previous_weight"] + blend * (data["target_weight"] - data["previous_weight"])
    data["target_weight"] = data["target_weight"].clip(lower=0)
    data["target_weight"] = data["target_weight"] / data["target_weight"].sum()
    return data.drop(columns=["previous_weight"])
