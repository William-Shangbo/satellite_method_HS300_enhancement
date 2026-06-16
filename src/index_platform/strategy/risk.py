from __future__ import annotations

import numpy as np
import pandas as pd


def performance_summary(
    returns: pd.DataFrame,
    *,
    portfolio_col: str = "portfolio_return",
    benchmark_col: str = "benchmark_return",
    periods_per_year: int = 252,
) -> pd.Series:
    data = returns[[portfolio_col, benchmark_col]].dropna().copy()
    active = data[portfolio_col] - data[benchmark_col]
    portfolio_curve = (1 + data[portfolio_col]).cumprod()
    benchmark_curve = (1 + data[benchmark_col]).cumprod()
    active_curve = (1 + active).cumprod()

    ann_return = portfolio_curve.iloc[-1] ** (periods_per_year / len(data)) - 1
    bench_ann_return = benchmark_curve.iloc[-1] ** (periods_per_year / len(data)) - 1
    active_ann_return_arithmetic = active.mean() * periods_per_year
    active_ann_return_geometric = active_curve.iloc[-1] ** (periods_per_year / len(data)) - 1
    annual_return_spread = ann_return - bench_ann_return
    tracking_error = active.std(ddof=1) * np.sqrt(periods_per_year)
    return pd.Series(
        {
            "annual_return": ann_return,
            "benchmark_annual_return": bench_ann_return,
            "annual_excess_return": active_ann_return_arithmetic,
            "annual_active_return_arithmetic": active_ann_return_arithmetic,
            "annual_active_return_geometric": active_ann_return_geometric,
            "annual_return_spread": annual_return_spread,
            "annual_volatility": data[portfolio_col].std(ddof=1) * np.sqrt(periods_per_year),
            "tracking_error": tracking_error,
            "information_ratio": active_ann_return_arithmetic / tracking_error if tracking_error else np.nan,
            "max_drawdown": max_drawdown(portfolio_curve),
            "max_active_drawdown": max_drawdown(active_curve),
            "period_win_rate": (active > 0).mean(),
        }
    )


def max_drawdown(curve: pd.Series) -> float:
    curve = curve.dropna()
    if curve.empty:
        return float("nan")
    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    return float(drawdown.min())


def trailing_tracking_error(
    returns: pd.DataFrame,
    *,
    active_return_col: str = "active_return",
    window: int = 60,
    periods_per_year: int = 252,
) -> pd.Series:
    active = returns[active_return_col].dropna()
    return active.rolling(window, min_periods=max(5, window // 3)).std() * np.sqrt(periods_per_year)


def should_reduce_risk(
    active_returns: pd.Series,
    *,
    max_active_drawdown_limit: float = -0.05,
    te_limit: float = 0.06,
    window: int = 60,
) -> bool:
    frame = pd.DataFrame({"active_return": active_returns.dropna()})
    if frame.empty:
        return False
    active_curve = (1 + frame["active_return"]).cumprod()
    current_drawdown = active_curve.iloc[-1] / active_curve.cummax().iloc[-1] - 1
    recent_te = trailing_tracking_error(frame, window=window).dropna()
    return bool(current_drawdown < max_active_drawdown_limit or (not recent_te.empty and recent_te.iloc[-1] > te_limit))
