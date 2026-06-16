from __future__ import annotations

import numpy as np
import pandas as pd


INFO_COLUMNS = {
    "ts_code",
    "trade_date",
    "industry",
    "benchmark_weight",
    "forward_return_1d",
    "forward_return_21d",
}


def winsorize_by_date(frame: pd.DataFrame, columns: list[str], limit: float = 0.025) -> pd.DataFrame:
    result = frame.copy()
    for col in columns:
        q_low = result.groupby("trade_date")[col].transform(lambda s: s.quantile(limit))
        q_high = result.groupby("trade_date")[col].transform(lambda s: s.quantile(1 - limit))
        result[col] = result[col].clip(q_low, q_high)
    return result


def zscore_by_date(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for col in columns:
        mean = result.groupby("trade_date")[col].transform("mean")
        std = result.groupby("trade_date")[col].transform("std").replace(0, np.nan)
        result[col] = (result[col] - mean) / std
    return result


def build_basic_factor_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Create first-pass daily factor features from Tushare daily and daily_basic data."""
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    if "adj_close" not in data.columns:
        latest_factor = data.groupby("ts_code")["adj_factor"].transform("last")
        data["adj_close"] = data["close"] * data["adj_factor"] / latest_factor
    data["return_1d"] = data.groupby("ts_code")["adj_close"].pct_change()
    data["forward_return_21d"] = data.groupby("ts_code")["adj_close"].pct_change(21).shift(-21)

    data["value_ep"] = _inverse_positive(data.get("pe_ttm"))
    data["value_bp"] = _inverse_positive(data.get("pb"))
    data["value_sp"] = _inverse_positive(data.get("ps_ttm"))
    data["dividend_yield"] = pd.to_numeric(data.get("dv_ttm"), errors="coerce")
    data["size_neg_log_mv"] = -np.log(pd.to_numeric(data.get("total_mv"), errors="coerce").where(lambda s: s > 0))

    grouped = data.groupby("ts_code", group_keys=False)
    data["momentum_60d"] = grouped["adj_close"].pct_change(60)
    data["reversal_20d"] = -grouped["adj_close"].pct_change(20)
    data["volatility_20d"] = -grouped["return_1d"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
    data["turnover_20d"] = -grouped["turnover_rate"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)

    if "roe" in data.columns:
        data["quality_roe"] = pd.to_numeric(data["roe"], errors="coerce")
    if "roa" in data.columns:
        data["quality_roa"] = pd.to_numeric(data["roa"], errors="coerce")
    if "grossprofit_margin" in data.columns:
        data["quality_gross_margin"] = pd.to_numeric(data["grossprofit_margin"], errors="coerce")

    factor_cols = factor_columns(data)
    data = winsorize_by_date(data, factor_cols)
    data = zscore_by_date(data, factor_cols)
    return data


def factor_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        "value_",
        "dividend_",
        "size_",
        "momentum_",
        "reversal_",
        "volatility_",
        "turnover_",
        "quality_",
        "growth_",
        "cashflow_",
        "leverage_",
    )
    return [col for col in frame.columns if col.startswith(prefixes)]


def composite_alpha(
    factor_panel: pd.DataFrame,
    *,
    factor_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    result = factor_panel.copy()
    cols = factor_columns(result)
    if not cols:
        raise ValueError("No factor columns found for composite alpha.")

    weights = pd.Series(factor_weights or {col: 1.0 for col in cols}, dtype=float)
    weights = weights.reindex(cols).fillna(0.0)
    if weights.abs().sum() == 0:
        raise ValueError("At least one factor weight must be non-zero.")
    weights = weights / weights.abs().sum()
    result["alpha_score"] = result[cols].mul(weights, axis=1).sum(axis=1, skipna=True)
    result = zscore_by_date(result, ["alpha_score"])
    return result


def rank_ic_by_date(
    factor_panel: pd.DataFrame,
    factor: str,
    forward_return_col: str = "forward_return_21d",
) -> pd.Series:
    data = factor_panel[["trade_date", factor, forward_return_col]].dropna().copy()
    return data.groupby("trade_date").apply(
        lambda df: df[factor].rank().corr(df[forward_return_col].rank())
    )


def factor_summary(
    factor_panel: pd.DataFrame,
    factors: list[str] | None = None,
    forward_return_col: str | None = None,
) -> pd.DataFrame:
    if forward_return_col is None:
        forward_return_col = "period_return" if "period_return" in factor_panel.columns else "forward_return_21d"
    rows = []
    for factor in factors or factor_columns(factor_panel):
        ic = rank_ic_by_date(factor_panel, factor, forward_return_col).dropna()
        if ic.empty:
            continue
        rows.append(
            {
                "factor": factor,
                "rank_ic_mean": ic.mean(),
                "rank_ic_std": ic.std(),
                "rank_ic_ir": ic.mean() / ic.std() if ic.std() else np.nan,
                "rank_ic_positive_rate": (ic > 0).mean(),
                "observations": len(ic),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ic_ir", ascending=False)


def _inverse_positive(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    return 1 / numeric.where(numeric > 0)
