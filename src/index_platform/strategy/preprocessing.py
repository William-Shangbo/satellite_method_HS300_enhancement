from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from index_platform.strategy.factors import factor_columns, winsorize_by_date, zscore_by_date


@dataclass(frozen=True)
class FactorPreprocessConfig:
    min_factor_coverage: float = 0.80
    min_stock_coverage: float = 0.60
    winsor_limit: float = 0.025
    correlation_threshold: float = 0.80
    require_industry: bool = True


@dataclass(frozen=True)
class FactorPreprocessResult:
    panel: pd.DataFrame
    coverage: pd.DataFrame
    selected_factors: list[str]
    dropped_factors: pd.DataFrame
    correlation_matrix: pd.DataFrame
    summary: pd.DataFrame


def preprocess_factor_panel(
    factor_monthly: pd.DataFrame,
    *,
    config: FactorPreprocessConfig = FactorPreprocessConfig(),
) -> FactorPreprocessResult:
    """Clean factor inputs on the actual rebalance universe before alpha construction."""
    result = factor_monthly.copy()
    factors = factor_columns(result)
    if not factors:
        raise ValueError("No factor columns found for preprocessing.")
    if config.require_industry and "industry" not in result.columns:
        raise ValueError("Industry column is required for industry neutralization.")

    coverage = factor_coverage_by_date(result, factors)
    selected = select_factors_by_coverage(coverage, min_coverage=config.min_factor_coverage)
    if not selected:
        raise ValueError("No factors passed the coverage filter.")

    result = result.copy()
    result["factor_valid_count_raw"] = result[selected].notna().sum(axis=1)
    result["factor_valid_ratio_raw"] = result["factor_valid_count_raw"] / len(selected)
    result["passes_stock_coverage"] = result["factor_valid_ratio_raw"] >= config.min_stock_coverage

    selected = [col for col in selected if col in result.columns]
    result = winsorize_by_date(result, selected, limit=config.winsor_limit)
    result = zscore_by_date(result, selected)
    result = neutralize_by_industry(result, selected, industry_col="industry")
    result = fill_factor_nans_by_date_industry(result, selected, industry_col="industry")
    result = zscore_by_date(result, selected)
    result = neutralize_by_industry(result, selected, industry_col="industry")
    result = zscore_by_date(result, selected)

    corr = mean_factor_correlation(result, selected)
    ic_summary = factor_ic_quality(result, selected)
    selected_corr, dropped = select_factors_by_correlation(
        corr,
        ic_summary,
        threshold=config.correlation_threshold,
    )
    if not selected_corr:
        raise ValueError("No factors passed the correlation filter.")

    dropped_rows = [
        {"factor": factor, "reason": "coverage", "paired_with": ""}
        for factor in factors
        if factor not in selected
    ]
    dropped_rows.extend(dropped.to_dict("records"))
    dropped_factors = pd.DataFrame(dropped_rows, columns=["factor", "reason", "paired_with"])

    result["factor_valid_count_final"] = result[selected_corr].notna().sum(axis=1)
    result["factor_valid_ratio_final"] = result["factor_valid_count_final"] / len(selected_corr)
    for factor in factors:
        if factor not in selected_corr:
            result = result.drop(columns=[factor], errors="ignore")

    summary = pd.DataFrame(
        [
            {
                "raw_factor_count": len(factors),
                "coverage_selected_count": len(selected),
                "final_selected_count": len(selected_corr),
                "dropped_factor_count": len(set(factors).difference(selected_corr)),
                "min_factor_coverage": config.min_factor_coverage,
                "min_stock_coverage": config.min_stock_coverage,
                "correlation_threshold": config.correlation_threshold,
                "rows_after_stock_coverage_filter": int(result["passes_stock_coverage"].sum()),
                "rows_preserved_for_benchmark_calendar": len(result),
                "rebalance_months_preserved": result["rebalance_date"].nunique(),
                "industry_count": result["industry"].nunique() if "industry" in result.columns else np.nan,
            }
        ]
    )
    return FactorPreprocessResult(
        panel=result,
        coverage=coverage,
        selected_factors=selected_corr,
        dropped_factors=dropped_factors,
        correlation_matrix=corr,
        summary=summary,
    )


def factor_coverage_by_date(frame: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    for date, current in frame.groupby("rebalance_date"):
        row = {"rebalance_date": date, "stock_count": len(current)}
        for factor in factors:
            row[factor] = current[factor].notna().mean()
        rows.append(row)
    coverage = pd.DataFrame(rows).sort_values("rebalance_date")
    long_rows = []
    for factor in factors:
        values = coverage[factor].dropna()
        long_rows.append(
            {
                "factor": factor,
                "mean_coverage": values.mean(),
                "min_coverage": values.min(),
                "coverage_below_80pct_months": int((values < 0.80).sum()),
                "observed_months": int(values.count()),
            }
        )
    return pd.DataFrame(long_rows).sort_values("mean_coverage", ascending=False)


def select_factors_by_coverage(coverage: pd.DataFrame, *, min_coverage: float) -> list[str]:
    keep = coverage[coverage["mean_coverage"] >= min_coverage]
    return keep["factor"].tolist()


def neutralize_by_industry(
    frame: pd.DataFrame,
    factors: list[str],
    *,
    industry_col: str = "industry",
) -> pd.DataFrame:
    result = frame.copy()
    if industry_col not in result.columns:
        raise ValueError(f"Missing {industry_col!r} for industry neutralization.")

    for factor in factors:
        industry_mean = result.groupby(["rebalance_date", industry_col])[factor].transform("mean")
        result[factor] = result[factor] - industry_mean
    return result


def fill_factor_nans_by_date_industry(
    frame: pd.DataFrame,
    factors: list[str],
    *,
    industry_col: str = "industry",
) -> pd.DataFrame:
    result = frame.copy()
    for factor in factors:
        industry_median = result.groupby(["rebalance_date", industry_col])[factor].transform("median")
        date_median = result.groupby("rebalance_date")[factor].transform("median")
        result[factor] = result[factor].fillna(industry_median).fillna(date_median).fillna(0.0)
    return result


def mean_factor_correlation(frame: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    matrices = []
    for _, current in frame.groupby("rebalance_date"):
        corr = current[factors].corr()
        if corr.notna().values.sum():
            matrices.append(corr)
    if not matrices:
        return pd.DataFrame(index=factors, columns=factors, dtype=float)
    stacked = pd.concat(matrices, keys=range(len(matrices)))
    return stacked.groupby(level=1).mean().reindex(index=factors, columns=factors)


def factor_ic_quality(frame: pd.DataFrame, factors: list[str]) -> pd.Series:
    scores = {}
    for factor in factors:
        values = []
        for _, current in frame.groupby("rebalance_date"):
            data = current[[factor, "period_return"]].dropna()
            if len(data) >= 50:
                values.append(data[factor].rank().corr(data["period_return"].rank()))
        ic = pd.Series(values, dtype=float).dropna()
        scores[factor] = ic.mean() / ic.std(ddof=1) if len(ic) >= 6 and ic.std(ddof=1) > 0 else 0.0
    return pd.Series(scores, dtype=float)


def select_factors_by_correlation(
    corr: pd.DataFrame,
    quality: pd.Series,
    *,
    threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    remaining = list(corr.columns)
    dropped_rows = []
    abs_corr = corr.abs()

    while True:
        high_pairs = []
        for i, left in enumerate(remaining):
            for right in remaining[i + 1 :]:
                value = abs_corr.loc[left, right]
                if pd.notna(value) and value >= threshold:
                    high_pairs.append((left, right, float(value)))
        if not high_pairs:
            break

        left, right, _ = max(high_pairs, key=lambda item: item[2])
        left_quality = quality.get(left, 0.0)
        right_quality = quality.get(right, 0.0)
        drop = left if left_quality < right_quality else right
        keep = right if drop == left else left
        remaining.remove(drop)
        dropped_rows.append({"factor": drop, "reason": "correlation", "paired_with": keep})

    return remaining, pd.DataFrame(dropped_rows, columns=["factor", "reason", "paired_with"])
