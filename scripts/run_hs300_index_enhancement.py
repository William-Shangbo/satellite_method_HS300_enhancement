from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index_platform.reporting.html_report import render_strategy_report
from index_platform.strategy.factors import (
    build_basic_factor_panel,
    composite_alpha,
    factor_columns,
    factor_summary,
)
from index_platform.strategy.optimization import OptimizationConstraints, heuristic_active_weight_optimizer
from index_platform.strategy.preprocessing import FactorPreprocessConfig, preprocess_factor_panel
from index_platform.strategy.risk import performance_summary, trailing_tracking_error
from index_platform.strategy.universe import month_end_rebalance_dates


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HS300 index enhancement research on downloaded Tushare data.")
    parser.add_argument("--data-dir", default="data/raw/tushare_hs300_20160101_20260101")
    parser.add_argument("--start-date", default="20160101")
    parser.add_argument("--end-date", default="20260101")
    parser.add_argument("--out-dir", default="artifacts/hs300_index_enhancement_20160101_20260101")
    parser.add_argument("--max-active-weight", type=float, default=0.01)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--active-share-budget", type=float, default=0.30)
    parser.add_argument("--max-turnover", type=float, default=0.50)
    parser.add_argument("--long-quantile", type=float, default=0.30)
    parser.add_argument("--short-quantile", type=float, default=0.30)
    parser.add_argument("--risk-control", choices=["none", "active_drawdown"], default="none")
    parser.add_argument("--risk-lookback", type=int, default=12)
    parser.add_argument("--risk-scale", type=float, default=0.50)
    parser.add_argument("--alpha-mode", choices=["equal", "rolling_ic", "rolling_ic_signed"], default="rolling_ic")
    parser.add_argument("--ic-window", type=int, default=36)
    parser.add_argument("--min-factor-coverage", type=float, default=0.80)
    parser.add_argument("--min-stock-coverage", type=float, default=0.60)
    parser.add_argument("--correlation-threshold", type=float, default=0.80)
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading raw data...")
    daily = read_stock_folder(data_dir / "daily_bar")
    adj = read_stock_folder(data_dir / "adjustment_factor")
    basic = read_stock_folder(data_dir / "daily_basic")
    weights = pd.read_csv(data_dir / "benchmark_weight.csv")
    calendar = pd.read_csv(data_dir / "trade_calendar.csv")
    financial_indicator = read_optional_stock_folder(data_dir / "financial_indicator")
    stock_basic = read_optional_csv(data_dir / "stock_basic.csv")

    start = pd.to_datetime(args.start_date, format="%Y%m%d")
    end = pd.to_datetime(args.end_date, format="%Y%m%d")
    for frame in [daily, adj, basic, weights]:
        frame["trade_date"] = parse_tushare_date(frame["trade_date"])
    daily = daily[(daily["trade_date"] >= start) & (daily["trade_date"] <= end)]
    adj = adj[(adj["trade_date"] >= start) & (adj["trade_date"] <= end)]
    basic = basic[(basic["trade_date"] >= start) & (basic["trade_date"] <= end)]
    weights = weights[(weights["trade_date"] >= start) & (weights["trade_date"] <= end)]
    if financial_indicator is not None:
        financial_indicator["ann_date"] = parse_tushare_date(financial_indicator["ann_date"])

    print("building merged panel...")
    panel = (
        daily.merge(adj, on=["ts_code", "trade_date"], how="left")
        .merge(basic, on=["ts_code", "trade_date"], how="left")
        .drop_duplicates(["ts_code", "trade_date"])
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )
    panel.to_parquet(out_dir / "merged_daily_panel.parquet", index=False)

    print("building factor panel...")
    factor_panel = build_basic_factor_panel(panel)
    factor_panel.to_parquet(out_dir / "factor_panel_daily.parquet", index=False)

    rebalance_dates = month_end_rebalance_dates(calendar)
    rebalance_dates = pd.DatetimeIndex([d for d in rebalance_dates if start <= d <= end])
    factor_monthly = nearest_rebalance_rows(factor_panel, rebalance_dates)
    factor_monthly = attach_benchmark_weights(factor_monthly, weights)
    if stock_basic is None:
        raise FileNotFoundError(
            f"Need stock_basic.csv with industry classifications for neutralization: {data_dir / 'stock_basic.csv'}"
        )
    factor_monthly = attach_industry(factor_monthly, stock_basic)
    if financial_indicator is not None:
        factor_monthly = attach_financial_indicators(factor_monthly, financial_indicator)
        factor_monthly = normalize_financial_factor_columns(factor_monthly)
    factor_monthly = factor_monthly.dropna(subset=["benchmark_weight"])
    preprocess_result = preprocess_factor_panel(
        factor_monthly,
        config=FactorPreprocessConfig(
            min_factor_coverage=args.min_factor_coverage,
            min_stock_coverage=args.min_stock_coverage,
            correlation_threshold=args.correlation_threshold,
        ),
    )
    preprocess_result.coverage.to_csv(out_dir / "factor_coverage.csv", index=False)
    preprocess_result.dropped_factors.to_csv(out_dir / "factor_dropped_factors.csv", index=False)
    preprocess_result.correlation_matrix.to_csv(out_dir / "factor_correlation_matrix.csv")
    preprocess_result.summary.to_csv(out_dir / "factor_preprocess_summary.csv", index=False)
    factor_monthly = preprocess_result.panel
    factor_monthly = build_alpha_scores(factor_monthly, mode=args.alpha_mode, ic_window=args.ic_window)
    factor_weight_history = factor_monthly.attrs.get("factor_weight_history")
    factor_monthly.attrs.clear()
    factor_monthly.to_parquet(out_dir / "factor_panel_monthly.parquet", index=False)
    if isinstance(factor_weight_history, pd.DataFrame):
        factor_weight_history.to_csv(out_dir / "factor_weight_history.csv", index=False)

    print("computing factor performance...")
    fac_summary = factor_summary(factor_monthly)
    fac_summary.to_csv(out_dir / "factor_ic_summary.csv", index=False)
    ic_ts = factor_ic_timeseries(factor_monthly)
    ic_ts.to_csv(out_dir / "factor_rank_ic_timeseries.csv", index=False)
    layer = layer_performance(factor_monthly, factor_columns(factor_monthly) + ["alpha_score"])
    layer.to_csv(out_dir / "factor_layer_performance.csv", index=False)

    print("running monthly optimization and backtest...")
    constraints = OptimizationConstraints(
        max_active_weight=args.max_active_weight,
        max_weight=args.max_weight,
        active_share_budget=args.active_share_budget,
        max_turnover=args.max_turnover,
        long_quantile=args.long_quantile,
        short_quantile=args.short_quantile,
    )
    returns, all_weights, diagnostics = run_backtest(
        factor_monthly,
        constraints,
        risk_control=args.risk_control,
        risk_lookback=args.risk_lookback,
        risk_scale=args.risk_scale,
    )
    returns.to_csv(out_dir / "portfolio_returns.csv", index=False)
    all_weights.to_csv(out_dir / "portfolio_weights.csv", index=False)
    diagnostics.to_csv(out_dir / "optimization_diagnostics.csv", index=False)

    perf = performance_summary(returns, periods_per_year=12)
    perf.to_frame("value").to_csv(out_dir / "performance_summary.csv")
    te = trailing_tracking_error(returns, window=12, periods_per_year=12)
    te.rename("trailing_tracking_error").to_frame().to_csv(out_dir / "trailing_tracking_error.csv")
    annual = annual_performance_table(returns)
    annual.to_csv(out_dir / "annual_performance.csv", index=False)
    monthly = monthly_return_table(returns)
    monthly.to_csv(out_dir / "monthly_active_return_table.csv")

    report_path = render_strategy_report(
        title="HS300 指数增强策略报告 2016-2026",
        performance=perf,
        factor_summary=fac_summary,
        returns=returns[["portfolio_return", "benchmark_return", "active_return"]],
        weights=all_weights[all_weights["rebalance_date"].eq(all_weights["rebalance_date"].max())],
        output_path=out_dir / "hs300_index_enhancement_report.html",
    )
    print("outputs:", out_dir)
    print("report:", report_path)


def read_stock_folder(path: Path) -> pd.DataFrame:
    frames = []
    for file in sorted(path.glob("*.csv")):
        frame = pd.read_csv(file)
        if "ts_code" not in frame.columns:
            frame["ts_code"] = file.stem
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No csv files found in {path}")
    return pd.concat(frames, ignore_index=True)


def read_optional_stock_folder(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    files = list(path.glob("*.csv"))
    if not files:
        return None
    return read_stock_folder(path)


def read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def parse_tushare_date(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")
    return pd.to_datetime(values.astype(str), format="%Y%m%d", errors="coerce")


def nearest_rebalance_rows(panel: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    trade_dates = pd.DatetimeIndex(sorted(panel["trade_date"].dropna().unique()))
    frames = []
    for date in rebalance_dates:
        visible = trade_dates[trade_dates <= date]
        if visible.empty:
            continue
        actual = visible[-1]
        current = panel[panel["trade_date"].eq(actual)].copy()
        if current.empty:
            continue
        next_visible = trade_dates[trade_dates > actual]
        if len(next_visible) < 21:
            continue
        exit_date = next_visible[min(20, len(next_visible) - 1)]
        next_px = panel.loc[panel["trade_date"].eq(exit_date), ["ts_code", "adj_close"]].rename(
            columns={"adj_close": "exit_adj_close"}
        )
        current = current.merge(next_px, on="ts_code", how="left")
        current["period_return"] = current["exit_adj_close"] / current["adj_close"] - 1
        current["rebalance_date"] = actual
        current["exit_date"] = exit_date
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def attach_benchmark_weights(factor_monthly: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    weights = weights.copy()
    weights["trade_date"] = parse_tushare_date(weights["trade_date"])
    weights = weights.rename(columns={"con_code": "ts_code", "weight": "benchmark_weight"})
    frames = []
    weight_dates = pd.DatetimeIndex(sorted(weights["trade_date"].dropna().unique()))
    for date, current in factor_monthly.groupby("rebalance_date"):
        visible = weight_dates[weight_dates <= pd.to_datetime(date)]
        if visible.empty:
            continue
        w_date = visible[-1]
        w = weights[weights["trade_date"].eq(w_date)][["ts_code", "benchmark_weight"]].copy()
        w["benchmark_weight"] = pd.to_numeric(w["benchmark_weight"], errors="coerce")
        w = w.dropna(subset=["benchmark_weight"])
        w["benchmark_weight"] = w["benchmark_weight"] / w["benchmark_weight"].sum()
        merged = current.drop(columns=["benchmark_weight"], errors="ignore").merge(w, on="ts_code", how="inner")
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def attach_industry(factor_monthly: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "industry"}
    missing = required.difference(stock_basic.columns)
    if missing:
        raise ValueError(f"stock_basic missing columns: {sorted(missing)}")
    industry = stock_basic[["ts_code", "industry"]].copy()
    industry["industry"] = industry["industry"].fillna("UNKNOWN").astype(str)
    industry = industry.drop_duplicates("ts_code", keep="last")
    merged = factor_monthly.drop(columns=["industry"], errors="ignore").merge(industry, on="ts_code", how="left")
    merged["industry"] = merged["industry"].fillna("UNKNOWN")
    return merged


def attach_financial_indicators(factor_monthly: pd.DataFrame, financial: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ts_code",
        "ann_date",
        "end_date",
        "roe",
        "roe_dt",
        "roa",
        "grossprofit_margin",
        "netprofit_margin",
        "assets_turn",
    ]
    available = [col for col in cols if col in financial.columns]
    fin = financial[available].copy()
    fin = fin.dropna(subset=["ts_code", "ann_date"])
    fin = fin.sort_values(["ts_code", "ann_date"]).drop_duplicates(["ts_code", "ann_date"], keep="last")
    frames = []
    for date, current in factor_monthly.groupby("rebalance_date"):
        visible = fin[fin["ann_date"] <= pd.to_datetime(date)]
        if visible.empty:
            frames.append(current)
            continue
        latest = visible.sort_values("ann_date").drop_duplicates("ts_code", keep="last")
        frames.append(current.merge(latest.drop(columns=["ann_date"], errors="ignore"), on="ts_code", how="left"))
    return pd.concat(frames, ignore_index=True)


def normalize_financial_factor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    mapping = {
        "roe": "quality_roe",
        "roe_dt": "quality_roe_dt",
        "roa": "quality_roa",
        "grossprofit_margin": "quality_gross_margin",
        "netprofit_margin": "quality_net_margin",
        "assets_turn": "quality_assets_turn",
    }
    for src, dst in mapping.items():
        if src in result.columns:
            result[dst] = pd.to_numeric(result[src], errors="coerce")
    fin_cols = [col for col in mapping.values() if col in result.columns]
    if fin_cols:
        from index_platform.strategy.factors import winsorize_by_date, zscore_by_date

        result = winsorize_by_date(result, fin_cols)
        result = zscore_by_date(result, fin_cols)
    return result


def build_alpha_scores(factor_monthly: pd.DataFrame, *, mode: str, ic_window: int) -> pd.DataFrame:
    if mode == "equal":
        return composite_alpha(factor_monthly)
    if mode not in {"rolling_ic", "rolling_ic_signed"}:
        raise ValueError(f"Unsupported alpha mode: {mode}")

    result = factor_monthly.copy()
    factors = factor_columns(result)
    result["alpha_score"] = np.nan
    result["alpha_weight_count"] = 0
    dates = sorted(result["rebalance_date"].dropna().unique())

    for i, date in enumerate(dates):
        current_idx = result["rebalance_date"].eq(date)
        history_dates = dates[max(0, i - ic_window) : i]
        weights = rolling_ic_weights(
            result[result["rebalance_date"].isin(history_dates)],
            factors,
            signed=(mode == "rolling_ic_signed"),
        )
        if weights.empty:
            weights = pd.Series(1.0 / len(factors), index=factors)
        current = result.loc[current_idx, factors].mul(weights, axis=1).sum(axis=1, skipna=True)
        result.loc[current_idx, "alpha_score"] = current
        result.loc[current_idx, "alpha_weight_count"] = int((weights != 0).sum())

    from index_platform.strategy.factors import zscore_by_date

    result = zscore_by_date(result, ["alpha_score"])
    factor_weight_rows = []
    for i, date in enumerate(dates):
        history_dates = dates[max(0, i - ic_window) : i]
        weights = rolling_ic_weights(
            result[result["rebalance_date"].isin(history_dates)],
            factors,
            signed=(mode == "rolling_ic_signed"),
        )
        if weights.empty:
            weights = pd.Series(1.0 / len(factors), index=factors)
        for factor, weight in weights.items():
            factor_weight_rows.append({"rebalance_date": date, "factor": factor, "weight": weight})
    result.attrs["factor_weight_history"] = pd.DataFrame(factor_weight_rows)
    return result


def rolling_ic_weights(history: pd.DataFrame, factors: list[str], *, signed: bool = False) -> pd.Series:
    if history.empty:
        return pd.Series(dtype=float)
    rows = []
    for factor in factors:
        ic = []
        for _, current in history.groupby("rebalance_date"):
            data = current[[factor, "period_return"]].dropna()
            if len(data) >= 50:
                ic.append(data[factor].rank().corr(data["period_return"].rank()))
        ic = pd.Series(ic, dtype=float).dropna()
        if len(ic) >= 6 and ic.std(ddof=1) > 0:
            rows.append((factor, ic.mean() / ic.std(ddof=1)))
    if not rows:
        return pd.Series(dtype=float)
    weights = pd.Series(dict(rows), dtype=float)
    if not signed:
        weights = weights.clip(lower=0)
        if weights.sum() <= 0:
            return pd.Series(dtype=float)
        return weights / weights.sum()
    if weights.abs().sum() <= 0:
        return pd.Series(dtype=float)
    return weights / weights.abs().sum()


def factor_ic_timeseries(factor_monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in factor_columns(factor_monthly) + ["alpha_score"]:
        for date, current in factor_monthly.groupby("rebalance_date"):
            data = current[[factor, "period_return"]].dropna()
            if len(data) < 20:
                continue
            rows.append(
                {
                    "rebalance_date": date,
                    "factor": factor,
                    "rank_ic": data[factor].rank().corr(data["period_return"].rank()),
                }
            )
    return pd.DataFrame(rows)


def layer_performance(factor_monthly: pd.DataFrame, factors: list[str], groups: int = 5) -> pd.DataFrame:
    rows = []
    for factor in factors:
        for date, current in factor_monthly.groupby("rebalance_date"):
            data = current[[factor, "period_return"]].dropna()
            if len(data) < groups * 10:
                continue
            data = data.copy()
            data["layer"] = pd.qcut(data[factor].rank(method="first"), groups, labels=False) + 1
            perf = data.groupby("layer")["period_return"].mean()
            for layer_id, value in perf.items():
                rows.append({"rebalance_date": date, "factor": factor, "layer": int(layer_id), "return": value})
    return pd.DataFrame(rows)


def run_backtest(
    factor_monthly: pd.DataFrame,
    constraints: OptimizationConstraints,
    *,
    risk_control: str = "none",
    risk_lookback: int = 12,
    risk_scale: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = []
    weights = []
    diagnostics = []
    previous = None

    for date, current in factor_monthly.groupby("rebalance_date"):
        current = current.dropna(subset=["alpha_score", "benchmark_weight", "period_return"])
        if len(current) < 100:
            continue
        current_constraints = constraints
        risk_multiplier = 1.0
        if risk_control == "active_drawdown" and len(returns) >= max(3, risk_lookback // 2):
            recent = pd.DataFrame(returns).tail(risk_lookback)
            active_curve = (1 + recent["active_return"]).cumprod()
            if active_curve.iloc[-1] < active_curve.cummax().iloc[-1]:
                risk_multiplier = risk_scale
                current_constraints = OptimizationConstraints(
                    max_active_weight=constraints.max_active_weight * risk_multiplier,
                    max_weight=constraints.max_weight,
                    active_share_budget=constraints.active_share_budget * risk_multiplier,
                    max_turnover=constraints.max_turnover,
                    long_quantile=constraints.long_quantile,
                    short_quantile=constraints.short_quantile,
                )
        result = heuristic_active_weight_optimizer(current, previous_weights=previous, constraints=current_constraints)
        w = result.weights.copy()
        w["rebalance_date"] = date
        previous = w[["ts_code", "target_weight"]]
        realized = current[["ts_code", "period_return", "exit_date"]].merge(
            w[["ts_code", "target_weight", "benchmark_weight"]], on="ts_code", how="inner"
        )
        portfolio_return = (realized["target_weight"] * realized["period_return"]).sum()
        benchmark_return = (realized["benchmark_weight"] * realized["period_return"]).sum()
        returns.append(
            {
                "rebalance_date": date,
                "exit_date": realized["exit_date"].iloc[0],
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "active_return": portfolio_return - benchmark_return,
            }
        )
        weights.append(w)
        diagnostics.append({"rebalance_date": date, "risk_multiplier": risk_multiplier, **result.diagnostics})

    return pd.DataFrame(returns), pd.concat(weights, ignore_index=True), pd.DataFrame(diagnostics)


def annual_performance_table(returns: pd.DataFrame) -> pd.DataFrame:
    data = returns.copy()
    data["year"] = pd.to_datetime(data["exit_date"]).dt.year
    rows = []
    for year, current in data.groupby("year"):
        perf = performance_summary(current, periods_per_year=12)
        rows.append({"year": year, **perf.to_dict()})
    return pd.DataFrame(rows)


def monthly_return_table(returns: pd.DataFrame) -> pd.DataFrame:
    data = returns.copy()
    data["exit_date"] = pd.to_datetime(data["exit_date"])
    data["year"] = data["exit_date"].dt.year
    data["month"] = data["exit_date"].dt.month
    return data.pivot_table(index="year", columns="month", values="active_return", aggfunc="sum")


if __name__ == "__main__":
    main()
