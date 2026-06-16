from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index_platform.strategy.factors import build_basic_factor_panel, zscore_by_date
from index_platform.strategy.risk import performance_summary
from index_platform.strategy.universe import month_end_rebalance_dates


IN_DOMAIN_FACTORS = [
    "quality_roe",
    "quality_roa",
    "quality_net_margin",
    "quality_assets_turn",
    "value_bp",
    "dividend_yield",
    "momentum_60d",
]
BASE_FACTORS = [
    "value_ep",
    "value_bp",
    "value_sp",
    "dividend_yield",
    "size_neg_log_mv",
    "momentum_60d",
    "reversal_20d",
    "volatility_20d",
    "turnover_20d",
    "quality_roe",
    "quality_roa",
    "quality_net_margin",
    "quality_assets_turn",
]
SMALL_GROWTH_FACTORS = [
    "size_neg_log_mv",
    "quality_roe",
    "quality_roa",
    "quality_net_margin",
    "momentum_60d",
]
GARP_VALUE_FACTORS = ["value_bp", "dividend_yield"]
GARP_GROWTH_FACTORS = ["quality_roe", "quality_roa", "quality_net_margin", "quality_assets_turn", "momentum_60d"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate the GTJA HS300 satellite enhancement framework.")
    parser.add_argument("--data-dir", default="data/raw/tushare_hs300_20160101_20260101")
    parser.add_argument("--start-date", default="20160101")
    parser.add_argument("--end-date", default="20260101")
    parser.add_argument("--out-dir", default="artifacts/gtja_hs300_satellite_replication_20160101_20260101")
    parser.add_argument("--ic-window", type=int, default=12)
    parser.add_argument("--cost-bps-one-way", type=float, default=20.0)
    parser.add_argument(
        "--benchmark-returns",
        default="artifacts/hs300_index_enhancement_rolling_ic_financial_20160101_20260101/portfolio_returns.csv",
        help="Reference benchmark return file. Keeps strategy experiments on the same benchmark calendar/return series.",
    )
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading data...")
    daily = read_panel_folder(data_dir / "daily_bar")
    adj = read_panel_folder(data_dir / "adjustment_factor")
    basic = read_panel_folder(data_dir / "daily_basic")
    weights = pd.read_csv(data_dir / "benchmark_weight.csv")
    calendar = pd.read_csv(data_dir / "trade_calendar.csv")
    financial_indicator = read_panel_folder(data_dir / "financial_indicator")
    stock_basic = pd.read_csv(data_dir / "stock_basic.csv") if (data_dir / "stock_basic.csv").exists() else None

    start = pd.to_datetime(args.start_date, format="%Y%m%d")
    end = pd.to_datetime(args.end_date, format="%Y%m%d")
    for frame in [daily, adj, basic, weights]:
        frame["trade_date"] = parse_tushare_date(frame["trade_date"])
    financial_indicator["ann_date"] = parse_tushare_date(financial_indicator["ann_date"])

    daily = daily[(daily["trade_date"] >= start) & (daily["trade_date"] <= end)]
    adj = adj[(adj["trade_date"] >= start) & (adj["trade_date"] <= end)]
    basic = basic[(basic["trade_date"] >= start) & (basic["trade_date"] <= end)]
    weights = weights[(weights["trade_date"] >= start) & (weights["trade_date"] <= end)]

    print("building monthly proxy universe...")
    panel = (
        daily.merge(adj, on=["ts_code", "trade_date"], how="left")
        .merge(basic, on=["ts_code", "trade_date"], how="left")
        .drop_duplicates(["ts_code", "trade_date"])
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )
    factor_panel = build_basic_factor_panel(panel)
    rebalance_dates = pd.DatetimeIndex([d for d in month_end_rebalance_dates(calendar) if start <= d <= end])
    monthly = nearest_rebalance_rows(factor_panel, rebalance_dates)
    monthly = attach_hs300_membership(monthly, weights)
    monthly = attach_financial_indicators(monthly, financial_indicator)
    monthly = normalize_financial_factor_columns(monthly)
    if stock_basic is not None:
        monthly = attach_industry(monthly, stock_basic)
    monthly.to_parquet(out_dir / "replication_factor_panel_monthly.parquet", index=False)

    print("building scores...")
    monthly = add_rolling_ic_score(
        monthly,
        BASE_FACTORS,
        "base_score",
        mask_col=None,
        ic_window=args.ic_window,
    )
    monthly = add_rolling_ic_score(
        monthly,
        IN_DOMAIN_FACTORS,
        "domain_in_score",
        mask_col="is_hs300_member",
        ic_window=args.ic_window,
    )
    monthly = add_equal_weight_score(monthly, SMALL_GROWTH_FACTORS, "small_growth_score")
    monthly = add_garp_score(monthly)
    monthly.to_parquet(out_dir / "replication_scored_panel_monthly.parquet", index=False)

    print("running article-style portfolios...")
    returns, weights_out = run_article_portfolios(monthly, cost_bps_one_way=args.cost_bps_one_way)
    returns = align_reference_benchmark_returns(returns, ROOT / args.benchmark_returns)
    returns.to_csv(out_dir / "article_strategy_returns.csv", index=False)
    weights_out.to_csv(out_dir / "article_strategy_weights.csv", index=False)

    summary = summarize_strategies(returns)
    summary.to_csv(out_dir / "article_performance_summary.csv", index=False)
    annual = annual_active_table(returns)
    annual.to_csv(out_dir / "article_annual_active_return.csv", index=False)
    grid = satellite_grid(returns)
    grid.to_csv(out_dir / "article_satellite_grid.csv", index=False)
    write_notes(out_dir, args, returns)

    print("outputs:", out_dir)
    print(summary.to_string(index=False))


def read_panel_folder(path: Path) -> pd.DataFrame:
    parquet_files = sorted(path.glob("*.parquet"))
    if parquet_files:
        return pd.concat((pd.read_parquet(file) for file in parquet_files), ignore_index=True)
    return read_stock_folder(path)


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


def attach_hs300_membership(monthly: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    weights = weights.copy()
    weights["trade_date"] = parse_tushare_date(weights["trade_date"])
    weights = weights.rename(columns={"con_code": "ts_code", "weight": "benchmark_weight"})
    weight_dates = pd.DatetimeIndex(sorted(weights["trade_date"].dropna().unique()))
    frames = []
    for date, current in monthly.groupby("rebalance_date"):
        visible = weight_dates[weight_dates <= pd.to_datetime(date)]
        if visible.empty:
            continue
        w_date = visible[-1]
        w = weights[weights["trade_date"].eq(w_date)][["ts_code", "benchmark_weight"]].copy()
        w["benchmark_weight"] = pd.to_numeric(w["benchmark_weight"], errors="coerce")
        w = w.dropna(subset=["benchmark_weight"])
        w["benchmark_weight"] = w["benchmark_weight"] / w["benchmark_weight"].sum()
        merged = current.drop(columns=["benchmark_weight"], errors="ignore").merge(w, on="ts_code", how="left")
        merged["benchmark_weight"] = merged["benchmark_weight"].fillna(0.0)
        merged["is_hs300_member"] = merged["benchmark_weight"] > 0
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def attach_financial_indicators(monthly: pd.DataFrame, financial: pd.DataFrame) -> pd.DataFrame:
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
    for date, current in monthly.groupby("rebalance_date"):
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
        "roa": "quality_roa",
        "netprofit_margin": "quality_net_margin",
        "assets_turn": "quality_assets_turn",
    }
    for src, dst in mapping.items():
        if src in result.columns:
            result[dst] = pd.to_numeric(result[src], errors="coerce")
    fin_cols = [col for col in mapping.values() if col in result.columns]
    if fin_cols:
        result = zscore_by_date(fill_by_date_median(result, fin_cols), fin_cols)
    return result


def attach_industry(monthly: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    industry = stock_basic[["ts_code", "industry"]].drop_duplicates("ts_code", keep="last")
    industry["industry"] = industry["industry"].fillna("UNKNOWN")
    result = monthly.drop(columns=["industry"], errors="ignore").merge(industry, on="ts_code", how="left")
    result["industry"] = result["industry"].fillna("UNKNOWN")
    return result


def fill_by_date_median(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for col in columns:
        date_median = result.groupby("rebalance_date")[col].transform("median")
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(date_median).fillna(0.0)
    return result


def add_rolling_ic_score(
    frame: pd.DataFrame,
    factors: list[str],
    score_col: str,
    *,
    mask_col: str | None,
    ic_window: int,
) -> pd.DataFrame:
    result = fill_by_date_median(frame, [factor for factor in factors if factor in frame.columns])
    result[score_col] = np.nan
    dates = sorted(result["rebalance_date"].dropna().unique())
    factors = [factor for factor in factors if factor in result.columns]
    for idx, date in enumerate(dates):
        history_dates = dates[max(0, idx - ic_window) : idx]
        history = result[result["rebalance_date"].isin(history_dates)]
        if mask_col is not None:
            history = history[history[mask_col]]
        weights = rolling_ic_weights(history, factors)
        if weights.empty:
            weights = pd.Series(1.0 / len(factors), index=factors)
        current_idx = result["rebalance_date"].eq(date)
        result.loc[current_idx, score_col] = result.loc[current_idx, factors].mul(weights, axis=1).sum(axis=1)
    return standardize_score(result, score_col)


def add_equal_weight_score(frame: pd.DataFrame, factors: list[str], score_col: str) -> pd.DataFrame:
    factors = [factor for factor in factors if factor in frame.columns]
    result = fill_by_date_median(frame, factors)
    result[score_col] = result[factors].mean(axis=1)
    return standardize_score(result, score_col)


def add_garp_score(frame: pd.DataFrame) -> pd.DataFrame:
    value_factors = [factor for factor in GARP_VALUE_FACTORS if factor in frame.columns]
    growth_factors = [factor for factor in GARP_GROWTH_FACTORS if factor in frame.columns]
    result = fill_by_date_median(frame, value_factors + growth_factors + ["turnover_20d"])
    result["garp_value_score"] = result[value_factors].mean(axis=1)
    result["garp_growth_score"] = result[growth_factors].mean(axis=1)
    result["garp_score"] = 0.5 * result["garp_value_score"] + 0.5 * result["garp_growth_score"]
    return standardize_score(result, "garp_score")


def rolling_ic_weights(history: pd.DataFrame, factors: list[str]) -> pd.Series:
    rows = []
    for factor in factors:
        values = []
        for _, current in history.groupby("rebalance_date"):
            data = current[[factor, "period_return"]].dropna()
            if len(data) >= 50:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    values.append(data[factor].rank().corr(data["period_return"].rank()))
        ic = pd.Series(values, dtype=float).dropna()
        if len(ic) >= 4 and ic.std(ddof=1) > 0:
            rows.append((factor, max(ic.mean() / ic.std(ddof=1), 0.0)))
    if not rows:
        return pd.Series(dtype=float)
    weights = pd.Series(dict(rows), dtype=float)
    if weights.sum() <= 0:
        return pd.Series(dtype=float)
    return weights / weights.sum()


def standardize_score(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    result = frame.copy()
    mean = result.groupby("rebalance_date")[score_col].transform("mean")
    std = result.groupby("rebalance_date")[score_col].transform("std").replace(0, np.nan)
    result[score_col] = ((result[score_col] - mean) / std).fillna(0.0)
    return result


def run_article_portfolios(frame: pd.DataFrame, *, cost_bps_one_way: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    weight_rows = []
    prev_weights: dict[str, pd.Series] = {}
    cost_rate = cost_bps_one_way / 10000

    for date, current in frame.groupby("rebalance_date"):
        current = current.dropna(subset=["period_return"]).copy()
        inside = current[current["is_hs300_member"]].copy()
        outside = current[~current["is_hs300_member"]].copy()
        if len(inside) < 100:
            continue

        benchmark_weights = inside.set_index("ts_code")["benchmark_weight"]
        benchmark_return = float((inside.set_index("ts_code")["period_return"] * benchmark_weights).sum())

        inside_core = benchmark_tilt_weights(inside, "base_score", active_share=0.30, max_active=0.01, max_weight=0.05)
        outside_base = top_equal_weights(outside, "base_score", n=50)
        domain_in_top30 = domain_in_top30_weights(inside)
        small_growth = top_equal_weights(outside, "small_growth_score", n=50)
        garp20 = garp_weights(outside, n=20)
        garp50 = garp_weights(outside, n=50)

        strategies = {
            "base_inside_proxy": inside_core,
            "base_outside_proxy": outside_base,
            "base_80in_20out_proxy": combine_weight_blocks({"base_inside_proxy": (inside_core, 0.80), "base_outside_proxy": (outside_base, 0.20)}),
            "domain_in_top30": domain_in_top30,
            "small_growth_outside_proxy": small_growth,
            "garp20_outside_proxy": garp20,
            "garp50_outside_proxy": garp50,
        }
        strategies["combo_in20_out10_small_proxy"] = combine_weight_blocks(
            {
                "base": (strategies["base_80in_20out_proxy"], 0.70),
                "domain_in_top30": (domain_in_top30, 0.20),
                "small_growth": (small_growth, 0.10),
            }
        )
        strategies["combo_in30_out10_small_proxy"] = combine_weight_blocks(
            {
                "base": (strategies["base_80in_20out_proxy"], 0.60),
                "domain_in_top30": (domain_in_top30, 0.30),
                "small_growth": (small_growth, 0.10),
            }
        )
        strategies["combo_in20_out10_garp_proxy"] = combine_weight_blocks(
            {
                "base": (strategies["base_80in_20out_proxy"], 0.70),
                "domain_in_top30": (domain_in_top30, 0.20),
                "garp20": (garp20, 0.10),
            }
        )
        strategies["combo_extreme_in_out_small_proxy"] = combine_weight_blocks(
            {"domain_in_top30": (domain_in_top30, 0.80), "small_growth": (small_growth, 0.20)}
        )

        row = {"rebalance_date": date, "exit_date": current["exit_date"].iloc[0], "benchmark_return": benchmark_return}
        for name, weights in strategies.items():
            gross = weighted_return(current, weights)
            turnover = portfolio_turnover(weights, prev_weights.get(name))
            net = gross - turnover * cost_rate
            row[f"{name}_return"] = net
            row[f"{name}_turnover"] = turnover
            prev_weights[name] = weights
            for code, weight in weights.items():
                if weight > 0:
                    weight_rows.append({"rebalance_date": date, "strategy": name, "ts_code": code, "weight": weight})
        rows.append(row)

    returns = pd.DataFrame(rows)
    return returns, pd.DataFrame(weight_rows)


def align_reference_benchmark_returns(returns: pd.DataFrame, benchmark_returns_path: Path) -> pd.DataFrame:
    if not benchmark_returns_path.exists():
        return returns
    reference = pd.read_csv(benchmark_returns_path)
    required = {"rebalance_date", "benchmark_return"}
    missing = required.difference(reference.columns)
    if missing:
        raise ValueError(f"Reference benchmark file missing columns: {sorted(missing)}")
    result = returns.copy()
    result["rebalance_date"] = pd.to_datetime(result["rebalance_date"])
    reference = reference[["rebalance_date", "benchmark_return"]].copy()
    reference["rebalance_date"] = pd.to_datetime(reference["rebalance_date"])
    merged = result.drop(columns=["benchmark_return"], errors="ignore").merge(
        reference,
        on="rebalance_date",
        how="left",
    )
    if merged["benchmark_return"].isna().any():
        missing_dates = merged.loc[merged["benchmark_return"].isna(), "rebalance_date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Reference benchmark missing rebalance dates: {missing_dates[:5]}")
    return merged


def benchmark_tilt_weights(
    frame: pd.DataFrame,
    score_col: str,
    *,
    active_share: float,
    max_active: float,
    max_weight: float,
) -> pd.Series:
    data = frame[["ts_code", "benchmark_weight", score_col]].dropna().copy()
    score = data[score_col] - data[score_col].mean()
    denom = score.abs().sum()
    active = score / denom * active_share if denom else score
    active = active.clip(-max_active, max_active)
    data["weight"] = (data["benchmark_weight"] + active).clip(lower=0, upper=max_weight)
    data["weight"] = data["weight"] / data["weight"].sum()
    return data.set_index("ts_code")["weight"]


def domain_in_top30_weights(inside: pd.DataFrame) -> pd.Series:
    pool = inside.sort_values("base_score", ascending=False).head(100)
    selected = pool.sort_values("domain_in_score", ascending=False).head(30)
    return top_cap_weighted(selected, "total_mv", cap=0.05)


def garp_weights(outside: pd.DataFrame, *, n: int) -> pd.Series:
    if outside.empty:
        return pd.Series(dtype=float)
    data = outside.copy()
    value_cut = data["garp_value_score"].quantile(0.20)
    turnover_cut = data["turnover_20d"].quantile(0.20)
    filtered = data[(data["garp_value_score"] > value_cut) & (data["turnover_20d"] > turnover_cut)]
    return top_equal_weights(filtered, "garp_score", n=n)


def top_equal_weights(frame: pd.DataFrame, score_col: str, *, n: int) -> pd.Series:
    data = frame.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(n)
    if data.empty:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(data), index=data["ts_code"])


def top_cap_weighted(frame: pd.DataFrame, weight_col: str, *, cap: float) -> pd.Series:
    data = frame[["ts_code", weight_col]].copy()
    raw = pd.to_numeric(data[weight_col], errors="coerce").where(lambda s: s > 0)
    raw = raw.fillna(1.0)
    weights = pd.Series(raw.to_numpy(dtype=float), index=data["ts_code"], dtype=float)
    weights = weights / weights.sum()
    fixed = pd.Series(False, index=weights.index)
    while (weights[~fixed] > cap).any():
        newly_fixed = (weights > cap) & ~fixed
        fixed = fixed | newly_fixed
        remaining_budget = 1.0 - fixed.sum() * cap
        if remaining_budget <= 0 or (~fixed).sum() == 0:
            weights.loc[fixed] = cap
            break
        weights.loc[fixed] = cap
        weights.loc[~fixed] = weights.loc[~fixed] / weights.loc[~fixed].sum() * remaining_budget
    return weights / weights.sum()


def combine_weight_blocks(blocks: dict[str, tuple[pd.Series, float]]) -> pd.Series:
    combined = pd.Series(dtype=float)
    for weights, allocation in [value for value in blocks.values()]:
        if weights.empty or allocation <= 0:
            continue
        combined = combined.add(weights * allocation, fill_value=0.0)
    return combined / combined.sum() if combined.sum() else combined


def weighted_return(frame: pd.DataFrame, weights: pd.Series) -> float:
    if weights.empty:
        return float("nan")
    returns = frame.set_index("ts_code")["period_return"]
    common = weights.index.intersection(returns.index)
    if common.empty:
        return float("nan")
    w = weights.loc[common]
    w = w / w.sum()
    return float((w * returns.loc[common]).sum())


def portfolio_turnover(current: pd.Series, previous: pd.Series | None) -> float:
    if previous is None or previous.empty:
        return 1.0
    index = current.index.union(previous.index)
    return float(0.5 * (current.reindex(index).fillna(0.0) - previous.reindex(index).fillna(0.0)).abs().sum())


def summarize_strategies(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        if not col.endswith("_return") or col == "benchmark_return":
            continue
        strategy = col.removesuffix("_return")
        perf = performance_summary(
            returns.rename(columns={col: "portfolio_return"}),
            portfolio_col="portfolio_return",
            benchmark_col="benchmark_return",
            periods_per_year=12,
        )
        rows.append({"strategy": strategy, **perf.to_dict()})
    return pd.DataFrame(rows).sort_values("annual_active_return_arithmetic", ascending=False)


def annual_active_table(returns: pd.DataFrame) -> pd.DataFrame:
    data = returns.copy()
    data["year"] = pd.to_datetime(data["exit_date"]).dt.year
    rows = []
    for year, current in data.groupby("year"):
        for col in returns.columns:
            if not col.endswith("_return") or col == "benchmark_return":
                continue
            active = current[col] - current["benchmark_return"]
            rows.append(
                {
                    "year": year,
                    "strategy": col.removesuffix("_return"),
                    "annual_active_return_arithmetic": active.mean() * 12,
                    "monthly_win_rate": (active > 0).mean(),
                }
            )
    return pd.DataFrame(rows)


def satellite_grid(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for in_weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for out_weight in [0.0, 0.1, 0.2]:
            if in_weight + out_weight > 1:
                continue
            name = f"grid_in{int(in_weight * 100)}_out{int(out_weight * 100)}_small_proxy"
            frame = returns.copy()
            frame["portfolio_return"] = (
                (1 - in_weight - out_weight) * frame["base_80in_20out_proxy_return"]
                + in_weight * frame["domain_in_top30_return"]
                + out_weight * frame["small_growth_outside_proxy_return"]
            )
            perf = performance_summary(frame, periods_per_year=12)
            rows.append({"strategy": name, "in_satellite_weight": in_weight, "out_satellite_weight": out_weight, **perf.to_dict()})
    return pd.DataFrame(rows).sort_values("information_ratio", ascending=False)


def write_notes(out_dir: Path, args: argparse.Namespace, returns: pd.DataFrame) -> None:
    text = f"""# GTJA HS300 Satellite Replication Notes

This run reproduces the article framework with the data currently available locally.

Period: {args.start_date}-{args.end_date}
Rebalance periods: {len(returns)}
Transaction cost: one-way {args.cost_bps_one_way:.1f} bps
Benchmark returns: {args.benchmark_returns}

Implemented:
- HS300 current constituent membership from Tushare index weights.
- In-domain model using available fundamental and momentum factors.
- In-domain Top30 satellite: current HS300 -> base-score Top100 -> in-domain-score Top30 -> market-cap weighted, single-name cap 5%.
- Outside satellite proxies using securities in the supplied data directory that are not current HS300 members on each rebalance date.
- Small-growth outside proxy and GARP outside proxy.
- Composite satellite allocations such as in-domain 20% + outside 10%.

Not strict GTJA replication yet:
- Analyst expectation factors are unavailable locally: expected net profit revision, EAV, SUE as defined by sell-side consensus.
- Intraday/order-flow factors are unavailable locally: post-open buying intensity, large-order return contribution, close-volume share, large-order net buy.
- R&D accumulated investment is unavailable in the current raw folders.
- The optimizer here is an article-style linear proxy, not the report's full risk model with Barra-like exposure constraints.

Therefore these outputs are for structure validation and implementation debugging, not a claim to reproduce the article's 12.6% annual excess number.
"""
    (out_dir / "replication_notes.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
