from __future__ import annotations

import argparse
import base64
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index_platform.strategy.risk import performance_summary


BASE_FACTOR_COLUMNS = [
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

SCORE_COLUMNS = [
    "base_score",
    "domain_in_score",
    "small_growth_score",
    "garp_value_score",
    "garp_growth_score",
    "garp_score",
]

STYLE_EXPOSURE_COLUMNS = [
    "size_neg_log_mv",
    "value_bp",
    "value_ep",
    "dividend_yield",
    "momentum_60d",
    "reversal_20d",
    "volatility_20d",
    "turnover_20d",
    "quality_roe",
    "quality_roa",
    "quality_net_margin",
    "quality_assets_turn",
]

SIZE_VALUE_COLUMNS = [
    "size_neg_log_mv",
    "total_mv",
    "circ_mv",
    "value_bp",
    "value_ep",
    "pe_ttm",
    "pb",
]

PLOT_STRATEGIES = [
    "benchmark",
    "base_80in_20out_proxy",
    "domain_in_top30",
    "combo_in20_out10_small_proxy",
    "combo_in30_out10_small_proxy",
    "combo_in20_out10_garp_proxy",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-signal tests, exposure diagnostics, and factor expansion.")
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101",
        help="Directory containing replication_scored_panel_monthly.parquet and article strategy outputs.",
    )
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--corr-threshold", type=float, default=0.75)
    parser.add_argument("--train-end", default="20221231")
    parser.add_argument("--redraw-only", action="store_true", help="Only redraw TE-controlled NAV figures from existing return CSV.")
    args = parser.parse_args()

    artifact_dir = ROOT / args.artifact_dir
    out_dir = artifact_dir / "factor_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(artifact_dir / "article_strategy_returns.csv")
    for col in ["rebalance_date", "exit_date"]:
        if col in returns.columns:
            returns[col] = pd.to_datetime(returns[col])
    if args.redraw_only:
        plot_te_controlled_strategy_curves(returns, artifact_dir)
        print(f"redrew TE-controlled figures in: {artifact_dir}")
        return

    panel = pd.read_parquet(artifact_dir / "replication_scored_panel_monthly.parquet")
    weights = pd.read_csv(artifact_dir / "article_strategy_weights.csv")

    for col in ["rebalance_date", "trade_date", "exit_date"]:
        if col in panel.columns:
            panel[col] = pd.to_datetime(panel[col])
    weights["rebalance_date"] = pd.to_datetime(weights["rebalance_date"])

    factor_cols = [col for col in BASE_FACTOR_COLUMNS + SCORE_COLUMNS if col in panel.columns]
    style_cols = [col for col in STYLE_EXPOSURE_COLUMNS if col in panel.columns]

    print("preprocessing factors for diagnostics...")
    panel = add_preprocessed_factors(panel, factor_cols)
    diagnostic_factor_cols = [f"{col}_ind_neutral" for col in factor_cols]

    print("running single-signal IC and layer tests...")
    ic_summary, ic_ts = single_signal_ic(panel, diagnostic_factor_cols)
    layer_returns, layer_summary = single_signal_layers(panel, diagnostic_factor_cols, layers=args.layers)
    ic_summary.to_csv(out_dir / "single_signal_ic_summary.csv", index=False)
    ic_ts.to_csv(out_dir / "single_signal_ic_timeseries.csv", index=False)
    layer_returns.to_csv(out_dir / "single_signal_layer_returns.csv", index=False)
    layer_summary.to_csv(out_dir / "single_signal_layer_summary.csv", index=False)

    print("running correlation control report...")
    corr_matrix, prune_report = factor_correlation_report(panel, diagnostic_factor_cols, args.corr_threshold)
    corr_matrix.to_csv(out_dir / "factor_correlation_matrix.csv")
    prune_report.to_csv(out_dir / "factor_correlation_prune_report.csv", index=False)

    print("running strategy exposure diagnostics...")
    industry_active, industry_summary = industry_exposure(panel, weights)
    style_active, style_summary = style_exposure(panel, weights, style_cols)
    size_value_active, size_value_summary = style_exposure(panel, weights, [col for col in SIZE_VALUE_COLUMNS if col in panel.columns])
    industry_active.to_csv(out_dir / "strategy_industry_active_weights.csv", index=False)
    industry_summary.to_csv(out_dir / "strategy_industry_deviation_summary.csv", index=False)
    style_active.to_csv(out_dir / "strategy_style_active_exposure.csv", index=False)
    style_summary.to_csv(out_dir / "strategy_style_exposure_summary.csv", index=False)
    size_value_active.to_csv(out_dir / "strategy_size_value_active_exposure.csv", index=False)
    size_value_summary.to_csv(out_dir / "strategy_size_value_exposure_summary.csv", index=False)

    print("running GP-like expression/kernel factor search...")
    gp_summary, gp_layers = gp_expression_search(panel, factor_cols, train_end=pd.to_datetime(args.train_end))
    gp_summary.to_csv(out_dir / "gp_candidate_factor_summary.csv", index=False)
    gp_layers.to_csv(out_dir / "gp_top_factor_layer_returns.csv", index=False)

    print("redrawing TE-controlled figures...")
    plot_layer_tests(layer_returns, layer_summary, out_dir)
    plot_exposures(industry_summary, style_summary, out_dir)
    plot_te_controlled_strategy_curves(returns, artifact_dir)
    write_report(
        out_dir,
        ic_summary,
        layer_summary,
        prune_report,
        industry_summary,
        style_summary,
        size_value_summary,
        gp_summary,
    )

    print(f"outputs: {out_dir}")


def add_preprocessed_factors(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    result = panel.copy()
    for factor in factor_cols:
        raw = pd.to_numeric(result[factor], errors="coerce")
        filled = raw.copy()
        by_date = result.groupby("rebalance_date", observed=True)
        median = by_date[factor].transform("median")
        filled = filled.fillna(median).fillna(0.0)
        result[f"{factor}_filled"] = filled
        result[f"{factor}_z"] = by_date[f"{factor}_filled"].transform(winsor_zscore)
        result[f"{factor}_demean"] = result[f"{factor}_z"] - by_date[f"{factor}_z"].transform("mean")
        result[f"{factor}_ind_neutral"] = result[f"{factor}_demean"] - result.groupby(
            ["rebalance_date", "industry"], observed=True
        )[f"{factor}_demean"].transform("mean")
        std = by_date[f"{factor}_ind_neutral"].transform("std").replace(0, np.nan)
        result[f"{factor}_ind_neutral"] = (result[f"{factor}_ind_neutral"] / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


def winsor_zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    lo = values.quantile(0.01)
    hi = values.quantile(0.99)
    clipped = values.clip(lo, hi)
    std = clipped.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (clipped - clipped.mean()) / std


def single_signal_ic(panel: pd.DataFrame, factor_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ts_rows = []
    scopes = {
        "all_a": pd.Series(True, index=panel.index),
        "hs300_inside": panel["is_hs300_member"].astype(bool),
        "outside_hs300": ~panel["is_hs300_member"].astype(bool),
    }
    for scope_name, scope_mask in scopes.items():
        scoped = panel.loc[scope_mask].copy()
        for factor in factor_cols:
            values = []
            for date, current in scoped.groupby("rebalance_date", observed=True):
                data = current[[factor, "period_return"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) < 50:
                    continue
                ic = data[factor].corr(data["period_return"])
                rank_ic = data[factor].rank().corr(data["period_return"].rank())
                values.append({"rebalance_date": date, "scope": scope_name, "factor": factor, "ic": ic, "rank_ic": rank_ic, "n": len(data)})
            if not values:
                continue
            fts = pd.DataFrame(values)
            ts_rows.append(fts)
            rows.append(
                {
                    "scope": scope_name,
                    "factor": factor,
                    "ic_mean": fts["ic"].mean(),
                    "ic_std": fts["ic"].std(ddof=1),
                    "icir": safe_div(fts["ic"].mean(), fts["ic"].std(ddof=1)),
                    "rank_ic_mean": fts["rank_ic"].mean(),
                    "rank_ic_std": fts["rank_ic"].std(ddof=1),
                    "rank_icir": safe_div(fts["rank_ic"].mean(), fts["rank_ic"].std(ddof=1)),
                    "positive_rank_ic_rate": (fts["rank_ic"] > 0).mean(),
                    "periods": len(fts),
                    "avg_coverage": fts["n"].mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["scope", "rank_icir"], ascending=[True, False]), pd.concat(ts_rows, ignore_index=True)


def single_signal_layers(panel: pd.DataFrame, factor_cols: list[str], *, layers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scopes = {
        "all_a": pd.Series(True, index=panel.index),
        "hs300_inside": panel["is_hs300_member"].astype(bool),
        "outside_hs300": ~panel["is_hs300_member"].astype(bool),
    }
    bench = panel.loc[panel["benchmark_weight"] > 0].groupby("rebalance_date", observed=True).apply(
        lambda x: np.average(x["period_return"], weights=x["benchmark_weight"]),
        include_groups=False,
    )
    for scope_name, scope_mask in scopes.items():
        scoped = panel.loc[scope_mask].copy()
        for factor in factor_cols:
            for date, current in scoped.groupby("rebalance_date", observed=True):
                data = current[["ts_code", factor, "period_return"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) < layers * 20:
                    continue
                data["layer"] = pd.qcut(data[factor].rank(method="first"), layers, labels=False, duplicates="drop") + 1
                universe_return = data["period_return"].mean()
                benchmark_return = bench.get(date, np.nan)
                for layer, layer_data in data.groupby("layer", observed=True):
                    ret = layer_data["period_return"].mean()
                    rows.append(
                        {
                            "rebalance_date": date,
                            "scope": scope_name,
                            "factor": factor,
                            "layer": int(layer),
                            "layer_return": ret,
                            "universe_return": universe_return,
                            "benchmark_return": benchmark_return,
                            "active_vs_universe": ret - universe_return,
                            "active_vs_benchmark": ret - benchmark_return if np.isfinite(benchmark_return) else np.nan,
                            "n": len(layer_data),
                        }
                    )
    layer_returns = pd.DataFrame(rows)
    summary_rows = []
    for keys, current in layer_returns.groupby(["scope", "factor"], observed=True):
        scope_name, factor = keys
        top = current[current["layer"].eq(layers)].sort_values("rebalance_date")
        bottom = current[current["layer"].eq(1)].sort_values("rebalance_date")
        merged = top[["rebalance_date", "layer_return", "active_vs_universe", "active_vs_benchmark"]].merge(
            bottom[["rebalance_date", "layer_return"]],
            on="rebalance_date",
            how="inner",
            suffixes=("_top", "_bottom"),
        )
        if merged.empty:
            continue
        long_short = merged["layer_return_top"] - merged["layer_return_bottom"]
        summary_rows.append(
            {
                "scope": scope_name,
                "factor": factor,
                "top_layer_annual_return": annualized_return(merged["layer_return_top"]),
                "top_layer_annual_active_vs_universe": merged["active_vs_universe"].mean() * 12,
                "top_layer_annual_active_vs_benchmark": merged["active_vs_benchmark"].mean() * 12,
                "long_short_annual_return_arithmetic": long_short.mean() * 12,
                "long_short_vol": long_short.std(ddof=1) * np.sqrt(12),
                "long_short_ir": safe_div(long_short.mean() * 12, long_short.std(ddof=1) * np.sqrt(12)),
                "top_win_rate_vs_universe": (merged["active_vs_universe"] > 0).mean(),
                "periods": len(merged),
            }
        )
    return layer_returns, pd.DataFrame(summary_rows).sort_values(["scope", "long_short_ir"], ascending=[True, False])


def factor_correlation_report(panel: pd.DataFrame, factor_cols: list[str], threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = panel[factor_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    corr = values.corr(method="spearman")
    ic_summary, _ = single_signal_ic(panel, factor_cols)
    score = ic_summary[ic_summary["scope"].eq("all_a")].set_index("factor")["rank_icir"].abs()
    pairs = []
    dropped = set()
    keep = set(factor_cols)
    for a, b in itertools.combinations(factor_cols, 2):
        rho = corr.loc[a, b]
        if abs(rho) < threshold:
            continue
        score_a = score.get(a, 0.0)
        score_b = score.get(b, 0.0)
        drop = b if score_a >= score_b else a
        dropped.add(drop)
        keep.discard(drop)
        pairs.append({"factor_a": a, "factor_b": b, "spearman_corr": rho, "drop_candidate": drop})
    report = pd.DataFrame(pairs)
    if report.empty:
        report = pd.DataFrame(columns=["factor_a", "factor_b", "spearman_corr", "drop_candidate"])
    report["threshold"] = threshold
    return corr, report.sort_values("spearman_corr", key=lambda s: s.abs(), ascending=False)


def industry_exposure(panel: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = weights.merge(panel[["rebalance_date", "ts_code", "industry", "benchmark_weight"]], on=["rebalance_date", "ts_code"], how="left")
    merged["industry"] = merged["industry"].fillna("UNKNOWN")
    strat = merged.groupby(["rebalance_date", "strategy", "industry"], observed=True)["weight"].sum().reset_index(name="strategy_weight")
    bench = panel[panel["benchmark_weight"] > 0].groupby(["rebalance_date", "industry"], observed=True)["benchmark_weight"].sum().reset_index(name="benchmark_weight")
    active = strat.merge(bench, on=["rebalance_date", "industry"], how="outer")
    active["strategy"] = active.groupby(["rebalance_date", "industry"], observed=True)["strategy"].transform(lambda s: s.ffill().bfill())
    active["strategy_weight"] = active["strategy_weight"].fillna(0.0)
    active["benchmark_weight"] = active["benchmark_weight"].fillna(0.0)
    # The outer merge can create benchmark-only rows without strategy names; expand those for complete diagnostics.
    strategies = sorted(weights["strategy"].dropna().unique())
    complete_rows = []
    for date, bench_date in bench.groupby("rebalance_date", observed=True):
        date_strat = strat[strat["rebalance_date"].eq(date)]
        industries = sorted(set(bench_date["industry"]).union(date_strat["industry"]))
        for strategy in strategies:
            s = date_strat[date_strat["strategy"].eq(strategy)].set_index("industry")["strategy_weight"]
            b = bench_date.set_index("industry")["benchmark_weight"]
            for industry in industries:
                complete_rows.append(
                    {
                        "rebalance_date": date,
                        "strategy": strategy,
                        "industry": industry,
                        "strategy_weight": float(s.get(industry, 0.0)),
                        "benchmark_weight": float(b.get(industry, 0.0)),
                    }
                )
    active = pd.DataFrame(complete_rows)
    active["active_weight"] = active["strategy_weight"] - active["benchmark_weight"]
    summary = active.groupby("strategy", observed=True).agg(
        mean_abs_industry_deviation=("active_weight", lambda s: s.abs().mean()),
        max_abs_industry_deviation=("active_weight", lambda s: s.abs().max()),
        p95_abs_industry_deviation=("active_weight", lambda s: s.abs().quantile(0.95)),
    )
    return active.sort_values(["strategy", "rebalance_date", "industry"]), summary.reset_index().sort_values("max_abs_industry_deviation", ascending=False)


def style_exposure(panel: pd.DataFrame, weights: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel[["rebalance_date", "ts_code", "benchmark_weight"] + columns].copy()
    for col in columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[f"{col}_z"] = data.groupby("rebalance_date", observed=True)[col].transform(winsor_zscore)
    zcols = [f"{col}_z" for col in columns]
    bench_rows = []
    for date, current in data[data["benchmark_weight"] > 0].groupby("rebalance_date", observed=True):
        w = current["benchmark_weight"] / current["benchmark_weight"].sum()
        row = {"rebalance_date": date}
        for col in zcols:
            row[col] = float((w * current[col].fillna(0.0)).sum())
        bench_rows.append(row)
    bench = pd.DataFrame(bench_rows).set_index("rebalance_date")
    merged = weights.merge(data[["rebalance_date", "ts_code"] + zcols], on=["rebalance_date", "ts_code"], how="left")
    rows = []
    for (date, strategy), current in merged.groupby(["rebalance_date", "strategy"], observed=True):
        w = current["weight"] / current["weight"].sum()
        brow = bench.loc[date] if date in bench.index else pd.Series(0.0, index=zcols)
        for col in zcols:
            exposure = float((w * current[col].fillna(0.0)).sum())
            rows.append(
                {
                    "rebalance_date": date,
                    "strategy": strategy,
                    "exposure": col.removesuffix("_z"),
                    "strategy_exposure_z": exposure,
                    "benchmark_exposure_z": float(brow.get(col, 0.0)),
                    "active_exposure_z": exposure - float(brow.get(col, 0.0)),
                }
            )
    active = pd.DataFrame(rows)
    summary = active.groupby(["strategy", "exposure"], observed=True).agg(
        mean_active_exposure_z=("active_exposure_z", "mean"),
        mean_abs_active_exposure_z=("active_exposure_z", lambda s: s.abs().mean()),
        max_abs_active_exposure_z=("active_exposure_z", lambda s: s.abs().max()),
        p95_abs_active_exposure_z=("active_exposure_z", lambda s: s.abs().quantile(0.95)),
    )
    return active.sort_values(["strategy", "exposure", "rebalance_date"]), summary.reset_index().sort_values(
        ["strategy", "max_abs_active_exposure_z"], ascending=[True, False]
    )


def gp_expression_search(panel: pd.DataFrame, factor_cols: list[str], *, train_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = [col for col in factor_cols if col in BASE_FACTOR_COLUMNS]
    data = panel[["rebalance_date", "ts_code", "period_return", "is_hs300_member", "industry"] + base_cols].copy()
    for col in base_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[f"gp_{col}"] = data.groupby("rebalance_date", observed=True)[col].transform(winsor_zscore).fillna(0.0)
    candidate_specs = []
    for col in base_cols:
        candidate_specs.append((f"id({col})", lambda df, c=col: df[f"gp_{c}"]))
        candidate_specs.append((f"neg({col})", lambda df, c=col: -df[f"gp_{c}"]))
    useful_pairs = [
        ("value_bp", "quality_roe"),
        ("value_ep", "quality_roe"),
        ("dividend_yield", "quality_roe"),
        ("momentum_60d", "volatility_20d"),
        ("momentum_60d", "turnover_20d"),
        ("reversal_20d", "volatility_20d"),
        ("quality_roe", "volatility_20d"),
        ("quality_roa", "turnover_20d"),
        ("size_neg_log_mv", "quality_roe"),
        ("size_neg_log_mv", "momentum_60d"),
        ("value_bp", "momentum_60d"),
        ("value_bp", "turnover_20d"),
    ]
    useful_pairs = [(a, b) for a, b in useful_pairs if a in base_cols and b in base_cols]
    for a, b in useful_pairs:
        candidate_specs.extend(
            [
                (f"add({a},{b})", lambda df, x=a, y=b: df[f"gp_{x}"] + df[f"gp_{y}"]),
                (f"sub({a},{b})", lambda df, x=a, y=b: df[f"gp_{x}"] - df[f"gp_{y}"]),
                (f"mul({a},{b})", lambda df, x=a, y=b: df[f"gp_{x}"] * df[f"gp_{y}"]),
                (f"ratio({a},{b})", lambda df, x=a, y=b: df[f"gp_{x}"] / (df[f"gp_{y}"].abs() + 0.5)),
            ]
        )
    rows = []
    layer_rows = []
    for name, func in candidate_specs:
        data["candidate"] = func(data).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        data["candidate"] = data.groupby("rebalance_date", observed=True)["candidate"].transform(winsor_zscore).fillna(0.0)
        data["candidate"] = data["candidate"] - data.groupby(["rebalance_date", "industry"], observed=True)["candidate"].transform("mean")
        data["candidate"] = data.groupby("rebalance_date", observed=True)["candidate"].transform(winsor_zscore).fillna(0.0)
        row = evaluate_candidate(data, name, train_end)
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, pd.DataFrame()
    summary = summary.sort_values(["test_rank_icir", "test_long_short_ir", "train_rank_icir"], ascending=False)
    for name in summary.head(10)["expression"]:
        spec = dict(candidate_specs)[name]
        data["candidate"] = spec(data).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        data["candidate"] = data.groupby("rebalance_date", observed=True)["candidate"].transform(winsor_zscore).fillna(0.0)
        for date, current in data.groupby("rebalance_date", observed=True):
            current = current[["candidate", "period_return"]].dropna()
            if len(current) < 100:
                continue
            current["layer"] = pd.qcut(current["candidate"].rank(method="first"), 5, labels=False, duplicates="drop") + 1
            for layer, layer_data in current.groupby("layer", observed=True):
                layer_rows.append({"rebalance_date": date, "expression": name, "layer": int(layer), "layer_return": layer_data["period_return"].mean()})
    return summary, pd.DataFrame(layer_rows)


def evaluate_candidate(data: pd.DataFrame, name: str, train_end: pd.Timestamp) -> dict[str, float | str]:
    def period_stats(frame: pd.DataFrame) -> tuple[float, float, float, float, float]:
        ics = []
        long_short = []
        for _, current in frame.groupby("rebalance_date", observed=True):
            current = current[["candidate", "period_return"]].dropna()
            if len(current) < 100:
                continue
            ics.append(current["candidate"].rank().corr(current["period_return"].rank()))
            current["layer"] = pd.qcut(current["candidate"].rank(method="first"), 5, labels=False, duplicates="drop") + 1
            top = current[current["layer"].eq(5)]["period_return"].mean()
            bottom = current[current["layer"].eq(1)]["period_return"].mean()
            long_short.append(top - bottom)
        ic = pd.Series(ics, dtype=float).dropna()
        ls = pd.Series(long_short, dtype=float).dropna()
        return (
            ic.mean(),
            safe_div(ic.mean(), ic.std(ddof=1)),
            ls.mean() * 12,
            safe_div(ls.mean() * 12, ls.std(ddof=1) * np.sqrt(12)),
            len(ic),
        )

    train = data[data["rebalance_date"] <= train_end]
    test = data[data["rebalance_date"] > train_end]
    train_rank_ic, train_rank_icir, train_ls, train_ls_ir, train_periods = period_stats(train)
    test_rank_ic, test_rank_icir, test_ls, test_ls_ir, test_periods = period_stats(test)
    return {
        "expression": name,
        "train_rank_ic": train_rank_ic,
        "train_rank_icir": train_rank_icir,
        "train_long_short_annual": train_ls,
        "train_long_short_ir": train_ls_ir,
        "train_periods": train_periods,
        "test_rank_ic": test_rank_ic,
        "test_rank_icir": test_rank_icir,
        "test_long_short_annual": test_ls,
        "test_long_short_ir": test_ls_ir,
        "test_periods": test_periods,
    }


def plot_layer_tests(layer_returns: pd.DataFrame, layer_summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    top = layer_summary[layer_summary["scope"].eq("hs300_inside")].head(4)
    if top.empty:
        top = layer_summary.head(4)
    fig, axes = plt.subplots(len(top), 1, figsize=(10, max(3, 2.8 * len(top))), sharex=True)
    if len(top) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, top.iterrows()):
        data = layer_returns[(layer_returns["scope"].eq(row["scope"])) & (layer_returns["factor"].eq(row["factor"]))]
        pivot = data.pivot_table(index="rebalance_date", columns="layer", values="layer_return", aggfunc="mean").sort_index()
        nav = (1 + pivot).cumprod()
        for col in nav.columns:
            ax.plot(nav.index, nav[col], label=f"L{col}", linewidth=1.2)
        ax.set_title(f"{row['scope']} {row['factor']} layer NAV")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "single_signal_top_layer_nav.png", dpi=180)
    plt.close(fig)


def plot_exposures(industry_summary: pd.DataFrame, style_summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    top = industry_summary.sort_values("max_abs_industry_deviation", ascending=False).head(10)
    ax.barh(top["strategy"], top["max_abs_industry_deviation"])
    ax.invert_yaxis()
    ax.set_title("Max absolute industry deviation")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "industry_deviation_summary.png", dpi=180)
    plt.close(fig)

    pivot = style_summary.pivot_table(index="strategy", columns="exposure", values="mean_abs_active_exposure_z", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(pivot.fillna(0.0), aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Mean absolute active style exposure (z)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "style_exposure_heatmap.png", dpi=180)
    plt.close(fig)


def plot_te_controlled_strategy_curves(returns: pd.DataFrame, artifact_dir: Path) -> None:
    import matplotlib.pyplot as plt

    frame = returns.sort_values("rebalance_date").copy()
    frame = frame.set_index("rebalance_date")
    nav_cols = {}
    nav_cols["benchmark"] = (1 + frame["benchmark_return"]).cumprod()
    for strategy in PLOT_STRATEGIES:
        if strategy == "benchmark":
            continue
        col = f"{strategy}_return"
        if col in frame.columns:
            nav_cols[strategy] = (1 + frame[col]).cumprod()
    nav = pd.DataFrame(nav_cols)
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in nav.columns:
        ax.plot(nav.index, nav[col], label=col, linewidth=1.5)
    ax.set_title("NAV curves, excluding standalone high-TE outside satellites")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    nav_path = artifact_dir / "nav_curves_te_controlled.png"
    fig.savefig(nav_path, dpi=180)
    fig.savefig(artifact_dir / "nav_curves.png", dpi=180)
    plt.close(fig)

    active_cols = {}
    for strategy in PLOT_STRATEGIES:
        if strategy == "benchmark":
            continue
        col = f"{strategy}_return"
        if col in frame.columns:
            active_cols[strategy] = (1 + (frame[col] - frame["benchmark_return"])).cumprod()
    active_nav = pd.DataFrame(active_cols)
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in active_nav.columns:
        ax.plot(active_nav.index, active_nav[col], label=col, linewidth=1.5)
    ax.set_title("Active NAV curves, excluding standalone high-TE outside satellites")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    active_path = artifact_dir / "active_nav_curves_te_controlled.png"
    fig.savefig(active_path, dpi=180)
    fig.savefig(artifact_dir / "active_nav_curves.png", dpi=180)
    plt.close(fig)
    write_embedded_nav_html(nav_path, active_path, artifact_dir / "nav_curves_te_controlled.html")


def write_embedded_nav_html(nav_path: Path, active_path: Path, out_path: Path) -> None:
    nav_b64 = base64.b64encode(nav_path.read_bytes()).decode("ascii")
    active_b64 = base64.b64encode(active_path.read_bytes()).decode("ascii")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HS300 NAV Curves</title>
  <style>
    body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #fff; color: #111; }}
    h1 {{ font-size: 22px; margin: 0 0 16px; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>HS300 Index Enhancement NAV Curves</h1>
  <h2>NAV</h2>
  <img alt="NAV curves" src="data:image/png;base64,{nav_b64}">
  <h2>Active NAV</h2>
  <img alt="Active NAV curves" src="data:image/png;base64,{active_b64}">
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def write_report(
    out_dir: Path,
    ic_summary: pd.DataFrame,
    layer_summary: pd.DataFrame,
    prune_report: pd.DataFrame,
    industry_summary: pd.DataFrame,
    style_summary: pd.DataFrame,
    size_value_summary: pd.DataFrame,
    gp_summary: pd.DataFrame,
) -> None:
    top_ic = ic_summary[ic_summary["scope"].eq("hs300_inside")].head(8)
    top_layer = layer_summary[layer_summary["scope"].eq("hs300_inside")].head(8)
    top_gp = gp_summary.head(12)
    text = f"""# Factor Diagnostics And Expansion

## Scope

This report uses the existing All-A monthly panel and strategy weights. It does not redownload data and does not alter the fixed HS300 benchmark return series.

## Preprocessing Used In This Diagnostic Round

- Missing factor values are filled by rebalance-date cross-sectional median, then by 0.
- Factors are 1%/99% winsorized and z-scored by rebalance date.
- Factors are cross-sectionally demeaned.
- Single-signal and GP tests use industry-demeaned factors: factor z-score minus same-date same-industry mean, then re-standardized.
- Rank transform is not used as an input transform; ranks are used only for RankIC and quantile layering diagnostics.
- Correlation control is reported through a Spearman correlation matrix and a threshold-based drop-candidate table.

## HS300-Inside Single Signal IC Leaders

{top_ic.to_markdown(index=False)}

## HS300-Inside Layer Test Leaders

{top_layer.to_markdown(index=False)}

## Correlation Control

High-correlation pairs above the configured threshold are written to `factor_correlation_prune_report.csv`.

Top pairs:

{prune_report.head(12).to_markdown(index=False) if not prune_report.empty else "No pairs above threshold."}

## Exposure Diagnostics

Industry active-weight summary is written to `strategy_industry_deviation_summary.csv`.

{industry_summary.head(10).to_markdown(index=False)}

Style exposure summary is written to `strategy_style_exposure_summary.csv`.

{style_summary.head(16).to_markdown(index=False)}

Size/value-specific exposure summary is written to `strategy_size_value_exposure_summary.csv`.

{size_value_summary.head(16).to_markdown(index=False)}

## GP-Like Factor Expansion

The current pass is a symbolic expression/kernel search over already-downloaded raw-data-derived fields. It tests unary transforms and interpretable pairwise kernels: add/subtract/multiply/ratio over value, quality, momentum, reversal, volatility, turnover, and size.

Top candidates:

{top_gp.to_markdown(index=False)}

## Output Figures

- `single_signal_top_layer_nav.png`
- `industry_deviation_summary.png`
- `style_exposure_heatmap.png`
- `../nav_curves_te_controlled.png`
- `../active_nav_curves_te_controlled.png`
"""
    (out_dir / "factor_diagnostics_report.md").write_text(text, encoding="utf-8")


def annualized_return(period_returns: pd.Series, periods_per_year: int = 12) -> float:
    period_returns = pd.Series(period_returns, dtype=float).dropna()
    if period_returns.empty:
        return np.nan
    return float((1 + period_returns).prod() ** (periods_per_year / len(period_returns)) - 1)


def safe_div(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or den == 0:
        return np.nan
    return float(num / den)


if __name__ == "__main__":
    main()
