from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TusharePanelSummary:
    raw_daily_rows: int
    raw_adj_factor_rows: int
    raw_daily_basic_rows: int
    raw_index_weight_rows: int
    cleaned_rows: int
    unique_stocks: int
    trade_dates: int
    duplicate_daily_keys_removed: int
    duplicate_adj_keys_removed: int
    duplicate_basic_keys_removed: int


def _deduplicate(frame: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, int]:
    dup_count = int(frame.duplicated(subset=keys).sum())
    cleaned = (
        frame.sort_values(keys)
        .drop_duplicates(subset=keys, keep="last")
        .reset_index(drop=True)
    )
    return cleaned, dup_count


def build_hs300_research_panel(
    daily: pd.DataFrame,
    adj_factor: pd.DataFrame,
    daily_basic: pd.DataFrame,
    index_weight: pd.DataFrame,
) -> tuple[pd.DataFrame, TusharePanelSummary]:
    daily = daily.copy()
    adj_factor = adj_factor.copy()
    daily_basic = daily_basic.copy()
    index_weight = index_weight.copy()

    for frame in [daily, adj_factor, daily_basic, index_weight]:
        if "trade_date" in frame.columns:
            frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")

    for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")

    if "adj_factor" in adj_factor.columns:
        adj_factor["adj_factor"] = pd.to_numeric(adj_factor["adj_factor"], errors="coerce")

    for col in [
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ]:
        if col in daily_basic.columns:
            daily_basic[col] = pd.to_numeric(daily_basic[col], errors="coerce")

    if "weight" in index_weight.columns:
        index_weight["weight"] = pd.to_numeric(index_weight["weight"], errors="coerce")

    daily, dup_daily = _deduplicate(daily, ["ts_code", "trade_date"])
    adj_factor, dup_adj = _deduplicate(adj_factor, ["ts_code", "trade_date"])
    daily_basic, dup_basic = _deduplicate(daily_basic, ["ts_code", "trade_date"])
    index_weight, _ = _deduplicate(index_weight, ["con_code", "trade_date"])

    latest_weight = (
        index_weight.sort_values("trade_date")
        .drop_duplicates(subset=["con_code"], keep="last")[["con_code", "weight"]]
        .rename(columns={"con_code": "ts_code", "weight": "benchmark_weight"})
    )
    latest_weight["benchmark_weight"] = latest_weight["benchmark_weight"] / latest_weight["benchmark_weight"].sum()

    panel = (
        daily.merge(adj_factor, on=["ts_code", "trade_date"], how="left")
        .merge(daily_basic, on=["ts_code", "trade_date"], how="left")
        .merge(latest_weight, on="ts_code", how="inner")
    )
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    latest_factor = panel.groupby("ts_code")["adj_factor"].transform("last")
    panel["adj_close"] = panel["close"] * panel["adj_factor"] / latest_factor
    panel["return_1d"] = panel.groupby("ts_code")["adj_close"].pct_change()
    panel["forward_return_1d"] = panel.groupby("ts_code")["return_1d"].shift(-1)
    panel["log_total_mv"] = panel["total_mv"].where(panel["total_mv"] > 0).map(lambda x: pd.NA if pd.isna(x) else x)
    panel["log_total_mv"] = pd.to_numeric(panel["log_total_mv"], errors="coerce")
    panel["log_total_mv"] = panel["log_total_mv"].map(lambda x: None if pd.isna(x) else x)
    panel["is_missing_pe_ttm"] = panel["pe_ttm"].isna().astype(int)
    panel["pe_ttm"] = panel.groupby("trade_date")["pe_ttm"].transform(lambda s: s.fillna(s.median()))

    cleaned = panel.dropna(subset=["adj_close", "return_1d", "forward_return_1d", "benchmark_weight"]).copy()
    summary = TusharePanelSummary(
        raw_daily_rows=len(daily),
        raw_adj_factor_rows=len(adj_factor),
        raw_daily_basic_rows=len(daily_basic),
        raw_index_weight_rows=len(index_weight),
        cleaned_rows=len(cleaned),
        unique_stocks=int(cleaned["ts_code"].nunique()),
        trade_dates=int(cleaned["trade_date"].nunique()),
        duplicate_daily_keys_removed=dup_daily,
        duplicate_adj_keys_removed=dup_adj,
        duplicate_basic_keys_removed=dup_basic,
    )
    return cleaned, summary
