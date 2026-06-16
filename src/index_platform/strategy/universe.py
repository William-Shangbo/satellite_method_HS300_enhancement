from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class UniverseSnapshot:
    as_of_date: pd.Timestamp
    stocks: pd.DataFrame


def month_end_rebalance_dates(trade_calendar: pd.DataFrame) -> pd.DatetimeIndex:
    """Return each calendar month's last open trading day."""
    calendar = trade_calendar.copy()
    calendar["cal_date"] = pd.to_datetime(calendar["cal_date"].astype(str), format="%Y%m%d", errors="coerce")
    if "is_open" in calendar.columns:
        calendar = calendar[calendar["is_open"].astype(int) == 1]
    dates = calendar["cal_date"].dropna().sort_values()
    return pd.DatetimeIndex(dates.groupby(dates.dt.to_period("M")).max().values)


def build_hs300_universe(
    index_weight: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    *,
    index_code: str = "399300.SZ",
) -> UniverseSnapshot:
    """Build the latest visible HS300 constituent snapshot on or before as_of_date."""
    weights = index_weight.copy()
    weights["trade_date"] = pd.to_datetime(weights["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    as_of = pd.to_datetime(as_of_date)
    if "index_code" in weights.columns:
        weights = weights[weights["index_code"] == index_code]
    weights = weights[weights["trade_date"] <= as_of]
    if weights.empty:
        raise ValueError(f"No index_weight rows visible on or before {as_of.date()}.")

    latest_date = weights["trade_date"].max()
    snapshot = weights[weights["trade_date"] == latest_date].copy()
    snapshot = snapshot.rename(columns={"con_code": "ts_code", "weight": "benchmark_weight"})
    snapshot["benchmark_weight"] = pd.to_numeric(snapshot["benchmark_weight"], errors="coerce")
    snapshot = snapshot.dropna(subset=["ts_code", "benchmark_weight"])
    snapshot["benchmark_weight"] = snapshot["benchmark_weight"] / snapshot["benchmark_weight"].sum()
    return UniverseSnapshot(as_of_date=latest_date, stocks=snapshot[["ts_code", "benchmark_weight"]])


def apply_tradability_filter(
    universe: pd.DataFrame,
    *,
    security_master: pd.DataFrame | None = None,
    suspension: pd.DataFrame | None = None,
    limit_status: pd.DataFrame | None = None,
    trade_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Apply basic listed/tradable filters while preserving benchmark weights."""
    result = universe.copy()
    date = pd.to_datetime(trade_date) if trade_date is not None else None

    if security_master is not None and "list_status" in security_master.columns:
        active = security_master[security_master["list_status"].eq("L")][["ts_code"]]
        result = result.merge(active, on="ts_code", how="inner")

    if suspension is not None and date is not None and not suspension.empty:
        susp = suspension.copy()
        susp["trade_date"] = pd.to_datetime(susp["trade_date"], errors="coerce")
        suspended = set(susp.loc[susp["trade_date"].eq(date), "ts_code"])
        result = result[~result["ts_code"].isin(suspended)]

    if limit_status is not None and date is not None and not limit_status.empty:
        limits = limit_status.copy()
        limits["trade_date"] = pd.to_datetime(limits["trade_date"], errors="coerce")
        blocked = set(limits.loc[limits["trade_date"].eq(date), "ts_code"])
        result = result[~result["ts_code"].isin(blocked)]

    result["benchmark_weight"] = result["benchmark_weight"] / result["benchmark_weight"].sum()
    return result.reset_index(drop=True)
