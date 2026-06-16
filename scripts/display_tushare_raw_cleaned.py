from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from index_platform.common.settings import load_settings
from index_platform.ingest.tushare_client import TushareHTTPClient, TushareRequest
from index_platform.panel.tushare_cleaning import build_hs300_research_panel


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Fetch a Tushare HS300 sample and display raw vs cleaned data.",
    )
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", default="20240131")
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "artifacts" / "tushare_raw_cleaned"),
    )
    parser.add_argument("--sample-rows", type=int, default=8)
    return parser.parse_args()


def _open_trade_dates(client: TushareHTTPClient, start_date: str, end_date: str) -> list[str]:
    calendar = client.fetch_frame(
        TushareRequest(
            api_name="trade_cal",
            params={"exchange": "SSE", "start_date": start_date, "end_date": end_date},
            fields=("exchange", "cal_date", "is_open", "pretrade_date"),
        )
    )
    return sorted(calendar.loc[calendar["is_open"] == "1", "cal_date"].tolist())


def _fetch_daily_by_trade_dates(client: TushareHTTPClient, trade_dates: list[str], api_name: str, fields: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        frames.append(
            client.fetch_frame(
                TushareRequest(
                    api_name=api_name,
                    params={"trade_date": trade_date},
                    fields=fields,
                )
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=fields)


def main() -> None:
    args = parse_args()
    settings = load_settings()
    client = TushareHTTPClient(settings)

    index_weight = client.fetch_frame(
        TushareRequest(
            api_name="index_weight",
            params={
                "index_code": settings.benchmark_index,
                "start_date": args.start_date,
                "end_date": args.end_date,
            },
            fields=("index_code", "con_code", "trade_date", "weight"),
        )
    )
    universe = set(index_weight["con_code"].dropna().tolist())
    trade_dates = _open_trade_dates(client, args.start_date, args.end_date)

    daily = _fetch_daily_by_trade_dates(
        client,
        trade_dates,
        "daily",
        ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"),
    )
    adj_factor = _fetch_daily_by_trade_dates(
        client,
        trade_dates,
        "adj_factor",
        ("ts_code", "trade_date", "adj_factor"),
    )
    daily_basic = _fetch_daily_by_trade_dates(
        client,
        trade_dates,
        "daily_basic",
        (
            "ts_code",
            "trade_date",
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
        ),
    )

    daily = daily.loc[daily["ts_code"].isin(universe)].copy()
    adj_factor = adj_factor.loc[adj_factor["ts_code"].isin(universe)].copy()
    daily_basic = daily_basic.loc[daily_basic["ts_code"].isin(universe)].copy()

    cleaned, summary = build_hs300_research_panel(daily, adj_factor, daily_basic, index_weight)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.head(args.sample_rows).to_csv(output_dir / "raw_daily_sample.csv", index=False)
    daily_basic.head(args.sample_rows).to_csv(output_dir / "raw_daily_basic_sample.csv", index=False)
    adj_factor.head(args.sample_rows).to_csv(output_dir / "raw_adj_factor_sample.csv", index=False)
    index_weight.head(args.sample_rows).to_csv(output_dir / "raw_index_weight_sample.csv", index=False)
    cleaned.head(args.sample_rows).to_csv(output_dir / "cleaned_panel_sample.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("RAW_DAILY_SHAPE", daily.shape)
    print("RAW_DAILY_BASIC_SHAPE", daily_basic.shape)
    print("RAW_ADJ_FACTOR_SHAPE", adj_factor.shape)
    print("RAW_INDEX_WEIGHT_SHAPE", index_weight.shape)
    print("\nRAW_DAILY_SAMPLE")
    print(daily.head(args.sample_rows).to_string(index=False))
    print("\nCLEANED_SUMMARY")
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    print("\nCLEANED_PANEL_SAMPLE")
    print(cleaned.head(args.sample_rows).to_string(index=False))


if __name__ == "__main__":
    main()
