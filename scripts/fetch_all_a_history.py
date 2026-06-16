from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index_platform.common.settings import load_settings
from index_platform.ingest.contracts import load_dataset_contracts
from index_platform.ingest.tushare_client import TushareAPIError, TushareHTTPClient, TushareRequest


DAILY_DATASETS = {
    "daily_bar": (
        "daily",
        ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"),
    ),
    "adjustment_factor": ("adj_factor", ("ts_code", "trade_date", "adj_factor")),
    "daily_basic": (
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
    ),
}

FINANCIAL_INDICATOR_FIELDS = (
    "ts_code",
    "ann_date",
    "end_date",
    "roe",
    "roe_dt",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "assets_turn",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch All-A data needed for HS300 satellite replication.")
    parser.add_argument("--start-date", default="20160101")
    parser.add_argument("--end-date", default="20260101")
    parser.add_argument("--index-code", default="399300.SZ")
    parser.add_argument("--out-dir", default="data/raw/tushare_all_a_20160101_20260101")
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument(
        "--stages",
        default="calendar,weights,stock_basic,daily,financial",
        help="Comma-separated: calendar,weights,stock_basic,daily,financial",
    )
    args = parser.parse_args()

    settings = load_settings(ROOT)
    contracts = load_dataset_contracts(ROOT / "configs/datasets/tushare_pipeline.yaml")
    client = TushareHTTPClient(settings, endpoint=contracts.endpoint, timeout_seconds=90)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}

    if "calendar" in stages:
        fetch_one(
            client,
            "trade_calendar",
            "trade_cal",
            {"exchange": "SSE", "start_date": args.start_date, "end_date": args.end_date},
            ("exchange", "cal_date", "is_open", "pretrade_date"),
            out_dir,
            sleep=args.sleep,
        )

    if "weights" in stages:
        frames = []
        for start, end in month_ranges(args.start_date, args.end_date):
            frame = fetch_frame(
                client,
                "index_weight",
                {"index_code": args.index_code, "start_date": start, "end_date": end},
                ("index_code", "con_code", "trade_date", "weight"),
                sleep=args.sleep,
            )
            frames.append(frame)
            print(f"benchmark_weight {start}-{end}: {len(frame)} rows")
        weights = pd.concat(frames, ignore_index=True).drop_duplicates()
        target = out_dir / "benchmark_weight.csv"
        weights.to_csv(target, index=False)
        print(f"benchmark_weight total: {len(weights)} rows -> {target}")

    stock_basic_path = out_dir / "stock_basic.csv"
    if "stock_basic" in stages:
        frames = []
        for status in ["L", "D", "P"]:
            frame = fetch_frame(
                client,
                "stock_basic",
                {"exchange": "", "list_status": status},
                ("ts_code", "symbol", "name", "area", "industry", "market", "exchange", "list_status", "list_date", "delist_date"),
                sleep=args.sleep,
            )
            frames.append(frame)
            print(f"stock_basic status={status}: {len(frame)} rows")
        stock_basic = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")
        stock_basic.to_csv(stock_basic_path, index=False)
        print(f"stock_basic total: {len(stock_basic)} rows -> {stock_basic_path}")

    if not stock_basic_path.exists():
        raise FileNotFoundError(f"Need stock_basic first: {stock_basic_path}")
    stock_basic = pd.read_csv(stock_basic_path)
    all_codes = sorted(stock_basic["ts_code"].dropna().astype(str).unique())
    (out_dir / "all_a_universe_codes.txt").write_text("\n".join(all_codes) + "\n", encoding="utf-8")
    print(f"all-a securities in stock_basic: {len(all_codes)}")

    if "daily" in stages:
        trade_dates = open_trade_dates(out_dir / "trade_calendar.csv", args.start_date, args.end_date)
        for dataset, (api_name, fields) in DAILY_DATASETS.items():
            dataset_dir = out_dir / dataset
            dataset_dir.mkdir(exist_ok=True)
            for idx, trade_date in enumerate(trade_dates, start=1):
                target = dataset_dir / f"{trade_date}.parquet"
                if target.exists():
                    continue
                frame = fetch_frame(
                    client,
                    api_name,
                    {"trade_date": trade_date},
                    fields,
                    sleep=args.sleep,
                )
                if not frame.empty:
                    frame = frame[frame["ts_code"].isin(all_codes)].copy()
                frame.to_parquet(target, index=False)
                print(f"{dataset} {idx}/{len(trade_dates)} {trade_date}: {len(frame)} rows")

    if "financial" in stages:
        dataset_dir = out_dir / "financial_indicator"
        dataset_dir.mkdir(exist_ok=True)
        for idx, code in enumerate(all_codes, start=1):
            target = dataset_dir / f"{code}.csv"
            if target.exists():
                continue
            frame = fetch_frame(
                client,
                "fina_indicator",
                {"ts_code": code, "start_date": args.start_date, "end_date": args.end_date},
                FINANCIAL_INDICATOR_FIELDS,
                sleep=args.sleep,
            )
            frame.to_csv(target, index=False)
            print(f"financial_indicator {idx}/{len(all_codes)} {code}: {len(frame)} rows")


def fetch_one(
    client: TushareHTTPClient,
    dataset: str,
    api_name: str,
    params: dict[str, str],
    fields: tuple[str, ...],
    out_dir: Path,
    *,
    sleep: float,
) -> pd.DataFrame:
    frame = fetch_frame(client, api_name, params, fields, sleep=sleep)
    target = out_dir / f"{dataset}.csv"
    frame.to_csv(target, index=False)
    print(f"{dataset}: {len(frame)} rows -> {target}")
    return frame


def fetch_frame(
    client: TushareHTTPClient,
    api_name: str,
    params: dict[str, str],
    fields: tuple[str, ...],
    *,
    sleep: float,
) -> pd.DataFrame:
    for attempt in range(5):
        try:
            frame = client.fetch_frame(TushareRequest(api_name=api_name, params=params, fields=fields))
            break
        except TushareAPIError as exc:
            if "频率超限" not in str(exc) or attempt == 4:
                raise
            wait_seconds = 75 + attempt * 15
            print(f"{api_name} rate limited; sleeping {wait_seconds}s before retry...")
            time.sleep(wait_seconds)
    if sleep:
        time.sleep(sleep)
    return frame


def open_trade_dates(calendar_path: Path, start_date: str, end_date: str) -> list[str]:
    if not calendar_path.exists():
        raise FileNotFoundError(f"Need trade calendar first: {calendar_path}")
    calendar = pd.read_csv(calendar_path)
    dates = calendar[calendar["is_open"].eq(1)]["cal_date"].astype(str)
    return dates[(dates >= start_date) & (dates <= end_date)].tolist()


def month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    starts = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="MS")
    if not len(starts) or starts[0] > pd.to_datetime(start_date):
        starts = starts.insert(0, pd.to_datetime(start_date))
    ranges = []
    end_ts = pd.to_datetime(end_date)
    for start in starts:
        end = min(start + pd.offsets.MonthEnd(0), end_ts)
        if start <= end:
            ranges.append((start.strftime("%Y%m%d"), end.strftime("%Y%m%d")))
    return ranges


if __name__ == "__main__":
    main()
