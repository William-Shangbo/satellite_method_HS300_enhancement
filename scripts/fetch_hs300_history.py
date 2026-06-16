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
from index_platform.ingest.tushare_client import TushareHTTPClient, TushareRequest


DAILY_DATASETS = {
    "daily_bar": ("daily", ("ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount")),
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

FINANCIAL_DATASETS = {
    "income_statement": ("income", ("ts_code", "ann_date", "end_date", "report_type", "total_revenue", "revenue", "operate_profit", "n_income", "n_income_attr_p")),
    "balance_sheet": ("balancesheet", ("ts_code", "ann_date", "end_date", "report_type", "total_assets", "total_hldr_eqy_exc_min_int", "total_liab", "money_cap", "inventories")),
    "cashflow_statement": ("cashflow", ("ts_code", "ann_date", "end_date", "report_type", "n_cashflow_act", "stot_cash_in_fnc_act", "stot_cashout_fnc_act")),
    "financial_indicator": ("fina_indicator", ("ts_code", "ann_date", "end_date", "roe", "roe_dt", "roa", "grossprofit_margin", "netprofit_margin", "assets_turn")),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch point-in-time HS300 research data from Tushare.")
    parser.add_argument("--start-date", default="20160101")
    parser.add_argument("--end-date", default="20260101")
    parser.add_argument("--index-code", default="399300.SZ")
    parser.add_argument("--out-dir", default="data/raw/tushare_hs300_20160101_20260101")
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument(
        "--stages",
        default="calendar,weights,stock_basic,daily,financial",
        help="Comma-separated: calendar,weights,stock_basic,daily,financial",
    )
    args = parser.parse_args()

    settings = load_settings(ROOT)
    contracts = load_dataset_contracts(ROOT / "configs/datasets/tushare_pipeline.yaml")
    client = TushareHTTPClient(settings, endpoint=contracts.endpoint, timeout_seconds=60)
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

    weights_path = out_dir / "benchmark_weight.csv"
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
        weights.to_csv(weights_path, index=False)
        print(f"benchmark_weight total: {len(weights)} rows -> {weights_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Need benchmark weights first: {weights_path}")
    universe_codes = sorted(pd.read_csv(weights_path)["con_code"].dropna().astype(str).unique())
    (out_dir / "hs300_universe_codes.txt").write_text("\n".join(universe_codes) + "\n", encoding="utf-8")
    print(f"unique historical HS300 constituents: {len(universe_codes)}")

    if "stock_basic" in stages:
        stock_basic = fetch_frame(
            client,
            "stock_basic",
            {"exchange": "", "list_status": "L"},
            ("ts_code", "symbol", "name", "area", "industry", "market", "list_date"),
            sleep=args.sleep,
        )
        stock_basic = stock_basic[stock_basic["ts_code"].isin(universe_codes)].copy()
        target = out_dir / "stock_basic.csv"
        stock_basic.to_csv(target, index=False)
        print(f"stock_basic total: {len(stock_basic)} rows -> {target}")

    if "daily" in stages:
        for dataset, (api_name, fields) in DAILY_DATASETS.items():
            dataset_dir = out_dir / dataset
            dataset_dir.mkdir(exist_ok=True)
            for idx, code in enumerate(universe_codes, start=1):
                target = dataset_dir / f"{code}.csv"
                if target.exists():
                    continue
                frame = fetch_frame(
                    client,
                    api_name,
                    {"ts_code": code, "start_date": args.start_date, "end_date": args.end_date},
                    fields,
                    sleep=args.sleep,
                )
                frame.to_csv(target, index=False)
                print(f"{dataset} {idx}/{len(universe_codes)} {code}: {len(frame)} rows")

    if "financial" in stages:
        for dataset, (api_name, fields) in FINANCIAL_DATASETS.items():
            dataset_dir = out_dir / dataset
            dataset_dir.mkdir(exist_ok=True)
            for idx, code in enumerate(universe_codes, start=1):
                target = dataset_dir / f"{code}.csv"
                if target.exists():
                    continue
                frame = fetch_frame(
                    client,
                    api_name,
                    {"ts_code": code, "start_date": args.start_date, "end_date": args.end_date},
                    fields,
                    sleep=args.sleep,
                )
                frame.to_csv(target, index=False)
                print(f"{dataset} {idx}/{len(universe_codes)} {code}: {len(frame)} rows")


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
    frame = client.fetch_frame(TushareRequest(api_name=api_name, params=params, fields=fields))
    if sleep:
        time.sleep(sleep)
    return frame


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
