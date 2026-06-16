from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from index_platform.common.settings import load_settings
from index_platform.ingest.contracts import load_dataset_contracts
from index_platform.ingest.tushare_client import TushareHTTPClient, TushareRequest


DATE_PARAM_BY_API = {
    "daily": ("start_date", "end_date"),
    "adj_factor": ("start_date", "end_date"),
    "daily_basic": ("start_date", "end_date"),
    "trade_cal": ("start_date", "end_date"),
    "index_weight": ("start_date", "end_date"),
    "income": ("start_date", "end_date"),
    "balancesheet": ("start_date", "end_date"),
    "cashflow": ("start_date", "end_date"),
    "fina_indicator": ("start_date", "end_date"),
    "suspend_d": ("start_date", "end_date"),
    "limit_list": ("start_date", "end_date"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch HS300 index enhancement datasets from Tushare.")
    parser.add_argument("--start-date", required=True, help="YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="YYYYMMDD")
    parser.add_argument("--contracts", default="configs/datasets/tushare_pipeline.yaml")
    parser.add_argument("--out-dir", default="data/raw/tushare")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names. Empty means required datasets.")
    args = parser.parse_args()

    settings = load_settings(ROOT)
    contracts = load_dataset_contracts(ROOT / args.contracts)
    client = TushareHTTPClient(settings, endpoint=contracts.endpoint)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {item.strip() for item in args.datasets.split(",") if item.strip()}
    for dataset in contracts.datasets:
        if selected and dataset.dataset not in selected:
            continue
        if not selected and not dataset.required:
            continue

        params = dict(dataset.params)
        if dataset.api_name in DATE_PARAM_BY_API:
            start_key, end_key = DATE_PARAM_BY_API[dataset.api_name]
            params[start_key] = args.start_date
            params[end_key] = args.end_date

        fields = dataset.fields or dataset.fields_hint
        request = TushareRequest(api_name=dataset.api_name, params=params, fields=fields)
        frame = client.fetch_frame(request)
        target = out_dir / f"{dataset.dataset}_{args.start_date}_{args.end_date}.csv"
        frame.to_csv(target, index=False)
        print(f"{dataset.dataset}: {len(frame)} rows -> {target}")


if __name__ == "__main__":
    main()
