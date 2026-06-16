from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import tempfile

import pandas as pd


RAW_COLUMN_MAP = {
    "代码": "ts_code",
    "简称": "name",
    "时间": "trade_date",
    "收盘价(元)": "close",
    "成交量(万股)": "volume_10k_shares",
    "PE市盈率(TTM)": "pe_ttm",
    "总市值(万元)": "total_mv_10k",
}


@dataclass(frozen=True)
class CleaningSummary:
    raw_rows: int
    cleaned_rows: int
    duplicate_keys_removed: int
    dropped_missing_key_rows: int
    unique_stocks: int
    date_min: str
    date_max: str


class HS300RawCleaner:
    def __init__(self, zip_path: str | Path) -> None:
        self.zip_path = Path(zip_path)

    def load_raw_panel(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        with ZipFile(self.zip_path) as zf:
            workbook_names = [name for name in zf.namelist() if name.endswith(".xlsx")]
            for name in workbook_names:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    target = Path(tmp_dir) / Path(name).name
                    target.write_bytes(zf.read(name))
                    frame = pd.read_excel(target)
                frame["source_file"] = Path(name).name
                frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"No workbook files found in {self.zip_path}")
        return pd.concat(frames, ignore_index=True)

    def clean_panel(self) -> tuple[pd.DataFrame, pd.DataFrame, CleaningSummary]:
        raw = self.load_raw_panel()
        working = raw.rename(columns=RAW_COLUMN_MAP)
        working = working[list(RAW_COLUMN_MAP.values()) + ["source_file"]].copy()

        working["trade_date"] = pd.to_datetime(working["trade_date"], errors="coerce")
        for col in ["close", "volume_10k_shares", "pe_ttm", "total_mv_10k"]:
            working[col] = pd.to_numeric(working[col], errors="coerce")

        raw_rows = len(working)
        before_key_drop = len(working)
        working = working.dropna(subset=["ts_code", "trade_date"]).copy()
        dropped_missing_key_rows = before_key_drop - len(working)

        duplicate_keys_removed = int(working.duplicated(subset=["trade_date", "ts_code"]).sum())
        working = (
            working.sort_values(["trade_date", "ts_code", "source_file"])
            .drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
            .sort_values(["ts_code", "trade_date"])
            .reset_index(drop=True)
        )

        working["return_1d"] = working.groupby("ts_code")["close"].pct_change()
        working["forward_return_1d"] = working.groupby("ts_code")["return_1d"].shift(-1)
        working["benchmark_weight"] = working.groupby("trade_date")["total_mv_10k"].transform(
            lambda s: s / s.sum()
        )
        working["volume_ma20"] = (
            working.groupby("ts_code")["volume_10k_shares"]
            .rolling(window=20, min_periods=5)
            .mean()
            .reset_index(level=0, drop=True)
        )
        working["turnover_proxy"] = working["volume_10k_shares"] / working["volume_ma20"]

        cleaned = working.dropna(subset=["return_1d", "forward_return_1d", "benchmark_weight"]).copy()
        summary = CleaningSummary(
            raw_rows=raw_rows,
            cleaned_rows=len(cleaned),
            duplicate_keys_removed=duplicate_keys_removed,
            dropped_missing_key_rows=dropped_missing_key_rows,
            unique_stocks=int(cleaned["ts_code"].nunique()),
            date_min=cleaned["trade_date"].min().strftime("%Y-%m-%d"),
            date_max=cleaned["trade_date"].max().strftime("%Y-%m-%d"),
        )
        return working, cleaned, summary
