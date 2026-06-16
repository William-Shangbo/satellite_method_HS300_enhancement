from pathlib import Path

from index_platform.panel.cleaning import HS300RawCleaner


def test_cleaner_removes_duplicate_trade_date_stock_keys():
    zip_path = Path("/Users/shangbo/study/P2025/ita/index_enforcement/src/沪深300成分股自2010以来的数据.zip")
    cleaner = HS300RawCleaner(zip_path)
    _, cleaned, summary = cleaner.clean_panel()

    assert summary.raw_rows > summary.cleaned_rows
    assert summary.duplicate_keys_removed > 0
    assert cleaned.duplicated(subset=["trade_date", "ts_code"]).sum() == 0


def test_cleaned_panel_has_core_research_columns_without_nulls():
    zip_path = Path("/Users/shangbo/study/P2025/ita/index_enforcement/src/沪深300成分股自2010以来的数据.zip")
    cleaner = HS300RawCleaner(zip_path)
    _, cleaned, _ = cleaner.clean_panel()

    required = ["ts_code", "trade_date", "close", "return_1d", "forward_return_1d", "benchmark_weight"]
    assert cleaned[required].isna().sum().sum() == 0
    assert cleaned["ts_code"].nunique() == 300
