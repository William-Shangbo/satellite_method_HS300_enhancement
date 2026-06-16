import pandas as pd

from index_platform.panel.tushare_cleaning import build_hs300_research_panel


def test_build_hs300_research_panel_removes_duplicate_keys_and_outputs_core_columns():
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
            "trade_date": ["20240102", "20240102", "20240103", "20240102", "20240103"],
            "open": [10, 10, 11, 20, 21],
            "high": [10.5, 10.5, 11.5, 20.5, 21.5],
            "low": [9.8, 9.8, 10.8, 19.8, 20.8],
            "close": [10.0, 10.0, 11.0, 20.0, 21.0],
            "pre_close": [9.8, 9.8, 10.0, 19.5, 20.0],
            "change": [0.2, 0.2, 1.0, 0.5, 1.0],
            "pct_chg": [2.0, 2.0, 10.0, 2.5, 5.0],
            "vol": [100, 100, 120, 200, 220],
            "amount": [1000, 1000, 1300, 4000, 4500],
        }
    )
    adj_factor = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
            "trade_date": ["20240102", "20240103", "20240102", "20240103"],
            "adj_factor": [1.0, 1.1, 1.0, 1.0],
        }
    )
    daily_basic = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
            "trade_date": ["20240102", "20240103", "20240102", "20240103"],
            "turnover_rate": [1, 1, 1, 1],
            "turnover_rate_f": [1, 1, 1, 1],
            "volume_ratio": [1, 1, 1, 1],
            "pe": [10, 11, 12, 13],
            "pe_ttm": [10, None, 12, 13],
            "pb": [1, 1, 1, 1],
            "ps": [1, 1, 1, 1],
            "ps_ttm": [1, 1, 1, 1],
            "dv_ratio": [1, 1, 1, 1],
            "dv_ttm": [1, 1, 1, 1],
            "total_share": [100, 100, 100, 100],
            "float_share": [80, 80, 80, 80],
            "free_share": [60, 60, 60, 60],
            "total_mv": [1000, 1100, 2000, 2100],
            "circ_mv": [800, 850, 1600, 1650],
        }
    )
    index_weight = pd.DataFrame(
        {
            "index_code": ["399300.SZ", "399300.SZ"],
            "con_code": ["000001.SZ", "000002.SZ"],
            "trade_date": ["20240131", "20240131"],
            "weight": [40.0, 60.0],
        }
    )

    cleaned, summary = build_hs300_research_panel(daily, adj_factor, daily_basic, index_weight)

    assert summary.duplicate_daily_keys_removed == 1
    assert cleaned.duplicated(subset=["ts_code", "trade_date"]).sum() == 0
    assert {"adj_close", "return_1d", "forward_return_1d", "benchmark_weight"} <= set(cleaned.columns)
    assert cleaned[["adj_close", "return_1d", "forward_return_1d", "benchmark_weight"]].isna().sum().sum() == 0
