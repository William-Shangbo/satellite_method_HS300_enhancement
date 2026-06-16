import pandas as pd

from index_platform.strategy.factors import build_basic_factor_panel, composite_alpha, factor_summary
from index_platform.strategy.optimization import OptimizationConstraints, heuristic_active_weight_optimizer
from index_platform.strategy.preprocessing import FactorPreprocessConfig, preprocess_factor_panel
from index_platform.strategy.risk import performance_summary
from index_platform.strategy.universe import build_hs300_universe, month_end_rebalance_dates


def test_month_end_rebalance_dates_uses_last_open_day():
    calendar = pd.DataFrame(
        {
            "cal_date": ["20240129", "20240130", "20240131", "20240228", "20240229"],
            "is_open": [1, 1, 0, 1, 1],
        }
    )
    dates = month_end_rebalance_dates(calendar)
    assert [d.strftime("%Y%m%d") for d in dates] == ["20240130", "20240229"]


def test_build_hs300_universe_uses_latest_visible_weight_snapshot():
    weights = pd.DataFrame(
        {
            "index_code": ["399300.SZ", "399300.SZ", "399300.SZ", "399300.SZ"],
            "con_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000003.SZ"],
            "trade_date": ["20240131", "20240131", "20240229", "20240229"],
            "weight": [40, 60, 50, 50],
        }
    )
    snapshot = build_hs300_universe(weights, "20240215")
    assert snapshot.as_of_date.strftime("%Y%m%d") == "20240131"
    assert set(snapshot.stocks["ts_code"]) == {"000001.SZ", "000002.SZ"}
    assert round(snapshot.stocks["benchmark_weight"].sum(), 10) == 1


def test_factor_optimizer_and_risk_pipeline_on_synthetic_data():
    rows = []
    for stock_idx, ts_code in enumerate(["000001.SZ", "000002.SZ", "000003.SZ"]):
        for day in range(90):
            rows.append(
                {
                    "ts_code": ts_code,
                    "trade_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "close": 10 + stock_idx + day * (0.01 + stock_idx * 0.002),
                    "adj_factor": 1.0,
                    "turnover_rate": 1 + stock_idx * 0.1,
                    "pe_ttm": 10 + stock_idx,
                    "pb": 1 + stock_idx * 0.2,
                    "ps_ttm": 2 + stock_idx * 0.1,
                    "dv_ttm": 1 + stock_idx * 0.1,
                    "total_mv": 1000 + stock_idx * 100,
                    "benchmark_weight": [0.4, 0.35, 0.25][stock_idx],
                }
            )
    panel = pd.DataFrame(rows)
    factors = composite_alpha(build_basic_factor_panel(panel))
    summary = factor_summary(factors)
    assert not summary.empty

    latest = factors[factors["trade_date"].eq(factors["trade_date"].max())]
    result = heuristic_active_weight_optimizer(
        latest,
        constraints=OptimizationConstraints(max_active_weight=0.05, max_weight=0.60),
    )
    assert round(result.weights["target_weight"].sum(), 10) == 1
    assert result.diagnostics["max_weight"] <= 0.60

    returns = pd.DataFrame(
        {
            "portfolio_return": [0.01, -0.002, 0.004, 0.003],
            "benchmark_return": [0.008, -0.001, 0.002, 0.001],
        }
    )
    perf = performance_summary(returns, periods_per_year=12)
    assert "information_ratio" in perf.index


def test_preprocess_neutralizes_industry_fills_missing_and_drops_correlated_factor():
    rows = []
    for date in pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30", "2024-05-31", "2024-06-28"]):
        for idx in range(12):
            industry = "Bank" if idx < 6 else "Tech"
            base = idx if industry == "Tech" else idx + 10
            rows.append(
                {
                    "ts_code": f"{idx:06d}.SZ",
                    "rebalance_date": date,
                    "trade_date": date,
                    "industry": industry,
                    "benchmark_weight": 1 / 12,
                    "period_return": idx / 100,
                    "value_ep": base,
                    "value_bp": base * 2,
                    "momentum_60d": None if idx == 0 else idx,
                }
            )
    panel = pd.DataFrame(rows)
    result = preprocess_factor_panel(
        panel,
        config=FactorPreprocessConfig(
            min_factor_coverage=0.80,
            min_stock_coverage=0.50,
            correlation_threshold=0.70,
        ),
    )

    assert "value_ep" in result.selected_factors
    assert "value_bp" not in result.selected_factors
    assert not result.panel[result.selected_factors].isna().any().any()
    industry_means = result.panel.groupby(["rebalance_date", "industry"])["value_ep"].mean().abs()
    assert industry_means.max() < 1e-12


def test_preprocess_preserves_rebalance_calendar_when_early_factors_are_missing():
    rows = []
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30", "2024-05-31", "2024-06-28"])
    for date_idx, date in enumerate(dates):
        for idx in range(12):
            rows.append(
                {
                    "ts_code": f"{idx:06d}.SZ",
                    "rebalance_date": date,
                    "trade_date": date,
                    "industry": "Bank" if idx < 6 else "Tech",
                    "benchmark_weight": 1 / 12,
                    "period_return": idx / 100,
                    "value_ep": idx,
                    "value_bp": None if date_idx < 2 else idx * 2,
                }
            )
    panel = pd.DataFrame(rows)
    result = preprocess_factor_panel(panel)

    assert set(result.panel["rebalance_date"]) == set(dates)
    assert result.panel["rebalance_date"].nunique() == len(dates)
