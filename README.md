# Satellite Method For HS300 Index Enhancement

This repository contains a reproducible research framework for an HS300 index enhancement strategy based on the "core + satellite" idea:

- HS300 in-domain core: benchmark-aware constituent portfolio.
- HS300 in-domain satellite: constituent Top30 portfolio selected by fundamental and momentum signals.
- Outside-HS300 satellite: non-constituent active sleeves such as small-growth and GARP proxies.
- Diagnostics: single-signal IC, factor layering, industry deviation, style exposure, size/value exposure, factor correlation control, and symbolic factor expansion.

The repository intentionally excludes raw data, generated artifacts, and figures. They can be regenerated locally with a Tushare token.

## Repository Layout

```text
.
├── configs/                         # Dataset config examples
├── docs/
│   ├── gtja_hs300_satellite_storyline.md
│   └── hs300_index_enhancement_design.md
├── scripts/
│   ├── fetch_all_a_history.py        # Download All-A Tushare raw data
│   ├── fetch_hs300_history.py        # Download HS300-only raw data
│   ├── run_gtja_hs300_satellite_replication.py
│   ├── run_factor_diagnostics.py
│   └── run_hs300_index_enhancement.py
├── src/index_platform/
│   ├── ingest/                       # Data contracts and Tushare client
│   ├── panel/                        # Raw-to-panel cleaning helpers
│   ├── reporting/                    # HTML report helpers
│   └── strategy/                     # Factors, universe, risk, optimization
├── tests/
├── pyproject.toml
└── docker-compose.yml
```

## Method Summary

The strategy follows the decomposition described in the GTJA/Guotai Haitong HS300 enhancement article:

1. Split an index enhancement portfolio into HS300 constituents and outside-HS300 stocks.
2. Treat the constituent sleeve as the lower-risk base portfolio.
3. Treat the outside-HS300 sleeve as a higher-elasticity satellite.
4. Build a separate in-domain factor model for HS300 constituents, where value, dividend, quality, and momentum-style signals are tested separately from the outside universe.
5. Evaluate whether satellite allocations improve annual active return, tracking error, information ratio, and active drawdown.

This is a proxy replication using fields available from the local Tushare download. It is not a strict replication of the article's proprietary analyst expectation, order-flow, intraday, or full risk-model inputs.

## Data

Raw data is not included in the repository.

Expected local data folders after download:

```text
data/raw/tushare_all_a_20160101_20260101/
├── daily_bar/
├── adjustment_factor/
├── daily_basic/
├── financial_indicator/
├── benchmark_weight.csv
├── stock_basic.csv
└── trade_calendar.csv
```

Set a Tushare token before running download scripts:

```bash
export TUSHARE_TOKEN="your_tushare_token"
```

Download All-A data:

```bash
python scripts/fetch_all_a_history.py \
  --start-date 20160101 \
  --end-date 20260101 \
  --out-dir data/raw/tushare_all_a_20160101_20260101 \
  --stages calendar,weights,stock_basic,daily,financial \
  --sleep 0.08
```

If API quota is limited, run stages separately:

```bash
python scripts/fetch_all_a_history.py --start-date 20160101 --end-date 20260101 \
  --out-dir data/raw/tushare_all_a_20160101_20260101 --stages calendar,weights,stock_basic

python scripts/fetch_all_a_history.py --start-date 20160101 --end-date 20260101 \
  --out-dir data/raw/tushare_all_a_20160101_20260101 --stages daily --sleep 0.08

python scripts/fetch_all_a_history.py --start-date 20160101 --end-date 20260101 \
  --out-dir data/raw/tushare_all_a_20160101_20260101 --stages financial --sleep 0.08
```

## Replication

Run the satellite-method replication:

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/run_gtja_hs300_satellite_replication.py \
  --data-dir data/raw/tushare_all_a_20160101_20260101 \
  --start-date 20160101 \
  --end-date 20260101 \
  --out-dir artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101 \
  --benchmark-returns artifacts/hs300_index_enhancement_rolling_ic_financial_20160101_20260101/portfolio_returns.csv
```

Main generated files:

```text
artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/
├── article_strategy_returns.csv
├── article_strategy_weights.csv
├── article_performance_summary.csv
├── article_annual_active_return.csv
├── article_satellite_grid.csv
├── replication_factor_panel_monthly.parquet
├── replication_scored_panel_monthly.parquet
└── replication_notes.md
```

## Diagnostics

Run factor diagnostics:

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/run_factor_diagnostics.py
```

Diagnostics include:

- Single-signal IC and RankIC.
- Single-factor layer returns.
- Industry active-weight deviation.
- Style exposure and size/value exposure.
- Spearman correlation matrix and drop-candidate report.
- GP-like symbolic expression factor expansion over current raw-data-derived fields.

Main outputs:

```text
artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/factor_diagnostics/
├── single_signal_ic_summary.csv
├── single_signal_layer_summary.csv
├── strategy_industry_deviation_summary.csv
├── strategy_style_exposure_summary.csv
├── strategy_size_value_exposure_summary.csv
├── factor_correlation_matrix.csv
├── factor_correlation_prune_report.csv
├── gp_candidate_factor_summary.csv
└── factor_diagnostics_report.md
```

To redraw only the TE-controlled NAV figures without recomputing diagnostics:

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/run_factor_diagnostics.py --redraw-only
```

## Factor Preprocessing

The diagnostic pipeline applies:

- Cross-sectional median fill by rebalance date.
- 1%/99% winsorization.
- Cross-sectional z-score.
- Cross-sectional demean.
- Industry demean and re-standardization.
- Rank only for RankIC and layer tests, not as the input transform.

## Important Caveats

- The implementation uses Tushare-available proxy factors.
- The current repository does not include raw data or generated artifacts.
- Analyst expectation factors, intraday/order-flow factors, and the article's full risk model are not included.
- Results should be interpreted as method validation and research infrastructure, not investment advice.

## Development

Install dependencies:

```bash
python -m pip install -e .
```

Run tests:

```bash
python -m pytest -q
```

