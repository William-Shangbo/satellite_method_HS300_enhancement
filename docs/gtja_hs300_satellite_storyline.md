# GTJA HS300 Satellite Replication Storyline

## 0. 这次复现的边界

这次目标不是声称完全复刻国泰海通原文的 12.6% 年化超额，而是在当前 Tushare 可用字段下，把文章的“沪深300域内底仓 + 域内卫星 + 域外卫星”框架复现出来，并用全 A universe 重新跑一版可检查、可扩展的结果。

核心约束：

- 回测区间：`20160101-20260101`
- 调仓频率：月度，取每月最后一个可交易日作为调仓日
- 收益频率：月频，持有约 21 个交易日
- 成本：单边 20 bps
- 基准：固定引用 `artifacts/hs300_index_enhancement_rolling_ic_financial_20160101_20260101/portfolio_returns.csv`
- 基准校验：119 期，`2016-01-29` 到 `2025-11-28`，新旧月度基准收益最大差异 `0.0`，基准年化 `10.0833%`

本次 All-A 数据已补齐：

| 数据 | 路径 | 完整性 |
|---|---|---:|
| 日行情 `daily_bar` | `data/raw/tushare_all_a_20160101_20260101/daily_bar` | 2430 个交易日 |
| 复权因子 `adjustment_factor` | `data/raw/tushare_all_a_20160101_20260101/adjustment_factor` | 2430 个交易日 |
| 每日估值 `daily_basic` | `data/raw/tushare_all_a_20160101_20260101/daily_basic` | 2430 个交易日 |
| 财务指标 `financial_indicator` | `data/raw/tushare_all_a_20160101_20260101/financial_indicator` | 5855 个证券文件 |

## 1. Story：为什么要拆成域内和域外

原文的出发点是：经典沪深300指增通常要求大部分权重留在指数成分股内，用来控制相对偏离；而指数外股票提供更强收益弹性，但风险也更高。因此组合应该被拆成两块看：

- 域内：沪深300成分股内，像“底仓 + 稳健卫星”。文章认为基本面和动量趋势因子在域内更有意义，小市值效应不一定有效。
- 域外：沪深300成分股外，像“弹性卫星”。文章测试了小市值高增长和 GARP 等主动量化组合。

所以这次复现按照三个层次来做：

1. 基础指增 proxy：指数内 benchmark tilt + 指数外 base top50。
2. 域内模型：在沪深300成分股内，用基本面 + 动量构造 top30 卫星。
3. 域外模型：在全 A 非沪深300内，用小市值高增长 proxy / GARP proxy 构造卫星。

## 2. 因子和预处理

当前可用因子不是原文完整因子集，而是用 Tushare 日频、估值、财务指标构造的 proxy。

### 2.1 基础因子

| 类型 | 当前字段 |
|---|---|
| 估值 | `value_ep`, `value_bp`, `value_sp`, `dividend_yield` |
| 市值 | `size_neg_log_mv` |
| 量价 | `momentum_60d`, `reversal_20d`, `volatility_20d`, `turnover_20d` |
| 质量 | `quality_roe`, `quality_roa`, `quality_net_margin`, `quality_assets_turn` |

关键代码：

```python
data["value_ep"] = _inverse_positive(data.get("pe_ttm"))
data["value_bp"] = _inverse_positive(data.get("pb"))
data["value_sp"] = _inverse_positive(data.get("ps_ttm"))
data["dividend_yield"] = pd.to_numeric(data.get("dv_ttm"), errors="coerce")
data["size_neg_log_mv"] = -np.log(pd.to_numeric(data.get("total_mv"), errors="coerce").where(lambda s: s > 0))

data["momentum_60d"] = grouped["adj_close"].pct_change(60)
data["reversal_20d"] = -grouped["adj_close"].pct_change(20)
data["volatility_20d"] = -grouped["return_1d"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
data["turnover_20d"] = -grouped["turnover_rate"].rolling(20, min_periods=10).mean().reset_index(level=0, drop=True)
```

### 2.2 缺失值、标准化、未来函数

当前处理：

- 财务指标只使用 `ann_date <= rebalance_date` 的最近一期，避免财报未来函数。
- 因子缺失值用当期截面中位数填补，再兜底为 0。
- 日频因子在每个交易日做 winsorize + zscore。
- 月度 score 再按 `rebalance_date` 做 zscore，本质上包含 cross-sectional demean。
- 没有做 rank transform；RankIC 只用于计算滚动 ICIR 权重。

关键代码：

```python
visible = fin[fin["ann_date"] <= pd.to_datetime(date)]
latest = visible.sort_values("ann_date").drop_duplicates("ts_code", keep="last")

date_median = result.groupby("rebalance_date")[col].transform("median")
result[col] = pd.to_numeric(result[col], errors="coerce").fillna(date_median).fillna(0.0)

mean = result.groupby("rebalance_date")[score_col].transform("mean")
std = result.groupby("rebalance_date")[score_col].transform("std").replace(0, np.nan)
result[score_col] = ((result[score_col] - mean) / std).fillna(0.0)
```

当前还没有完成的严格处理：

- 行业中性化：已 attach `industry`，但当前组合 proxy 没有做行业内去均值/回归中性化。
- 因子相关性控制：当前没有对高相关因子做聚类、正交化或相关阈值剔除。
- 完整风险模型：当前是线性 proxy，不是 Barra-like 风格/行业/个股约束优化器。

这些应该作为下一轮改进优先项。

## 3. 组合构建

### 3.1 基础组合

基础模型分成指数内和指数外：

- 指数内：以沪深300权重为底，按 `base_score` 做 30% active tilt，个股主动偏离上限 1%，个股权重上限 5%。
- 指数外：全 A 非沪深300股票中，按 `base_score` 选 top50 等权。
- 基础指增：80% 指数内 + 20% 指数外。

```python
inside_core = benchmark_tilt_weights(
    inside, "base_score", active_share=0.30, max_active=0.01, max_weight=0.05
)
outside_base = top_equal_weights(outside, "base_score", n=50)
base_80in_20out = combine_weight_blocks({
    "inside": (inside_core, 0.80),
    "outside": (outside_base, 0.20),
})
```

### 3.2 域内 top30 卫星

复现文章的输出端调整思路：

1. 在沪深300成分股内，先按全 A base score 选 top100。
2. 再用域内模型 `domain_in_score` 选 top30。
3. 按市值加权，单票 5% cap。

```python
def domain_in_top30_weights(inside: pd.DataFrame) -> pd.Series:
    pool = inside.sort_values("base_score", ascending=False).head(100)
    selected = pool.sort_values("domain_in_score", ascending=False).head(30)
    return top_cap_weighted(selected, "total_mv", cap=0.05)
```

### 3.3 域外小市值高增长 proxy

原文小市值高增长包含 SUE、EAV、预期净利润调整、累计研发投入、PB_INT、小市值、尾盘成交占比、开盘后大单净买入金额占比等。

当前没有预期、研发、分钟/order-flow 数据，所以 proxy 用：

- `size_neg_log_mv`
- `quality_roe`
- `quality_roa`
- `quality_net_margin`
- `momentum_60d`

并在全 A 非沪深300中选 top50 等权。

### 3.4 GARP proxy

当前 GARP proxy：

- value：`value_bp`, `dividend_yield`
- growth：`quality_roe`, `quality_roa`, `quality_net_margin`, `quality_assets_turn`, `momentum_60d`
- 剔除价值分数最低 20% 和换手分数最低 20%，再选 top20/top50 等权。

```python
def garp_weights(outside: pd.DataFrame, *, n: int) -> pd.Series:
    data = outside.copy()
    value_cut = data["garp_value_score"].quantile(0.20)
    turnover_cut = data["turnover_20d"].quantile(0.20)
    filtered = data[(data["garp_value_score"] > value_cut) & (data["turnover_20d"] > turnover_cut)]
    return top_equal_weights(filtered, "garp_score", n=n)
```

## 4. 复现结果

输出目录：

`artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101`

核心文件：

- `article_strategy_returns.csv`
- `article_strategy_weights.csv`
- `article_performance_summary.csv`
- `article_annual_active_return.csv`
- `article_satellite_grid.csv`
- `replication_factor_panel_monthly.parquet`
- `replication_scored_panel_monthly.parquet`

### 4.1 图

![NAV](../artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/nav_curves.png)

![Active NAV](../artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/active_nav_curves.png)

![Annual Active Returns](../artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/annual_active_returns.png)

![Satellite Grid IR](../artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/satellite_grid_ir.png)

### 4.2 绩效汇总

| strategy | annual_return | benchmark_annual_return | annual_active_return | tracking_error | IR | max_active_drawdown | win_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `small_growth_outside_proxy` | 22.56% | 10.08% | 14.58% | 29.56% | 0.49 | -45.33% | 52.10% |
| `base_outside_proxy` | 23.42% | 10.08% | 12.36% | 18.98% | 0.65 | -28.06% | 52.10% |
| `domain_in_top30` | 17.52% | 10.08% | 6.37% | 14.07% | 0.45 | -27.61% | 57.14% |
| `base_80in_20out_proxy` | 13.40% | 10.08% | 2.79% | 5.01% | 0.56 | -9.22% | 56.30% |
| `combo_in20_out10_small_proxy` | 15.67% | 10.08% | 4.69% | 6.50% | 0.72 | -9.91% | 59.66% |
| `combo_in30_out10_small_proxy` | 16.13% | 10.08% | 5.05% | 7.14% | 0.71 | -12.36% | 63.87% |
| `combo_in20_out10_garp_proxy` | 14.68% | 10.08% | 3.80% | 6.74% | 0.56 | -10.42% | 54.62% |

读法：

- 单独域外组合收益弹性很强，但 TE 和回撤都明显过大，不能直接当低风险指增。
- 基础 80/20 proxy 年化超额 2.79%，TE 5.01%，IR 0.56。
- 加入域内 20% + 域外小市值高增长 10% 后，年化超额提升到 4.69%，TE 6.50%，IR 0.72。
- 域内 30% + 域外 10% 年化超额 5.05%，但 TE 和相对回撤继续上升。

### 4.3 年度超额

| year | base_80in_20out_proxy | combo_in20_out10_small_proxy | combo_in30_out10_small_proxy |
|---:|---:|---:|---:|
| 2016 | 5.30% | 13.79% | 13.86% |
| 2017 | -0.88% | 0.40% | 0.16% |
| 2018 | 1.35% | 0.00% | -0.49% |
| 2019 | -3.34% | -5.82% | -7.31% |
| 2020 | -1.13% | 2.11% | 3.88% |
| 2021 | 9.71% | 13.19% | 13.78% |
| 2022 | 13.23% | 13.68% | 15.34% |
| 2023 | 10.13% | 14.85% | 15.80% |
| 2024 | -4.68% | -3.65% | -1.81% |
| 2025 | -1.17% | -0.39% | -1.40% |

问题也很清楚：

- 2019 和 2024 是当前 proxy 的主要拖累年份。
- 域外小市值高增长提高收益，但稳定性不够，尤其不是原文所说的低 TE 高 IR 形态。
- 这说明“结构复现”完成了，但 alpha 质量和风险模型还远未到券商报告版本。

### 4.4 卫星配置 grid

按 `base_80in_20out_proxy` 为底，叠加域内 top30 和域外 small-growth proxy。IR 排名前几：

| strategy | 域内卫星 | 域外卫星 | 年化超额 | TE | IR | 最大相对回撤 |
|---|---:|---:|---:|---:|---:|---:|
| `grid_in30_out20_small_proxy` | 30% | 20% | 6.22% | 8.40% | 0.74 | -13.14% |
| `grid_in20_out20_small_proxy` | 20% | 20% | 5.86% | 7.98% | 0.74 | -11.27% |
| `grid_in40_out20_small_proxy` | 40% | 20% | 6.58% | 8.95% | 0.73 | -15.51% |
| `grid_in20_out10_small_proxy` | 20% | 10% | 4.69% | 6.50% | 0.72 | -9.92% |

当前比较稳妥的展示组合是 `combo_in20_out10_small_proxy`：相对基础组合有提升，同时 TE 和相对回撤没有失控。

## 5. 和原文差距

原文提到域内 30%、域外 10% 卫星配置下，2016 年以来年化超额 12.6%，TE 5.2%，IR 2.38。当前 All-A proxy 的 `combo_in30_out10_small_proxy` 是年化超额 5.05%，TE 7.14%，IR 0.71。

差距主要来自：

1. 因子缺失：没有分析师预期净利润调整、EAV、严格定义的 SUE、研发投入、开盘后买入意愿、大单推动涨幅、尾盘成交占比等。
2. 风险模型缺失：当前只有简单权重拼接和 benchmark tilt，没有行业偏离、风格暴露、个股偏离和交易约束优化。
3. 域外组合太粗：small-growth proxy 收益高但 TE 极高，不像原文通过优化/组合替代后把风险压住。
4. 行业和相关性没有严格控制：当前 attach 了行业，但没有做行业中性化；因子间相关性也没做剔除/正交化。
5. 股票池过滤不够：还需要 ST、上市未满 N 日、低流动性、极端停牌/一字板等交易约束。

## 6. 下一步提高超额的方向

优先级从高到低：

1. 补真正的原文因子：`SUE/EAV/预期净利润调整/研发投入/order-flow/分钟成交`。当前结果证明结构能跑，但 alpha 不够。
2. 做行业中性和因子相关性控制：行业内 zscore，相关性超过阈值的因子做聚类择优或残差正交。
3. 上组合优化器：目标最大化 alpha，约束 80% 指数内、个股主动偏离、行业偏离、size/value 暴露、换手和流动性。
4. 域外卫星降风险：小市值高增长不直接等权 top50，可加入行业分散、波动率缩放、流动性 cap、回撤止损。
5. 做 walk-forward 因子筛选：按滚动 ICIR 分域内/域外分别筛因子，避免一个全 A 权重套所有场景。
6. 单独诊断 2019/2024：拆分因子贡献、行业暴露、域内/域外超额，找当前 proxy 失效来源。

## 7. 本轮新增诊断：single_signal_test、分层、暴露、因子扩充

新增脚本：

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/run_factor_diagnostics.py
```

输出目录：

`artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/factor_diagnostics`

### 7.1 本轮诊断口径

这次不重跑下载、不重建基准、不改原策略收益，只读取已有 All-A 月频面板和策略权重。

单因子和 GP 候选因子统一使用以下预处理：

- `NaN`：按调仓日截面中位数填补，再兜底为 0。
- 去极值：每个调仓日做 1%/99% winsorize。
- 标准化：每个调仓日做 zscore。
- demean：每个调仓日做截面去均值。
- 行业中性：在同一调仓日、同一行业内再去均值，然后重新 zscore。
- rank transform：不作为输入变换，只在 RankIC 和分层测试里用于排序。
- correlation control：输出 Spearman 相关矩阵，并给出高相关因子的 drop candidate。

### 7.2 single_signal_test 和分层测试

核心输出：

- `single_signal_ic_summary.csv`
- `single_signal_ic_timeseries.csv`
- `single_signal_layer_returns.csv`
- `single_signal_layer_summary.csv`
- `single_signal_top_layer_nav.png`

域内，也就是沪深300成分股内，RankIC 排名前几：

| factor | RankIC | RankICIR | 正 RankIC 月份占比 |
|---|---:|---:|---:|
| `dividend_yield_ind_neutral` | 5.05% | 0.34 | 58.82% |
| `garp_score_ind_neutral` | 4.97% | 0.32 | 63.03% |
| `base_score_ind_neutral` | 4.70% | 0.32 | 58.82% |
| `domain_in_score_ind_neutral` | 4.80% | 0.30 | 64.71% |
| `value_ep_ind_neutral` | 4.54% | 0.30 | 63.87% |

域内分层里，表现更像“可用”的是估值和 GARP：

| factor | top layer 年化 | 多空年化 | 多空 IR |
|---|---:|---:|---:|
| `value_ep_ind_neutral` | 13.65% | 9.83% | 0.69 |
| `domain_in_score_ind_neutral` | 13.25% | 8.03% | 0.52 |
| `garp_score_ind_neutral` | 12.13% | 7.17% | 0.49 |
| `base_score_ind_neutral` | 11.20% | 5.43% | 0.38 |
| `dividend_yield_ind_neutral` | 12.12% | 4.82% | 0.35 |

域外，也就是非沪深300，因子强度明显更高：

| factor | RankIC | RankICIR |
|---|---:|---:|
| `base_score_ind_neutral` | 10.02% | 0.93 |
| `turnover_20d_ind_neutral` | 8.45% | 0.83 |
| `volatility_20d_ind_neutral` | 8.56% | 0.75 |
| `garp_value_score_ind_neutral` | 6.22% | 0.74 |
| `value_bp_ind_neutral` | 6.21% | 0.69 |

这验证了文章里的拆分思路：域内更适合低风险地做估值/质量/分红/GARP，域外 alpha 弹性更强，但风险也更难压。

### 7.3 行业偏离、风格暴露、市值/估值暴露

核心输出：

- `strategy_industry_active_weights.csv`
- `strategy_industry_deviation_summary.csv`
- `strategy_style_active_exposure.csv`
- `strategy_style_exposure_summary.csv`
- `strategy_size_value_active_exposure.csv`
- `strategy_size_value_exposure_summary.csv`
- `industry_deviation_summary.png`
- `style_exposure_heatmap.png`

行业偏离最大值：

| strategy | max abs industry deviation |
|---|---:|
| `domain_in_top30` | 65.49% |
| `combo_extreme_in_out_small_proxy` | 50.02% |
| `garp20_outside_proxy` | 44.50% |
| `base_outside_proxy` | 29.47% |
| `small_growth_outside_proxy` | 25.47% |
| `combo_in20_out10_small_proxy` | 14.83% |
| `base_80in_20out_proxy` | 6.97% |

风格暴露结论：

- `base_80in_20out_proxy` 相对基准有明显小市值、价值、分红暴露。
- `base_80in_20out_proxy` 的 `total_mv` / `circ_mv` 主动暴露约为 `-0.93z`，即小盘暴露已经比较明显。
- 域外单腿组合的市值暴露极端，`base_outside_proxy` 的 `circ_mv` 主动暴露约 `-4.75z`；这解释了为什么域外收益弹性强，但 TE 很难低。
- 组合层面需要把行业偏离、size/value 暴露放进优化器约束，而不是只做 topN。

### 7.4 相关性控制

高相关因子对已经输出到：

`factor_correlation_prune_report.csv`

当前阈值 0.75 下，主要冲突：

| factor_a | factor_b | corr | drop candidate |
|---|---|---:|---|
| `quality_roe_ind_neutral` | `quality_roa_ind_neutral` | 0.91 | `quality_roe_ind_neutral` |
| `value_bp_ind_neutral` | `garp_value_score_ind_neutral` | 0.84 | `value_bp_ind_neutral` |
| `garp_value_score_ind_neutral` | `garp_score_ind_neutral` | 0.81 | `garp_score_ind_neutral` |
| `domain_in_score_ind_neutral` | `garp_score_ind_neutral` | 0.76 | `domain_in_score_ind_neutral` |
| `quality_roa_ind_neutral` | `quality_net_margin_ind_neutral` | 0.75 | `quality_roa_ind_neutral` |

下一轮组合打分建议不要把高度重叠的质量因子和 GARP 复合分都无脑放进去，应在域内/域外分别做择优或正交。

### 7.5 最近一年因子研究映射：第一轮扩充

本轮外部研究的可执行结论，不是直接照搬文章因子，而是把近一年里反复出现的方向映射到当前已下载 raw data：

| 外部研究方向 | 当前可落地字段 | 本轮处理 |
|---|---|---|
| value / dividend / low-vol 防守因子 | `pe_ttm`, `pb`, `ps_ttm`, `dv_ttm`, `return_1d` rolling vol | 已测 `value_ep`, `value_bp`, `value_sp`, `dividend_yield`, `volatility_20d` |
| factor diversification / regime rotation | 所有已构造因子的月度 IC、分层收益 | 已做域内/域外分开 single signal test |
| crowding / implementation cost | `turnover_rate`, `amount`, `total_mv`, `circ_mv` | 已测低换手、流动性和市值暴露 |
| machine / symbolic alpha mining | 已构造的估值、质量、量价、市值字段 | 已做 GP-like 表达式搜索 |
| 因果和稳健性筛选 | train/test split、分层、相关性控制 | 已把 GP 候选拆成 2016-2022 train、2023-2025 test |

公开研究对本轮的启发：

- 因子投资常用维度仍集中在 size、value、momentum、quality、low volatility；但实现上要警惕 data mining、拥挤和交易成本。
- 2025 年以来海外 factor tracking 也显示单一因子年度表现差异很大，value 和 low volatility 在部分市场更强，momentum/alpha 不总是领先。
- 量化机构近年更多使用机器学习来做因子加权和模式识别，但生产环境仍要强调可解释性和回撤阶段的诊断。

当前 raw data 暂时不能支持的原文因子：

- 分析师一致预期类：`预期净利润调整`, `EAV`。
- 严格公告预期差类：原文定义下的 `SUE` 需要更完整的公告和预测基准。
- order-flow / intraday：`开盘后买入意愿强度`, `大单推动涨幅`, `尾盘成交占比`, `大单净买入金额占比`。
- 研发投入累计：当前 financial_indicator 不够直接，需要利润表/现金流/研发费用明细。

参考来源：

- 国泰海通金工：用户给定文章《再论沪深300增强：从增强组合成分股内外收益分解说起》。
- Financial Times, 2025-06-04, [AQR 使用 AI/机器学习增强量化交易决策](https://www.ft.com/content/e62c85cb-e3c8-4df3-b115-e3e11eeaa266)。
- Barron's, 2025-05-27, [2025 年低波动股票在波动市中表现突出](https://www.barrons.com/articles/low-volatility-stocks-outperform-dcccabbe)。
- MarketWatch, 2025-06-11, [2025 年 value investing 在国际市场相对占优](https://www.marketwatch.com/story/value-investing-is-finally-excelling-again-in-2025-but-there-is-one-catch-for-americans-62571e4c)。
- CFA Institute Research Foundation, 2025-09-22, Marcos López de Prado and Vincent Zoonekynd, `Causality and Factor Investing: A Primer`, DOI `10.56227/25.1.30`。

### 7.6 GP-like / kernel search：第二轮扩充

核心输出：

- `gp_candidate_factor_summary.csv`
- `gp_top_factor_layer_returns.csv`

这不是完整遗传规划引擎，而是一个可解释的 symbolic kernel search：对当前字段做 `id/neg/add/sub/mul/ratio`，然后按 train/test 的 RankICIR 和分层多空收益筛选。

Top 候选：

| expression | train RankICIR | test RankICIR | test 多空年化 | test 多空 IR |
|---|---:|---:|---:|---:|
| `ratio(value_bp,quality_roe)` | 0.58 | 0.97 | 14.08% | 1.80 |
| `id(value_bp)` | 0.55 | 0.96 | 15.57% | 1.95 |
| `ratio(value_bp,momentum_60d)` | 0.55 | 0.93 | 11.77% | 1.74 |
| `add(value_bp,turnover_20d)` | 0.73 | 0.84 | 13.74% | 1.10 |
| `ratio(value_bp,turnover_20d)` | 0.42 | 0.78 | 9.96% | 1.33 |
| `sub(size_neg_log_mv,momentum_60d)` | 0.37 | 0.63 | 24.87% | 1.64 |

可落地建议：

- 第一批纳入候选池：`value_bp`, `value_bp / (abs(quality_roe)+0.5)`, `value_bp / (abs(momentum_60d)+0.5)`, `value_bp + turnover_20d`。
- `sub(size_neg_log_mv,momentum_60d)` 收益弹性高，但明显带小盘暴露，必须放进 size/行业约束后再用于组合。
- `turnover_20d` 和 `volatility_20d` 在域外 IC 强，但单腿 TE 会很高，适合做风险调节或二级筛选，不适合直接等权 topN。

### 7.7 high-TE small_growth 图形处理

`small_growth_outside_proxy` 单腿 TE 为 `29.56%`，远高于指增可展示口径，所以新图已经去掉 standalone small-growth，仅保留基础组合、域内卫星和组合策略：

![TE-controlled NAV](../artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/nav_curves_te_controlled.png)

![TE-controlled Active NAV](../artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/active_nav_curves_te_controlled.png)

诊断图：

![Single Signal Layer NAV](../artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/factor_diagnostics/single_signal_top_layer_nav.png)

![Industry Deviation](../artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/factor_diagnostics/industry_deviation_summary.png)

![Style Exposure](../artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/factor_diagnostics/style_exposure_heatmap.png)

## 8. 可复现实验命令

数据补齐：

```bash
TUSHARE_TOKEN='***' python scripts/fetch_all_a_history.py \
  --start-date 20160101 \
  --end-date 20260101 \
  --out-dir data/raw/tushare_all_a_20160101_20260101 \
  --stages daily \
  --sleep 0.08

TUSHARE_TOKEN='***' python scripts/fetch_all_a_history.py \
  --start-date 20160101 \
  --end-date 20260101 \
  --out-dir data/raw/tushare_all_a_20160101_20260101 \
  --stages financial \
  --sleep 0.08
```

复现：

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/run_gtja_hs300_satellite_replication.py \
  --data-dir data/raw/tushare_all_a_20160101_20260101 \
  --start-date 20160101 \
  --end-date 20260101 \
  --out-dir artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101 \
  --benchmark-returns artifacts/hs300_index_enhancement_rolling_ic_financial_20160101_20260101/portfolio_returns.csv
```

基准不变校验：

```python
new = pd.read_csv("artifacts/gtja_hs300_satellite_replication_all_a_20160101_20260101/article_strategy_returns.csv", parse_dates=["rebalance_date"])
old = pd.read_csv("artifacts/hs300_index_enhancement_rolling_ic_financial_20160101_20260101/portfolio_returns.csv", parse_dates=["rebalance_date"])
m = new.merge(old[["rebalance_date", "benchmark_return"]], on="rebalance_date", suffixes=("_new", "_old"))
assert (m.benchmark_return_new - m.benchmark_return_old).abs().max() == 0
```
