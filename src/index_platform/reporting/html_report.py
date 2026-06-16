from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def render_strategy_report(
    *,
    title: str,
    performance: pd.Series,
    factor_summary: pd.DataFrame,
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    active = returns.get("active_return", returns["portfolio_return"] - returns["benchmark_return"])
    curves = pd.DataFrame(
        {
            "portfolio": (1 + returns["portfolio_return"]).cumprod(),
            "benchmark": (1 + returns["benchmark_return"]).cumprod(),
            "active": (1 + active).cumprod(),
        }
    ).reset_index(drop=True)

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .story {{ max-width: 920px; line-height: 1.6; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d8dee4; padding: 12px; border-radius: 6px; }}
    .metric strong {{ display: block; font-size: 20px; margin-top: 6px; }}
    svg {{ width: 100%; height: 280px; border: 1px solid #d8dee4; background: #fff; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="story">这份报告按研究故事展开：先证明单因子是否有横截面解释力，再展示优化是否把 alpha 转换成风险受控的主动持仓，最后用组合、基准和主动净值说明指数增强结果。</p>
  <h2>核心指标</h2>
  {metrics_grid(performance)}
  <h2>净值与主动净值</h2>
  {line_svg(curves)}
  <h2>单因子构造成果</h2>
  {factor_summary.head(20).to_html(index=False, float_format=lambda x: f"{x:.4f}")}
  <h2>组合优化成果</h2>
  {weights.head(30).to_html(index=False, float_format=lambda x: f"{x:.4f}")}
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    return output


def metrics_grid(performance: pd.Series) -> str:
    labels = {
        "annual_return": "组合年化",
        "benchmark_annual_return": "基准年化",
        "annual_excess_return": "年化超额",
        "tracking_error": "跟踪误差",
        "information_ratio": "信息比率",
        "max_drawdown": "最大回撤",
        "max_active_drawdown": "主动最大回撤",
        "period_win_rate": "周期胜率",
    }
    cards = []
    for key, label in labels.items():
        value = performance.get(key)
        if pd.isna(value):
            text = "NA"
        elif "ratio" in key:
            text = f"{value:.2f}"
        else:
            text = f"{value:.2%}"
        cards.append(f"<div class='metric'>{escape(label)}<strong>{escape(text)}</strong></div>")
    return "<div class='metric-grid'>" + "".join(cards) + "</div>"


def line_svg(curves: pd.DataFrame) -> str:
    width, height, pad = 920, 260, 28
    y_min = curves.min().min()
    y_max = curves.max().max()
    if y_max == y_min:
        y_max = y_min + 1
    colors = {"portfolio": "#2563eb", "benchmark": "#6b7280", "active": "#dc2626"}
    paths = []
    for col in curves.columns:
        points = []
        series = curves[col].reset_index(drop=True)
        for i, value in series.items():
            x = pad + i * (width - 2 * pad) / max(len(series) - 1, 1)
            y = height - pad - (value - y_min) * (height - 2 * pad) / (y_max - y_min)
            points.append(f"{x:.2f},{y:.2f}")
        paths.append(
            f"<polyline fill='none' stroke='{colors[col]}' stroke-width='2' points='{' '.join(points)}' />"
        )
    legend = "".join(
        f"<text x='{pad + i * 150}' y='18' fill='{color}' font-size='12'>{name}</text>"
        for i, (name, color) in enumerate(colors.items())
    )
    return f"<svg viewBox='0 0 {width} {height}' role='img'>{legend}{''.join(paths)}</svg>"
