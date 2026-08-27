"""
IC 분석에서 나온 신호가 진짜인지 파고드는 후속 진단.

run_multi_source_ic.py 결과에서 가격 계열 팩터가 하나같이 음의 IC를 보였다.
그런데 이것들(모멘텀/이격도/초과수익)은 사실상 '최근에 얼마나 올랐나'를 조금씩
다르게 잰 것이라, 20개 독립 검정으로 보고 Bonferroni를 적용하는 건 지나치게
보수적일 수 있다. 여기서 확인하는 것:

1) 팩터들이 실제로 얼마나 겹치는가 (상관관계 -> 유효 검정 수)
2) 변동성 팩터가 규모(size)의 대리변수에 불과한가 (소형주가 더 변동성이 크므로)
3) 가장 일관된 신호가 시간 구간을 나눠도 유지되는가 (안정성)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import phase1
from src.data_loader.krx_openapi import build_close_panel, build_panel
from src.data_loader.krx_panel import trading_dates
from src.data_loader.universe import market_cap_universe_mask
from src.features.multi_source import (
    build_excess_return_factors,
    build_price_factors,
    build_size_factor,
    neutralize_by_size,
)
from src.research.ic_analysis import non_overlapping_ic, summarize_ic

TOP_K = 200
MIN_OBS = 30
HORIZON = 20

WATCH = [
    "volatility_60",
    "momentum_20",
    "momentum_60",
    "disparity_20",
    "disparity_60",
    "excess_return_20",
    "excess_return_60",
]


def average_cross_sectional_correlation(
    panels: dict[str, pd.DataFrame], universe: pd.DataFrame
) -> pd.DataFrame:
    """
    날짜마다 팩터간 순위상관을 구해 평균낸다.
    '이 팩터들이 사실상 같은 것을 재고 있는가'를 보는 지표다.
    """
    names = list(panels)
    total = pd.DataFrame(0.0, index=names, columns=names)
    count = 0

    sample_dates = universe.index[::HORIZON]
    for date in sample_dates:
        members = universe.columns[universe.loc[date]]
        frame = pd.DataFrame(
            {name: panel.loc[date, panel.columns.intersection(members)] for name, panel in panels.items()}
        )
        frame = frame.dropna()
        if len(frame) < MIN_OBS:
            continue
        total += frame.corr(method="spearman")
        count += 1

    return total / count


def main() -> None:
    dates = trading_dates(phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END)
    close = build_close_panel(dates)
    volume = build_panel("volume", dates)
    market_cap = build_panel("market_cap", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

    weights = market_cap.shift(1)
    weights = weights.div(weights.sum(axis=1), axis=0)
    market_index = (1 + (close.pct_change() * weights).sum(axis=1, min_count=1).fillna(0)).cumprod()

    panels: dict[str, pd.DataFrame] = {}
    panels.update(build_price_factors(close, volume))
    panels.update(build_excess_return_factors(close, market_index))
    panels.update(build_size_factor(market_cap))

    watched = {name: panels[name] for name in WATCH}

    print("=== 1. 팩터간 평균 순위상관 (같은 걸 재고 있는가) ===")
    corr = average_cross_sectional_correlation(watched, universe)
    with pd.option_context("display.float_format", "{:.2f}".format):
        print(corr.to_string())

    print("\n=== 2. 변동성이 규모의 대리변수인가 ===")
    log_cap = panels["log_market_cap"]
    fwd = close.pct_change(HORIZON).shift(-HORIZON).where(universe)

    vol = panels["volatility_60"]
    vol_corr = average_cross_sectional_correlation(
        {"volatility_60": vol, "log_market_cap": log_cap}, universe
    ).loc["volatility_60", "log_market_cap"]
    print(f"  변동성 vs 로그시총 평균 순위상관: {vol_corr:.2f}")

    rows = []
    for label, panel in [
        ("변동성 (원본)", vol),
        ("변동성 (규모중립화)", neutralize_by_size(vol.where(universe), log_cap)),
    ]:
        ic = non_overlapping_ic(panel.reindex_like(fwd).where(universe), fwd, HORIZON, min_obs=MIN_OBS)
        summary = summarize_ic(ic)
        summary["팩터"] = label
        rows.append(summary)

    table = pd.DataFrame(rows).set_index("팩터")[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]]
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())

    print("\n=== 3. 기간을 반으로 갈라도 유지되는가 ===")
    midpoint = len(dates) // 2
    first_half, second_half = dates[:midpoint], dates[midpoint:]

    rows = []
    for name in WATCH:
        panel = panels[name].reindex_like(fwd).where(universe)
        for label, window in [("전반부", first_half), ("후반부", second_half)]:
            ic = non_overlapping_ic(
                panel.loc[window], fwd.loc[window], HORIZON, min_obs=MIN_OBS
            )
            summary = summarize_ic(ic)
            rows.append({"factor": name, "구간": label, "평균 IC": summary["평균 IC"], "t-stat": summary["t-stat"]})

    table = pd.DataFrame(rows).pivot(index="factor", columns="구간", values=["평균 IC", "t-stat"])
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


if __name__ == "__main__":
    main()
