"""
여러 정보원의 팩터들이 실제로 미래 수익률을 예측하는지 한 번에 비교한다.

이 프로젝트에서 반복해서 확인한 교훈 두 가지를 그대로 반영한다:
1) 전략을 만들기 전에 팩터의 예측력부터 데이터로 확인한다 (임의로 정한 가중치로
   전략을 만들고 성과를 보는 순서는 무엇이 문제인지 알려주지 못한다)
2) 일별 IC를 그대로 평균내면 t-stat이 심하게 부풀려진다. 겹치지 않게 샘플링하고,
   여러 팩터를 동시에 검정한다는 점(다중검정)도 감안해서 본다

정보원:
- 가격/거래량 (KRX Open API 일별매매정보)
- 규모        (같은 API의 시가총액)
- 밸류에이션  (DART 재무제표 / 시가총액 -- PER/PBR을 직접 주는 API가 없어서 직접 계산)
- 펀더멘털    (DART 재무제표, 공시일 기준으로 펼침)

유니버스는 각 분기 시작일의 실제 시가총액 상위 200종목(point-in-time)이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scipy import stats

from config import phase1
from src.data_loader.dart_loader import load_fundamentals_bulk
from src.data_loader.krx_openapi import build_panel
from src.data_loader.krx_panel import trading_dates
from src.data_loader.universe import fetch_candidate_pool, market_cap_universe_mask
from src.features.fundamental import (
    FACTOR_COLUMNS,
    VALUATION_LEVEL_COLUMNS,
    build_fundamental_factors,
    build_valuation_from_fundamentals,
    to_daily_factors,
)
from src.features.multi_source import build_price_factors, build_size_factor
from src.research.ic_analysis import non_overlapping_ic, summarize_ic

HORIZONS = [20, 60]
TOP_K = 200
MIN_OBS = 30


def build_fundamental_panels(
    tickers: list[str], dates: pd.DatetimeIndex
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """DART 재무제표에서 (펀더멘털 팩터 패널, 밸류에이션 수준값 패널)을 만든다."""
    raw = load_fundamentals_bulk(tickers, 2016, 2024, verbose=False)

    factor_series: dict[str, dict[str, pd.Series]] = {f: {} for f in FACTOR_COLUMNS}
    level_series: dict[str, dict[str, pd.Series]] = {c: {} for c in VALUATION_LEVEL_COLUMNS}

    for ticker, group in raw.groupby("ticker"):
        computed = build_fundamental_factors(group)

        daily_factors = to_daily_factors(computed, dates)
        for name in FACTOR_COLUMNS:
            factor_series[name][ticker] = daily_factors[name]

        daily_levels = to_daily_factors(computed, dates, columns=VALUATION_LEVEL_COLUMNS)
        for name in VALUATION_LEVEL_COLUMNS:
            level_series[name][ticker] = daily_levels[name]

    factors = {name: pd.DataFrame(cols) for name, cols in factor_series.items()}
    levels = {name: pd.DataFrame(cols) for name, cols in level_series.items()}
    print(f"  DART 재무데이터 확보: {raw['ticker'].nunique()}종목", flush=True)
    return factors, levels


def main() -> None:
    dates = trading_dates(phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END)
    print(f"인샘플 거래일 {len(dates)}일 ({phase1.IN_SAMPLE_START} ~ {phase1.IN_SAMPLE_END})", flush=True)

    close = build_panel("close", dates)
    volume = build_panel("volume", dates)
    market_cap = build_panel("market_cap", dates)
    print(f"가격/시총 패널: {close.shape[0]}일 x {close.shape[1]}종목", flush=True)

    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)
    print(f"시점별 유니버스: 일별 {universe.sum(axis=1).mean():.0f}종목", flush=True)

    factors: dict[str, pd.DataFrame] = {}
    groups: dict[str, str] = {}

    def register(group: str, panels: dict[str, pd.DataFrame]) -> None:
        for name, panel in panels.items():
            factors[name] = panel
            groups[name] = group

    register("가격", build_price_factors(close, volume))
    register("규모", build_size_factor(market_cap))
    # 시장초과수익 팩터는 제외했다. 시장수익률은 그날 전 종목에 동일한 상수라
    # 빼도 종목간 순위가 안 바뀌고, 실제로 momentum과 순위상관 1.00이 나왔다.
    # (build_excess_return_factors docstring 참고)

    print("DART 재무데이터 로딩 중...", flush=True)
    candidates = fetch_candidate_pool()["ticker"].tolist()
    fundamental_panels, level_panels = build_fundamental_panels(candidates, dates)

    register("펀더멘털", fundamental_panels)
    register(
        "밸류에이션",
        build_valuation_from_fundamentals(
            level_panels["equity"],
            level_panels["net_income_ttm"],
            level_panels["revenue_ttm"],
            market_cap,
        ),
    )

    print(f"팩터 {len(factors)}개 준비 완료", flush=True)

    for horizon in HORIZONS:
        print(f"\n===== forward return horizon = {horizon}일 =====")
        fwd = close.pct_change(horizon).shift(-horizon).where(universe)

        rows = []
        for name, panel in factors.items():
            aligned = panel.reindex_like(fwd).where(universe)
            ic = non_overlapping_ic(aligned, fwd, horizon, min_obs=MIN_OBS)
            summary = summarize_ic(ic)
            summary["구분"] = groups[name]
            summary["factor"] = name
            rows.append(summary)

        table = pd.DataFrame(rows).set_index(["구분", "factor"])
        table = table[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]]
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(table.sort_values("t-stat", key=abs, ascending=False).to_string())

        threshold = stats.norm.ppf(1 - 0.05 / (2 * len(rows)))
        print(f"  팩터 {len(rows)}개 동시검정 -> Bonferroni 보정 임계값 |t| > {threshold:.2f}")

    print(
        "\n주의: 유니버스 자체는 시점별 시가총액으로 구성해 생존편향이 없지만, "
        "펀더멘털/밸류에이션 팩터는 '현재' 시총 상위에서 뽑은 후보군의 DART 데이터만 "
        "있어서 그 부분에는 생존편향이 남아있다."
    )


if __name__ == "__main__":
    main()
