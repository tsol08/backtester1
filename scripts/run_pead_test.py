"""
가설 검정: 실적발표 후 표류(PEAD)가 한국 시장에 존재하는가.

**사전에 정한 단일 가설이다.** 이 프로젝트는 이미 40건 넘게 검정했으므로, 우연히
t 3을 넘는 것이 나오는 게 정상인 상태다. 그래서 여러 변형을 돌려보고 좋은 것을
고르는 방식을 쓰지 않는다. 가설과 측정 방식을 먼저 못박고 한 번만 본다.

가설: 어닝 서프라이즈(SUE)가 큰 종목은 공시 이후 수 주간 초과수익을 낸다.
근거: 전 세계에서 반복 확인된 이상현상이며, 소멸하지 않는 이유로 '정보 처리 비용'이
      지목된다 - 분기 재무제표를 공시일 기준으로 정렬해 서프라이즈를 계산하는 일
      자체가 장벽이다.

측정: 공시 후 60일 이내에만 SUE를 유효하게 두고(PEAD는 발표 직후 현상이므로),
      20일 forward return에 대한 cross-sectional IC와 롱온리 분위 초과를 본다.
      공매도를 하지 않으므로 분위 쪽이 실질적 판단 기준이다.

유니버스: 시점별 시가총액 상위 200종목. DART 재무는 그 유니버스에 한 번이라도
      편입된 466종목 전체를 수집했으므로, 이전 분석에 있던 생존편향이 없다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader.dart_loader import load_corp_codes, load_fundamentals_bulk
from src.data_loader.krx_openapi import build_close_panel, build_panel, cached_trading_dates
from src.data_loader.universe import market_cap_universe_mask
from src.features.earnings_surprise import build_sue_panel
from src.research.ic_analysis import daily_cross_sectional_ic, summarize_ic
from src.research.quantile_analysis import (
    long_only_edge,
    monotonicity,
    quantile_forward_returns,
    summarize_quantiles,
)

TOP_K = 200
HORIZON = 20
N_QUANTILES = 5
MIN_OBS = 30
DRIFT_WINDOW = 60  # 공시 후 며칠까지 신호로 볼 것인가
PERIODS_PER_YEAR = 252 / HORIZON

# SUE는 8분기 표준편차 + 4분기 시차가 필요하므로 2015년 데이터로는 2018년부터 나온다
START, END = "2018-01-01", "2026-08-26"


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def main() -> None:
    dates = cached_trading_dates(START, END)
    close = build_close_panel(dates)
    market_cap = build_panel("market_cap", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

    print(f"거래일 {len(dates)}일 ({dates.min().date()} ~ {dates.max().date()})", flush=True)
    print(f"비겹침 관측 약 {len(dates) // HORIZON}개", flush=True)

    members = sorted(universe.columns[universe.any(axis=0)])
    dart_codes = set(load_corp_codes()["stock_code"])
    tickers = [t for t in members if t in dart_codes]

    fundamentals = load_fundamentals_bulk(tickers, 2015, 2024, verbose=False)
    print(f"DART 재무: {len(fundamentals)}행 / {fundamentals['ticker'].nunique()}종목", flush=True)

    sue = build_sue_panel(fundamentals, dates, drift_window=DRIFT_WINDOW)
    print(f"SUE 패널: {sue.shape}", flush=True)

    in_universe = sue.reindex_like(close).where(universe)
    coverage = in_universe.notna().sum(axis=1)
    print(f"유니버스 내 SUE 보유 종목(일평균): {coverage.mean():.0f} / {TOP_K}", flush=True)

    fwd = close.pct_change(HORIZON).shift(-HORIZON)

    print("\n" + "=" * 74)
    print("1) cross-sectional IC")
    print("=" * 74)

    ic = daily_cross_sectional_ic(in_universe, fwd.where(universe), min_obs=MIN_OBS)
    ic = ic.iloc[::HORIZON]
    show("[SUE, forward 20일]", pd.DataFrame([summarize_ic(ic)]).set_index("관측일수"))

    print("\n" + "=" * 74)
    print("2) 롱온리 분위 (공매도를 하지 않으므로 이쪽이 실질 기준)")
    print("=" * 74)

    qr = quantile_forward_returns(
        in_universe, fwd, universe, n_quantiles=N_QUANTILES, sample_every=HORIZON
    )
    if qr.empty:
        print("분위를 만들 수 있는 날이 없음")
        return

    show(f"[SUE 분위별 {HORIZON}일 수익률]", summarize_quantiles(qr, PERIODS_PER_YEAR))

    edge = long_only_edge(qr, PERIODS_PER_YEAR)
    print(f"\n  단조성: {monotonicity(qr):.2f}")
    print(
        f"  롱온리 초과(상위분위 - 유니버스평균): 연 {edge['상위분위 초과(연율화)']:.2%}"
        f" (t {edge['상위분위 초과 t-stat']:.2f})"
    )
    print(
        f"  참고: 상하위 스프레드(공매도 필요): 연 {edge['상하위 스프레드(연율화)']:.2%}"
        f" (t {edge['상하위 스프레드 t-stat']:.2f})"
    )

    print(
        "\n판정 기준: 사전에 정한 단일 가설이므로 다중검정 보정 없이 |t| > 1.96을 쓴다."
        "\n분위 검정의 검정력은 미국 모멘텀 보정 기준 이 표본 크기에서 60% 안팎이므로,"
        "\n유의하지 않게 나와도 '모멘텀급 효과는 없다' 이상으로 해석하지 않는다."
    )


if __name__ == "__main__":
    main()
