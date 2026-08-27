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

---

**2026-08-28 2차 사전 등록 (유니버스 상위 500위).**

1차(상위 200위)는 롱온리 초과 연 11.43% t 2.05, 비용 차감 후 t 1.71로 미달이었다.
그런데 구간별 초과의 분산을 쪼개보니 **76%가 "상위분위가 22종목뿐이라" 생기는
개별종목 잡음**이고, 효과 자체의 시간에 따른 변동은 24%였다:

    관측된 전체 표준편차   0.0315
    종목이 적어서(잡음)    0.0274   <- 종목 수로 줄인다
    효과의 실제 변동       0.0156   <- 시간으로만 줄어든다

즉 병목은 햇수가 아니라 종목 수다. 종목을 2배로 하는 것이 햇수를 4배로 하는 것과
같다. 그래서 유니버스를 넓힌다. **근거는 이 분산 분해이지 수익률이 아니다.**

  유니버스   시총 상위 500위 (편입이력 1,258종목, 일평균 거래대금 중앙값 62억).
             1000위까지 넓히면 종목은 더 늘지만 거래대금 중앙값이 13억으로 떨어져
             비용을 감당하기 어렵고, 이미 기각된 소형주 가설의 구간에 들어간다.
  동일하게 둠 신호 정의, 표류 창 60일, 5분위 상위, 20일 비겹침 - 1차와 전부 같다.
             여기서 뭔가 더 바꾸면 무엇이 결과를 바꿨는지 알 수 없게 된다.
  판정       **|t| > 2.24** (Bonferroni 2건). 같은 가설의 두 번째 유니버스다.
  2차 관문   비용 차감 후에도 양수여야 한다.

  **정량 예측**: 분산 분해가 맞다면 종목이 3배쯤 되어 t가 **3.0~3.4**로 올라야 한다.
  2.24는 넘되 3.0에 한참 못 미치면, 늘어난 종목이 잡음을 줄인 것이 아니라 신호를
  희석했다는 뜻이므로 그것도 실패로 기록한다. 방향만 맞히는 것보다 반증하기 쉽다.

  **반증 확인(검정 아님)**: SUE를 가진 종목 **전체**(분위 무관)가 유니버스 평균을
  이기면 안 된다. 이기면 우리가 재는 것이 '어닝 서프라이즈'가 아니라 '분기 재무가
  깨끗하게 잡히는 회사'라는 뜻이다. 자사주 검정에서 취득과 처분이 둘 다 양수로 나와
  기각된 것과 같은 함정이다 - 그때 t 2.14짜리 신호의 고유분이 t 0.56이었다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import pandas as pd

from config import settings
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

HORIZON = settings.FORWARD_HORIZON
N_QUANTILES = settings.N_QUANTILES
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=settings.UNIVERSE_TOP_K)
    parser.add_argument("--threshold", type=float, default=1.96)
    args = parser.parse_args()
    TOP_K = args.top_k

    dates = cached_trading_dates(START, END)
    close = build_close_panel(dates)
    market_cap = build_panel("market_cap", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

    print(f"거래일 {len(dates)}일 ({dates.min().date()} ~ {dates.max().date()})", flush=True)
    print(f"비겹침 관측 약 {len(dates) // HORIZON}개", flush=True)

    members = sorted(universe.columns[universe.any(axis=0)])
    dart_codes = set(load_corp_codes()["stock_code"])
    tickers = [t for t in members if t in dart_codes]

    # 끝 연도를 박아두면 해가 바뀌어도 신호가 조용히 과거에서 멈춘다
    fundamentals = load_fundamentals_bulk(tickers, 2015, dates[-1].year, verbose=False)
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

    print("\n" + "=" * 74)
    print("3) 반증 확인: SUE를 가진 종목 전체가 유니버스 평균을 이기는가")
    print("=" * 74)
    print(
        "\n이기면 우리가 재는 것이 '어닝 서프라이즈'가 아니라 '분기 재무가 깨끗하게"
        "\n잡히는 회사'라는 뜻이다. 자사주 검정이 정확히 그렇게 무너졌다 - 취득과"
        "\n처분이 둘 다 양수여서, t 2.14짜리 신호의 고유분이 t 0.56이었다."
    )

    has_sue = in_universe.notna() & universe
    rows = {}
    for date in dates[::HORIZON]:
        covered, pool = has_sue.loc[date], universe.loc[date]
        if covered.sum() < MIN_OBS:
            continue
        returns = fwd.loc[date]
        with_sue, everything = returns[covered].mean(), returns[pool].mean()
        if pd.notna(with_sue) and pd.notna(everything):
            rows[date] = with_sue - everything

    coverage_excess = pd.Series(rows)
    if len(coverage_excess) > 1 and coverage_excess.std() > 0:
        t_cov = coverage_excess.mean() / (coverage_excess.std() / np.sqrt(len(coverage_excess)))
        print(
            f"\n  SUE 보유 종목 - 유니버스 평균: 구간당 {coverage_excess.mean():+.3%}"
            f" (연 {(1 + coverage_excess.mean()) ** PERIODS_PER_YEAR - 1:+.2%}),"
            f" t {t_cov:.2f}, 관측 {len(coverage_excess)}"
        )
        verdict = "경고: 커버리지 자체가 우위를 낸다" if abs(t_cov) > 1.96 else "이상 없음"
        print(f"  -> {verdict}")

    print(
        f"\n판정 기준: |t| > {args.threshold:.2f}."
        "\n분위 검정의 검정력은 미국 모멘텀 보정 기준 이 표본 크기에서 60% 안팎이므로,"
        "\n유의하지 않게 나와도 '모멘텀급 효과는 없다' 이상으로 해석하지 않는다."
    )


if __name__ == "__main__":
    main()
