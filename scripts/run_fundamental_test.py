"""
가설 검정: 밸류 / 수익성 / 자산성장이 한국 시장에 롱온리 우위를 주는가.

**세 개를 미리 선언하고 한 번만 본다.** 하나씩 보고 실패하면 다음 것으로 넘어가는
방식은 검정 횟수가 숨겨져서, 셋을 선언하고 보정하는 것보다 나쁘다. 이 프로젝트는
이미 40건 넘게 검정해서 우연히 t 3이 나오는 게 정상인 상태다.

  H1  book_to_market  자본총계/시가총액이 **높은** 종목이 초과수익  (Fama-French HML)
  H2  roe             최근 4분기 순이익/자본총계가 **높은** 종목     (RMW)
  H3  asset_growth    자산총계 증가율이 **낮은** 종목                (CMA)

부호까지 미리 못박는 이유: 반대 부호로 유의하게 나오는 것은 발견이 아니라 잡음이다.
H3은 낮은 쪽을 사는 가설이므로 부호를 뒤집어 '상위 분위 = 저성장'이 되게 한다.

**판정 기준**: 3개 검정에 대한 Bonferroni 보정, |t| > 2.39 (양측 5%).
**측정**: 20일 forward return에 대한 롱온리 분위 초과(상위분위 - 유니버스 평균).
공매도를 하지 않으므로 IC가 아니라 이쪽이 판단 기준이다. 비겹침 20일 간격 샘플링.

왜 이 셋인가: 임의로 고른 조합이 아니라 Fama-French 5팩터가 존재를 주장하는
팩터들이고, 우리 도구는 French 데이터에서 이들을 검출할 수 있음을 이미 확인했다
(run_french_validation.py). PEAD를 정당한 가설로 삼았던 것과 같은 근거다.

표본이 PEAD보다 낫다: 재무상태표 항목이라 워밍업이 필요 없어 2016년부터 시작하고,
SUE와 달리 공시 계절에 따른 공백이 없다(월별 커버리지 138~153종목으로 균일).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scipy import stats

from config import settings
from src.data_loader.dart_loader import load_corp_codes, load_fundamentals_bulk
from src.data_loader.panels import Panels
from src.features.fundamental_factors import build_fundamental_panels
from src.research.ic_analysis import daily_cross_sectional_ic, summarize_ic
from src.research.power_analysis import minimum_detectable_ic
from src.research.quantile_analysis import (
    long_only_edge,
    monotonicity,
    quantile_forward_returns,
    summarize_quantiles,
)

HORIZON = settings.FORWARD_HORIZON
N_QUANTILES = settings.N_QUANTILES
PERIODS_PER_YEAR = 252 / HORIZON
MIN_OBS = 30

N_TESTS = 3
THRESHOLD = float(stats.norm.ppf(1 - 0.05 / (2 * N_TESTS)))

# (팩터, 사는 방향). +1이면 값이 큰 쪽, -1이면 작은 쪽을 산다.
HYPOTHESES = [
    ("book_to_market", +1, "H1 밸류 (자본총계/시가총액 높은 쪽)"),
    ("roe", +1, "H2 수익성 (ROE 높은 쪽)"),
    ("asset_growth", -1, "H3 자산성장 (증가율 낮은 쪽)"),
]


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def main() -> None:
    print("데이터 로딩...", flush=True)
    panels = Panels.load(start="2015-01-01")
    dates = panels.dates

    dart_codes = set(load_corp_codes()["stock_code"])
    fundamentals = load_fundamentals_bulk(
        [t for t in panels.members if t in dart_codes], 2015, dates[-1].year, verbose=False
    )
    factors = build_fundamental_panels(fundamentals, dates, panels.market_cap)

    forward = panels.close.pct_change(HORIZON, fill_method=None).shift(-HORIZON)
    universe = panels.universe

    coverage = factors["book_to_market"].where(universe).notna().sum(axis=1)
    usable = dates[dates >= coverage[coverage >= MIN_OBS].index[0]]

    print("\n" + "=" * 74)
    print(f"검정 조건 (사전 등록, Bonferroni {N_TESTS}건 -> |t| > {THRESHOLD:.2f})")
    print("=" * 74)
    print(f"\n평가 구간: {usable[0].date()} ~ {usable[-1].date()}"
          f"  -> 비겹침 20일 관측 {len(usable) // HORIZON}개")
    print(f"유니버스 {settings.UNIVERSE_TOP_K}종목 중 팩터 보유(일평균): {coverage.mean():.0f}")
    print("  (참고: PEAD는 신호 구간 61개, 1년의 43%가 공백이었다)")

    print("\n" + "=" * 74)
    print("0) 이 표본으로 무엇을 볼 수 있는가 - 결과를 보기 전에 먼저 잰다")
    print("=" * 74)
    power = minimum_detectable_ic(
        forward.loc[usable],
        universe.loc[usable],
        horizon=HORIZON,
        candidates=[0.02, 0.03, 0.05, 0.08],
        n_trials=100,
        thresholds=(THRESHOLD,),
    )
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(f"\n{power.to_string()}")
    print("\n검출률이 낮은 구간에서 유의하지 않게 나오면 '없다'가 아니라 '못 본다'이다.")

    print("\n" + "=" * 74)
    print("판정")
    print("=" * 74)

    verdicts = []
    for name, direction, label in HYPOTHESES:
        signal = (direction * factors[name]).where(universe).loc[usable]
        quantile_returns = quantile_forward_returns(
            signal, forward.loc[usable], universe.loc[usable],
            n_quantiles=N_QUANTILES, min_obs=MIN_OBS, sample_every=HORIZON,
        )
        if quantile_returns.empty or quantile_returns.shape[1] < N_QUANTILES:
            verdicts.append({"가설": label, "관측": 0})
            continue

        edge = long_only_edge(quantile_returns, PERIODS_PER_YEAR)
        ic = daily_cross_sectional_ic(
            signal, forward.loc[usable].where(universe.loc[usable]), min_obs=MIN_OBS
        ).iloc[::HORIZON]

        print(f"\n{'-' * 74}\n{label}\n{'-' * 74}")
        show("", summarize_quantiles(quantile_returns, PERIODS_PER_YEAR))
        print(f"\n  단조성: {monotonicity(quantile_returns):.2f}")
        print(f"  IC: {summarize_ic(ic)}")

        t_stat = edge["상위분위 초과 t-stat"]
        verdicts.append({
            "가설": label,
            "관측": edge["관측"],
            "롱온리초과(연)": edge["상위분위 초과(연율화)"],
            "t-stat": t_stat,
            "단조성": monotonicity(quantile_returns),
            "판정": "통과" if abs(t_stat) > THRESHOLD and t_stat > 0 else "미달",
        })

    print("\n" + "=" * 74)
    show(f"[사전 등록 기준 |t| > {THRESHOLD:.2f}, 부호는 양수여야 함]",
         pd.DataFrame(verdicts).set_index("가설"))
    print("\n부호가 반대인데 유의한 경우는 '통과'가 아니다 - 가설이 예측한 방향이 아니다.")


if __name__ == "__main__":
    main()
