"""
무상증자 제외가 t 3.03인 이유를 찾는다. **전략 판정이 아니라 진단이다.**

## 왜 이걸 하나

유상증자 제외 검정(2026-08-28 (9))에서 반증 확인 대조군이던 **무상증자 제외**가
초과 연 1.12%, **t 3.03**으로 본 가설(1.42)보다 강했다. 사전 등록대로 본 가설은
기각했지만 이 숫자는 남았다.

**그리고 이 숫자는 앞뒤가 안 맞는다.** 제외형의 산술이

    전략 초과 = 제외비중/(1-제외비중) x 제외 종목의 부진폭

이고 제외 비중이 7.8%였으므로, 역산하면 **제외 종목이 연 13% 부진**해야 한다.
3년 창에서 연 13%짜리 이상현상은 실재하기엔 너무 크다. 그리고 무상증자는 주주에게
공짜 주식을 주는 **양의 이벤트**다. 피하는 게 낫다는 것 자체가 이상하다.

이 저장소에서 가장 큰 t(4.12)는 버그였고, 그걸 잡은 것은 새 검정이 아니라
"숫자가 안 맞는다"를 그냥 넘기지 않은 것이었다. 같은 순서로 간다:
**수익을 쫓기 전에 데이터부터 본다.**

## 사전 등록 — 결과를 보기 전에 무엇을 볼지와 각 결과가 뜻하는 바를 정한다

**이것은 배포 가설이 아니다.** 여기서 무엇이 나오든 그것으로 전략을 판정하지
않는다. 원인이 밝혀지고 그것이 여전히 신호로 보인다면, **그때 별도로 사전
등록해서 한 번 본다.**

  D1 데이터 정렬 실사
     무상증자 공시 후 120일 안에, 주식수 변화(+-3일)를 동반한 |일간수익률| > 10%
     일수를 센다. 주가 조정일과 주식수 갱신일이 하루라도 어긋나면 보정이 안 걸리고
     가짜 하락이 남는다. 그러면 그 종목을 제외하는 것이 좋아 보인다.
     -> 다수 발견되면 **데이터 문제**다. 고치고 전부 재계산한다.

  D2 산술 확인
     무상증자 종목 바스켓 자체의 초과수익을 직접 잰다. 위 역산이 맞다면 연 -13%
     근처여야 한다.
     -> 크게 어긋나면 산술이나 배선이 틀린 것이므로 그것부터 본다.

  D3 반전인가 발행인가
     공시 직전 6개월 수익률로 무상증자 종목을 상/하위 절반으로 나눈다.
     한국에서 무상증자는 주가가 급등한 소형주가 하는 경우가 많다고 알려져 있다.
     -> 효과가 **급등한 쪽에만** 있으면 이건 발행 효과가 아니라 반전 효과다.
        무상증자는 '최근 급등'의 대리변수일 뿐이다.
     -> 양쪽에 고르게 있으면 발행 자체와 관련이 있다는 쪽으로 기운다.

  D4 규모/변동성 중립화
     제외 마스크를 규모·변동성에 회귀시킨 잔차로 바꿔 같은 것을 잰다.
     -> 사라지면 무상증자가 아니라 소형/고변동 종목을 피한 것이다.

읽는 순서가 정해져 있다. **D1이 걸리면 D2~D4는 의미가 없다.**
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import settings
from src.data_loader.dart_filings import event_dates, load_filings, recent_event_mask
from src.data_loader.krx_openapi import build_panels, cached_trading_dates
from src.data_loader.panels import Panels
from src.strategy.base import periodic_schedule

from scripts.run_backtest import t_stat

TOP_K = 500
HORIZON = settings.FORWARD_HORIZON
PERIODS_PER_YEAR = 252 / HORIZON
EVENT_WINDOW = 1095
FILING_START, FILING_END = "2015-01-01", "2026-08-31"
BONUS = ["무상증자결정"]

D1_WINDOW = 120  # 공시 후 며칠 안을 볼 것인가 (권리락일이 보통 이 안에 온다)
D1_MOVE = 0.10  # 이보다 큰 일간 변동을 의심 대상으로 본다
D1_ALIGN = 3  # 주식수 변화가 +-이 며칠 안에 있으면 '동반'으로 본다
LOOKBACK = 126  # D3의 직전 6개월 (거래일)


def annualize(x: float) -> float:
    return (1 + x) ** PERIODS_PER_YEAR - 1


def excess_of(mask: pd.DataFrame, fwd: pd.DataFrame, universe: pd.DataFrame,
              schedule: list[pd.Timestamp], min_names: int = 5) -> pd.Series:
    """마스크에 든 종목 동일가중의 구간 초과수익(vs 같은 날 유니버스 동일가중)."""
    rows = {}
    for date in schedule:
        in_universe = universe.loc[date]
        picked = mask.loc[date] & in_universe
        if int(picked.sum()) < min_names:
            continue
        returns = fwd.loc[date]
        picked_mean, universe_mean = returns[picked].mean(), returns[in_universe].mean()
        if pd.notna(picked_mean) and pd.notna(universe_mean):
            rows[date] = picked_mean - universe_mean
    return pd.Series(rows)


def report(label: str, series: pd.Series) -> None:
    if series.empty:
        print(f"  {label:34s} 관측 없음")
        return
    print(f"  {label:34s} 관측 {len(series):3d}  연 {annualize(series.mean()):+7.2%}"
          f"  t {t_stat(series):+6.2f}")


def main() -> None:
    print("데이터 로딩...", flush=True)
    panels = Panels.load(top_k=TOP_K)
    dates = panels.dates
    schedule = periodic_schedule(dates, HORIZON)
    members = set(panels.members)

    filings = load_filings(FILING_START, FILING_END)
    events = event_dates(filings, BONUS)
    inside = events[events["ticker"].isin(members)]
    bonus_mask = recent_event_mask(inside, dates, panels.close.columns, EVENT_WINDOW)
    bonus_mask &= panels.universe

    fwd = panels.close.pct_change(HORIZON, fill_method=None).shift(-HORIZON)

    print("\n" + "=" * 80)
    print("무상증자 제외 t 3.03의 원인 진단 — 전략 판정이 아니다")
    print("=" * 80)
    print(f"유니버스 상위 {TOP_K}위 | 창 {EVENT_WINDOW}일 | 리밸런싱 {len(schedule)}회")
    print(f"무상증자 원본 {len(events):,}건 중 유니버스 내 {len(inside):,}건")
    print(f"제외 비중 평균"
          f" {(bonus_mask.loc[schedule].sum(axis=1) / panels.universe.loc[schedule].sum(axis=1)).mean():.1%}")

    # ---------------- D1. 데이터 정렬 실사 ----------------
    print("\n" + "-" * 80)
    print("D1  주가 조정일과 주식수 갱신일이 어긋나 남은 가짜 변동이 있는가")
    print("-" * 80)

    raw = build_panels(["close", "listed_shares"], cached_trading_dates(
        settings.DATA_START, dates[-1].strftime("%Y-%m-%d")))
    raw_close = raw["close"].reindex(index=dates, columns=panels.close.columns)
    shares = raw["listed_shares"].reindex(index=dates, columns=panels.close.columns)

    # 보정된 패널에 남아있는 큰 하락
    adjusted_move = panels.close.pct_change(fill_method=None)
    shares_change = (shares / shares.shift(1) - 1).abs() > 0.05
    # +-D1_ALIGN일 안에 주식수 변화가 있었는가
    nearby = shares_change.rolling(2 * D1_ALIGN + 1, center=True, min_periods=1).max().astype(bool)

    suspicious = (adjusted_move.abs() > D1_MOVE) & nearby & bonus_mask
    n_suspicious = int(suspicious.sum().sum())
    print(f"  무상증자 창 안에서, 주식수 변화를 +-{D1_ALIGN}일 안에 동반한"
          f" |일간변동| > {D1_MOVE:.0%} 일수: **{n_suspicious:,}건**")

    if n_suspicious:
        r, c = np.where(suspicious.to_numpy())
        detail = pd.DataFrame({
            "date": dates[r], "ticker": panels.close.columns[c],
            "보정후변동": [adjusted_move.iat[i, j] for i, j in zip(r, c)],
            "원본변동": [raw_close.iat[i, j] / raw_close.iat[i - 1, j] - 1 if i else np.nan
                     for i, j in zip(r, c)],
            "주식수배율": [shares.iat[i, j] / shares.iat[i - 1, j] if i else np.nan
                      for i, j in zip(r, c)],
        })
        down = detail[detail["보정후변동"] < 0]
        print(f"  그중 하락 {len(down):,}건, 상승 {len(detail) - len(down):,}건."
              f" 하락 중앙값 {down['보정후변동'].median():.2%}"
              if len(down) else "  전부 상승이다")
        print("\n  가장 큰 하락 8건")
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(detail.nsmallest(8, "보정후변동").to_string(index=False))
        print("\n  ** D1에서 건수가 나왔다. 이것이 원인일 수 있으므로 D2~D4보다")
        print("     먼저 이 건들의 성격을 봐야 한다.")
    else:
        print("  없다. 정렬 문제는 아니다.")

    # ---------------- D2. 산술 확인 ----------------
    print("\n" + "-" * 80)
    print("D2  제외 바스켓 자체는 얼마나 부진한가 (역산은 연 -13%를 함의한다)")
    print("-" * 80)
    basket = excess_of(bonus_mask, fwd, panels.universe, schedule)
    report("무상증자 종목 바스켓", basket)
    share = (bonus_mask.loc[schedule].sum(axis=1)
             / panels.universe.loc[schedule].sum(axis=1)).mean()
    implied = annualize(basket.mean()) * share / (1 - share) if not basket.empty else np.nan
    print(f"  -> 이 부진폭이면 제외 전략 초과는 연 {implied:+.2%}"
          f"  (실측 +1.12%와 맞아야 한다)")

    # ---------------- D3. 반전인가 발행인가 ----------------
    print("\n" + "-" * 80)
    print("D3  공시 직전 6개월 수익률로 나누면 어느 쪽에 몰려 있는가")
    print("-" * 80)
    prior = panels.close.pct_change(LOOKBACK, fill_method=None)
    prior_rank = prior.where(panels.universe).rank(axis=1, pct=True)

    hot = bonus_mask & (prior_rank > 0.5)
    cold = bonus_mask & (prior_rank <= 0.5)
    report("급등한 쪽 (직전 6개월 상위 절반)", excess_of(hot, fwd, panels.universe, schedule))
    report("안 급등한 쪽 (하위 절반)", excess_of(cold, fwd, panels.universe, schedule))
    print("  -> 급등한 쪽에만 몰려 있으면 발행 효과가 아니라 반전 효과다.")

    # ---------------- D4. 규모/변동성 ----------------
    print("\n" + "-" * 80)
    print("D4  무상증자 종목은 어떤 종목들인가 (규모·변동성 분위)")
    print("-" * 80)
    size_rank = panels.market_cap.where(panels.universe).rank(axis=1, pct=True)
    vol = panels.close.pct_change(fill_method=None).rolling(60).std()
    vol_rank = vol.where(panels.universe).rank(axis=1, pct=True)

    for label, ranks in (("시가총액 백분위", size_rank), ("60일 변동성 백분위", vol_rank)):
        picked = ranks.where(bonus_mask).loc[schedule].stack().mean()
        print(f"  무상증자 종목의 평균 {label}: {picked:.2f}  (유니버스 평균 0.50)")

    small = bonus_mask & (size_rank <= 0.5)
    large = bonus_mask & (size_rank > 0.5)
    report("소형 절반", excess_of(small, fwd, panels.universe, schedule))
    report("대형 절반", excess_of(large, fwd, panels.universe, schedule))


if __name__ == "__main__":
    main()
