"""
저변동성 팩터가 진짜 신호인지, 특정 국면에서만 작동하는 것인지 판별한다.

지금까지 확인된 것: 저변동성은 유일하게 아웃오브샘플에서 유의했지만(t 2.56),
인샘플 전반부에서는 무신호(t 0.45)였다. 국면 의존성 의심이 남아있다.

여기서 던지는 질문은 세 가지고, 세 번째가 가장 중요하다:

1) 연도별로 보면 어떤가 — 특정 해가 전체를 만든 건 아닌가
2) 시장이 오를 때 vs 내릴 때 — 하락장에서만 작동하는가
3) **저변동성은 저베타의 다른 이름 아닌가**

3번이 결정적인 이유: 변동성이 낮은 종목은 대개 베타도 낮다. 그러면 시장이 빠질 때
덜 빠지는 게 당연하고, 이건 종목을 고르는 능력(cross-sectional 알파)이 아니라
그냥 시장 방향에 베팅한 것이다. 표본 기간에 하락장이 많았다면 후자를 전자로
착각하기 쉽다. 베타로 중립화한 뒤에도 IC가 남는지가 그 구분선이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader.krx_openapi import build_close_panel, build_panel
from src.data_loader.krx_panel import trading_dates
from src.data_loader.universe import market_cap_universe_mask
from src.features.multi_source import build_price_factors, neutralize, rolling_beta
from src.research.composite_factor import combine
from src.research.ic_analysis import daily_cross_sectional_ic, summarize_ic

TOP_K = 200
MIN_OBS = 30
HORIZON = 20

FULL_START = "2018-01-01"
FULL_END = "2024-12-31"


def market_index_from(close: pd.DataFrame, market_cap: pd.DataFrame) -> pd.Series:
    """시가총액 가중 시장지수 (KRX 지수 API는 승인 대상이 아니라 직접 합성)."""
    weights = market_cap.shift(1)
    weights = weights.div(weights.sum(axis=1), axis=0)
    market_return = (close.pct_change() * weights).sum(axis=1, min_count=1)
    return (1 + market_return.fillna(0)).cumprod()


def sampled_ic(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """겹치지 않게 샘플링한 IC 시계열."""
    ic = daily_cross_sectional_ic(signal.reindex_like(fwd), fwd, min_obs=MIN_OBS)
    return ic.iloc[::HORIZON]


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def main() -> None:
    dates = trading_dates(FULL_START, FULL_END)
    close = build_close_panel(dates)
    volume = build_panel("volume", dates)
    market_cap = build_panel("market_cap", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

    price = build_price_factors(close, volume)
    low_vol = combine(
        {"volatility_60": price["volatility_60"]}, universe, signs={"volatility_60": -1}
    )

    fwd = close.pct_change(HORIZON).shift(-HORIZON).where(universe)
    ic = sampled_ic(low_vol, fwd)

    print("=" * 70)
    print("1) 연도별 IC - 특정 해가 전체를 만든 건 아닌가")
    print("=" * 70)

    by_year = pd.DataFrame(
        [
            {"연도": year, **summarize_ic(group)}
            for year, group in ic.groupby(ic.index.year)
        ]
    ).set_index("연도")
    show("[저변동성, horizon 20일]", by_year[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]])

    print("\n" + "=" * 70)
    print("2) 시장 상승기 vs 하락기 - 하락장에서만 작동하는가")
    print("=" * 70)

    market = market_index_from(close, market_cap)

    # (a) 사후적 분류: 그 다음 20일간 시장이 실제로 어땠는지로 나눈다.
    #     미래 정보를 쓰므로 매매에는 못 쓴다. "언제 작동하는가"를 설명하는 진단용이다.
    market_fwd = market.pct_change(HORIZON).shift(-HORIZON).reindex(ic.index)
    forward_regime = pd.Series("상승장", index=ic.index)
    forward_regime[market_fwd < 0] = "하락장"

    by_regime = pd.DataFrame(
        [{"국면": label, **summarize_ic(group)} for label, group in ic.groupby(forward_regime)]
    ).set_index("국면")
    show(
        "[사후적 분류: 이후 20일 시장수익률 기준 - 진단용, 매매 불가]",
        by_regime[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]],
    )

    # (b) 사전적 분류: 이미 지나간 60일 시장수익률로 나눈다. 판단 시점에 알 수 있는
    #     정보만 쓰므로, 여기서 차이가 나면 실제로 활용 가능한 조건부 신호가 된다.
    market_past = market.pct_change(60).reindex(ic.index)
    past_regime = pd.Series("직전 상승", index=ic.index)
    past_regime[market_past < 0] = "직전 하락"

    by_past = pd.DataFrame(
        [{"국면": label, **summarize_ic(group)} for label, group in ic.groupby(past_regime)]
    ).set_index("국면")
    show(
        "[사전적 분류: 직전 60일 시장수익률 기준 - 실제 활용 가능]",
        by_past[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]],
    )

    print("\n" + "=" * 70)
    print("3) 저변동성 vs 저베타 - 시장방어 효과를 걷어내도 남는가")
    print("=" * 70)

    beta = rolling_beta(close, market)
    raw_volatility = price["volatility_60"].where(universe)

    beta_neutral = neutralize(raw_volatility, beta.where(universe))
    beta_neutral_signal = combine(
        {"v": beta_neutral}, universe, signs={"v": -1}
    )

    rows = []
    for label, signal in [
        ("저변동성 (원본)", low_vol),
        ("저변동성 (베타중립화)", beta_neutral_signal),
        ("저베타 단독", combine({"b": beta}, universe, signs={"b": -1})),
    ]:
        summary = summarize_ic(sampled_ic(signal, fwd))
        summary["신호"] = label
        rows.append(summary)

    show(
        "[horizon 20일, 2018~2024 전체]",
        pd.DataFrame(rows).set_index("신호")[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]],
    )

    corr = (
        raw_volatility.corrwith(beta.where(universe), axis=1, method="spearman")
        .dropna()
        .mean()
    )
    print(f"\n  변동성 vs 베타 평균 순위상관: {corr:.2f}")


if __name__ == "__main__":
    main()
