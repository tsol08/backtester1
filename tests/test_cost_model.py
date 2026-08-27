"""
거래비용 모델 검증.

핵심은 **비용이 조용히 사라지거나 조용히 폭발하지 않는가**다. 두 방향 모두 실제로
겪었다: 거래정지 종목(평균거래대금 0)에서 참여율이 무한대가 되어 백테스트 전체가
-inf가 됐고, 반대로 이력이 짧은 종목은 NaN이 0으로 해석되어 매매가 공짜였다.

그리고 조용히 **틀리지도** 않는가: 조정 종가 x 원본 거래량은 분할 이력이 있는
종목의 거래대금을 수십 배 부풀린다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.costs.cost_model import (
    FALLBACK_VOLATILITY,
    MAX_PARTICIPATION,
    CostModel,
    transaction_tax_rate,
)


def make_prices(
    volume: list[float] | None = None,
    close: float = 1000.0,
    trading_value: float | None = None,
    daily_range: float | None = None,
    n: int = 90,
) -> pd.DataFrame:
    """
    일정한 가격/거래량/변동폭을 가진 종목.

    daily_range는 로그 변동폭 ln(고가/저가)이다. Parkinson 추정량이 로그 비율을
    쓰므로, 이렇게 잡아야 변동폭과 추정 변동성이 정확히 비례한다.
    """
    index = pd.date_range("2020-01-01", periods=len(volume) if volume else n, freq="B")
    df = pd.DataFrame({"close": close, "volume": volume if volume else 1_000.0}, index=index)
    if trading_value is not None:
        df["trading_value"] = trading_value
    if daily_range is not None:
        df["high"] = close * np.exp(daily_range / 2)
        df["low"] = close * np.exp(-daily_range / 2)
    return df


def test_suspended_stock_does_not_produce_infinite_cost():
    """거래정지(거래량 0)라도 비용은 유한해야 한다. inf는 백테스트를 통째로 날린다."""
    df = make_prices(volume=[0.0] * 90)
    order = pd.Series(1_000_000.0, index=df.index)

    cost = CostModel().total_cost_rate(df, order)

    assert np.isfinite(cost).all()


def test_participation_is_capped_not_unbounded():
    """평균거래대금보다 큰 주문은 상한에서 잘린다 - 그 너머는 모델이 아는 바가 없다."""
    df = make_prices(volume=[100.0] * 90)
    huge = pd.Series(1e12, index=df.index)

    assert CostModel().participation_rate(df, huge).max() == MAX_PARTICIPATION


def test_short_history_is_not_free_to_trade():
    """
    이력이 윈도우에 못 미치는 종목도 비용을 낸다.

    rolling(20)이 NaN을 주고 그 NaN이 참여율 0으로 해석되던 시절에는, 신규 편입
    종목의 첫 매수가 충격비용 없이 체결됐다.
    """
    df = make_prices(volume=[1_000.0] * 3)
    order = pd.Series(500_000.0, index=df.index)
    model = CostModel()

    impact = model.total_cost_rate(df, order) - (model.commission_rate + model.slippage_rate)

    assert (impact > 0).all()


def test_no_order_costs_nothing_even_when_untradeable():
    """주문이 없으면 상한에 걸린 종목이라도 실제 지불액은 0이다."""
    df = make_prices(volume=[0.0] * 90)
    no_order = pd.Series(0.0, index=df.index)

    paid = CostModel().total_cost_rate(df, no_order) * no_order

    assert (paid == 0).all()


def test_impact_grows_with_order_size_but_decelerates():
    """참여율이 4배면 충격은 2배 (제곱근 법칙)."""
    df = make_prices(volume=[1_000.0] * 90)  # 평균거래대금 = 1,000,000원
    model = CostModel()

    small = model.market_impact_rate(df, pd.Series(10_000.0, index=df.index)).iloc[-1]
    large = model.market_impact_rate(df, pd.Series(40_000.0, index=df.index)).iloc[-1]

    assert large == pytest.approx(2 * small)


def test_impact_scales_with_the_stocks_own_volatility():
    """
    같은 참여율이라도 잘 흔들리는 종목이 더 많이 밀린다.

    이 항이 없던 시절에는 상수 하나가 '변동성 x Y'를 통째로 흡수했고, 그 값이
    얼마여야 하는지 아무도 몰랐다.
    """
    order = pd.Series(10_000.0, index=make_prices(daily_range=0.02).index)
    model = CostModel()

    calm = model.market_impact_rate(make_prices(daily_range=0.01), order).iloc[-1]
    wild = model.market_impact_rate(make_prices(daily_range=0.04), order).iloc[-1]

    assert wild == pytest.approx(4 * calm)


def test_parkinson_volatility_recovers_a_known_range():
    """고가/저가 폭이 2%면 Parkinson 추정량도 그 부근을 돌려줘야 한다."""
    df = make_prices(close=10_000.0, daily_range=0.02)

    sigma = CostModel().daily_volatility(df).iloc[-1]

    # sigma = ln(고가/저가) / (2 sqrt(ln 2))
    assert sigma == pytest.approx(0.02 / (2 * np.sqrt(np.log(2))))


def test_volatility_falls_back_when_high_low_missing():
    df = make_prices(close=10_000.0)  # 고가/저가 없음, 종가도 상수라 표준편차 0

    assert (CostModel().daily_volatility(df) == FALLBACK_VOLATILITY).all()


def test_actual_trading_value_beats_price_times_volume():
    """
    trading_value 컬럼이 있으면 그걸 쓴다.

    조정 종가 x 원본 거래량은 분할 이력이 있는 종목에서 거래대금을 수십 배
    부풀린다(삼성전자 최대 52배). 그러면 참여율이 그만큼 작아져 충격비용이 사라진다.
    """
    adjusted = make_prices(close=50_000.0, volume=[1_000.0] * 90, trading_value=1_000_000.0)

    with_actual = CostModel().average_trading_value(adjusted).iloc[-1]
    without = CostModel().average_trading_value(adjusted.drop(columns="trading_value")).iloc[-1]

    assert with_actual == pytest.approx(1_000_000.0)
    assert without == pytest.approx(50_000_000.0)  # 50배 부풀려진 값


def test_transaction_tax_follows_the_schedule_by_date():
    dates = pd.DatetimeIndex(["2018-06-01", "2019-06-03", "2024-03-01", "2026-01-02"])

    rates = transaction_tax_rate(dates)

    assert list(rates) == [0.0030, 0.0025, 0.0018, 0.0020]


def test_transaction_tax_can_be_turned_off_for_comparison():
    dates = pd.DatetimeIndex(["2024-03-01"])

    assert CostModel(apply_transaction_tax=False).sell_tax_rate(dates).iloc[0] == 0.0
    assert CostModel().sell_tax_rate(dates).iloc[0] > 0.0
