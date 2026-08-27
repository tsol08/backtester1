"""
거래비용 모델 검증.

핵심은 **비용이 조용히 사라지거나 조용히 폭발하지 않는가**다. 두 방향 모두 실제로
겪었다: 거래정지 종목(평균거래대금 0)에서 참여율이 무한대가 되어 백테스트 전체가
-inf가 됐고, 반대로 이력이 짧은 종목은 NaN이 0으로 해석되어 매매가 공짜였다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.costs.cost_model import MAX_PARTICIPATION, CostModel, transaction_tax_rate


def make_prices(volume: list[float], close: float = 1000.0) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(volume), freq="B")
    return pd.DataFrame({"close": close, "volume": volume}, index=index)


def test_suspended_stock_does_not_produce_infinite_cost():
    """거래정지(거래량 0)라도 비용은 유한해야 한다. inf는 백테스트를 통째로 날린다."""
    df = make_prices([0.0] * 30)
    order = pd.Series(1_000_000.0, index=df.index)

    cost = df.pipe(lambda d: CostModel().total_cost_rate(d, order))

    assert np.isfinite(cost).all()
    assert cost.max() == pytest.approx(0.00015 + 0.0005 + 0.1)


def test_participation_is_capped_not_unbounded():
    """평균거래대금보다 큰 주문은 상한에서 잘린다 - 그 너머는 모델이 아는 바가 없다."""
    df = make_prices([100.0] * 30)  # 평균거래대금 = 100,000원
    huge = pd.Series(1e12, index=df.index)

    assert CostModel().participation_rate(df, huge).max() == MAX_PARTICIPATION


def test_short_history_is_not_free_to_trade():
    """
    이력이 20일이 안 되는 종목도 비용을 낸다.

    rolling(20)이 NaN을 주고 그 NaN이 참여율 0으로 해석되던 시절에는, 신규 편입
    종목의 첫 매수가 충격비용 없이 체결됐다.
    """
    df = make_prices([1_000.0] * 3)  # 20일 윈도우에 한참 못 미침
    order = pd.Series(500_000.0, index=df.index)

    impact = CostModel().total_cost_rate(df, order) - (0.00015 + 0.0005)

    assert (impact > 0).all()


def test_no_order_costs_nothing_even_when_untradeable():
    """주문이 없으면 상한에 걸린 종목이라도 실제 지불액은 0이다."""
    df = make_prices([0.0] * 30)
    no_order = pd.Series(0.0, index=df.index)

    paid = CostModel().total_cost_rate(df, no_order) * no_order

    assert (paid == 0).all()


def test_impact_grows_with_order_size_but_decelerates():
    """참여율이 4배면 충격은 2배 (sqrt 법칙)."""
    df = make_prices([1_000.0] * 30)  # 평균거래대금 = 1,000,000원
    model = CostModel(commission_rate=0.0, slippage_rate=0.0)

    small = model.total_cost_rate(df, pd.Series(10_000.0, index=df.index)).iloc[-1]
    large = model.total_cost_rate(df, pd.Series(40_000.0, index=df.index)).iloc[-1]

    assert large == pytest.approx(2 * small)


def test_transaction_tax_follows_the_schedule_by_date():
    dates = pd.DatetimeIndex(["2018-06-01", "2019-06-03", "2024-03-01", "2026-01-02"])

    rates = transaction_tax_rate(dates)

    assert list(rates) == [0.0030, 0.0025, 0.0018, 0.0020]


def test_transaction_tax_can_be_turned_off_for_comparison():
    dates = pd.DatetimeIndex(["2024-03-01"])

    assert CostModel(apply_transaction_tax=False).sell_tax_rate(dates).iloc[0] == 0.0
    assert CostModel().sell_tax_rate(dates).iloc[0] > 0.0
