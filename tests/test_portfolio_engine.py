"""
포트폴리오 엔진의 동일가중 배분/암묵적 리밸런싱 로직을 손계산 값과 비교 검증.

가격은 전부 상수(수익률 0)로 둬서, weights/trades 계산 로직만 순수하게 분리해서 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.costs.cost_model import CostModel
from src.portfolio.portfolio_engine import run_portfolio_backtest


def _flat_price_df(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"close": [100.0] * len(dates), "volume": [1_000_000] * len(dates)}, index=dates
    )


def test_equal_weight_and_implicit_rebalance():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")

    price_by_ticker = {t: _flat_price_df(dates) for t in ["A", "B", "C"]}
    signal_by_ticker = {
        "A": pd.Series([1, 1, 1, 1, 1], index=dates, dtype=float),
        "B": pd.Series([0, 0, 1, 1, 1], index=dates, dtype=float),
        "C": pd.Series([0, 0, 0, 0, 1], index=dates, dtype=float),
    }

    result = run_portfolio_backtest(price_by_ticker, signal_by_ticker, CostModel())

    # position = signal.shift(1) 이므로 A는 day1부터, B는 day3부터 활성.
    # day0: 아무도 활성 아님 -> 전부 0
    # day1,2: A만 활성 -> A=1.0
    # day3,4: A,B 둘 다 활성 -> 각 0.5 (C는 이 구간에서 한 번도 활성화 안 됨)
    expected_weights = pd.DataFrame(
        {
            "A": [0.0, 1.0, 1.0, 0.5, 0.5],
            "B": [0.0, 0.0, 0.0, 0.5, 0.5],
            "C": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=dates,
    )
    pd.testing.assert_frame_equal(result.weights, expected_weights, check_names=False)

    # day3에 B가 새로 들어오면서 A 비중이 1.0->0.5로 줄어드는 것도 "거래"로 잡혀야 한다
    # (A 자신의 신호는 그대로인데도, 다른 종목이 들어와서 비중이 재조정된 것).
    expected_trades_day3 = pd.Series({"A": -0.5, "B": 0.5, "C": 0.0})
    pd.testing.assert_series_equal(
        result.trades.loc[dates[3]], expected_trades_day3, check_names=False
    )

    # 가격이 전부 상수라 수익률은 항상 0 -> 비용만큼만 자산이 줄어야 한다.
    assert (result.returns <= 0).all()
    assert result.cost_paid.sum() > 0


def test_fixed_position_size_ignores_other_tickers():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")

    price_by_ticker = {t: _flat_price_df(dates) for t in ["A", "B"]}
    signal_by_ticker = {
        "A": pd.Series([1, 1, 1, 1, 1], index=dates, dtype=float),
        "B": pd.Series([0, 0, 1, 1, 1], index=dates, dtype=float),
    }

    result = run_portfolio_backtest(
        price_by_ticker, signal_by_ticker, CostModel(), position_size=0.3
    )

    # 고정 비중 모드에서는 B가 들어와도 A의 비중(0.3)이 전혀 안 바뀌어야 한다
    # (동일가중 모드였다면 A는 1.0->0.5로 희석됐을 상황).
    expected_weights = pd.DataFrame(
        {
            "A": [0.0, 0.3, 0.3, 0.3, 0.3],
            "B": [0.0, 0.0, 0.0, 0.3, 0.3],
        },
        index=dates,
    )
    pd.testing.assert_frame_equal(result.weights, expected_weights, check_names=False)

    # B가 진입하는 day3에 A의 비중 변화는 0이어야 한다 (희석 거래 없음).
    assert result.trades.loc[dates[3], "A"] == 0.0


def test_fixed_position_size_caps_when_too_many_active():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")

    price_by_ticker = {t: _flat_price_df(dates) for t in ["A", "B", "C"]}
    signal_by_ticker = {
        "A": pd.Series([1, 1, 1, 1, 1], index=dates, dtype=float),
        "B": pd.Series([0, 1, 1, 1, 1], index=dates, dtype=float),
        "C": pd.Series([0, 0, 1, 1, 1], index=dates, dtype=float),
    }

    result = run_portfolio_backtest(
        price_by_ticker, signal_by_ticker, CostModel(), position_size=0.4
    )

    # position = signal.shift(1) -> A는 day1부터, B는 day2부터, C는 day3부터 활성.
    # day1: A만 활성, 0.4*1=0.4 <= 1.0 -> 캡 없음
    # day2: A,B 활성, 0.4*2=0.8 <= 1.0 -> 캡 없음
    # day3,4: A,B,C 다 활성, 0.4*3=1.2 > 1.0 -> 1.2를 1.0으로 맞추기 위해 1/1.2배로 축소
    scale = 1.0 / 1.2
    expected_weights = pd.DataFrame(
        {
            "A": [0.0, 0.4, 0.4, 0.4 * scale, 0.4 * scale],
            "B": [0.0, 0.0, 0.4, 0.4 * scale, 0.4 * scale],
            "C": [0.0, 0.0, 0.0, 0.4 * scale, 0.4 * scale],
        },
        index=dates,
    )
    pd.testing.assert_frame_equal(result.weights, expected_weights, check_names=False)

    # 캡이 걸리는 날엔 A,B의 비중도 (자기 신호는 안 바뀌었는데) 줄어드는 게 맞다 -
    # "신호가 몰릴 때만" 발생하는 예외적 축소이므로.
    assert result.trades.loc[dates[3], "A"] < 0
    # 반대로 캡이 안 걸리는 구간(day1->day2)에서는 A의 비중이 그대로 유지돼야 한다.
    assert result.trades.loc[dates[2], "A"] == 0.0
