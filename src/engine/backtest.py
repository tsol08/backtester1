"""
Vectorized 단일 종목 백테스트 엔진.

핵심 불변조건(룩어헤드 방지): t일 종가까지의 정보로 계산된 시그널은 t+1일에만 체결된다.
이를 코드 한 곳(signal.shift(1))에서만 강제해서, 다른 곳에서 실수로 당일 체결을 만들지
않도록 한다. tests/test_no_lookahead.py 에서 이 불변조건을 검증한다.

포지션 사이징은 Phase 1에서는 단순화해 "매 시점 초기자본 대비 비중"으로 계산한다
(복리 재투자 없음). 포트폴리오 단위 복리/리밸런싱은 이후 phase(portfolio 모듈)에서 다룬다.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.costs.cost_model import CostModel


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series  # 비용 차감 후 일별 순수익률
    positions: pd.Series  # 실제 체결된 포지션 비중 (0~1 등)
    trades: pd.Series  # 일별 포지션 변화량
    cost_paid: pd.Series  # 일별 비용(원화)


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_model: CostModel,
    initial_capital: float = 100_000_000,
) -> BacktestResult:
    """
    df: OHLCV (close 컬럼 필수)
    signal: 날짜별 원하는 포지션 비중 (예: 0~1). t일 값은 t일 종가까지의 정보로 계산된 것으로 간주.
    """
    aligned_signal = signal.reindex(df.index).fillna(0.0)
    position = aligned_signal.shift(1).fillna(0.0)  # <-- 룩어헤드 방지: 유일한 shift 지점

    daily_return = df["close"].pct_change().fillna(0.0)
    gross_return = position * daily_return

    position_change = position.diff()
    position_change.iloc[0] = position.iloc[0]

    order_value = position_change.abs() * initial_capital
    cost_rate = cost_model.total_cost_rate(df, order_value)
    cost_drag = cost_rate * position_change.abs()

    net_return = gross_return - cost_drag
    equity_curve = initial_capital * (1 + net_return).cumprod()

    return BacktestResult(
        equity_curve=equity_curve,
        returns=net_return,
        positions=position,
        trades=position_change,
        cost_paid=cost_drag * initial_capital,
    )
