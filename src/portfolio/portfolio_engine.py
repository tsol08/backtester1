"""
N개 종목을 동시에 다루는 포트폴리오 백테스트 엔진.

각 종목의 (0/1) 신호를 독립적으로 t+1에 체결한다. 이후 비중을 어떻게 배분하는지는
position_size 인자로 고른다:

- position_size=None(기본): 그날 활성인 종목들에 동일가중 배분(1/활성종목수). 항상
  자본을 100% 굴리지만, 활성 종목 수가 바뀔 때마다 "내 신호는 안 바뀌었는데 남이
  들어와서 내 비중이 희석"되는 암묵적 리밸런싱 거래가 발생하고 비용이 붙는다.
- position_size=고정값(예: 1/N): 종목당 비중을 고정한다. 한 종목의 비중은 원칙적으로
  그 종목 자신의 신호가 바뀔 때만 변해서 희석 거래가 사라지지만, 활성 종목이 적은
  날엔 자본 일부가 현금으로 논다(자본 활용률 저하).

  position_size를 1/N보다 크게 잡으면(예: 1/3) 평소엔 자본을 더 많이 굴리면서도,
  "여러 종목이 우연히 한꺼번에 활성화돼서 합계가 max_gross_exposure(기본 100%)를
  넘는" 드문 날에만 전 종목 비중을 비례적으로 줄인다(capping). 즉 평소엔 종목별
  비중이 서로 독립적이라 희석 거래가 거의 없고, 신호가 정말 몰리는 날에만 예외적으로
  전체를 줄이는 거래가 발생 — "억지로 매일 분산"과 "절대 서로 안 건드림" 사이의
  절충안.

단일 종목 엔진(src/engine/backtest.py)과 동일한 단순화를 그대로 따른다: 포지션
사이징은 복리 재투자 없이 매 시점 initial_capital 대비 비중으로 계산한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.costs.cost_model import CostModel


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    returns: pd.Series
    weights: pd.DataFrame  # 종목별 실제 체결된 비중
    trades: pd.DataFrame  # 종목별 비중 변화량
    cost_paid: pd.Series  # 일별 총비용(원)


def run_portfolio_backtest(
    price_by_ticker: dict[str, pd.DataFrame],
    signal_by_ticker: dict[str, pd.Series],
    cost_model: CostModel,
    initial_capital: float = 100_000_000,
    position_size: float | None = None,
    max_gross_exposure: float = 1.0,
) -> PortfolioResult:
    tickers = list(price_by_ticker.keys())

    index = price_by_ticker[tickers[0]].index
    for t in tickers[1:]:
        index = index.union(price_by_ticker[t].index)

    positions = pd.DataFrame(index=index, columns=tickers, dtype=float)
    daily_returns = pd.DataFrame(index=index, columns=tickers, dtype=float)

    for t in tickers:
        df = price_by_ticker[t].reindex(index)
        signal = signal_by_ticker[t].reindex(index).fillna(0.0)
        positions[t] = signal.shift(1).fillna(0.0)
        # 거래정지 등으로 종가가 비는 날은 수익률 0으로 두고, 재개된 날 그동안의 변동을
        # 한꺼번에 인식한다(ffill 후 변화율). 구멍을 그냥 NaN으로 두면 재개일의 움직임까지
        # 통째로 사라져서 정지 기간의 손익이 없던 일이 된다.
        daily_returns[t] = df["close"].ffill().pct_change(fill_method=None).fillna(0.0)

    if position_size is None:
        n_active = (positions > 0).sum(axis=1)
        weights = positions.div(n_active.replace(0, np.nan), axis=0).fillna(0.0)
    else:
        raw_weights = positions * position_size
        total_exposure = raw_weights.sum(axis=1)

        # 합계가 한도를 넘는 날에만 비례적으로 줄인다 (넘지 않는 날은 scale=1, 즉
        # 각 종목 비중이 다른 종목과 무관하게 결정됨 -> 희석 거래 없음).
        scale = pd.Series(1.0, index=total_exposure.index)
        over_limit = total_exposure > max_gross_exposure
        scale[over_limit] = max_gross_exposure / total_exposure[over_limit]

        weights = raw_weights.mul(scale, axis=0)

    return _simulate(weights, price_by_ticker, cost_model, initial_capital)


def run_weighted_backtest(
    target_weights: pd.DataFrame,
    price_by_ticker: dict[str, pd.DataFrame],
    cost_model: CostModel,
    initial_capital: float = 100_000_000,
) -> PortfolioResult:
    """
    목표 비중 패널(날짜 x 종목)을 그대로 받는 진입점.

    비중을 직접 정하는 전략(src/strategy)은 이쪽을 쓴다. t+1 체결은 여기서 강제하므로
    전략이 목표를 언제 계산했든 그 다음 날 종가에 체결된다.
    """
    tickers = [t for t in target_weights.columns if t in price_by_ticker]
    weights = target_weights[tickers].shift(1).fillna(0.0)
    return _simulate(weights, {t: price_by_ticker[t] for t in tickers}, cost_model, initial_capital)


# 주문금액을 평가금액에 맞추기 위한 반복 횟수. 비용이 평가금액에 의존하고 평가금액이
# 다시 비용에 의존하는데, 되먹임이 시장충격을 통해서만 일어나 아주 약하다(충격은
# 주문금액의 제곱근에 비례하고, 전체 비용의 15% 남짓이다). 두세 번이면 수렴한다.
EQUITY_PASSES = 3


def _simulate(
    weights: pd.DataFrame,
    price_by_ticker: dict[str, pd.DataFrame],
    cost_model: CostModel,
    initial_capital: float,
) -> PortfolioResult:
    """체결된 비중이 주어졌을 때의 손익. 이 아래로는 신호도 전략도 모른다."""
    index = weights.index
    tickers = list(weights.columns)

    daily_returns = pd.DataFrame(index=index, columns=tickers, dtype=float)
    for t in tickers:
        df = price_by_ticker[t].reindex(index)
        # 거래정지 등으로 종가가 비는 날은 수익률 0으로 두고, 재개된 날 그동안의 변동을
        # 한꺼번에 인식한다(ffill 후 변화율).
        daily_returns[t] = df["close"].ffill().pct_change(fill_method=None).fillna(0.0)

    weight_change = weights.diff()
    weight_change.iloc[0] = weights.iloc[0]
    traded = weight_change.abs()

    gross_return = (weights * daily_returns).sum(axis=1)

    # 증권거래세는 매도할 때만 붙으므로, 비중이 줄어든 부분(음의 변화)에만 적용한다.
    sell_weight = (-weight_change).clip(lower=0.0)
    tax_drag = sell_weight.sum(axis=1) * cost_model.sell_tax_rate(index)

    frames = {t: price_by_ticker[t].reindex(index) for t in tickers}

    # 주문은 '어제 종가 기준 평가금액'으로 낸다. 초기자본으로 고정하면 자산이 불어난
    # 뒤의 주문 크기를 과소평가해 시장충격이 실제보다 싸게 잡힌다.
    order_base = pd.Series(initial_capital, index=index)
    cost_drag = pd.Series(0.0, index=index)

    for _ in range(EQUITY_PASSES):
        order_value = traded.mul(order_base, axis=0)

        cost_rate = pd.DataFrame(index=index, columns=tickers, dtype=float)
        for t in tickers:
            cost_rate[t] = cost_model.total_cost_rate(frames[t], order_value[t])

        cost_drag = (cost_rate * traded).sum(axis=1) + tax_drag
        equity = initial_capital * (1 + gross_return - cost_drag).cumprod()
        order_base = equity.shift(1).fillna(initial_capital)

    # 마지막 pass의 cost_drag는 그 직전 order_base로 계산된 값이므로, 지불액도 같은
    # 기준으로 환산해야 한다. 한 번 더 돌려 둘을 맞춘다.
    order_value = traded.mul(order_base, axis=0)
    cost_rate = pd.DataFrame(index=index, columns=tickers, dtype=float)
    for t in tickers:
        cost_rate[t] = cost_model.total_cost_rate(frames[t], order_value[t])
    cost_drag = (cost_rate * traded).sum(axis=1) + tax_drag

    net_return = gross_return - cost_drag
    equity_curve = initial_capital * (1 + net_return).cumprod()

    return PortfolioResult(
        equity_curve=equity_curve,
        returns=net_return,
        weights=weights,
        trades=weight_change,
        cost_paid=cost_drag * order_base,
    )
