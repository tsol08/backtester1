"""
저변동성 롱온리 전략 백테스트.

IC 분석은 "신호가 미래 수익률의 순위를 맞히는가"만 말해준다. 실제로 돈이 되는지는
별개 문제다 — 거래할 때마다 수수료·슬리피지·시장충격·증권거래세가 나가고, IC 0.07은
"두 종목 중 어느 쪽이 오를지 52% 확률로 맞힌다" 정도의 얇은 우위이기 때문이다.
이 스크립트가 답하려는 건 그 얇은 우위가 비용을 내고도 남느냐다.

규칙:
- 20거래일마다 리밸런싱
- 그 시점 시가총액 상위 200종목 중, 최근 60일 변동성이 가장 낮은 N종목을 동일가중 보유
- 리밸런싱 사이에는 그대로 들고 간다

과적합을 피하기 위해 파라미터는 앞선 분석에서 이미 정해진 값을 그대로 쓴다
(변동성 60일, 리밸런싱 20일, 유니버스 200종목). 보유 종목 수만 몇 가지로 나눠보는데,
이건 튜닝이 아니라 "집중할수록 신호는 진해지지만 비용·리스크가 커진다"는 트레이드오프가
어디서 뒤집히는지 보기 위한 것이다. 인샘플/아웃오브샘플을 나눠 보고한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import phase1
from src.costs.cost_model import CostModel
from src.data_loader.krx_openapi import build_close_panel, build_panel
from src.data_loader.krx_panel import trading_dates
from src.data_loader.universe import market_cap_universe_mask
from src.features.multi_source import build_price_factors

TOP_K = 200
REBALANCE_EVERY = 20
WARMUP_DAYS = 400
PORTFOLIO_SIZES = [20, 40, 60]


def build_holdings(
    volatility: pd.DataFrame, universe: pd.DataFrame, n_holdings: int
) -> pd.DataFrame:
    """
    리밸런싱일마다 변동성 하위 n_holdings 종목을 동일가중으로 담고,
    다음 리밸런싱까지 그 비중을 유지한다.

    t일에 관측된 변동성으로 t일에 비중을 정하지만, 실제 체결은 엔진이 t+1로 미룬다
    (portfolio_engine의 signal.shift(1)). 즉 종가를 보고 그 종가에 사는 일은 없다.
    """
    weights = pd.DataFrame(0.0, index=volatility.index, columns=volatility.columns)
    candidates = volatility.where(universe)

    rebalance_dates = volatility.index[::REBALANCE_EVERY]
    for i, date in enumerate(rebalance_dates):
        row = candidates.loc[date].dropna()
        if len(row) < n_holdings:
            continue

        picked = row.nsmallest(n_holdings).index
        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None

        window = weights.index >= date
        if end is not None:
            window &= weights.index < end
        weights.loc[window, picked] = 1.0 / n_holdings

    return weights


def backtest(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    cost_model: CostModel,
    initial_capital: float,
) -> dict:
    """
    비중 패널을 받아 비용 차감 후 자산곡선을 계산한다.

    portfolio_engine은 종목별 dict를 받는 구조라 수백 종목에는 무겁다. 여기서는
    같은 계산(t+1 체결, 비중변화에 비용 부과, 매도분에 거래세)을 패널 연산으로 한다.
    """
    positions = weights.shift(1).fillna(0.0)
    daily_returns = close.pct_change().fillna(0.0)

    weight_change = positions.diff()
    weight_change.iloc[0] = positions.iloc[0]

    order_value = weight_change.abs() * initial_capital

    trading_value = (close * volume).rolling(cost_model.impact_window).mean()
    participation = (order_value / trading_value).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rate = (
        cost_model.commission_rate
        + cost_model.slippage_rate
        + cost_model.impact_coefficient * np.sqrt(participation.clip(lower=0))
    )

    sell_weight = (-weight_change).clip(lower=0.0)
    tax_drag = sell_weight.sum(axis=1) * cost_model.sell_tax_rate(positions.index)

    cost_drag = (rate * weight_change.abs()).sum(axis=1) + tax_drag

    gross_return = (positions * daily_returns).sum(axis=1)
    net_return = gross_return - cost_drag

    equity = initial_capital * (1 + net_return).cumprod()
    years = len(net_return) / 252

    peak = equity.cummax()
    drawdown = equity / peak - 1

    return {
        "CAGR": (equity.iloc[-1] / initial_capital) ** (1 / years) - 1,
        "Sharpe": net_return.mean() / net_return.std() * np.sqrt(252)
        if net_return.std() > 0
        else 0.0,
        "MDD": drawdown.min(),
        "총비용(만원)": cost_drag.sum() * initial_capital / 10_000,
        "연회전율": weight_change.abs().sum(axis=1).sum() / 2 / years,
        "비용전 CAGR": (initial_capital * (1 + gross_return).cumprod().iloc[-1] / initial_capital)
        ** (1 / years)
        - 1,
        "최종자산(만원)": equity.iloc[-1] / 10_000,
    }


def market_benchmark(
    close: pd.DataFrame,
    market_cap: pd.DataFrame,
    universe: pd.DataFrame,
    initial_capital: float,
    weighting: str = "cap",
) -> dict:
    """
    비교 기준: 유니버스 전체를 사서 그냥 들고 있기 (거래비용 미반영).

    가중방식을 두 가지로 나눈다. 전략은 동일가중인데 시총가중 지수와만 비교하면
    '저변동성 효과'와 '동일가중 효과'가 섞여서, 무엇 때문에 차이가 났는지 알 수 없다.
    동일가중 기준을 같이 두면 저변동성 종목선택 자체의 기여만 떼어볼 수 있다.
    """
    if weighting == "cap":
        weights = market_cap.where(universe)
    else:
        weights = universe.astype(float).replace(0.0, np.nan)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)

    gross_return = (weights.shift(1) * close.pct_change().fillna(0.0)).sum(axis=1)
    equity = initial_capital * (1 + gross_return).cumprod()
    years = len(gross_return) / 252
    drawdown = equity / equity.cummax() - 1

    return {
        "CAGR": (equity.iloc[-1] / initial_capital) ** (1 / years) - 1,
        "Sharpe": gross_return.mean() / gross_return.std() * np.sqrt(252),
        "MDD": drawdown.min(),
        "총비용(만원)": 0.0,
        "연회전율": 0.0,
        "비용전 CAGR": (equity.iloc[-1] / initial_capital) ** (1 / years) - 1,
        "최종자산(만원)": equity.iloc[-1] / 10_000,
    }


def run_period(label: str, start: str, end: str) -> None:
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(WARMUP_DAYS, unit="D")).strftime("%Y-%m-%d")
    dates = trading_dates(warmup_start, end)
    eval_dates = dates[dates >= pd.Timestamp(start)]

    close = build_close_panel(dates)
    volume = build_panel("volume", dates)
    market_cap = build_panel("market_cap", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

    volatility = build_price_factors(close, volume)["volatility_60"]

    cost_model = CostModel()
    capital = phase1.INITIAL_CAPITAL

    rows = {}
    for n in PORTFOLIO_SIZES:
        weights = build_holdings(volatility, universe, n).loc[eval_dates]
        rows[f"저변동성 {n}종목"] = backtest(
            weights, close.loc[eval_dates], volume.loc[eval_dates], cost_model, capital
        )

    for benchmark_label, weighting in [("시총가중 보유(기준)", "cap"), ("동일가중 보유(기준)", "equal")]:
        rows[benchmark_label] = market_benchmark(
            close.loc[eval_dates],
            market_cap.loc[eval_dates],
            universe.loc[eval_dates],
            capital,
            weighting=weighting,
        )

    table = pd.DataFrame(rows).T
    print(f"\n===== {label} ({start} ~ {end}, {len(eval_dates)}거래일) =====")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def main() -> None:
    run_period("인샘플", phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END)
    run_period("아웃오브샘플", phase1.OUT_OF_SAMPLE_START, phase1.OUT_OF_SAMPLE_END)


if __name__ == "__main__":
    main()
