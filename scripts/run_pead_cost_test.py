"""
PEAD가 거래비용을 내고도 남는가.

run_pead_test.py에서 상위분위 롱온리 초과가 연 13.41%(t 2.42)로 나왔고, 진단
(run_pead_diagnostics.py)도 통과했다. 남은 관문은 둘이다.

1) **비용**: 이 프로젝트에는 전례가 있다. 저변동성은 비용 전 초과 1.88%p 중 74%가
   비용으로 사라졌다. 분위 초과는 마찰이 없는 세계의 값이다.

2) **실제로 굴릴 수 있는 형태인가**: 분위 분석은 '분위가 성립하는 날'만 골라서 재고,
   그 값을 1년 내내 그 상태였던 것처럼 연율화한다. SUE는 공시 후 60일만 유효하므로
   한국 공시 일정상 신호가 통째로 비는 계절이 있다. 비어 있는 동안 무엇을 들고
   있었는지에 따라 실제 수익은 달라진다.

**사전에 정한 것** (돌려보고 좋은 걸 고르지 않기 위해 먼저 못박는다):

  전략 A(현금 대기)     분위가 성립하면 상위분위 동일가중, 아니면 현금
  전략 B(유니버스 대기)  분위가 성립하면 상위분위 동일가중, 아니면 유니버스 동일가중

  A는 분위 분석을 글자 그대로 옮긴 것이고, B는 실제로 굴린다면 택할 형태다. 둘 다
  각자의 벤치마크(같은 날 유니버스 동일가중)와 비교하며, 결과가 어느 쪽이든 둘 다 적는다.

  판정: 비용 차감 후 초과수익의 t가 1.96을 넘고 크기가 의미 있는가.
        관측은 분위 분석과 같은 비겹침 20일 구간이다.

**엔진의 단순화 두 가지**(해석에 필요하므로 밝혀둔다):
  - 보유 중 가격이 움직여 생기는 비중 표류를 되돌리는 거래에는 비용을 물리지 않는다.
    20일 들고 있다가 통째로 재구성하는 방식이라 실제로도 중간 리밸런싱이 없다.
  - 체결은 신호 다음 날이다. 분위 분석이 재는 구간(d ~ d+20)과 하루 어긋난다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import settings
from src.costs.cost_model import MAX_PARTICIPATION, CostModel
from src.data_loader.dart_loader import load_corp_codes, load_fundamentals_bulk
from src.data_loader.krx_openapi import build_close_panel, build_panel, cached_trading_dates
from src.data_loader.universe import market_cap_universe_mask
from src.features.earnings_surprise import build_sue_panel
from src.portfolio.portfolio_engine import run_portfolio_backtest
from src.reporting.metrics import cagr, max_drawdown, sharpe_ratio
from src.research.quantile_analysis import (
    assign_quantiles,
    long_only_edge,
    quantile_forward_returns,
)

TOP_K = settings.UNIVERSE_TOP_K
HORIZON = settings.FORWARD_HORIZON
N_QUANTILES = settings.N_QUANTILES
DRIFT_WINDOW = 60
PERIODS_PER_YEAR = 252 / HORIZON
CAPITAL = settings.INITIAL_CAPITAL
START, END = "2018-01-01", "2026-08-26"

FREE = CostModel(
    commission_rate=0.0, slippage_rate=0.0, impact_coefficient=0.0, apply_transaction_tax=False
)
REAL = CostModel()

STRATEGY_A = "전략A(현금 대기)"
BENCHMARK_A = "벤치마크A(같은 날 유니버스)"
STRATEGY_B = "전략B(유니버스 대기)"
BENCHMARK_B = "벤치마크B(상시 유니버스)"


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def hold_until_next_rebalance(
    members_by_date: dict[pd.Timestamp, pd.Index],
    schedule: list[pd.Timestamp],
    dates: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    """
    리밸런싱일에 고른 종목을 다음 리밸런싱일 직전까지 들고 있는 보유 패널.

    보유 기간을 정하는 것은 members_by_date의 키가 아니라 schedule이다. 신호가 있는
    날만 키로 넘기면 "다음 신호가 뜰 때까지" 들고 있게 되어, 20일 보유를 잰 분위
    분석과 전혀 다른 전략(길면 몇 달 보유)이 된다.
    """
    target = pd.DataFrame(False, index=dates, columns=columns)
    for i, start in enumerate(schedule):
        members = members_by_date.get(start, pd.Index([]))
        if len(members) == 0:
            continue
        window = dates >= start
        if i + 1 < len(schedule):
            window &= dates < schedule[i + 1]
        target.loc[window, members] = True
    return target


def backtest(
    target: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    cost_model: CostModel,
    capital: float = CAPITAL,
):
    tickers = sorted(target.columns[target.any(axis=0)])
    price_by = {t: pd.DataFrame({"close": close[t], "volume": volume[t]}) for t in tickers}
    signal_by = {t: target[t].astype(float) for t in tickers}
    return run_portfolio_backtest(price_by, signal_by, cost_model, initial_capital=capital)


def annual_turnover(trades: pd.DataFrame) -> float:
    """연 단방향 회전율. 비중 변화 절댓값 합의 절반이 실제로 사고판 규모다."""
    one_way = trades.abs().sum(axis=1) / 2
    years = len(trades) / 252
    return float(one_way.sum() / years) if years else float("nan")


def window_returns(returns: pd.Series, schedule: list[pd.Timestamp]) -> pd.Series:
    """리밸런싱 구간별 복리수익률. 구간이 겹치지 않으므로 t-stat이 부풀지 않는다."""
    rows = {}
    for i, start in enumerate(schedule):
        window = returns.index > start
        if i + 1 < len(schedule):
            window &= returns.index <= schedule[i + 1]
        chunk = returns[window]
        if len(chunk):
            rows[start] = (1 + chunk).prod() - 1
    return pd.Series(rows)


def t_stat(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 2 or series.std() == 0:
        return 0.0
    return float(series.mean() / (series.std() / np.sqrt(len(series))))


def performance(label: str, result) -> dict:
    return {
        "구분": label,
        "CAGR": cagr(result.equity_curve),
        "Sharpe": sharpe_ratio(result.returns),
        "MDD": max_drawdown(result.equity_curve),
        "연회전율": annual_turnover(result.trades),
        "총비용(자본대비)": float(result.cost_paid.sum()) / CAPITAL,
    }


def main() -> None:
    dates = cached_trading_dates(START, END)
    close = build_close_panel(dates)
    volume = build_panel("volume", dates)
    market_cap = build_panel("market_cap", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

    members = sorted(universe.columns[universe.any(axis=0)])
    dart_codes = set(load_corp_codes()["stock_code"])
    fundamentals = load_fundamentals_bulk(
        [t for t in members if t in dart_codes], 2015, 2024, verbose=False
    )
    sue = build_sue_panel(fundamentals, dates, drift_window=DRIFT_WINDOW).reindex_like(close)
    signal = sue.where(universe)

    quantiles = assign_quantiles(signal, universe, N_QUANTILES)
    top_quantile = quantiles == N_QUANTILES - 1
    tradeable = close.notna()

    rebalance_dates = list(dates[::HORIZON])
    universe_picks, invested = {}, {}
    for date in rebalance_dates:
        pool = universe.loc[date] & tradeable.loc[date]
        universe_picks[date] = pool.index[pool]
        picks = top_quantile.loc[date] & tradeable.loc[date]
        if picks.any():
            invested[date] = picks.index[picks]

    schedule = sorted(invested)

    print("=" * 74)
    print("1) 이 전략은 실제로 어떤 모습인가")
    print("=" * 74)

    sizes = pd.Series({d: len(v) for d, v in invested.items()})
    print(
        f"\n리밸런싱 시점 {len(rebalance_dates)}개 중 분위가 성립한 날: {len(schedule)}개"
        f" ({len(schedule) / len(rebalance_dates):.0%})"
    )
    print(f"성립한 날의 보유 종목 수: 평균 {sizes.mean():.1f} (최소 {sizes.min()}, 최대 {sizes.max()})")
    print(f"신호가 존재한 구간: {min(schedule).date()} ~ {max(schedule).date()}")

    coverage = signal.notna().sum(axis=1)
    by_month = coverage.groupby(coverage.index.month).mean().round(0).astype(int)
    print(f"\n유니버스 {TOP_K}종목 중 유효 SUE 보유 수 (월평균):")
    print("  " + "  ".join(f"{m:>2}월 {v:>3}" for m, v in by_month.items()))
    print("  -> 분위를 나누려면 30종목이 필요하다. 공시가 없는 계절엔 신호가 통째로 빈다.")

    # 분위 분석이 실제로 몇 구간을 쟀고 그것을 몇 배로 연율화했는지 드러낸다
    forward = close.pct_change(HORIZON, fill_method=None).shift(-HORIZON)
    qr = quantile_forward_returns(signal, forward, universe, N_QUANTILES, sample_every=HORIZON)
    edge = long_only_edge(qr, PERIODS_PER_YEAR)
    per_period = (qr[qr.columns[-1]] - qr.mean(axis=1)).dropna()
    years = (max(schedule) - min(schedule)).days / 365.25
    per_year = len(per_period) / years
    print(f"\n분위 분석의 초과: 20일 구간당 {per_period.mean():.2%}, 관측 {len(per_period)}개")
    print(f"  연 {edge['상위분위 초과(연율화)']:.2%}  <- {PERIODS_PER_YEAR:.1f}개 구간 내내 투자 상태라고 보고 연율화")
    print(
        f"  연 {(1 + per_period.mean()) ** per_year - 1:.2%}"
        f"  <- 실제 성립 빈도({per_year:.1f}구간/년)로 연율화"
    )

    # 분위 분석의 '유니버스 평균'은 SUE가 있는 종목만의 평균이다(분위에 든 종목들).
    # 롱온리 투자자가 실제로 살 수 있는 대안은 유니버스 200종목 전체이므로, 아래
    # 백테스트는 그쪽을 벤치마크로 쓴다. 재현을 대조할 때 이 차이를 알고 봐야 한다.
    whole_universe = forward.where(universe).mean(axis=1)
    versus_universe = (qr[qr.columns[-1]] - whole_universe.reindex(qr.index)).dropna()
    print(
        f"\n  같은 초과를 '유니버스 {TOP_K}종목 전체' 대비로 다시 재면 구간당"
        f" {versus_universe.mean():.2%} (t {t_stat(versus_universe):.2f})"
    )
    print("  -> SUE가 있는 종목만의 평균보다 유니버스 전체 평균이 낮았다는 뜻이다.")

    print("\n" + "=" * 74)
    print("2) 비용 전후 성과")
    print("=" * 74)

    empty = pd.Index([])
    designs = {
        STRATEGY_A: invested,
        BENCHMARK_A: {d: universe_picks[d] if d in invested else empty for d in rebalance_dates},
        STRATEGY_B: {**universe_picks, **invested},
        BENCHMARK_B: universe_picks,
    }

    results, rows = {}, []
    for label, picks in designs.items():
        target = hold_until_next_rebalance(picks, rebalance_dates, dates, close.columns)
        for tag, model in (("비용 전", FREE), ("비용 후", REAL)):
            result = backtest(target, close, volume, model)
            results[(label, tag)] = result
            rows.append(performance(f"{label} / {tag}", result))
    show("[성과]", pd.DataFrame(rows).set_index("구분"))

    print("\n" + "=" * 74)
    print("3) 초과수익은 비용을 내고도 남는가 (비겹침 20일 구간)")
    print("=" * 74)

    rows = []
    for strategy, benchmark, only_invested, note in [
        (STRATEGY_A, BENCHMARK_A, True, "투자한 구간만"),
        (STRATEGY_B, BENCHMARK_B, False, "전 구간"),
    ]:
        for tag in ("비용 전", "비용 후"):
            strategy_returns = window_returns(results[(strategy, tag)].returns, rebalance_dates)
            benchmark_returns = window_returns(results[(benchmark, tag)].returns, rebalance_dates)
            excess = (strategy_returns - benchmark_returns).dropna()
            if only_invested:
                excess = excess[excess.index.isin(schedule)]
            rows.append(
                {
                    "구분": f"{strategy} / {tag}",
                    "관측": len(excess),
                    "구간당 초과": excess.mean(),
                    "연율화": (1 + excess.mean()) ** PERIODS_PER_YEAR - 1,
                    "t-stat": t_stat(excess),
                    "기준": note,
                }
            )
    show("[초과수익]", pd.DataFrame(rows).set_index("구분"))
    print("\n  비용 전 전략A가 분위 분석의 구간당 1.00%를 재현해야 배선이 맞은 것이다.")

    print("\n" + "=" * 74)
    print("4) 비용은 어디서 나가는가 (전략B)")
    print("=" * 74)

    target_b = hold_until_next_rebalance(designs[STRATEGY_B], rebalance_dates, dates, close.columns)
    parts = {
        "수수료+슬리피지": CostModel(impact_coefficient=0.0, apply_transaction_tax=False),
        "시장충격": CostModel(commission_rate=0.0, slippage_rate=0.0, apply_transaction_tax=False),
        "증권거래세": CostModel(commission_rate=0.0, slippage_rate=0.0, impact_coefficient=0.0),
    }
    total = float(results[(STRATEGY_B, "비용 후")].cost_paid.sum())
    rows = []
    for label, model in parts.items():
        paid = float(backtest(target_b, close, volume, model).cost_paid.sum())
        rows.append({"항목": label, "자본대비": paid / CAPITAL, "비중": paid / total})
    rows.append(
        {
            "항목": "합계",
            "자본대비": sum(r["자본대비"] for r in rows),
            "비중": sum(r["비중"] for r in rows),
        }
    )
    show("[비용 분해]", pd.DataFrame(rows).set_index("항목"))
    print(f"  (전체 모델 실측 자본대비 {total / CAPITAL:.4f} - 분해 합계와 같아야 한다)")

    print("\n" + "=" * 74)
    print("5) 자본이 커지면 (시장충격은 주문 크기에 붙는다)")
    print("=" * 74)

    target_a = hold_until_next_rebalance(invested, rebalance_dates, dates, close.columns)
    rows = []
    for capital in (1e8, 1e9, 1e10, 1e11):
        result = backtest(target_a, close, volume, REAL, capital=capital)
        rows.append(
            {
                "자본": f"{capital / 1e8:,.0f}억",
                "CAGR": cagr(result.equity_curve),
                "총비용(자본대비)": float(result.cost_paid.sum()) / capital,
            }
        )
    show("[전략A 기준]", pd.DataFrame(rows).set_index("자본"))

    trades = results[(STRATEGY_A, "비용 후")].trades
    capped = 0
    for ticker in trades.columns:
        order = trades[ticker].abs() * CAPITAL
        if not (order > 0).any():
            continue
        prices = pd.DataFrame({"close": close[ticker], "volume": volume[ticker]})
        rate = REAL.participation_rate(prices, order)
        capped += int(((rate >= MAX_PARTICIPATION) & (order > 0)).sum())
    print(
        f"\n참여율 상한({MAX_PARTICIPATION:.0%})에 걸린 주문: {capped}건"
        " (거래정지 종목을 사고팔려 한 경우)"
    )

    print("\n" + "=" * 74)
    print("6) 판정이 무엇에 달려 있는가 - 시장충격 계수는 보정된 적이 없다")
    print("=" * 74)
    print(
        "\n비용 세 가지 중 수수료와 증권거래세는 고시된 값이고 슬리피지 0.05%는 통념이다."
        "\n반면 시장충격 계수 0.1은 이 프로젝트가 처음부터 들고 있던 기본값일 뿐,"
        "\n한국 시장 데이터로 맞춰본 적이 없다. 그런데 위에서 전체 비용의 45%를 차지한다."
        "\n\n참고로 문헌의 제곱근 법칙은 충격 = Y x 일간변동성 x sqrt(참여율) 꼴이다."
        "\n한국 대형주 일간변동성 2%에 Y=1이면 실효 계수는 0.02 근처가 된다."
        "\n아래는 '어느 계수부터 결론이 바뀌는가'이지, 낮은 쪽이 옳다는 주장이 아니다."
    )

    rows = []
    for coefficient in (0.0, 0.02, 0.05, 0.1):
        model = CostModel(impact_coefficient=coefficient)
        row = {"충격계수": coefficient}
        for strategy, benchmark, only_invested, tag in [
            (STRATEGY_A, BENCHMARK_A, True, "A"),
            (STRATEGY_B, BENCHMARK_B, False, "B"),
        ]:
            excess = None
            for label in (strategy, benchmark):
                target = hold_until_next_rebalance(
                    designs[label], rebalance_dates, dates, close.columns
                )
                returns = window_returns(backtest(target, close, volume, model).returns,
                                         rebalance_dates)
                excess = returns if excess is None else (excess - returns).dropna()
            if only_invested:
                excess = excess[excess.index.isin(schedule)]
            row[f"전략{tag} 연율화"] = (1 + excess.mean()) ** PERIODS_PER_YEAR - 1
            row[f"전략{tag} t"] = t_stat(excess)
        rows.append(row)
    show("[비용 후 초과수익]", pd.DataFrame(rows).set_index("충격계수"))


if __name__ == "__main__":
    main()
