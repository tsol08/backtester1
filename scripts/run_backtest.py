"""
전략 하나를 처음부터 끝까지 돌려서 성과를 낸다.

연구 스크립트(run_pead_test.py 등)는 "이 신호가 실재하는가"를 묻는다. 이쪽은
"그래서 이걸 굴렸으면 어떻게 됐는가"를 묻는다. 둘은 다른 질문이고, 후자에는
체결 시점·거래비용·현금 대기 구간이 전부 들어간다.

전략은 --strategy로 갈아끼운다. 도구는 전략을 모르고, 전략은 체결을 모른다.

사용:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --strategy pead-cash --capital 5000
    python scripts/run_backtest.py --strategy universe
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import settings
from src.costs.cost_model import CostModel
from src.data_loader.panels import Panels
from src.portfolio.portfolio_engine import run_weighted_backtest
from src.reporting.metrics import cagr, max_drawdown, sharpe_ratio
from src.strategy.base import EqualWeightUniverse, build_weight_panel
from src.strategy.pead import PeadStrategy

STRATEGIES = {
    "pead": lambda: PeadStrategy(hold_universe_when_idle=True),
    "pead-cash": lambda: PeadStrategy(hold_universe_when_idle=False),
    "universe": EqualWeightUniverse,
}


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def annual_turnover(trades: pd.DataFrame) -> float:
    """연 단방향 회전율. 비중 변화 절댓값 합의 절반이 실제로 사고판 규모다."""
    one_way = trades.abs().sum(axis=1) / 2
    years = len(trades) / 252
    return float(one_way.sum() / years) if years else float("nan")


def window_returns(returns: pd.Series, schedule: list[pd.Timestamp]) -> pd.Series:
    """리밸런싱 구간별 복리수익률. 구간이 겹치지 않아 t-stat이 부풀지 않는다."""
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


def restrict(result, start: pd.Timestamp, capital: float):
    """
    평가 구간만 남기고 자산곡선을 그 시점 기준으로 다시 세운다.

    패널은 워밍업 때문에 리밸런싱 격자보다 앞에서 시작한다. 그 구간은 포지션이
    없어 수익률이 0인데, 그대로 두면 연도별 표가 0으로 채워지고 CAGR도 희석된다.
    """
    window = result.returns.index >= start
    returns = result.returns[window]
    return SimpleNamespace(
        returns=returns,
        equity_curve=capital * (1 + returns).cumprod(),
        trades=result.trades[window],
        cost_paid=result.cost_paid[window],
    )


def summarize(label: str, result, capital: float) -> dict:
    return {
        "구분": label,
        "CAGR": cagr(result.equity_curve),
        "Sharpe": sharpe_ratio(result.returns),
        "MDD": max_drawdown(result.equity_curve),
        "연회전율": annual_turnover(result.trades),
        "총비용(자본대비)": float(result.cost_paid.sum()) / capital,
        "최종자산(억)": float(result.equity_curve.iloc[-1]) / 1e8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="pead", choices=sorted(STRATEGIES))
    parser.add_argument("--end", default=None, help="평가 종료일 (기본: 수집된 마지막 날)")
    parser.add_argument("--capital", type=float, default=settings.INITIAL_CAPITAL / 1e4,
                        help="초기자본, 만원 단위 (기본 10000 = 1억)")
    args = parser.parse_args()

    capital = args.capital * 1e4

    print(f"데이터 로딩...", flush=True)
    # 구간을 잘라 싣지 않는다. 리밸런싱 격자가 앵커(2018-01-02) 기준이라 패널을
    # 어디서부터 실어도 같은 날짜에 갈아타지만, 성과 비교의 출발점이 달라지면
    # 벤치마크와 어긋난다. 부분 구간은 아래 연도별 표로 본다.
    panels = Panels.load(end=args.end)

    strategy = STRATEGIES[args.strategy]()
    benchmark = EqualWeightUniverse()
    for engine in (strategy, benchmark):
        engine.prepare(panels)

    print(f"\n{'=' * 74}")
    print(f"전략: {strategy.name}")
    print(f"기간: {panels.dates[0].date()} ~ {panels.dates[-1].date()}"
          f" ({len(panels.dates)}거래일)")
    print(f"자본: {capital / 1e8:,.2f}억원")
    print("=" * 74)

    if isinstance(strategy, PeadStrategy):
        latest = strategy.latest_signal_date()
        available = strategy.top_quantile.loc[strategy.rebalance_dates()].any(axis=1)
        print(f"\n분위가 성립한 리밸런싱 시점: {int(available.sum())} / {len(available)}"
              f" ({available.mean():.0%})")
        print(f"신호가 살아있는 마지막 날: {latest.date() if latest is not None else '없음'}")
        stale = (panels.dates[-1] - latest).days if latest is not None else None
        if stale is not None and stale > 120:
            print(f"  ** 경고: {stale}일째 신호 없음. DART 재무가 최신인지 확인할 것"
                  " (scripts/collect_dart_fundamentals.py)")

    schedule = strategy.rebalance_dates()
    results = {}
    for label, engine in (("전략", strategy), ("벤치마크", benchmark)):
        weights = build_weight_panel(engine, panels)
        held = sorted(weights.columns[(weights != 0).any(axis=0)])
        raw = run_weighted_backtest(
            weights, panels.price_frames(held), CostModel(), initial_capital=capital
        )
        results[label] = restrict(raw, schedule[0], capital)

    print(f"평가 구간: {schedule[0].date()} ~ {panels.dates[-1].date()}"
          f" (리밸런싱 {len(schedule)}회)")

    show("[성과]", pd.DataFrame(
        [summarize(f"{label}: {engine.name}", results[label], capital)
         for label, engine in (("전략", strategy), ("벤치마크", benchmark))]
    ).set_index("구분"))

    print("\n" + "=" * 74)
    print("연도별 수익률")
    print("=" * 74)
    yearly = pd.DataFrame({
        label: result.returns.groupby(result.returns.index.year).apply(lambda r: (1 + r).prod() - 1)
        for label, result in results.items()
    })
    yearly["초과"] = yearly["전략"] - yearly["벤치마크"]
    show("", yearly)

    print("\n" + "=" * 74)
    print("초과수익 (비겹침 리밸런싱 구간)")
    print("=" * 74)
    excess = (
        window_returns(results["전략"].returns, schedule)
        - window_returns(results["벤치마크"].returns, schedule)
    ).dropna()
    periods_per_year = 252 / getattr(strategy, "horizon", settings.FORWARD_HORIZON)
    print(f"\n  관측 {len(excess)}개, 구간당 {excess.mean():.2%}"
          f" -> 연 {(1 + excess.mean()) ** periods_per_year - 1:.2%}")
    print(f"  t-stat {t_stat(excess):.2f}"
          f"  ({'기준 1.96 통과' if abs(t_stat(excess)) > 1.96 else '기준 1.96 미달'})")

    print("\n" + "=" * 74)
    print("최근 리밸런싱")
    print("=" * 74)
    weights = build_weight_panel(strategy, panels)
    rows = []
    for date in schedule[-6:]:
        row = weights.loc[date]
        holding = row[row > 0]
        rows.append({
            "날짜": date.date(),
            "종목수": len(holding),
            "종목당 비중": holding.mean() if len(holding) else 0.0,
            "신호": "상위분위" if getattr(strategy, "signal_available", lambda d: True)(date)
                    else "대기",
        })
    show("", pd.DataFrame(rows).set_index("날짜"))


if __name__ == "__main__":
    main()
