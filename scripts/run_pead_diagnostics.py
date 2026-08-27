"""
PEAD 결과를 깨보는 진단.

run_pead_test.py에서 상위분위 롱온리 초과가 연 13.41%(t 2.42), 단조성 0.90으로
사전 기준(|t| > 1.96)을 넘었다. 이 프로젝트에서 그럴듯해 보였다가 무너진 신호가
네 개다 — 반전, 저변동성, 내부자, 소형주. 같은 잣대를 여기에도 댄다.

특히 설명이 필요한 대목이 있다: **IC는 유의하지 않은데(t 0.71) 분위는 유의하다(t 2.42).**
저변동성 때와 정반대 패턴이다. 신호가 극단에만 몰려 있다면 이론(큰 서프라이즈만
표류한다)과 맞지만, 소수 종목의 우연일 수도 있다.

확인하는 것:
1) 구간을 반으로 갈라도 유지되는가
2) 규모/변동성의 대리변수는 아닌가
3) 표류 창(30/60/90일) 선택에 따라 결과가 뒤집히는가 — 60일이 운 좋은 선택이었는지
4) 분위당 종목 수가 결론을 낼 만한가
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import settings
from src.data_loader.dart_loader import load_corp_codes, load_fundamentals_bulk
from src.data_loader.krx_openapi import build_close_panel, build_panel, cached_trading_dates
from src.data_loader.universe import market_cap_universe_mask
from src.features.earnings_surprise import build_sue_panel
from src.features.multi_source import build_price_factors, build_size_factor, neutralize
from src.research.quantile_analysis import (
    assign_quantiles,
    long_only_edge,
    monotonicity,
    quantile_forward_returns,
)

TOP_K = settings.UNIVERSE_TOP_K
HORIZON = settings.FORWARD_HORIZON
N_QUANTILES = settings.N_QUANTILES
PERIODS_PER_YEAR = 252 / HORIZON
START, END = "2018-01-01", "2026-08-26"


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def evaluate(signal, fwd, universe, label: str) -> dict:
    qr = quantile_forward_returns(
        signal, fwd, universe, n_quantiles=N_QUANTILES, sample_every=HORIZON
    )
    if qr.empty or qr.shape[1] < N_QUANTILES:
        return {"구분": label, "관측": 0}
    edge = long_only_edge(qr, PERIODS_PER_YEAR)
    return {
        "구분": label,
        "관측": len(qr),
        "상위분위(연)": (1 + qr[qr.columns[-1]].mean()) ** PERIODS_PER_YEAR - 1,
        "롱온리초과(연)": edge["상위분위 초과(연율화)"],
        "t-stat": edge["상위분위 초과 t-stat"],
        "단조성": monotonicity(qr),
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

    fwd = close.pct_change(HORIZON).shift(-HORIZON)

    print("=" * 74)
    print("1) 표류 창 선택에 민감한가 (60일이 운 좋은 선택이었나)")
    print("=" * 74)

    panels = {}
    rows = []
    for window in (30, 60, 90, None):
        sue = build_sue_panel(fundamentals, dates, drift_window=window)
        aligned = sue.reindex_like(close).where(universe)
        panels[window] = aligned
        label = f"{window}일" if window else "제한 없음"
        row = evaluate(aligned, fwd, universe, label)
        row["일평균 종목"] = aligned.notna().sum(axis=1).mean()
        rows.append(row)

    show("[표류 창별]", pd.DataFrame(rows).set_index("구분"))

    base = panels[60]

    print("\n" + "=" * 74)
    print("2) 구간을 반으로 갈라도 유지되는가")
    print("=" * 74)

    midpoint = len(dates) // 2
    rows = [evaluate(base, fwd, universe, "전체")]
    for label, window in [("전반부", dates[:midpoint]), ("후반부", dates[midpoint:])]:
        rows.append(
            evaluate(base.loc[window], fwd.loc[window], universe.loc[window], label)
        )
    show("[표류 창 60일]", pd.DataFrame(rows).set_index("구분"))

    print("\n" + "=" * 74)
    print("3) 규모/변동성의 대리변수는 아닌가")
    print("=" * 74)

    log_cap = build_size_factor(market_cap)["log_market_cap"]
    volatility = build_price_factors(close, volume)["volatility_60"]

    rows = [evaluate(base, fwd, universe, "원본")]
    for label, control in [("규모중립화", log_cap), ("변동성중립화", volatility)]:
        rows.append(
            evaluate(neutralize(base, control.where(universe)), fwd, universe, label)
        )
    show("[표류 창 60일]", pd.DataFrame(rows).set_index("구분"))

    for label, control in [("로그시총", log_cap), ("변동성", volatility)]:
        corr = base.corrwith(control.where(universe), axis=1, method="spearman").dropna().mean()
        print(f"  SUE vs {label} 평균 순위상관: {corr:.2f}")

    print("\n" + "=" * 74)
    print("4) 분위당 종목 수가 결론을 낼 만한가")
    print("=" * 74)

    quantiles = assign_quantiles(base, universe, N_QUANTILES).iloc[::HORIZON]
    counts = quantiles.apply(lambda row: row.value_counts(), axis=1).mean()
    print(f"\n분위당 평균 종목 수: {counts.mean():.1f}")
    print(f"  (분위별: {', '.join(f'Q{int(q)+1}={counts[q]:.0f}' for q in sorted(counts.index))})")


if __name__ == "__main__":
    main()
