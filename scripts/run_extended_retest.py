"""
표본을 2010년까지 늘린 뒤, 검정력이 부족해 보류했던 판정들을 다시 본다.

경위: 미국 100년 데이터로 확인한 결과, 모멘텀처럼 견고한 팩터조차 5년 구간에서는
37%만 검출됐다. 즉 한국 5년 표본으로 내린 롱온리 판정들은 "신호가 없다"가 아니라
"이 표본으로는 알 수 없다"였다. 2010~2016을 추가 수집해 비겹침 관측을 61개에서
204개로 늘렸으므로, 이제 제대로 물을 수 있다.

두 가지를 순서대로 한다:
1) 늘어난 표본의 검출 하한 재측정 — 무엇을 볼 수 있게 됐는지 먼저 알아야 한다
2) 저변동성 롱온리 재판정 — 보류했던 핵심 질문

표본 분할: 2010~2017을 '신규 구간'으로 따로 본다. 이 구간은 이번에 처음 수집해서
한 번도 들여다본 적이 없다. 2018년 이후는 이미 여러 번 분석한 구간이라, 거기서
나오는 숫자는 선택 편향에서 자유롭지 않다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader.krx_openapi import build_close_panel, build_panel, cached_trading_dates
from src.data_loader.universe import market_cap_universe_mask
from src.features.multi_source import build_price_factors
from src.research.power_analysis import minimum_detectable_ic
from src.research.quantile_analysis import (
    long_only_edge,
    monotonicity,
    quantile_forward_returns,
    summarize_quantiles,
)

TOP_K = 200
HORIZON = 20
N_QUANTILES = 5
PERIODS_PER_YEAR = 252 / HORIZON

FULL_START, FULL_END = "2010-01-01", "2026-08-26"
PERIODS = [
    ("신규 구간 2010~2017 (미탐색)", "2010-01-01", "2017-12-31"),
    ("기탐색 구간 2018~2026", "2018-01-01", "2026-08-26"),
    ("전체 2010~2026", FULL_START, FULL_END),
]

CANDIDATE_ICS = [0.01, 0.02, 0.03, 0.05]


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def main() -> None:
    dates = cached_trading_dates(FULL_START, FULL_END)
    print(f"거래일 {len(dates)}일 ({dates.min().date()} ~ {dates.max().date()})", flush=True)

    close = build_close_panel(dates)
    volume = build_panel("volume", dates)
    market_cap = build_panel("market_cap", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

    volatility = build_price_factors(close, volume)["volatility_60"]
    low_vol = -volatility  # 클수록 유리하게 방향 통일
    fwd = close.pct_change(HORIZON).shift(-HORIZON)

    print("\n" + "=" * 78)
    print("1) 늘어난 표본으로 무엇을 검출할 수 있는가")
    print("=" * 78)

    table = minimum_detectable_ic(
        fwd,
        universe,
        horizon=HORIZON,
        candidates=CANDIDATE_ICS,
        n_trials=150,
        thresholds=(1.96, 2.87),
    )
    print(f"\n비겹침 관측 {len(dates) // HORIZON}개 (이전 인샘플 61개)")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.to_string())

    print("\n" + "=" * 78)
    print("2) 저변동성 롱온리 재판정")
    print("=" * 78)

    rows = []
    for label, start, end in PERIODS:
        window = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
        qr = quantile_forward_returns(
            low_vol.loc[window],
            fwd.loc[window],
            universe.loc[window],
            n_quantiles=N_QUANTILES,
            sample_every=HORIZON,
        )
        if qr.empty:
            continue

        show(f"[{label}] 분위별 {HORIZON}일 수익률", summarize_quantiles(qr, PERIODS_PER_YEAR))
        edge = long_only_edge(qr, PERIODS_PER_YEAR)
        rows.append(
            {
                "구간": label,
                "관측": len(qr),
                "상위분위(연)": (1 + qr[qr.columns[-1]].mean()) ** PERIODS_PER_YEAR - 1,
                "롱온리초과(연)": edge["상위분위 초과(연율화)"],
                "t-stat": edge["상위분위 초과 t-stat"],
                "단조성": monotonicity(qr),
            }
        )

    show("[요약]", pd.DataFrame(rows).set_index("구간"))


if __name__ == "__main__":
    main()
