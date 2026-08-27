"""
지금까지 나온 신호들을 '롱온리 관점'에서 다시 본다.

공매도를 하지 않기로 했으므로, IC(전체 순위상관)는 더 이상 우리가 최적화할 지표가
아니다. IC는 아래쪽 순위가 정확해도 올라가는데, 그건 공매도를 해야 쓸 수 있다.
롱온리에서 중요한 건 **상위 분위가 유니버스 평균보다 나은가** 하나뿐이다.

비교 기준을 '유니버스 동일가중 평균'으로 잡는 이유: 신호를 아예 안 쓰고 전 종목을
똑같이 사는 것이 가장 정직한 대안이기 때문이다. 흔히 쓰는 '상위-하위 스프레드'는
공매도를 해야 얻는 값이라 롱온리 성과를 부풀린다. 둘 다 찍어서 차이를 보여준다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import phase1
from src.data_loader.krx_openapi import build_close_panel, build_panel
from src.data_loader.krx_panel import trading_dates
from src.data_loader.universe import market_cap_universe_mask
from src.features.multi_source import build_price_factors
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


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def analyze(label: str, signal: pd.DataFrame, close: pd.DataFrame, universe: pd.DataFrame) -> None:
    fwd = close.pct_change(HORIZON).shift(-HORIZON)

    qr = quantile_forward_returns(
        signal, fwd, universe, n_quantiles=N_QUANTILES, sample_every=HORIZON
    )
    if qr.empty:
        print(f"\n[{label}] 분위를 만들 수 있는 날이 없음")
        return

    show(f"[{label}] 분위별 {HORIZON}일 수익률", summarize_quantiles(qr, PERIODS_PER_YEAR))

    edge = long_only_edge(qr, PERIODS_PER_YEAR)
    print(f"  단조성(분위 vs 수익 순위상관): {monotonicity(qr):.2f}")
    print(f"  롱온리 초과(상위분위 - 유니버스평균): 연 {edge['상위분위 초과(연율화)']:.2%}"
          f" (t {edge['상위분위 초과 t-stat']:.2f})")
    print(f"  참고: 상하위 스프레드(공매도 필요): 연 {edge['상하위 스프레드(연율화)']:.2%}"
          f" (t {edge['상하위 스프레드 t-stat']:.2f})")


def main() -> None:
    for label, start, end in [
        ("인샘플", phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END),
        ("아웃오브샘플", phase1.OUT_OF_SAMPLE_START, phase1.OUT_OF_SAMPLE_END),
    ]:
        warmup = (pd.Timestamp(start) - pd.Timedelta(400, unit="D")).strftime("%Y-%m-%d")
        dates = trading_dates(warmup, end)
        eval_dates = dates[dates >= pd.Timestamp(start)]

        close = build_close_panel(dates)
        volume = build_panel("volume", dates)
        market_cap = build_panel("market_cap", dates)
        universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

        # 저변동성: 변동성이 낮을수록 유리하므로 부호를 뒤집어 '클수록 유리'로 통일
        volatility = build_price_factors(close, volume)["volatility_60"]
        signal = -volatility

        print("\n" + "=" * 70)
        print(f"{label} ({start} ~ {end})")
        print("=" * 70)
        analyze(
            "저변동성",
            signal.loc[eval_dates],
            close.loc[eval_dates],
            universe.loc[eval_dates],
        )


if __name__ == "__main__":
    main()
