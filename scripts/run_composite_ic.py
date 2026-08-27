"""
반전 + 저변동성 결합 팩터의 예측력을 검증한다.

앞선 진단(run_factor_diagnostics.py)에서 확인한 것:
- 반전 계열끼리는 상관 0.7~0.83 (사실상 같은 팩터)
- 변동성은 반전 계열과 상관 -0.02~0.12 (거의 독립)
- 개별로는 어느 것도 다중검정 임계값을 넘지 못함

독립적인 두 신호를 합치면 개별보다 나아지는지 본다. 구조상 두 단계다:
반전 계열 4종을 먼저 하나로 평균내고(안 그러면 같은 팩터에 4배 가중이 된다),
그 결과와 변동성을 동일가중으로 결합한다.

검증 순서를 지킨다:
1) 인샘플 전체 -> 개별 대비 결합이 나아졌는가
2) 인샘플 전/후반 -> 특정 국면에만 의존하지 않는가
3) 아웃오브샘플 -> 위 두 개를 통과했을 때만 의미가 있는 최종 확인

가중치는 동일가중으로 고정이다. 인샘플 결과를 보고 가중치를 조정하면 3)이
검증이 아니게 된다.
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
from src.research.composite_factor import combine
from src.research.ic_analysis import non_overlapping_ic, summarize_ic

TOP_K = 200
MIN_OBS = 30
HORIZONS = [20, 60]

# 팩터 계산에 필요한 과거 구간. 평가 시작일부터 데이터를 로드하면 rolling(60) 등이
# 초반 구간에서 전부 NaN이 되어 관측치가 통째로 날아간다(아웃오브샘플처럼 짧은
# 구간에서는 치명적). 평가 구간 앞에 이만큼을 덧붙여 팩터를 계산한 뒤, 평가는
# 원래 구간에서만 한다. 과거 데이터로 팩터를 만드는 것이므로 look-ahead가 아니다.
WARMUP_DAYS = 400

REVERSAL_PARTS = ["momentum_20", "momentum_60", "disparity_20", "disparity_60"]


def load_with_warmup(start: str, end: str) -> tuple[pd.DataFrame, ...]:
    """
    평가 구간 + 워밍업 구간의 패널과, 평가 구간 날짜만 담은 인덱스를 반환한다.

    수집된 패널이 2018년부터라 그 이전을 워밍업으로 요청하면 조용히 비어버린다.
    실제로 확보된 워밍업 일수를 찍어서, 워밍업이 걸렸는지 아닌지 눈에 보이게 한다.
    """
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(WARMUP_DAYS, unit="D")).strftime("%Y-%m-%d")
    dates = trading_dates(warmup_start, end)
    eval_dates = dates[dates >= pd.Timestamp(start)]

    close = build_close_panel(dates)
    volume = build_panel("volume", dates)
    market_cap = build_panel("market_cap", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

    actual_warmup = (close.index < pd.Timestamp(start)).sum()
    note = "" if actual_warmup >= 60 else "  <- 부족: 평가 초반 팩터가 NaN이 된다"
    print(f"  평가 {len(eval_dates)}일 / 확보된 워밍업 {actual_warmup}일{note}", flush=True)

    return close, volume, market_cap, universe, eval_dates


def build_signals(
    close: pd.DataFrame, volume: pd.DataFrame, universe: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """반전 / 저변동성 / 결합 세 신호를 만든다 (모두 높을수록 유리한 방향)."""
    price = build_price_factors(close, volume)

    reversal = combine(
        {name: price[name] for name in REVERSAL_PARTS},
        universe,
        signs={name: -1 for name in REVERSAL_PARTS},
    )
    low_volatility = combine(
        {"volatility_60": price["volatility_60"]}, universe, signs={"volatility_60": -1}
    )
    combined = combine(
        {"reversal": reversal, "low_volatility": low_volatility},
        universe,
        signs={"reversal": 1, "low_volatility": 1},
    )

    return {"반전": reversal, "저변동성": low_volatility, "반전+저변동성": combined}


def evaluate(
    signals: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    universe: pd.DataFrame,
    horizon: int,
    window: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    fwd = close.pct_change(horizon).shift(-horizon).where(universe)
    if window is not None:
        fwd = fwd.loc[window]

    rows = []
    for name, panel in signals.items():
        aligned = panel.reindex_like(fwd).where(universe.reindex_like(fwd))
        ic = non_overlapping_ic(aligned, fwd, horizon, min_obs=MIN_OBS)
        summary = summarize_ic(ic)
        summary["신호"] = name
        rows.append(summary)

    return pd.DataFrame(rows).set_index("신호")[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]]


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def main() -> None:
    print("=" * 70)
    print("1) 인샘플 전체 - 결합이 개별보다 나은가")
    print("=" * 70)

    close, volume, _, universe, eval_dates = load_with_warmup(
        phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END
    )
    signals = build_signals(close, volume, universe)

    for horizon in HORIZONS:
        show(
            f"[horizon {horizon}일]",
            evaluate(signals, close, universe, horizon, eval_dates),
        )

    print("\n" + "=" * 70)
    print("2) 인샘플 전/후반 - 특정 국면에만 의존하는가")
    print("=" * 70)

    midpoint = len(eval_dates) // 2
    for label, window in [("전반부", eval_dates[:midpoint]), ("후반부", eval_dates[midpoint:])]:
        show(f"[{label}, horizon 20일]", evaluate(signals, close, universe, 20, window))

    print("\n" + "=" * 70)
    print("3) 아웃오브샘플 - 위를 통과했을 때만 의미가 있는 최종 확인")
    print("=" * 70)

    oos_close, oos_volume, _, oos_universe, oos_eval = load_with_warmup(
        phase1.OUT_OF_SAMPLE_START, phase1.OUT_OF_SAMPLE_END
    )
    oos_signals = build_signals(oos_close, oos_volume, oos_universe)

    for horizon in HORIZONS:
        show(
            f"[horizon {horizon}일]",
            evaluate(oos_signals, oos_close, oos_universe, horizon, oos_eval),
        )


if __name__ == "__main__":
    main()
