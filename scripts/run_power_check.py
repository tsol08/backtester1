"""
우리 표본으로 어느 정도 크기의 신호까지 검출할 수 있었는가.

이 프로젝트는 네 개의 팩터를 차례로 기각했다. 그런데 기각에는 두 가지 의미가 있다:
"신호가 없다"와 "우리 표본으로는 못 본다". 이 둘을 구분하지 않으면 지금까지의
결론 전체가 공중에 뜬다.

여기서는 실제 한국 주가 데이터의 미래수익률에 **강도를 아는 신호를 심어놓고**,
우리가 실제로 쓴 파이프라인(비겹침 샘플링 -> cross-sectional IC -> t-stat)이
그것을 몇 번이나 찾아내는지 센다.

검출 기준을 두 개 쓴다:
- |t| > 1.96 : 단일 검정 기준. 팩터 하나만 보고 판단할 때.
- |t| > 2.87 : 다중검정 보정 기준. 실제로 우리가 팩터를 판정할 때 쓴 잣대.

읽는 법: 검정력 80%가 관례적 기준이다. 그 선을 넘는 최소 IC가, 우리가 이 표본으로
'있다면 봤을' 신호의 하한이다. 실제 팩터들의 IC가 그 아래였다면, 우리의 기각은
증거의 부재일 뿐 부재의 증거가 아니다.
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
from src.research.power_analysis import minimum_detectable_ic

TOP_K = 200
HORIZON = 20
N_TRIALS = 150
CANDIDATE_ICS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]

# 이 프로젝트에서 실제로 측정된 팩터 IC들 (비교용)
OBSERVED = {
    "저변동성(인샘플)": 0.067,
    "저변동성(OOS)": 0.101,
    "반전(인샘플)": 0.041,
    "내부자 정제후": 0.014,
}


def run(label: str, start: str, end: str) -> None:
    warmup = (pd.Timestamp(start) - pd.Timedelta(400, unit="D")).strftime("%Y-%m-%d")
    dates = trading_dates(warmup, end)
    eval_dates = dates[dates >= pd.Timestamp(start)]

    close = build_close_panel(dates)
    market_cap = build_panel("market_cap", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K).loc[eval_dates]

    fwd = close.pct_change(HORIZON).shift(-HORIZON).loc[eval_dates]

    n_obs = len(eval_dates) // HORIZON
    print(f"\n{'=' * 72}")
    print(f"{label} ({start} ~ {end}) - 거래일 {len(eval_dates)}일, 비겹침 관측 약 {n_obs}개")
    print("=" * 72)

    for threshold, basis in [(1.96, "단일검정 5%"), (2.87, "다중검정 보정")]:
        table = minimum_detectable_ic(
            fwd,
            universe,
            horizon=HORIZON,
            candidates=CANDIDATE_ICS,
            n_trials=N_TRIALS,
            threshold=threshold,
        )
        print(f"\n[검출 기준 |t| > {threshold} ({basis})]")
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(table[["실현 IC(평균)", "t-stat 중앙값", "검출률", "검정력 달성"]].to_string())


def main() -> None:
    run("인샘플", phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END)
    run("아웃오브샘플", phase1.OUT_OF_SAMPLE_START, phase1.OUT_OF_SAMPLE_END)

    print("\n" + "=" * 72)
    print("참고: 이 프로젝트에서 실제로 측정된 팩터 IC")
    print("=" * 72)
    for name, ic in OBSERVED.items():
        print(f"  {name:20s} {ic:.3f}")


if __name__ == "__main__":
    main()
