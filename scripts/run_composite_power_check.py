"""
밸류+PEAD 합성 검정의 검출 하한. **결과를 보기 전에 돌린다.**

experiments/plan_value_pead_composite.md 5장의 1번 단계다. 사전 등록이
"미달로 끝났을 때 '없다'와 '이 표본으로는 못 본다'를 구분해서 기록한다"고
정해놨고(CLAUDE.md 규율 7), 그 구분은 검정력을 **먼저** 재야 가능하다.

여기서 재는 것은 합성 스코어의 성과가 아니라 **구성(top-500, 2018~2026, 20일
비겹침)이 어느 크기의 신호까지 볼 수 있는가**이다. 실제 신호는 쓰지 않는다 -
알려진 강도의 인공 신호를 실제 수익률에 심어 몇 번이나 검출하는지 센다.

문턱을 두 개 같이 본다:
  1.96  단일 검정 기준. 참고용.
  2.50  이번 검정의 사전 등록 문턱 (Bonferroni 4건).

미국 모멘텀조차 5년 구간에서 37%만 검출된다는 것이 이 저장소의 기준점이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import settings
from src.data_loader.panels import Panels
from src.research.power_analysis import minimum_detectable_ic

TOP_K = 500  # PEAD 1~3차를 판정한 구성
HORIZON = settings.FORWARD_HORIZON
START, END = "2018-01-01", "2026-08-26"
THRESHOLDS = (1.96, 2.50)  # 2.50이 이번 사전 등록 문턱
CANDIDATE_ICS = [0.01, 0.02, 0.03, 0.04, 0.05]


def main() -> None:
    print("데이터 로딩...", flush=True)
    panels = Panels.load(top_k=TOP_K).slice(START, END)

    fwd = panels.close.pct_change(HORIZON).shift(-HORIZON)
    n_obs = len(panels.dates[::HORIZON])

    print()
    print("=" * 70)
    print("밸류+PEAD 합성 검정의 검출 하한 (결과를 보기 전에 측정)")
    print("=" * 70)
    print(f"유니버스 시총 상위 {TOP_K}위, {START} ~ {END}")
    print(f"비겹침 {HORIZON}일 관측 약 {n_obs}개, 일평균 유니버스"
          f" {panels.universe.sum(axis=1).mean():.0f}종목")
    print()

    table = minimum_detectable_ic(
        fwd, panels.universe, HORIZON, CANDIDATE_ICS, thresholds=THRESHOLDS
    )
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(table.to_string())

    print()
    print("읽는 법: 검정력 80%가 관례적 기준이다. 그 선을 넘는 가장 작은 IC가")
    print("이 표본이 '있다'고 말할 수 있는 하한이고, 그보다 작은 신호는 미달로")
    print("나와도 '없다'가 아니라 '이 표본으로는 못 본다'로 기록해야 한다.")


if __name__ == "__main__":
    main()
