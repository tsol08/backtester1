"""
종목 수가 몇 개면 롱온리 이벤트 전략이 검출 가능한가.

`run_event_coverage_survey.py`가 남긴 질문에 답한다. 이벤트 후보들은 유니버스를
1,500위까지 넓혀도 리밸런싱 시점당 **3~9종목**밖에 안 나온다. 그런 포트폴리오는
개별 종목 잡음이 커서, 효과가 실재해도 t가 안 나올 수 있다.

**이벤트 수익률을 보지 않고 답한다.** 같은 유니버스에서 N종목을 **무작위로** 뽑은
동일가중 포트폴리오의 추적오차를 잰다. 무작위이므로 기대 초과는 0이고, 어떤
이벤트에 대해서도 아무 정보를 주지 않는다. 여기서 나오는 것은 순수한 표본 성질이다:

    t = (연 초과 / 추적오차) * sqrt(년수)
    -> 검출에 필요한 연 초과 = 문턱 * 추적오차 / sqrt(년수)

이 표가 "이벤트 축을 열 수 있는가"를 결과를 보기 전에 판정한다. 이 저장소가
PEAD covered에서 역산한 값이 추적오차 6.2%, IR 0.43이었다(종목 수 ~52).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import settings
from src.data_loader.panels import Panels
from src.strategy.base import periodic_schedule

TOP_K = 500  # 실행 가능한 크기. 1500위는 1,185종목을 들어야 해서 소액으로 불가능하다
HORIZON = settings.FORWARD_HORIZON
PERIODS_PER_YEAR = 252 / HORIZON
SIZES = [3, 5, 8, 10, 20, 50, 100]
N_TRIALS = 300
SEED = 0

# 문턱: 새 가설 단일 사전 등록이면 1.96이 옳은 기준이다.
THRESHOLD = 1.96


def main() -> None:
    print("데이터 로딩...", flush=True)
    panels = Panels.load(top_k=TOP_K)
    schedule = periodic_schedule(panels.dates, HORIZON)

    # 리밸런싱 구간 수익률. 구간이 겹치지 않아 t가 부풀지 않는다.
    forward = panels.close.pct_change(HORIZON, fill_method=None).shift(-HORIZON)
    forward = forward.loc[schedule]
    universe = panels.universe.loc[schedule]
    masked = forward.where(universe)

    years = (schedule[-1] - schedule[0]).days / 365.25
    benchmark = masked.mean(axis=1)

    print("\n" + "=" * 84)
    print("종목 수별 검출 하한 — 무작위 포트폴리오로 잰다 (이벤트 수익률을 보지 않는다)")
    print("=" * 84)
    print(f"유니버스 상위 {TOP_K}위, 리밸런싱 {len(schedule)}회, {years:.1f}년")
    print(f"시행 {N_TRIALS}회/크기, 문턱 |t| > {THRESHOLD}")

    rng = np.random.default_rng(SEED)
    values = masked.to_numpy()
    valid = ~np.isnan(values)

    rows = []
    for size in SIZES:
        excesses = []
        for _ in range(N_TRIALS):
            picks = np.full(len(schedule), np.nan)
            for i in range(len(schedule)):
                available = np.flatnonzero(valid[i])
                if len(available) < size:
                    continue
                chosen = rng.choice(available, size=size, replace=False)
                picks[i] = values[i, chosen].mean()
            excess = pd.Series(picks, index=schedule) - benchmark
            excesses.append(excess.std())

        # 구간 표준편차 -> 연율화. 구간이 독립이므로 sqrt(구간수/년) 배다.
        tracking_error = float(np.mean(excesses)) * np.sqrt(PERIODS_PER_YEAR)
        required = THRESHOLD * tracking_error / np.sqrt(years)
        rows.append({
            "종목수": size,
            "추적오차(연)": tracking_error,
            "검출에 필요한 연초과": required,
            "필요 IR": THRESHOLD / np.sqrt(years),
        })

    table = pd.DataFrame(rows).set_index("종목수")
    print()
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())

    print("\n" + "-" * 84)
    print("읽는 법: 그 종목 수의 '검출에 필요한 연초과'를 넘길 만한 이벤트 효과가")
    print("문헌에 보고된 적이 있는지를 보면 된다. 없으면 그 검정은 돌리기 전에")
    print("이미 검정력이 없는 것이고, 미달이 나와도 '없다'가 아니라 '못 본다'다.")

    print("\n" + "=" * 84)
    print("제외형은 사정이 다르다 — 유니버스에서 일부를 빼면 대부분이 상쇄된다")
    print("=" * 84)
    print("보유형은 소수 종목을 들어 벤치마크와 크게 어긋나지만, 제외형은")
    print("'벤치마크 빼기 한 조각'이라 추적오차가 구조적으로 작다. 여기서도")
    print("**무작위로** 제외해서 잰다 - 어떤 이벤트에 대해서도 정보를 주지 않는다.")

    rows = []
    for share in (0.03, 0.05, 0.10, 0.15, 0.21, 0.26):
        stds = []
        for _ in range(N_TRIALS):
            picks = np.full(len(schedule), np.nan)
            for i in range(len(schedule)):
                available = np.flatnonzero(valid[i])
                n_drop = int(round(len(available) * share))
                if len(available) - n_drop < 2:
                    continue
                dropped = rng.choice(available, size=n_drop, replace=False)
                kept = np.setdiff1d(available, dropped, assume_unique=False)
                picks[i] = values[i, kept].mean()
            stds.append((pd.Series(picks, index=schedule) - benchmark).std())

        tracking_error = float(np.mean(stds)) * np.sqrt(PERIODS_PER_YEAR)
        required = THRESHOLD * tracking_error / np.sqrt(years)
        # 초과 = w/(1-w) x (제외 종목의 부진). 뒤집으면 필요한 부진 폭이 나온다.
        needed_gap = required * (1 - share) / share
        rows.append({
            "제외비중": share,
            "추적오차(연)": tracking_error,
            "검출에 필요한 연초과": required,
            "그러려면 제외종목이 연": needed_gap,
        })

    table = pd.DataFrame(rows).set_index("제외비중")
    print()
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())

    print()
    print("마지막 열이 핵심이다: 제외한 종목들이 유니버스보다 연 그만큼 못해야")
    print("검출된다. 문헌의 발행 후 장기 부진이 그 크기인지를 보면 된다.")


if __name__ == "__main__":
    main()
