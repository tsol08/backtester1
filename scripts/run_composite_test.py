"""
가설 검정: 밸류와 PEAD를 스코어 단계에서 합치면 각각으로는 못 넘던 선을 넘는가.

**사전 등록: experiments/plan_value_pead_composite.md — 결과를 보기 전에 커밋됐다.**
아래 기준은 그 문서에 먼저 적힌 것이고 결과를 본 뒤 바꾸지 않는다.

왜: 이 저장소에서 롱온리 초과가 양수로 측정된 신호는 밸류(+4.77%/년, t 1.33,
항상 켜짐)와 PEAD(+9.55%/년, t 2.69, 리밸런싱의 58%에만 존재) 둘뿐이고, 약점이
서로 반대다. PEAD가 죽은 원인은 강도가 아니라 공백이었다 - 회전율의 79%가
'신호<->유휴' 전환이었고 그 전환은 신호 노출을 전혀 주지 않았다.

**이것은 PEAD 4차다.** pead.py에 "네 번째 변형을 만들지 않는다"고 적어놨고, 이
검정은 그 선언에 걸린다. 무시하지 않고 4차로 세고 문턱을 그에 맞춘다.
**미달이면 PEAD를 닫는다. 5차는 없다.**

  주 판정   비용 차감 후 롱온리 초과(vs 후보 집합)의 |t| > 2.50
            (Bonferroni 4건. 1건 1.96 -> 2건 2.24 -> 3건 2.39 -> 4건 2.50)

  반증 확인 (셋 다 통과해야 한다. 하나라도 걸리면 t가 얼마든 기각)
    FC1  합성 초과 > 밸류 단독 초과. 아니면 PEAD 성분이 아무것도 더하지 않은 것이다.
         **이게 핵심 반증** - 밸류를 PEAD로 포장한 것에 불과한지를 가른다.
    FC2  하위 5분위 초과가 음수. 반대쪽이 반대 부호가 아니면 분위 구조가 아니라 잡음이다.
    FC3  연회전율 <= 3.0. 이 가설의 메커니즘 주장이 '전환 마찰이 사라진다'이므로,
         회전율이 PEAD covered(5.58) 근처면 초과가 양수여도 주장한 이유 때문이 아니다.

  정량 예측 (대조용): 회전율 1.0~3.0, 합성이 밸류 단독 +1.5%p 이상.
  **사전 예상은 '미달'이다** - 합성이 밸류 지배적이면 t 1.33에서 크게 못 오른다.
  실패했을 때 사후에 "그럴 줄 알았다"고 말하지 않으려고 미리 적는다.

측정 코드는 run_backtest.py의 것을 그대로 import한다. PEAD 1~3차를 판정한 계산과
**같은 잣대**여야 비교가 성립한다.

사용:
    python scripts/run_composite_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scipy import stats

from config import settings
from src.costs.cost_model import CostModel
from src.data_loader.panels import Panels
from src.portfolio.portfolio_engine import run_weighted_backtest
from src.strategy.base import EqualWeightUniverse, SubsetEqualWeight, build_weight_panel
from src.strategy.composite import CompositeStrategy

# PEAD 1~3차를 판정한 것과 같은 계산을 쓴다. 복사하면 갈라진다.
from scripts.run_backtest import restrict, summarize, t_stat, window_returns

TOP_K = 500
HORIZON = settings.FORWARD_HORIZON
CAPITAL = settings.INITIAL_CAPITAL
PERIODS_PER_YEAR = 252 / HORIZON

N_TESTS = 4  # PEAD 4차
THRESHOLD = float(stats.norm.ppf(1 - 0.05 / (2 * N_TESTS)))
MAX_TURNOVER = 3.0  # FC3


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def annualize(period_return: float) -> float:
    return (1 + period_return) ** PERIODS_PER_YEAR - 1


def main() -> None:
    print("데이터 로딩...", flush=True)
    panels = Panels.load(top_k=TOP_K)

    composite = CompositeStrategy(components=("value", "pead"))
    composite.prepare(panels)

    value_only = CompositeStrategy(components=("value",))
    value_only.prepare(panels)

    benchmark = composite.benchmark()
    benchmark.prepare(panels)

    universe_benchmark = EqualWeightUniverse()
    universe_benchmark.prepare(panels)

    # FC2: 하위 5분위. 전략에 손잡이를 다는 대신 여기서 마스크로 만든다 -
    # 사전 등록된 반증 확인이지 전략의 변형이 아니다.
    bottom = SubsetEqualWeight(
        composite.quantiles == 0, "합성 하위 5분위 (FC2)", HORIZON
    )
    bottom.prepare(panels)

    schedule = composite.rebalance_dates()

    print("\n" + "=" * 78)
    print(f"밸류+PEAD 합성 — 사전 등록 검정 (PEAD 4차, |t| > {THRESHOLD:.2f})")
    print("=" * 78)
    print(f"유니버스 시총 상위 {TOP_K}위 | 기간 {schedule[0].date()} ~ {panels.dates[-1].date()}"
          f" | 리밸런싱 {len(schedule)}회")
    print(f"후보 집합(B/M 보유) 일평균 {composite.covered.sum(axis=1).mean():.0f}종목")

    pead_on = composite.pead_coverage.loc[schedule] > 0
    print(f"유효 SUE가 있는 리밸런싱 시점: {int(pead_on.sum())} / {len(pead_on)}"
          f" ({pead_on.mean():.0%})   <- 합성이 메우려는 공백")
    print(f"그 시점의 SUE 보유 종목 일평균:"
          f" {composite.pead_coverage.loc[schedule][pead_on].mean():.0f}종목")

    engines = [
        ("합성", composite),
        ("밸류단독", value_only),
        ("벤치마크", benchmark),
        ("하위분위", bottom),
        ("유니버스", universe_benchmark),
    ]

    results = {}
    for label, engine in engines:
        weights = build_weight_panel(engine, panels)
        held = sorted(weights.columns[(weights != 0).any(axis=0)])
        raw = run_weighted_backtest(
            weights, panels.price_frames(held), CostModel(), initial_capital=CAPITAL
        )
        results[label] = restrict(raw, schedule[0], CAPITAL)

    show("[성과] 비용 차감 후", pd.DataFrame(
        [summarize(f"{label}: {engine.name}", results[label], CAPITAL)
         for label, engine in engines]
    ).set_index("구분"))

    # 초과수익은 비겹침 리밸런싱 구간으로 잰다. 매일 재면 t가 부풀려진다.
    bench_windows = window_returns(results["벤치마크"].returns, schedule)
    excess = {}
    for label in ("합성", "밸류단독", "하위분위"):
        excess[label] = (window_returns(results[label].returns, schedule) - bench_windows).dropna()

    print("\n" + "=" * 78)
    print("초과수익 vs 후보 집합 (비겹침 20일 구간)")
    print("=" * 78)
    table = pd.DataFrame([
        {
            "구분": label,
            "관측": len(series),
            "구간당": series.mean(),
            "연율화": annualize(series.mean()),
            "t": t_stat(series),
        }
        for label, series in excess.items()
    ]).set_index("구분")
    show("", table)

    composite_excess = annualize(excess["합성"].mean())
    value_excess = annualize(excess["밸류단독"].mean())
    composite_t = t_stat(excess["합성"])
    turnover = summarize("", results["합성"], CAPITAL)["연회전율"]

    print("\n" + "=" * 78)
    print("판정 — 사전 등록 기준과 대조")
    print("=" * 78)

    main_pass = abs(composite_t) > THRESHOLD
    fc1_pass = composite_excess > value_excess
    fc2_pass = annualize(excess["하위분위"].mean()) < 0
    fc3_pass = turnover <= MAX_TURNOVER

    print(f"\n주 판정   |t| = {abs(composite_t):.2f} vs {THRESHOLD:.2f}"
          f"   -> {'통과' if main_pass else '미달'}")
    print(f"\nFC1  합성 {composite_excess:+.2%} vs 밸류 단독 {value_excess:+.2%}"
          f"   -> {'통과' if fc1_pass else '실패'}"
          f"   (차이 {composite_excess - value_excess:+.2%}p, 예측 +1.5%p 이상)")
    print(f"FC2  하위분위 초과 {annualize(excess['하위분위'].mean()):+.2%}"
          f"   -> {'통과' if fc2_pass else '실패'} (음수여야 한다)")
    print(f"FC3  연회전율 {turnover:.2f} vs 상한 {MAX_TURNOVER}"
          f"   -> {'통과' if fc3_pass else '실패'} (예측 1.0~3.0)")

    verdict = main_pass and fc1_pass and fc2_pass and fc3_pass
    print(f"\n>>> {'통과' if verdict else '기각'}"
          f" — 주 판정과 반증 확인 셋을 모두 충족해야 통과다.")
    if not verdict:
        print(">>> 사전 등록대로 PEAD를 닫는다. 5차는 없다.")

    print("\n" + "=" * 78)
    print("연도별 수익률 (해석용, 판정 기준 아님)")
    print("=" * 78)
    yearly = pd.DataFrame({
        label: result.returns.groupby(result.returns.index.year).apply(lambda r: (1 + r).prod() - 1)
        for label, result in results.items()
    })
    yearly["합성-벤치"] = yearly["합성"] - yearly["벤치마크"]
    show("", yearly)


if __name__ == "__main__":
    main()
