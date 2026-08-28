"""
가설 검정: 유상증자를 공시한 종목을 안 사면 유니버스를 이기는가.

**사전 등록: experiments/plan_seo_exclusion.md — 수익률을 보기 전에 커밋됐다(ed0d096).**
아래 기준은 그 문서에 먼저 적힌 것이고 결과를 본 뒤 바꾸지 않는다.

왜 이 형태인가. 이 저장소가 열 번 검정해서 얻은 가장 견고한 사실은 한국 대형주의
횡단면 예측력이 **하위 구간에 몰려 있다**는 것이다(밸류 62%, 저변동성 대부분,
합성 67%). 롱온리가 그 정보를 못 쓴다고 계속 적어왔는데, 정확히는 **'사서는'**
못 쓰는 것이다. **안 사는 것으로는 쓸 수 있다.** 이번이 그 통로의 첫 검정이다.

제외형을 고른 것은 검정력 때문이다. 결과를 보기 전에 무작위 포트폴리오로 쟀다:

    보유형  8종목이면 추적오차 연 17.0% -> 검출에 연 11.3% 필요. 그런 이벤트 효과는 없다.
    제외형  21% 제외면 추적오차 연 1.13% -> 검출에 연 0.75% 필요.
            = 제외 종목이 연 2.83%만 부진하면 된다. 문헌 범위 안이다.

창 3년은 문헌(신주발행 후 장기 부진은 3~5년 현상)과 검정력을 보고 골랐다. 60일은
PEAD의 창이고 발행 이벤트에 쓸 근거가 없었다. **수익률을 보고 고른 것이 아니다.**

  주 판정   비용 차감 후 초과(vs 유니버스 동일가중)의 t > 1.96, **부호는 양수**
            (사전 등록된 단일 신규 가설. PEAD 1차·자사주 1차와 같은 기준)

  반증 확인 (셋 다 통과해야 한다)
    FC1  무상증자를 같은 방식으로 제외하면 효과가 없어야 한다. **핵심 반증.**
         무상증자는 주식 수만 늘고 현금 조달이 없어 희석이 아니다. 둘 다 부진하면
         '주식 수를 늘리는 회사' 효과이지 유상증자 고유분이 아니다.
         자사주가 t 2.14로 통과할 뻔했다가 이 장치에 걸렸다(고유분 t 0.56).
    FC2  제외한 바스켓 자체가 유니버스보다 부진해야 한다. 부호가 반대면 기각.
    FC3  연회전율 1.5 이하. 제외형의 근거가 저회전이다.

  정량 예측: 제외 종목 부진 연 3~8%, 전략 초과 연 0.8~2.1%, 회전율 0.5~1.2.
  **사전 예상은 '경계선, 확률 반반 아래'.**

측정 코드는 run_backtest.py에서 import한다. 앞선 판정들과 같은 잣대여야 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import settings
from src.costs.cost_model import CostModel
from src.data_loader.dart_filings import event_dates, load_filings, recent_event_mask
from src.data_loader.panels import Panels
from src.portfolio.portfolio_engine import run_weighted_backtest
from src.strategy.base import EqualWeightUniverse, SubsetEqualWeight, build_weight_panel

from scripts.run_backtest import restrict, summarize, t_stat, window_returns

TOP_K = 500
HORIZON = settings.FORWARD_HORIZON
CAPITAL = settings.INITIAL_CAPITAL
PERIODS_PER_YEAR = 252 / HORIZON
EVENT_WINDOW = 1095  # 3년. 문헌의 관측 구간
FILING_START, FILING_END = "2015-01-01", "2026-08-31"

SEO = ["유상증자결정"]
BONUS = ["무상증자결정"]  # FC1 대조군

THRESHOLD = 1.96
MAX_TURNOVER = 1.5  # FC3


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def annualize(period_return: float) -> float:
    return (1 + period_return) ** PERIODS_PER_YEAR - 1


def main() -> None:
    print("데이터 로딩...", flush=True)
    panels = Panels.load(top_k=TOP_K)
    members = set(panels.members)
    filings = load_filings(FILING_START, FILING_END)

    def flag(kinds: list[str]) -> pd.DataFrame:
        events = event_dates(filings, kinds)
        inside = events[events["ticker"].isin(members)]
        mask = recent_event_mask(inside, panels.dates, panels.close.columns, EVENT_WINDOW)
        return mask & panels.universe

    seo = flag(SEO)
    bonus = flag(BONUS)

    strategy = SubsetEqualWeight(panels.universe & ~seo, "유상증자 제외", HORIZON)
    benchmark = EqualWeightUniverse(HORIZON)
    control = SubsetEqualWeight(panels.universe & ~bonus, "무상증자 제외 (FC1 대조)", HORIZON)
    basket = SubsetEqualWeight(seo, "제외 바스켓 (FC2)", HORIZON)

    engines = [("전략", strategy), ("벤치마크", benchmark),
               ("무상증자대조", control), ("제외바스켓", basket)]
    for _, engine in engines:
        engine.prepare(panels)

    schedule = strategy.rebalance_dates()

    print("\n" + "=" * 80)
    print(f"유상증자 제외 필터 — 사전 등록 검정 (t > {THRESHOLD}, 부호 양수)")
    print("=" * 80)
    print(f"유니버스 시총 상위 {TOP_K}위 | 창 {EVENT_WINDOW}일(3년)"
          f" | {schedule[0].date()} ~ {panels.dates[-1].date()} | 리밸런싱 {len(schedule)}회")

    # 배선 검증. 수익률이 아니라 마스크가 의도대로 도는지 본다.
    universe_size = panels.universe.loc[schedule].sum(axis=1)
    excluded = seo.loc[schedule].sum(axis=1)
    kept = (panels.universe & ~seo).loc[schedule].sum(axis=1)
    print(f"\n[배선] 유니버스 {universe_size.mean():.0f} = 제외 {excluded.mean():.0f}"
          f" + 보유 {kept.mean():.0f}"
          f"  (합 일치: {bool((universe_size == excluded + kept).all())})")
    print(f"[배선] 제외 비중 평균 {(excluded / universe_size).mean():.1%}"
          f" | 무상증자 대조군 제외 비중 {(bonus.loc[schedule].sum(axis=1) / universe_size).mean():.1%}")

    by_year = (excluded / universe_size).groupby(pd.Index(schedule).year).mean()
    print(f"[배선] 연도별 제외 비중 (2018은 공시 이력이 3년치가 안 차서 낮은 것이 정상):")
    print("        " + "  ".join(f"{y}:{v:.0%}" for y, v in by_year.items()))

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

    bench_windows = window_returns(results["벤치마크"].returns, schedule)
    excess = {
        label: (window_returns(results[label].returns, schedule) - bench_windows).dropna()
        for label in ("전략", "무상증자대조", "제외바스켓")
    }

    print("\n" + "=" * 80)
    print("초과수익 vs 유니버스 동일가중 (비겹침 20일 구간)")
    print("=" * 80)
    show("", pd.DataFrame([
        {"구분": label, "관측": len(s), "구간당": s.mean(),
         "연율화": annualize(s.mean()), "t": t_stat(s)}
        for label, s in excess.items()
    ]).set_index("구분"))

    strategy_excess = annualize(excess["전략"].mean())
    strategy_t = t_stat(excess["전략"])
    control_t = t_stat(excess["무상증자대조"])
    basket_excess = annualize(excess["제외바스켓"].mean())
    turnover = summarize("", results["전략"], CAPITAL)["연회전율"]

    print("\n" + "=" * 80)
    print("판정 — 사전 등록 기준과 대조")
    print("=" * 80)

    main_pass = strategy_t > THRESHOLD  # 부호까지 미리 정했다
    fc1_pass = control_t < strategy_t
    fc2_pass = basket_excess < 0
    fc3_pass = turnover <= MAX_TURNOVER

    print(f"\n주 판정   t = {strategy_t:+.2f} vs {THRESHOLD} (양수여야 함)"
          f"   -> {'통과' if main_pass else '미달'}   초과 연 {strategy_excess:+.2%}")
    print(f"\nFC1  무상증자 대조 t {control_t:+.2f} < 유상증자 t {strategy_t:+.2f}"
          f"   -> {'통과' if fc1_pass else '실패'}")
    print(f"FC2  제외 바스켓 초과 {basket_excess:+.2%}"
          f"   -> {'통과' if fc2_pass else '실패'} (음수여야 한다)")
    print(f"FC3  연회전율 {turnover:.2f} vs 상한 {MAX_TURNOVER}"
          f"   -> {'통과' if fc3_pass else '실패'}")

    verdict = main_pass and fc1_pass and fc2_pass and fc3_pass
    print(f"\n>>> {'통과' if verdict else '기각'}"
          f" — 주 판정과 반증 확인 셋을 모두 충족해야 통과다.")

    print("\n" + "=" * 80)
    print("연도별 수익률 (해석용, 판정 기준 아님)")
    print("=" * 80)
    yearly = pd.DataFrame({
        label: r.returns.groupby(r.returns.index.year).apply(lambda x: (1 + x).prod() - 1)
        for label, r in results.items()
    })
    yearly["전략-벤치"] = yearly["전략"] - yearly["벤치마크"]
    show("", yearly)


if __name__ == "__main__":
    main()
