"""
가설 검정: 순 발행이 많은 기업을 안 사면 유니버스를 이기는가. (12번째 가설)

**이 docstring이 사전 등록이고, 수익률을 보기 전에 커밋한다.**
아래 기준은 결과를 본 뒤 바꾸지 않는다.

## 왜 이 형태인가

11번째 가설(유상증자 제외)에서 제외형의 산술이 나왔다:

    전략 초과 = 제외비중/(1-제외비중) x 제외 종목의 부진폭

유상증자 제외는 **고르는 데는 성공했다** — 제외 바스켓이 실제로 연 3.22% 부진했다
(반증 확인 FC2 통과). 그런데 제외 비중 20.8%에 희석되어 초과가 0.85%(t 1.42)로
줄었다. 그리고 유상증자는 '발행의 한 가지 원인'이라 그 이상 넓힐 수가 없다.

**주식 수 증가율 자체를 재면 상위 5분위 = 제외 비중 20%가 정의상 보장된다.**
문헌에서도 순 발행(net share issuance)이 개별 발행 사건보다 강한 예측력을 갖는
것으로 보고된다 - 유상증자만이 아니라 전환사채 전환·신주인수권 행사까지 한 번에
잡히기 때문이다.

**그리고 이 프로젝트가 방금 배운 것과 깨끗하게 갈린다.** 무상증자 종목이 이후
연 12.55% 부진하지만 그 원인은 발행이 아니라 '최근 급등한 소형주'라는 표지였다
(2026-08-28 (10)). 기업행위(분할·무상증자)는 주주 지분이 그대로라 발행이 아니므로
`net_issuance`가 구조적으로 제외한다. **그 분리가 FC1의 근거이기도 하다.**

## 사전에 잰 것 (수익률을 보기 전)

- 유니버스 내 순발행 계산 가능 종목 일평균 **452 / 500**
- 순발행 중앙값 0 (대부분 발행하지 않는다), 80분위 1.4%, 90분위 10.0%
- 순발행 > 1%인 종목-일이 **18.9%** -> 상위 5분위이 '실제로 발행한 기업'과 거의 일치
- 기업행위 몫 > 1%는 2.1%로 깨끗하게 분리된다

## 구성 — 새로 고른 파라미터가 없다

| 항목 | 값 | 근거 |
|---|---|---|
| 유니버스 | 시총 상위 500 | 11번째 가설과 동일 |
| 신호 | 순 발행 = 주식수 증가율 / 기업행위 몫 - 1 | |
| 측정 구간 | 252거래일(1년) | 이 저장소가 자산성장을 잰 구간과 동일 |
| 제외 | 순 발행 **상위 5분위** (w = 20%) | `N_QUANTILES = 5`, 기존 전 검정과 동일 |
| 벤치마크 | 유니버스 동일가중 | 전 종목이 후보이고 신호가 일부를 뺀다. 규율 5 충족 |
| 리밸런싱 | 20거래일 | 기존과 동일 |
| 비용 | `CostModel()` 기본 + 증권거래세 | 기존과 동일 |
| 기간 | 2018-01-02 ~ 2026-08-26 | 기존과 동일 |

## 판정 기준

**주 판정: 비용 차감 후 초과(vs 유니버스)의 t > 2.24, 부호는 양수.**

문턱을 1.96이 아니라 **2.24(Bonferroni 2건)**로 둔다. 11번째 가설이 같은 주장
('발행하면 이후 부진하다')을 유상증자라는 다른 측정으로 이미 한 번 검정했기
때문이다. 게다가 그때 반증 확인 FC1이 **발행 해석을 기각했다**(희석 없는 무상증자가
희석하는 유상증자보다 네 배 나빴다). 그 반대 증거를 알고도 다시 보는 것이므로
느슨한 문턱을 쓸 자격이 없다.

### 반증 확인 — 넷 다 통과해야 한다

**(FC1) 기업행위 상위 5분위 제외는 효과가 없어야 한다. 핵심 반증.**
분할·무상증자는 주주 지분이 그대로라 경제적 사건이 아니다. 그쪽도 효과가 있으면
'주식 수가 늘어난 회사' 효과이지 발행 고유분이 아니다. 자사주가 이 장치에 걸렸고
(취득·처분 둘 다 양수 -> 고유분 t 0.56), 11번째도 이 장치에 걸렸다.
판정: 기업행위 제외의 t가 순 발행 제외의 t보다 **작아야** 한다.

**(FC2) 제외 바스켓 자체가 유니버스보다 부진해야 한다.** 부호가 반대면 기각.

**(FC3) 연회전율 1.5 이하.** 주식 수는 천천히 변하므로 낮아야 정상이다. 높으면
분위 경계에서 들락거리는 것이고, 제외형의 저비용 근거가 사라진다.

**(FC4) 자산성장 상위 5분위 제외와 구별돼야 한다.** 자산성장은 이 저장소에서
IC 0.002로 기각됐다(2026-08-28 (4)). 둘 다 유의하면 같은 것을 다른 이름으로 본
것이므로, 순 발행 고유분을 주장할 수 없다.
판정: 자산성장 제외의 t가 순 발행 제외의 t보다 **작아야** 한다.

### 정량 예측 (대조용, 판정 아님)

- 제외 바스켓 부진 **연 4~8%** (유상증자 단독의 3.22%보다는 커야 말이 된다.
  순 발행은 전환사채 전환·신주인수권 행사까지 잡으므로)
- 전략 초과 연 **1.0~2.0%**, 회전율 0.4~1.0
- **사전 예상: 미달.** 11번째의 FC1이 발행 해석에 이미 반대 증거를 냈다. 그래도
  검정할 값어치가 있는 것은, 그때의 반대 증거가 '무상증자가 더 나쁘다'였고
  이번 구성은 **바로 그 무상증자를 신호에서 제거**하기 때문이다. 확률은 낮게 본다.

## 실패 시

12번 연속 기각이면 이 저장소의 결론은 **"공개 데이터 + 롱온리 + 이 표본으로는
비용을 넘는 우위를 찾지 못했다"**로 굳는다. 그건 실패가 아니라 결과다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import settings
from src.costs.cost_model import CostModel
from src.data_loader.dart_loader import load_corp_codes, load_fundamentals_bulk
from src.data_loader.krx_openapi import build_panels
from src.data_loader.panels import Panels
from src.features.fundamental_factors import build_fundamental_panels
from src.features.share_issuance import corporate_action_growth, net_issuance
from src.portfolio.portfolio_engine import run_weighted_backtest
from src.research.quantile_analysis import assign_quantiles
from src.strategy.base import EqualWeightUniverse, SubsetEqualWeight, build_weight_panel

from scripts.run_backtest import restrict, summarize, t_stat, window_returns

TOP_K = 500
HORIZON = settings.FORWARD_HORIZON
N_QUANTILES = settings.N_QUANTILES
CAPITAL = settings.INITIAL_CAPITAL
PERIODS_PER_YEAR = 252 / HORIZON
LOOKBACK = 252
MIN_CROSS_SECTION = 30

THRESHOLD = 2.24  # Bonferroni 2건. 11번째가 같은 주장을 이미 한 번 검정했다
MAX_TURNOVER = 1.5  # FC3


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def annualize(x: float) -> float:
    return (1 + x) ** PERIODS_PER_YEAR - 1


def top_quintile(signal: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    quantiles = assign_quantiles(signal, universe, N_QUANTILES, MIN_CROSS_SECTION)
    return (quantiles == N_QUANTILES - 1).fillna(False) & universe


def main() -> None:
    print("데이터 로딩...", flush=True)
    panels = Panels.load(top_k=TOP_K)
    dates, columns = panels.dates, panels.close.columns

    raw = build_panels(["close", "listed_shares"], dates)
    raw_close = raw["close"].reindex(index=dates, columns=columns)
    shares = raw["listed_shares"].reindex(index=dates, columns=columns)

    issuance = net_issuance(raw_close, shares, LOOKBACK)
    action = corporate_action_growth(raw_close, shares, LOOKBACK)

    dart_codes = set(load_corp_codes()["stock_code"])
    fundamentals = load_fundamentals_bulk(
        [t for t in panels.members if t in dart_codes], 2015, dates[-1].year, verbose=False
    )
    asset_growth = build_fundamental_panels(
        fundamentals, dates, panels.market_cap
    )["asset_growth"].reindex(index=dates, columns=columns)

    issuers = top_quintile(issuance.where(panels.universe), panels.universe)
    action_heavy = top_quintile(action.where(panels.universe), panels.universe)
    growers = top_quintile(asset_growth.where(panels.universe), panels.universe)

    strategy = SubsetEqualWeight(panels.universe & ~issuers, "순 발행 상위분위 제외", HORIZON)
    benchmark = EqualWeightUniverse(HORIZON)
    fc1 = SubsetEqualWeight(panels.universe & ~action_heavy, "기업행위 제외 (FC1)", HORIZON)
    fc2 = SubsetEqualWeight(issuers, "제외 바스켓 (FC2)", HORIZON)
    fc4 = SubsetEqualWeight(panels.universe & ~growers, "자산성장 제외 (FC4)", HORIZON)

    engines = [("전략", strategy), ("벤치마크", benchmark),
               ("기업행위", fc1), ("제외바스켓", fc2), ("자산성장", fc4)]
    for _, engine in engines:
        engine.prepare(panels)

    schedule = strategy.rebalance_dates()

    print("\n" + "=" * 82)
    print(f"순 발행 제외 — 사전 등록 검정 (12번째 가설, t > {THRESHOLD}, 부호 양수)")
    print("=" * 82)
    print(f"유니버스 상위 {TOP_K}위 | 측정 {LOOKBACK}거래일"
          f" | {schedule[0].date()} ~ {dates[-1].date()} | 리밸런싱 {len(schedule)}회")

    # 배선 검증 — 수익률이 아니라 마스크가 의도대로 도는지 본다
    size = panels.universe.loc[schedule].sum(axis=1)
    print(f"\n[배선] 순발행 계산 가능 {issuance.where(panels.universe).notna().loc[schedule].sum(axis=1).mean():.0f}"
          f" / {size.mean():.0f}종목")
    for label, mask in (("순 발행 상위분위", issuers), ("기업행위 상위분위", action_heavy),
                        ("자산성장 상위분위", growers)):
        n = mask.loc[schedule].sum(axis=1)
        print(f"[배선] {label:16s} 제외 {n.mean():5.1f}종목 = 유니버스의 {(n / size).mean():.1%}")
    overlap = (issuers & action_heavy).loc[schedule].sum(axis=1).mean()
    print(f"[배선] 순 발행 상위분위와 기업행위 상위분위가 겹치는 종목: {overlap:.1f}개"
          f"  (분리가 됐으면 작아야 한다)")

    results = {}
    for label, engine in engines:
        weights = build_weight_panel(engine, panels)
        held = sorted(weights.columns[(weights != 0).any(axis=0)])
        raw_result = run_weighted_backtest(
            weights, panels.price_frames(held), CostModel(), initial_capital=CAPITAL
        )
        results[label] = restrict(raw_result, schedule[0], CAPITAL)

    show("[성과] 비용 차감 후", pd.DataFrame(
        [summarize(f"{label}: {engine.name}", results[label], CAPITAL)
         for label, engine in engines]
    ).set_index("구분"))

    bench_windows = window_returns(results["벤치마크"].returns, schedule)
    excess = {
        label: (window_returns(results[label].returns, schedule) - bench_windows).dropna()
        for label in ("전략", "기업행위", "제외바스켓", "자산성장")
    }

    print("\n" + "=" * 82)
    print("초과수익 vs 유니버스 동일가중 (비겹침 20일 구간)")
    print("=" * 82)
    show("", pd.DataFrame([
        {"구분": label, "관측": len(s), "구간당": s.mean(),
         "연율화": annualize(s.mean()), "t": t_stat(s)}
        for label, s in excess.items()
    ]).set_index("구분"))

    strategy_t = t_stat(excess["전략"])
    turnover = summarize("", results["전략"], CAPITAL)["연회전율"]

    print("\n" + "=" * 82)
    print("판정 — 사전 등록 기준과 대조")
    print("=" * 82)
    checks = [
        ("주 판정", strategy_t > THRESHOLD,
         f"t {strategy_t:+.2f} vs {THRESHOLD} (양수여야 함), 초과 연 {annualize(excess['전략'].mean()):+.2%}"),
        ("FC1 기업행위", t_stat(excess["기업행위"]) < strategy_t,
         f"기업행위 t {t_stat(excess['기업행위']):+.2f} < 순발행 t {strategy_t:+.2f}"),
        ("FC2 바스켓", annualize(excess["제외바스켓"].mean()) < 0,
         f"제외 바스켓 {annualize(excess['제외바스켓'].mean()):+.2%} (음수여야 함)"),
        ("FC3 회전율", turnover <= MAX_TURNOVER, f"{turnover:.2f} vs 상한 {MAX_TURNOVER}"),
        ("FC4 자산성장", t_stat(excess["자산성장"]) < strategy_t,
         f"자산성장 t {t_stat(excess['자산성장']):+.2f} < 순발행 t {strategy_t:+.2f}"),
    ]
    print()
    for name, passed, detail in checks:
        print(f"  {name:14s} {'통과' if passed else '실패'}   {detail}")

    verdict = all(passed for _, passed, _ in checks)
    print(f"\n>>> {'통과' if verdict else '기각'} — 주 판정과 반증 확인 넷을 모두 충족해야 한다.")

    print("\n" + "=" * 82)
    print("연도별 수익률 (해석용, 판정 기준 아님)")
    print("=" * 82)
    yearly = pd.DataFrame({
        label: r.returns.groupby(r.returns.index.year).apply(lambda x: (1 + x).prod() - 1)
        for label, r in results.items()
    })
    yearly["전략-벤치"] = yearly["전략"] - yearly["벤치마크"]
    show("", yearly)


if __name__ == "__main__":
    main()
