"""
가설 검정: 자사주 취득을 공시한 종목이 이후 초과수익을 내는가.

**왜 이벤트인가.** 지금까지 일곱 번 검정해서 얻은 가장 견고한 결론은, 한국 대형주의
횡단면 예측력이 **하위 구간에 몰려 있다**는 것이다. 밸류는 스프레드의 62%가 공매도
쪽에 있었고(IC t 2.95인데 롱온리 초과 t 1.33), 저변동성은 거의 전부가 그랬다.
랭킹 팩터는 양쪽 꼬리를 다 써야 하는데 우리는 한쪽을 못 쓴다.

이벤트 공시는 다르다. "그 일이 일어난 종목만 산다"라서 하위 꼬리를 쓸 일이 애초에
없다. 살아남은 유일한 후보인 PEAD도 이 종류다(스프레드의 61%가 살 수 있는 쪽).

가설의 근거: 자사주 매입 공시 후의 초과수익은 Ikenberry-Lakonishok-Vermaelen(1995)
이후 여러 시장에서 반복 확인됐다. 설명은 경영진이 자기 회사가 저평가됐다고 볼 때
사들인다는 신호 효과다. 한국은 2024년 밸류업 프로그램 이후 특히 흔해졌다.

**사전에 정한 것** (돌려보고 고르지 않기 위해 먼저 못박는다):

  H1  자기주식취득결정 또는 자기주식취득신탁계약체결결정을 공시한 종목을
      공시일부터 EVENT_WINDOW일간 동일가중 보유하면, 같은 날 유니버스 동일가중보다
      높은 수익을 낸다.

  판정: |t| > 1.96. 사전에 정한 단일 가설이므로 다중검정 보정을 쓰지 않는다.
  측정: 20일 forward return, 비겹침 20일 간격 샘플링. PEAD·밸류와 같은 잣대다.
  창:  60일. PEAD의 표류 창과 같게 둔다 - 여기서 다른 값을 고르면 그것부터가 선택이다.
  최소 종목: 5. 그보다 적으면 한 종목이 결과를 좌우한다.

  **반증 확인(검정 아님)**: 자기주식'처분'은 회사가 물량을 시장에 내놓는 일이라
  방향이 반대여야 한다. 처분도 양수로 나오면 우리가 재는 것이 자사주 효과가 아니라
  '공시를 많이 하는 회사' 같은 다른 무엇이라는 뜻이다. 이건 가설로 주장하지 않으므로
  검정 횟수에 넣지 않는다 - 부호만 본다.

---

**2026-08-28 1차 결과 (시총 1~200위): 미달.** 연 6.98%, t 1.29, 관측 92개.
부호는 이론대로였고(처분 -1.88%) 크기도 연 7%였지만 유의하지 않았다.

그런데 자사주 취득 사건 4,692건 중 **상위 200 유니버스 안에 든 것은 1,054건(22%)**
뿐이었다. 가격 패널에는 4,595건(98%)이 있으므로 데이터가 없어서가 아니라 유니버스를
잘라서 버린 것이다. 자사주 매입은 원래 중소형주에서 흔하다 - 대형주는 저평가 신호를
보낼 일이 적다. 현상이 가장 약한 구간에서 찾은 셈이다.

**2차 사전 등록 (시총 201~1000위).** 결과를 보기 전에 다음을 못박았고, 근거는
전부 사건 분포와 유동성이지 수익률이 아니다:

  유니버스   시총 201~1000위. 사건 1,910건(상위200의 3.3배)이 들어오면서
             일평균거래대금 중앙값이 13~31억으로 개인 규모에 감당된다.
             1001위 밖은 사건이 가장 많지만(2,071건) 거래대금 중앙값이 4억이라
             비용을 감당할 수 없고, 이미 기각된 소형주 가설이 살던 구간이다.
  유동성 하한 최근 20일 평균 거래대금 10억원. 그 구간 하위 25%가 6.7억이므로
             아래 3분의 1가량을 자른다.
  판정       **|t| > 2.24** (Bonferroni 2건). 같은 현상을 두 번째 유니버스에서
             보는 것이므로 1.96을 그대로 쓰면 두 번의 기회를 한 번인 척하게 된다.
  2차 관문   비용 차감 후에도 양수여야 한다. 이건 별도 가설이 아니라 필요조건이다.

창(60일)·최소 종목수(5)·호라이즌(20일 비겹침)은 1차와 같게 둔다. 여기서 바꾸면
무엇이 결과를 바꿨는지 알 수 없게 된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import pandas as pd

from config import settings
from src.data_loader.dart_filings import event_dates, load_filings, recent_event_mask
from src.data_loader.panels import Panels
from src.data_loader.universe import market_cap_universe_mask

HORIZON = settings.FORWARD_HORIZON
PERIODS_PER_YEAR = 252 / HORIZON
EVENT_WINDOW = 60  # 공시 후 며칠까지 신호로 볼 것인가 (PEAD의 표류 창과 동일)
MIN_EVENT_NAMES = 5
LIQUIDITY_WINDOW = 20  # 거래대금 평균 구간. 과거만 쓰므로 look-ahead가 없다

ACQUISITION = ["자기주식취득결정", "자기주식취득신탁계약체결결정"]
DISPOSAL = ["자기주식처분결정"]


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def event_excess(
    mask: pd.DataFrame, forward: pd.DataFrame, universe: pd.DataFrame
) -> pd.DataFrame:
    """
    리밸런싱 시점마다 [이벤트 종목 평균, 유니버스 평균, 종목 수].

    비교 대상이 유니버스 전체 평균인 이유는, 신호를 안 쓰고 유니버스를 통째로
    동일가중 보유하는 것이 롱온리 투자자에게 가장 정직한 대안이기 때문이다.
    """
    rows = {}
    for date in forward.index[::HORIZON]:
        in_universe = universe.loc[date]
        in_event = mask.loc[date] & in_universe
        if in_event.sum() < MIN_EVENT_NAMES:
            continue
        returns = forward.loc[date]
        event_mean = returns[in_event].mean()
        universe_mean = returns[in_universe].mean()
        if pd.isna(event_mean) or pd.isna(universe_mean):
            continue
        rows[date] = {
            "이벤트": event_mean, "유니버스": universe_mean, "종목수": int(in_event.sum())
        }
    return pd.DataFrame(rows).T


def summarize(excess: pd.Series, label: str, names: pd.Series) -> dict:
    if excess.empty:
        return {"구분": label, "관측": 0}
    t_stat = (
        excess.mean() / (excess.std() / np.sqrt(len(excess))) if excess.std() > 0 else 0.0
    )
    return {
        "구분": label,
        "관측": len(excess),
        "평균 종목수": names.mean(),
        "구간당 초과": excess.mean(),
        "연율화": (1 + excess.mean()) ** PERIODS_PER_YEAR - 1,
        "t-stat": t_stat,
        "양수 비율": (excess > 0).mean(),
    }


def build_universe(panels: Panels, rank_from: int, rank_to: int, min_adv: float) -> pd.DataFrame:
    """시총 순위 구간 + 유동성 하한. 거래대금은 과거 20일 평균이라 미래를 안 본다."""
    band = market_cap_universe_mask(panels.market_cap, top_k=rank_to, rank_from=rank_from - 1)
    liquid = panels.trading_value.rolling(LIQUIDITY_WINDOW, min_periods=10).mean() >= min_adv
    return band & liquid & panels.tradeable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-from", type=int, default=201)
    parser.add_argument("--rank-to", type=int, default=1000)
    parser.add_argument("--min-adv", type=float, default=10.0, help="일평균거래대금 하한(억원)")
    parser.add_argument("--threshold", type=float, default=2.24)
    args = parser.parse_args()

    print("데이터 로딩...", flush=True)
    panels = Panels.load(start="2015-01-01")
    dates = panels.dates
    forward = panels.close.pct_change(HORIZON, fill_method=None).shift(-HORIZON)

    universe = build_universe(panels, args.rank_from, args.rank_to, args.min_adv * 1e8)
    members = sorted(universe.columns[universe.any(axis=0)])

    filings = load_filings("2015-01-01", dates[-1].strftime("%Y-%m-%d"))
    print(f"주요사항보고서 {len(filings):,}건 / {filings['stock_code'].nunique():,}종목")
    print(f"유니버스: 시총 {args.rank_from}~{args.rank_to}위, 일평균거래대금 {args.min_adv:.0f}억 이상")
    print(f"  편입 이력 종목 {len(members):,}개, 일평균 {universe.sum(axis=1).mean():.0f}종목")

    print("\n" + "=" * 74)
    print("자사주 관련 공시 (원본만, 정정·연장 제외)")
    print("=" * 74)
    buyback = filings[filings["kind"].str.contains("자기주식", na=False)]
    counts = (
        buyback[buyback["is_original"]]
        .groupby([buyback["event_date"].dt.year, "kind"])
        .size()
        .unstack(fill_value=0)
    )
    show("[연도별 건수]", counts)
    print(f"\n  정정·연장 등 제외분: {(~buyback['is_original']).sum():,}건"
          f" / 전체 {len(buyback):,}건")

    results, detail = [], {}
    for label, kinds in (("H1 자사주 취득", ACQUISITION), ("(반증확인) 자사주 처분", DISPOSAL)):
        events = event_dates(filings, kinds)
        in_universe = events[events["ticker"].isin(members)]
        mask = recent_event_mask(in_universe, dates, panels.close.columns, EVENT_WINDOW)
        table = event_excess(mask, forward, universe)
        if table.empty:
            results.append({"구분": label, "관측": 0})
            continue
        excess = table["이벤트"] - table["유니버스"]
        detail[label] = table
        results.append(summarize(excess, label, table["종목수"]))
        print(f"\n{label}: 사건 {len(events):,}건 중 유니버스 내 {len(in_universe):,}건")

    print("\n" + "=" * 74)
    print(f"판정 (사전 기준 |t| > {args.threshold:.2f}, 창 {EVENT_WINDOW}일)")
    print("=" * 74)
    show("", pd.DataFrame(results).set_index("구분"))

    if "H1 자사주 취득" in detail:
        table = detail["H1 자사주 취득"]
        excess = (table["이벤트"] - table["유니버스"]).dropna()
        yearly = excess.groupby(excess.index.year).agg(["count", "mean"])
        show("[H1 연도별 초과]", yearly)

    print(
        "\n반증 확인: 처분이 양수로 나오면 자사주 효과가 아니라 다른 것을 재고 있다는 뜻이다."
        "\n창 길이에 대한 민감도는 이 검정에 포함하지 않는다 - 60일을 먼저 정하고 한 번만 본다."
    )


if __name__ == "__main__":
    main()
