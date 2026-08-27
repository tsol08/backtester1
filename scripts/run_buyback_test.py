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
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import settings
from src.data_loader.dart_filings import event_dates, load_filings, recent_event_mask
from src.data_loader.panels import Panels

HORIZON = settings.FORWARD_HORIZON
PERIODS_PER_YEAR = 252 / HORIZON
EVENT_WINDOW = 60  # 공시 후 며칠까지 신호로 볼 것인가 (PEAD의 표류 창과 동일)
MIN_EVENT_NAMES = 5

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


def main() -> None:
    print("데이터 로딩...", flush=True)
    panels = Panels.load(start="2015-01-01")
    dates = panels.dates
    forward = panels.close.pct_change(HORIZON, fill_method=None).shift(-HORIZON)

    filings = load_filings("2015-01-01", dates[-1].strftime("%Y-%m-%d"))
    print(f"주요사항보고서 {len(filings):,}건 / {filings['stock_code'].nunique():,}종목")

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
        in_universe = events[events["ticker"].isin(panels.members)]
        mask = recent_event_mask(in_universe, dates, panels.close.columns, EVENT_WINDOW)
        table = event_excess(mask, forward, panels.universe & panels.tradeable)
        if table.empty:
            results.append({"구분": label, "관측": 0})
            continue
        excess = table["이벤트"] - table["유니버스"]
        detail[label] = table
        results.append(summarize(excess, label, table["종목수"]))
        print(f"\n{label}: 사건 {len(events):,}건 중 유니버스 내 {len(in_universe):,}건")

    print("\n" + "=" * 74)
    print(f"판정 (사전 기준 |t| > 1.96, 창 {EVENT_WINDOW}일)")
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
