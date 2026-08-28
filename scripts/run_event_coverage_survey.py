"""
이벤트 후보들의 **표본과 커버리지만** 잰다. 수익률은 보지 않는다.

왜 이걸 먼저 하나: 자사주 검정에서 **이벤트의 78%를 유니버스가 버렸다.** 그걸
모르고 사전 등록하면 검정력 없는 검정을 하게 된다(CLAUDE.md 규율 7). 그래서
가설을 고르기 **전에** 각 후보가 이 유니버스에서 몇 건이나 살아남는지 센다.

**이 스크립트는 수익률을 계산하지 않는다.** 여러 이벤트의 성과를 보고 좋은 것을
고르면 그 순간 데이터 마이닝이다. 여기서 고르는 기준은 오직 **표본 크기와
실현 가능성**이고, 그건 결과와 무관한 정보다.

두 가지 쓰임새를 나눠서 잰다. 롱온리에서 이벤트는 두 방향으로 쓸 수 있다:

  **보유형** (양의 이벤트) - 그 일이 일어난 종목을 산다. 자사주·PEAD가 이 형태였다.
      결정적 수치: 사건이 5종목 이상 살아있는 리밸런싱 시점이 몇 번인가.

  **제외형** (음의 이벤트) - 그 일이 일어난 종목을 유니버스에서 뺀다. 공매도를
      못 하는 대신 '안 사는' 것으로 하위 꼬리 정보를 쓰는 유일한 방법이다.
      결정적 수치: **유니버스의 몇 %를 제외하게 되는가.**

      제외형의 기대 초과 = 제외비중 x (유니버스수익 - 제외종목수익)
      이므로, 제외비중이 3%면 제외 종목이 연 10%p 못해도 초과는 0.3%p다.
      **검출 하한(연 3.4%) 근처에도 못 간다.** 이 비중이 실현 가능성을 정한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import settings
from src.data_loader.dart_filings import event_dates, load_filings, recent_event_mask
from src.data_loader.panels import Panels
from src.strategy.base import periodic_schedule

TOP_K = 500
HORIZON = settings.FORWARD_HORIZON
EVENT_WINDOW = 60  # 공시 후 며칠까지 신호로 볼 것인가 (PEAD·자사주와 동일)
MIN_EVENT_NAMES = 5  # 이보다 적으면 그 시점은 이벤트 포트폴리오를 못 만든다
FILING_START, FILING_END = "2015-01-01", "2026-08-31"

# 후보. 자사주 취득(신탁계약 포함)은 이미 두 번 검정해 기각했으므로 뺀다.
CANDIDATES = [
    ("유상증자결정", ["유상증자결정"], "제외형", "희석. 문헌상 발행 후 장기 부진"),
    ("전환사채권발행결정", ["전환사채권발행결정"], "제외형", "잠재 희석"),
    ("자기주식처분결정", ["자기주식처분결정"], "제외형", "물량이 시장에 나옴"),
    ("회사합병결정", ["회사합병결정"], "보유형", "인수기업 효과는 문헌상 모호"),
    ("타법인주식양수결정", ["타법인주식및출자증권양수결정"], "보유형", "인수"),
    ("무상증자결정", ["무상증자결정"], "보유형", "희석 없음. 한국서 양의 공시효과 보고"),
    ("감자결정", ["감자결정"], "보유형", "자본감소. 부실 신호일 수도"),
    ("유형자산양수결정", ["유형자산양수결정"], "제외형", "설비투자 = 자산성장"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="유니버스 크기. 이벤트는 소형주에 몰려 있어 넓히면 커버리지가 오른다")
    parser.add_argument("--window", type=int, default=EVENT_WINDOW,
                        help="이벤트 창(달력일). 문헌의 발행/매입 드리프트는 1~4년에 걸쳐 보고된다")
    args = parser.parse_args()

    print("데이터 로딩...", flush=True)
    panels = Panels.load(top_k=args.top_k)
    dates = panels.dates
    members = set(panels.members)
    schedule = periodic_schedule(dates, HORIZON)

    filings = load_filings(FILING_START, FILING_END)
    universe_size = panels.universe.loc[schedule].sum(axis=1)

    print("\n" + "=" * 92)
    print("이벤트 후보 커버리지 조사 — 표본만 본다. 수익률은 계산하지 않는다.")
    print("=" * 92)
    print(f"유니버스 시총 상위 {args.top_k}위 (일평균 {universe_size.mean():.0f}종목),"
          f" 이벤트 창 {args.window}일")
    print(f"리밸런싱 {len(schedule)}회 ({schedule[0].date()} ~ {schedule[-1].date()})")
    print(f"편입 이력 종목 {len(members):,}개")

    rows = []
    for label, kinds, usage, note in CANDIDATES:
        events = event_dates(filings, kinds)
        inside = events[events["ticker"].isin(members)]
        mask = recent_event_mask(inside, dates, panels.close.columns, args.window)

        flagged = (mask.loc[schedule] & panels.universe.loc[schedule])
        n_names = flagged.sum(axis=1)
        share = (n_names / universe_size).replace([float("inf")], float("nan"))

        rows.append({
            "이벤트": label,
            "쓰임": usage,
            "원본사건": len(events),
            "유니버스내": len(inside),
            "커버리지": len(inside) / len(events) if len(events) else float("nan"),
            f"창내{MIN_EVENT_NAMES}종목+": int((n_names >= MIN_EVENT_NAMES).sum()),
            "시점당종목": n_names.mean(),
            "유니버스비중": share.mean(),
        })

    table = pd.DataFrame(rows).set_index("이벤트")
    print()
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(table.to_string())

    print("\n" + "-" * 92)
    print("읽는 법")
    print("-" * 92)
    print(f"  커버리지     원본 사건 중 이 유니버스에 든 비율. 자사주 때 0.22였다.")
    print(f"  창내{MIN_EVENT_NAMES}종목+   {len(schedule)}회 중 포트폴리오를 만들 수 있는 시점 수."
          f" 보유형의 검정력을 정한다.")
    print(f"  유니버스비중  제외형이 빼게 되는 비중. **여기가 3% 아래면 제외형은")
    print(f"               기대 초과가 검출 하한(연 3.4%)에 못 미친다.**")
    print()
    print("이 표만 보고 하나를 고른다. 수익률을 본 적이 없으므로 고르는 행위가")
    print("결과에 오염되지 않는다.")


if __name__ == "__main__":
    main()
