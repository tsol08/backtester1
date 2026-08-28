"""
분할 보정이 놓치는 기업행위가 얼마나 되는가.

**왜 이걸 보나.** 유상증자 제외 검정(run_seo_exclusion_test.py)에서 대조군이던
무상증자 제외가 t 3.05로 본 가설(t 1.53)보다 잘 나왔다. 무상증자는 주주에게
공짜 주식을 주는 **양의 이벤트**인데 그걸 피하는 게 낫다는 건 앞뒤가 안 맞는다.
이 저장소는 이미 액면분할 미보정으로 결과 전체를 재계산한 적이 있다(2026-08-27).

의심되는 기계적 원인: `split_adjustment_factor`가 분할을 인정하는 조건에
**`raw_move > 0.30`**(원본 수익률이 30% 넘게 급변)이 들어 있다. 무상증자 20%면
주가가 1/1.2로 16.7%만 떨어지므로 이 조건에 안 걸리고 **보정되지 않는다.**
주주는 주식 수가 1.2배가 되어 손익이 0인데, 패널에는 -16.7%로 남는다.

    무상증자 100% -> 주가 -50.0% -> 보정됨
    무상증자  50% -> 주가 -33.3% -> 보정됨 (겨우)
    무상증자  20% -> 주가 -16.7% -> **안 됨**
    무상증자  10% -> 주가  -9.1% -> **안 됨**

여기서는 가설을 검정하지 않는다. **패널 전체에서 '주식수가 늘었고 주가가 그만큼
빠졌는데 보정되지 않은 날'을 세기만 한다.** 수익률 비교가 아니라 데이터 실사다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader.krx_openapi import build_panels, cached_trading_dates
from src.data_loader.price_adjust import MIN_SHARES_CHANGE, is_corporate_action

START, END = "2010-01-01", "2026-08-26"

# '주식수 변화가 주가 변화를 설명하는가'의 허용 오차. 보정 후 수익률이 이 안이면
# 그 날 움직임은 사실상 전부 기업행위 때문이라고 본다.
EXPLAINED = 0.05


def main() -> None:
    print("데이터 로딩...", flush=True)
    dates = cached_trading_dates(START, END)
    raw = build_panels(["close", "listed_shares"], dates)
    close, shares = raw["close"], raw["listed_shares"]

    price_ratio = close / close.shift(1)
    shares_ratio = shares / shares.shift(1)

    raw_move = (price_ratio - 1).abs()
    adjusted_move = (price_ratio * shares_ratio - 1).abs()

    # 현재 코드가 보정 대상으로 인정하는 날. 조건을 여기 다시 적지 않고 그대로
    # 부른다 - 두 곳에 적으면 갈라지고, 갈라진 것을 잡은 게 이 실사다.
    recognised = is_corporate_action(close, shares)

    # 기업행위가 분명한 날: 주식수가 5% 넘게 변했고, 주식수로 보정하면 움직임이
    # 거의 사라진다(= 주가 변동이 주식수 변화로 설명된다).
    corporate_action = (
        ((shares_ratio - 1).abs() > MIN_SHARES_CHANGE)
        & (adjusted_move < EXPLAINED)
        & (raw_move > EXPLAINED)
    )

    missed = corporate_action & ~recognised

    print("\n" + "=" * 78)
    print("기업행위 보정 실사 — 가설 검정이 아니라 데이터 확인이다")
    print("=" * 78)
    print(f"기간 {dates[0].date()} ~ {dates[-1].date()}, {len(dates):,}거래일,"
          f" {close.shape[1]:,}종목")
    print(f"\n현재 코드가 분할로 인정한 날      {int(recognised.sum().sum()):,}건")
    print(f"주식수로 설명되는 기업행위일       {int(corporate_action.sum().sum()):,}건")
    print(f"  그중 **보정되지 않은 것**       {int(missed.sum().sum()):,}건"
          f"  ({missed.sum().sum() / max(corporate_action.sum().sum(), 1):.0%})")

    if not missed.any().any():
        print("\n놓친 것이 없다. 의심이 틀렸다.")
        return

    rows, cols = np.where(missed.to_numpy())
    detail = pd.DataFrame({
        "date": close.index[rows],
        "ticker": close.columns[cols],
        "주가변동": [price_ratio.iat[r, c] - 1 for r, c in zip(rows, cols)],
        "주식수배율": [shares_ratio.iat[r, c] for r, c in zip(rows, cols)],
        "보정후": [price_ratio.iat[r, c] * shares_ratio.iat[r, c] - 1
                 for r, c in zip(rows, cols)],
    })

    print("\n놓친 건들의 주가변동 분포 (이 값이 그대로 가짜 수익률로 남는다)")
    print(detail["주가변동"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

    print(f"\n연도별 건수")
    print(detail.groupby(detail["date"].dt.year).size().to_string())

    print(f"\n영향받은 종목 {detail['ticker'].nunique():,}개."
          f" 가짜 하락(주가변동 < 0) {int((detail['주가변동'] < 0).sum()):,}건,"
          f" 가짜 상승 {int((detail['주가변동'] > 0).sum()):,}건")

    print("\n가장 큰 것 15건")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(detail.reindex(detail["주가변동"].abs().sort_values(ascending=False).index)
              .head(15).to_string(index=False))

    print("\n" + "-" * 78)
    print("남은 건이 있으면 판정 조건이 아직 뭔가를 놓치고 있다는 뜻이다.")


if __name__ == "__main__":
    main()
