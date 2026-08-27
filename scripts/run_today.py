"""
지금 이 전략이 들고 있어야 할 포트폴리오.

백테스트가 "굴렸으면 어땠을까"라면 이쪽은 "그래서 지금 뭘 들고 있어야 하는가"다.
그리고 **매번 돌릴 때마다 그 판단을 기록으로 남긴다** - 이 프로젝트에 없는 것이
진짜 아웃오브샘플인데, 표본이 쌓이기를 기다리는 방법은 굴리면서 기록하는 것뿐이다.
기록해두지 않으면 나중에 "그때 정말 이걸 샀을까"를 확인할 수 없다.

주의: 이 파일은 매매 지시가 아니다. PEAD는 비겹침 관측 61개에 t 2.5인 가설이고
확정된 것이 아니다. 근거와 유보는 experiments/log.md에 있다.

사용:
    python scripts/run_today.py
    python scripts/run_today.py --capital 3000     # 3천만원
    python scripts/run_today.py --strategy pead-cash
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import settings
from src.data_loader.krx_openapi import fetch_base_info, fetch_daily_trading
from src.data_loader.panels import Panels
from src.strategy.base import EqualWeightUniverse, build_weight_panel
from src.strategy.pead import PeadStrategy

LIVE_DIR = PROJECT_ROOT / "data" / "processed" / "live"
STALE_SIGNAL_DAYS = 120  # 분기 공시 주기보다 길면 데이터가 멈춘 것이다

STRATEGIES = {
    "pead": lambda: PeadStrategy(hold_universe_when_idle=True),
    "pead-cash": lambda: PeadStrategy(hold_universe_when_idle=False),
    "universe": EqualWeightUniverse,
}


def ticker_names() -> pd.Series:
    try:
        # fetch_base_info는 이미 ticker를 인덱스로 돌려준다
        return fetch_base_info()["name"].str.replace("보통주$", "", regex=True)
    except Exception as error:  # 이름은 있으면 좋은 것이지 없다고 멈출 일은 아니다
        print(f"  (종목명 조회 실패: {type(error).__name__} - 코드로만 표시한다)")
        return pd.Series(dtype=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="pead", choices=sorted(STRATEGIES))
    parser.add_argument("--top-k", type=int, default=settings.UNIVERSE_TOP_K,
                        help="유니버스 크기 (시총 상위 N종목)")
    parser.add_argument("--capital", type=float, default=settings.INITIAL_CAPITAL / 1e4,
                        help="투자금, 만원 단위 (기본 10000 = 1억)")
    parser.add_argument("--no-record", action="store_true", help="기록을 남기지 않는다")
    args = parser.parse_args()

    capital = args.capital * 1e4

    print("데이터 로딩...", flush=True)
    panels = Panels.load(top_k=args.top_k)
    strategy = STRATEGIES[args.strategy]()
    strategy.prepare(panels)

    as_of = panels.dates[-1]
    today = pd.Timestamp.today().normalize()
    schedule = strategy.rebalance_dates()
    rebalance_date = schedule[-1]
    since_rebalance = int((panels.dates > rebalance_date).sum())
    horizon = getattr(strategy, "horizon", settings.FORWARD_HORIZON)

    print("\n" + "=" * 74)
    print(f"{strategy.name}  |  투자금 {capital / 1e8:,.2f}억원")
    print("=" * 74)
    print(f"\n가격 데이터 기준일: {as_of.date()}  (오늘 {today.date()})")
    if (today - as_of).days > 4:
        print(f"  ** {(today - as_of).days}일 지난 데이터다."
              " scripts/collect_openapi_panel.py를 먼저 돌릴 것")

    if isinstance(strategy, PeadStrategy):
        latest = strategy.latest_signal_date()
        if latest is None:
            print("  ** 유효한 SUE 신호가 하나도 없다. DART 재무를 확인할 것")
            return
        stale = (as_of - latest).days
        print(f"신호가 살아있는 마지막 날: {latest.date()} ({stale}일 전)")
        if stale > STALE_SIGNAL_DAYS:
            print(f"  ** 신호가 {stale}일째 멈춰 있다. 최신 공시가 안 받아졌을 가능성이 높다:"
                  "\n     python scripts/collect_dart_fundamentals.py")

    print(f"\n마지막 리밸런싱일: {rebalance_date.date()} ({since_rebalance}거래일 전)")
    print(f"다음 리밸런싱: {horizon - since_rebalance}거래일 뒤"
          if since_rebalance < horizon else
          f"다음 리밸런싱: **지금이 리밸런싱 시점이다** ({since_rebalance - horizon}거래일 지남)")

    target = strategy.target_weights(rebalance_date)
    if target.empty:
        print("\n목표 포트폴리오: **현금 100%** (신호가 없는 구간)")
        return

    signalled = getattr(strategy, "signal_available", lambda d: True)(rebalance_date)
    print(f"\n상태: {'상위분위 보유' if signalled else '신호 없음 - 유니버스 대기'}")

    raw = fetch_daily_trading(as_of)
    names = ticker_names()

    holdings = pd.DataFrame({"비중%": (target * 100).round(2)})
    holdings["배분액"] = (target * capital).round(0)
    holdings["종가"] = raw["close"].reindex(holdings.index)
    holdings["주식수"] = (holdings["배분액"] / holdings["종가"]).fillna(0).astype(int)
    holdings.insert(0, "종목명", names.reindex(holdings.index).fillna(""))
    holdings = holdings.sort_values("비중%", ascending=False)

    print(f"\n목표 포트폴리오: {len(holdings)}종목, 종목당 {target.iloc[0]:.2%}")
    shown = holdings if len(holdings) <= 40 else holdings.head(15)
    if len(shown) < len(holdings):
        print(f"  (종목이 많아 상위 {len(shown)}개만 표시. 전체는 아래 기록 파일에 있다)")
    print(shown.to_string(formatters={
        "비중%": "{:.2f}".format, "배분액": "{:,.0f}".format, "종가": "{:,.0f}".format,
    }))

    unpriced = holdings[holdings["종가"].isna()]
    if len(unpriced):
        print(f"\n  ** 종가를 못 구한 종목 {len(unpriced)}개: {list(unpriced.index)}")

    # 실행 가능성. 한 주도 못 사는 종목이 있으면 이 포트폴리오는 종이 위에만 존재한다.
    unaffordable = holdings[(holdings["주식수"] == 0) & holdings["종가"].notna()]
    if len(unaffordable):
        needed = holdings["종가"].max() * len(holdings)
        print(f"\n  ** 한 주도 못 사는 종목 {len(unaffordable)}개 / {len(holdings)}개."
              f" 이 투자금({capital / 1e4:,.0f}만원)으로는 실행할 수 없는 포트폴리오다.")
        print(f"     전 종목 1주 이상이 되려면 최소 {needed / 1e4:,.0f}만원이 필요하다"
              f" (가장 비싼 종목 {holdings['종가'].max():,.0f}원 x {len(holdings)}종목).")
        print("     백테스트는 비중을 소수로 다루므로 이 제약을 반영하지 않는다 -"
              " 소액에서는 백테스트 성과를 그대로 얻을 수 없다는 뜻이다.")

    if not args.no_record:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        path = LIVE_DIR / f"{args.strategy}_{rebalance_date.strftime('%Y%m%d')}.parquet"
        record = holdings.assign(
            rebalance_date=rebalance_date, as_of=as_of, recorded_at=pd.Timestamp.now()
        )
        record.to_parquet(path)
        print(f"\n기록: {path.relative_to(PROJECT_ROOT)}")
        print(f"  누적 기록 {len(list(LIVE_DIR.glob(f'{args.strategy}_*.parquet')))}건."
              " 이게 쌓이면 사후에 '그때 실제로 이걸 골랐는가'를 대조할 수 있다.")


if __name__ == "__main__":
    main()
