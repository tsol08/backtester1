"""
DART 분기 재무제표를 수집한다 (SUE/PEAD 분석의 입력).

**대상 선정이 이 스크립트의 핵심이다.** '오늘 시가총액 상위' 종목만 받으면 생존편향이
들어간다 — 그때는 컸지만 지금은 상장폐지·축소·합병된 기업이 통째로 빠지고, 살아남은
기업만 남아서 펀더멘털 검정이 실제보다 좋아 보인다. 실제로 확인해보니 시점별 상위
200위에 한 번이라도 편입된 종목이 473개인데 '오늘 기준' 후보군은 255개뿐이었다.

그래서 **시점별 유니버스에 한 번이라도 들어간 종목 전체**를 대상으로 한다.

수집 범위는 2015년부터다. DART는 대량조회(fnlttMultiAcnt)와 개별조회(fnlttSinglAcntAll)
모두 그 이전 데이터를 주지 않는다(2011·2013년 0건 확인). SUE가 8분기 표준편차 +
4분기 시차를 요구하므로 실제 신호는 2018년부터 나온다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.dart_loader import load_corp_codes, load_fundamentals_bulk
from src.data_loader.krx_openapi import build_panel, cached_trading_dates
from src.data_loader.universe import market_cap_universe_mask

TOP_K = 200
START_YEAR, END_YEAR = 2015, 2024
UNIVERSE_START, UNIVERSE_END = "2015-01-01", "2026-12-31"


def main() -> None:
    dates = cached_trading_dates(UNIVERSE_START, UNIVERSE_END)
    if len(dates) == 0:
        print("가격 패널이 없다. collect_openapi_panel.py를 먼저 실행할 것.")
        return

    market_cap = build_panel("market_cap", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)
    members = sorted(universe.columns[universe.any(axis=0)])

    dart_codes = set(load_corp_codes()["stock_code"])
    tickers = [t for t in members if t in dart_codes]

    print(f"시점별 상위{TOP_K}에 편입된 적 있는 종목: {len(members)}개")
    print(f"  그중 DART 고유번호 보유: {len(tickers)}개")
    print(f"수집 범위: {START_YEAR}~{END_YEAR}년", flush=True)

    started = time.time()
    fundamentals = load_fundamentals_bulk(tickers, START_YEAR, END_YEAR, verbose=True)

    print(
        f"\n완료: {len(fundamentals)}행 / {fundamentals['ticker'].nunique()}종목,"
        f" {(time.time() - started) / 60:.1f}분"
    )
    print("\n연도별 행수:")
    print(fundamentals.groupby("fiscal_year").size().to_string())


if __name__ == "__main__":
    main()
