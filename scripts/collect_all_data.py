"""
시점별 유니버스(344종목)에 대해 모든 정보원의 데이터를 수집한다.

정보원별로 성격이 다르다:
- 가격/거래량: 기술적 팩터
- 밸류에이션(PER/PBR): 싸게 거래되는가
- 수급: 외국인/기관이 사는가
- 공매도: 하락 베팅 규모
- 시가총액: 규모 팩터 + 유니버스 필터
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader.krx_extra import (
    load_index_ohlcv,
    load_investor_flow,
    load_market_cap,
    load_shorting,
    load_valuation,
)
from src.data_loader.krx_loader import load_ohlcv

START = "2016-01-01"
END = "2024-12-31"

SOURCES = [
    ("가격", load_ohlcv),
    ("밸류에이션", load_valuation),
    ("시가총액", load_market_cap),
    ("수급", load_investor_flow),
    ("공매도", load_shorting),
]


def universe_tickers() -> list[str]:
    snapshot_dir = PROJECT_ROOT / "data" / "raw" / "universe_snapshots"
    frames = [pd.read_parquet(p) for p in sorted(snapshot_dir.glob("*.parquet"))]
    return sorted(pd.concat(frames)["ticker"].unique())


def main() -> None:
    tickers = universe_tickers()
    print(f"대상: {len(tickers)}종목 ({START} ~ {END})", flush=True)

    print("\nKOSPI 지수 수집...", flush=True)
    index_df = load_index_ohlcv("1001", START, END)
    print(f"  KOSPI: {len(index_df)}일", flush=True)

    for source_name, loader in SOURCES:
        ok, failed = 0, []
        print(f"\n{source_name} 수집 시작...", flush=True)
        for i, ticker in enumerate(tickers):
            try:
                df = loader(ticker, START, END)
                if len(df):
                    ok += 1
                else:
                    failed.append(ticker)
            except Exception as exc:
                failed.append(f"{ticker}({type(exc).__name__})")

            if (i + 1) % 50 == 0:
                print(f"  진행 {i + 1}/{len(tickers)} (성공 {ok})", flush=True)

        print(f"  {source_name} 완료: 성공 {ok}/{len(tickers)}, 실패 {len(failed)}", flush=True)
        if failed:
            print(f"    실패 예시: {failed[:8]}", flush=True)


if __name__ == "__main__":
    main()
