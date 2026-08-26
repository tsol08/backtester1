"""
전종목 일자별 패널 데이터를 수집한다.

날짜당 소스별 1회 호출로 전 종목을 받으므로, 종목별로 긴 기간을 요청하는 방식보다
훨씬 빠르고 유니버스 전체(900~2800종목)를 커버한다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.krx_panel import SOURCES, fetch_cross_section, trading_dates

START = "2017-01-01"
END = "2024-12-31"


def main() -> None:
    dates = trading_dates(START, END)
    print(f"거래일 {len(dates)}일 ({START} ~ {END})", flush=True)
    print(f"소스 {len(SOURCES)}개: {list(SOURCES)}", flush=True)

    for kind in SOURCES:
        print(f"\n[{kind}] 수집 시작", flush=True)
        started = time.time()
        ok, empty = 0, 0

        for i, date in enumerate(dates):
            try:
                df = fetch_cross_section(kind, date)
                if len(df):
                    ok += 1
                else:
                    empty += 1
            except Exception as exc:
                print(f"  {date.date()} 실패: {type(exc).__name__}", flush=True)

            if (i + 1) % 200 == 0:
                elapsed = time.time() - started
                rate = (i + 1) / elapsed
                remaining = (len(dates) - i - 1) / rate
                print(
                    f"  {i + 1}/{len(dates)} (성공 {ok}) "
                    f"- {rate:.1f}일/초, 남은 시간 약 {remaining / 60:.1f}분",
                    flush=True,
                )

        print(f"[{kind}] 완료: 성공 {ok}, 빈값 {empty}, 소요 {(time.time() - started) / 60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
