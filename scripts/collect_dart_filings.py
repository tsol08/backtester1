"""
DART 공시목록(주요사항보고서)을 월 단위로 수집한다.

재무제표 수집과 달리 종목별로 돌지 않는다. list.json은 날짜 구간으로 전 종목을
한꺼번에 주므로, 한 달에 5~6번 호출이면 그 달 전체가 받아진다.

중단해도 다시 실행하면 이어받는다(월별 parquet 캐시).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader.dart_filings import fetch_filing_month

START = "2015-01-01"


def main() -> None:
    end = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.today().strftime("%Y-%m-%d")
    months = pd.period_range(pd.Timestamp(START), pd.Timestamp(end), freq="M")
    print(f"수집 범위: {months[0]} ~ {months[-1]} ({len(months)}개월)", flush=True)

    started, total = time.time(), 0
    for i, month in enumerate(months, 1):
        total += len(fetch_filing_month(month))
        if i % 12 == 0 or i == len(months):
            print(f"  {month}  누적 {total:,}건  ({(time.time() - started) / 60:.1f}분)", flush=True)

    print(f"\n완료: {total:,}건 / {(time.time() - started) / 60:.1f}분")


if __name__ == "__main__":
    main()
