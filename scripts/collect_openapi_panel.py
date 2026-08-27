"""
KRX Open API로 전종목 일별 시세+시가총액 패널을 수집한다.

정식 채널이다. 인증키를 발급받고 서비스별 이용 승인을 거친 Open API를 쓴다.
pykrx로 data.krx.co.kr을 스크래핑하던 방식은 이용약관 위반으로 계정이 차단된 적이
있고, 그때 KRX 안내문이 직접 권한 정식 경로가 바로 이것이다. 이 스크립트는 pykrx를
전혀 import하지 않는다.

거래일 달력을 따로 받지 않고 **영업일 전체를 순회**하는 이유: Open API는 휴장일에
빈 응답을 준다. 즉 응답 자체가 달력 역할을 한다. pykrx 기반 달력은 2014년 중반부터만
있어서 2010~2014 구간을 커버하지 못한다.

빈 결과도 캐싱되므로 재실행하면 이미 받은 날짜는 건너뛴다. 중간에 끊겨도 다시 돌리면
빠진 것만 채운다.

사용법:
    python scripts/collect_openapi_panel.py [시작일] [종료일]
    (기본값: 2010-01-01 ~ 오늘)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader.krx_openapi import fetch_daily_trading

DEFAULT_START = "2010-01-01"
PROGRESS_EVERY = 200


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_START
    end = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.today().strftime("%Y-%m-%d")

    days = pd.bdate_range(start, end)
    print(f"영업일 {len(days)}일 ({start} ~ {end}) 수집 시작", flush=True)

    started = time.time()
    traded = holiday = failed = 0

    for i, date in enumerate(days):
        try:
            if len(fetch_daily_trading(date)):
                traded += 1
            else:
                holiday += 1
        except Exception as exc:
            failed += 1
            print(f"  {date.date()} 실패: {type(exc).__name__}", flush=True)

        if (i + 1) % PROGRESS_EVERY == 0:
            rate = (i + 1) / (time.time() - started)
            remaining = (len(days) - i - 1) / rate / 60
            print(
                f"  {i + 1}/{len(days)} (거래일 {traded}, 휴장 {holiday}, 실패 {failed})"
                f" - {rate:.1f}일/초, 남은시간 약 {remaining:.0f}분",
                flush=True,
            )

    elapsed = (time.time() - started) / 60
    print(f"완료: 거래일 {traded}, 휴장 {holiday}, 실패 {failed}, {elapsed:.1f}분", flush=True)

    if failed:
        print("실패분은 이 스크립트를 다시 실행하면 채워진다 (캐시된 날짜는 건너뜀).", flush=True)


if __name__ == "__main__":
    main()
