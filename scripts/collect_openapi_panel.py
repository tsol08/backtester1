"""
KRX Open API(공식 승인 채널)로 전종목 일별 시세+시가총액 패널을 수집한다.

krx_panel.py(pykrx의 data.krx.co.kr 로그인 스크래핑)는 계정이 밴당해 더 이상
쓰지 않는다. 이 스크립트는 그 대신 인증키 발급+서비스 승인을 받은 정식
Open API로 같은 종류의 데이터(OHLCV, 시가총액, 상장주식수)를 받는다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.krx_openapi import fetch_daily_trading
from src.data_loader.krx_panel import trading_dates

START = "2018-01-01"
END = "2024-12-31"


def main() -> None:
    dates = trading_dates(START, END)
    print(f"거래일 {len(dates)}일 ({START} ~ {END})", flush=True)

    started = time.time()
    ok, empty = 0, 0

    for i, date in enumerate(dates):
        try:
            df = fetch_daily_trading(date)
            if len(df):
                ok += 1
            else:
                empty += 1
        except Exception as exc:
            print(f"  {date.date()} 실패: {type(exc).__name__}: {exc}", flush=True)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - started
            rate = (i + 1) / elapsed
            remaining = (len(dates) - i - 1) / rate
            print(
                f"  {i + 1}/{len(dates)} (성공 {ok}, 빈값 {empty}) "
                f"- {rate:.1f}일/초, 남은 시간 약 {remaining / 60:.1f}분",
                flush=True,
            )

    print(f"완료: 성공 {ok}, 빈값 {empty}, 소요 {(time.time() - started) / 60:.1f}분", flush=True)


if __name__ == "__main__":
    main()
