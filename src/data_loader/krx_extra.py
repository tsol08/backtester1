"""
KRX 로그인 이후 사용 가능해진 데이터 중, 지금도 쓰이는 것만 남긴 모듈.

원래 여기 있던 종목별(ticker) x 기간(start~end) 방식의 밸류에이션/시총/수급/공매도/
지수 로더들은 src/data_loader/krx_panel.py의 "특정 날짜의 전종목" 방식으로 대체됐다
(종목 하나 5년치를 받는 데 16초 걸리던 게, 날짜 하나에 전종목을 받는 방식으로는
0.5초면 된다). 그 함수들은 제거했다.

load_all_market_cap만 남아있는데, 이건 애초에 "날짜 하나 x 전종목" 방식이라 애초부터
빠르고, 시점별(point-in-time) 유니버스 스냅샷(src/data_loader/universe.py)이
지금도 이 함수를 쓰고 있다.
"""
from __future__ import annotations

import time

import pandas as pd

from src.data_loader.env import import_pykrx_stock

stock = import_pykrx_stock()

REQUEST_INTERVAL = 0.2

CAP_COLUMNS = {"시가총액": "market_cap", "거래대금": "trading_value", "상장주식수": "shares_outstanding"}


def load_all_market_cap(date: str) -> pd.DataFrame:
    """
    특정 일자의 전종목 시가총액. 시점별(point-in-time) 유니버스 구성에 쓴다.
    '오늘 기준 대형주'가 아니라 '그 시점 기준 대형주'를 뽑을 수 있어 생존편향을 줄인다.
    """
    raw = stock.get_market_cap(pd.Timestamp(date).strftime("%Y%m%d"))
    time.sleep(REQUEST_INTERVAL)
    available = {k: v for k, v in CAP_COLUMNS.items() if k in raw.columns}
    result = raw.rename(columns=available)[list(available.values())]
    result.index.name = "ticker"
    return result
