"""
pykrx 래퍼 + 로컬 캐싱.

pykrx 호출은 느리고 KRX 서버에 부담을 주기 때문에, 한 번 받아온 종목의 OHLCV는
data/raw/{ticker}.parquet 에 저장해두고 같은 요청이 다시 오면 디스크에서 읽는다.
요청 범위가 캐시 범위를 벗어나면 필요한 만큼만 새로 받아 캐시를 갱신한다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader.env import import_pykrx_stock

stock = import_pykrx_stock()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

_COLUMN_MAP = {
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
    "등락률": "change_pct",
}
_COLUMNS = list(_COLUMN_MAP.values())


def _cache_path(ticker: str) -> Path:
    return RAW_DIR / f"{ticker}.parquet"


def _fetch(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = stock.get_market_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ticker)
    raw = raw.rename(columns=_COLUMN_MAP)[_COLUMNS]
    raw.index.name = "date"
    raw = raw.sort_index()
    return raw[~raw.index.duplicated(keep="last")]


def load_ohlcv(ticker: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    종목의 일별 OHLCV를 [start, end] 구간으로 반환한다 (컬럼: open/high/low/close/volume/change_pct).

    force_refresh=True 이면 캐시를 무시하고 pykrx에서 다시 받아온다.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    cache_path = _cache_path(ticker)

    cached = None
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        if len(cached) > 0 and start_ts >= cached.index.min() and end_ts <= cached.index.max():
            return cached.loc[start_ts:end_ts].copy()

    fetch_start = start_ts if cached is None or len(cached) == 0 else min(start_ts, cached.index.min())
    fetch_end = end_ts if cached is None or len(cached) == 0 else max(end_ts, cached.index.max())

    fresh = _fetch(ticker, fetch_start, fetch_end)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(cache_path)

    return fresh.loc[start_ts:end_ts].copy()
