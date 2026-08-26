"""
KRX 로그인 이후 사용 가능해진 데이터: 밸류에이션(PER/PBR), 시가총액, 투자자별 수급,
공매도, 시장지수.

가격(OHLCV)과 달리 이 데이터들은 성격이 다른 별개의 정보원이다:
- 밸류에이션: 이익/자산 대비 주가가 싼가 (펀더멘털)
- 수급: 외국인/기관이 사는가 (가격에 안 담긴 주체별 행동)
- 공매도: 하락에 베팅하는 자금 규모
- 시장지수: 개별 종목의 초과수익(시장 대비)을 계산하는 기준

모두 일별 시계열이며 종목별 parquet으로 캐싱한다.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.data_loader.env import ensure_krx_credentials

ensure_krx_credentials()

from pykrx import stock  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

REQUEST_INTERVAL = 0.2

FUNDAMENTAL_COLUMNS = {"BPS": "bps", "PER": "per", "PBR": "pbr", "EPS": "eps", "DIV": "div_yield", "DPS": "dps"}
CAP_COLUMNS = {"시가총액": "market_cap", "거래대금": "trading_value", "상장주식수": "shares_outstanding"}
INVESTOR_COLUMNS = {"기관합계": "institution_net", "외국인합계": "foreign_net", "개인": "individual_net"}
SHORT_COLUMNS = {"공매도": "short_volume", "비중": "short_ratio"}


def _cache_path(kind: str, ticker: str) -> Path:
    return RAW_DIR / kind / f"{ticker}.parquet"


def _load_cached(
    kind: str, ticker: str, start: str, end: str, fetch, column_map: dict, force_refresh: bool = False
) -> pd.DataFrame:
    """pykrx 호출 + parquet 캐싱 공통 로직 (krx_loader.load_ohlcv와 같은 정책)."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    path = _cache_path(kind, ticker)

    cached = None
    if path.exists() and not force_refresh:
        cached = pd.read_parquet(path)
        if len(cached) and start_ts >= cached.index.min() and end_ts <= cached.index.max():
            return cached.loc[start_ts:end_ts].copy()

    fetch_start = start_ts if cached is None or not len(cached) else min(start_ts, cached.index.min())
    fetch_end = end_ts if cached is None or not len(cached) else max(end_ts, cached.index.max())

    raw = fetch(fetch_start.strftime("%Y%m%d"), fetch_end.strftime("%Y%m%d"), ticker)
    time.sleep(REQUEST_INTERVAL)

    available = {k: v for k, v in column_map.items() if k in raw.columns}
    fresh = raw.rename(columns=available)[list(available.values())]
    fresh.index.name = "date"
    fresh = fresh.sort_index()
    fresh = fresh[~fresh.index.duplicated(keep="last")]

    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(path)
    return fresh.loc[start_ts:end_ts].copy()


def load_valuation(ticker: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """PER/PBR/EPS/BPS/배당수익률. 밸류에이션 팩터의 원천."""
    return _load_cached(
        "valuation", ticker, start, end, stock.get_market_fundamental, FUNDAMENTAL_COLUMNS, force_refresh
    )


def load_market_cap(ticker: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """시가총액/거래대금/상장주식수. 유니버스 구성과 시장충격비용 계산에 쓴다."""
    return _load_cached("cap", ticker, start, end, stock.get_market_cap, CAP_COLUMNS, force_refresh)


def load_investor_flow(ticker: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """투자자 주체별 순매수 대금(기관/외국인/개인)."""
    return _load_cached(
        "investor",
        ticker,
        start,
        end,
        stock.get_market_trading_value_by_date,
        INVESTOR_COLUMNS,
        force_refresh,
    )


def load_shorting(ticker: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """공매도 거래량과 거래대금 대비 비중."""
    return _load_cached(
        "shorting", ticker, start, end, stock.get_shorting_volume_by_date, SHORT_COLUMNS, force_refresh
    )


def load_index_ohlcv(index_code: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    시장지수 OHLCV. 1001=KOSPI, 2001=KOSDAQ.
    개별 종목 수익률에서 시장 수익률을 빼면 '시장 대비 초과수익'이 된다.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    path = _cache_path("index", index_code)

    if path.exists() and not force_refresh:
        cached = pd.read_parquet(path)
        if len(cached) and start_ts >= cached.index.min() and end_ts <= cached.index.max():
            return cached.loc[start_ts:end_ts].copy()

    raw = stock.get_index_ohlcv(start_ts.strftime("%Y%m%d"), end_ts.strftime("%Y%m%d"), index_code)
    time.sleep(REQUEST_INTERVAL)

    column_map = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
    available = {k: v for k, v in column_map.items() if k in raw.columns}
    fresh = raw.rename(columns=available)[list(available.values())]
    fresh.index.name = "date"
    fresh = fresh.sort_index()

    path.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_parquet(path)
    return fresh.loc[start_ts:end_ts].copy()


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
