"""
전종목 일자별(cross-section) 데이터 수집.

종목별로 긴 기간을 요청하는 방식(get_market_fundamental(start, end, ticker))은
한 종목 5년치에 16초나 걸린다. 반면 "특정 날짜의 전종목"을 받는 방식은 0.5초 남짓에
900~2800종목을 한 번에 준다. 팩터 분석은 어차피 '같은 날 여러 종목을 비교'하는
cross-sectional 작업이라, 후자가 필요한 데이터 모양과도 정확히 일치한다.

날짜별 parquet으로 캐싱하고, 나중에 팩터별 (날짜 x 종목) 패널로 조립한다.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.data_loader.env import import_pykrx_stock

stock = import_pykrx_stock()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_DIR = PROJECT_ROOT / "data" / "raw" / "panel"

REQUEST_INTERVAL = 0.15

# 소스별: (하위폴더, pykrx 호출, 컬럼 매핑)
SOURCES: dict[str, tuple[str, object, dict[str, str]]] = {
    "ohlcv": (
        "ohlcv",
        lambda d: stock.get_market_ohlcv(d),
        {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume", "거래대금": "trading_value"},
    ),
    "valuation": (
        "valuation",
        lambda d: stock.get_market_fundamental(d),
        {"BPS": "bps", "PER": "per", "PBR": "pbr", "EPS": "eps", "DIV": "div_yield", "DPS": "dps"},
    ),
    "cap": (
        "cap",
        lambda d: stock.get_market_cap(d),
        {"시가총액": "market_cap", "거래대금": "trading_value", "상장주식수": "shares_outstanding"},
    ),
    "shorting": (
        "shorting",
        lambda d: stock.get_shorting_volume_by_ticker(d),
        {"공매도": "short_volume", "비중": "short_ratio"},
    ),
}


def _path(kind: str, date: pd.Timestamp) -> Path:
    return PANEL_DIR / kind / f"{date.strftime('%Y%m%d')}.parquet"


def fetch_cross_section(kind: str, date: pd.Timestamp, force_refresh: bool = False) -> pd.DataFrame:
    """특정 날짜의 전종목 데이터를 받아온다 (인덱스=종목코드)."""
    path = _path(kind, date)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    _, fetch, column_map = SOURCES[kind]
    raw = fetch(date.strftime("%Y%m%d"))
    time.sleep(REQUEST_INTERVAL)

    if raw is None or len(raw) == 0:
        result = pd.DataFrame()
    else:
        available = {k: v for k, v in column_map.items() if k in raw.columns}
        result = raw.rename(columns=available)[list(available.values())]
        result.index.name = "ticker"
        result = result[~result.index.duplicated(keep="last")]

    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path)
    return result


def build_panel(kind: str, column: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    캐시된 날짜별 데이터를 (날짜 x 종목) 패널로 조립한다.
    IC 분석과 백테스트가 요구하는 모양이다.
    """
    rows = {}
    for date in dates:
        path = _path(kind, date)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if len(df) and column in df.columns:
            rows[date] = df[column]

    if not rows:
        return pd.DataFrame()

    panel = pd.DataFrame(rows).T
    panel.index.name = "date"
    return panel.sort_index()


def trading_dates(start: str, end: str) -> pd.DatetimeIndex:
    """
    KRX 실제 거래일 목록 (휴장일 제외).

    지수 API(get_index_ohlcv)는 로그인이 있어야 응답하는데, 우리는 로그인을
    쓰지 않기로 했다. 대신 어차피 캐시돼 있는 삼성전자 OHLCV의 날짜 인덱스를
    거래일 달력으로 쓴다 — 시가총액 1위 종목이 거래정지될 일은 사실상 없으니
    실제 거래일과 동일하다.
    """
    from src.data_loader.krx_loader import load_ohlcv

    prices = load_ohlcv("005930", start, end)
    return pd.DatetimeIndex(prices.index)
