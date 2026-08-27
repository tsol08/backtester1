"""
KRX Open API(openapi.krx.co.kr) 기반 데이터 수집.

data.krx.co.kr 웹사이트 로그인 스크래핑(krx_extra.py)은 KRX 이용약관을 위반해
계정이 일시 차단된 적이 있다. 이 모듈은 그 대신 KRX가 공식 승인한 Open API
채널을 쓴다 — 인증키 발급 + 서비스별 활용 승인을 거친 정식 경로라 밴 위험이 없다.

무료 API가 주는 항목은 제한적이다: 일별 시세(OHLCV) + 시가총액/상장주식수,
종목 기본정보뿐이다. PER/PBR 같은 밸류에이션, 공매도, 투자자별 매매동향은
Open API 서비스 목록에 아예 없다(공식 서비스 목록 페이지에서 확인 완료).
시가총액이 daily trading 응답에 포함돼 있으므로, size 팩터는 이걸로 커버되고
book-to-market 등 밸류에이션은 DART 재무제표(자본총계 등)와 조합해 직접
계산해야 한다.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from src.data_loader.env import get_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENAPI_DIR = PROJECT_ROOT / "data" / "raw" / "openapi"

BASE_URL = "http://data-dbg.krx.co.kr/svc/apis"
# 0.2초 간격으로 80분 넘게 쉬지 않고 돌리면 DNS 조회 자체가 실패하기 시작하는
# 패턴이 두 번 반복 관측됐다(연속 요청량에 따른 일시적 제한으로 추정). 간격을
# 늘려 연속 요청 부담을 줄인다.
REQUEST_INTERVAL = 0.5

TRADING_ENDPOINTS = {
    "KOSPI": "sto/stk_bydd_trd",
    "KOSDAQ": "sto/ksq_bydd_trd",
}
BASE_INFO_ENDPOINTS = {
    "KOSPI": "sto/stk_isu_base_info",
    "KOSDAQ": "sto/ksq_isu_base_info",
}

TRADING_COLUMNS = {
    "TDD_OPNPRC": "open",
    "TDD_HGPRC": "high",
    "TDD_LWPRC": "low",
    "TDD_CLSPRC": "close",
    "ACC_TRDVOL": "volume",
    "ACC_TRDVAL": "trading_value",
    "MKTCAP": "market_cap",
    "LIST_SHRS": "listed_shares",
}
NUMERIC_TRADING_COLS = list(TRADING_COLUMNS.values())


def _get(endpoint: str, params: dict) -> list[dict]:
    headers = {"AUTH_KEY": get_api_key("KRX_API_KEY")}
    resp = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("OutBlock_1", [])


def _trading_path(date: pd.Timestamp) -> Path:
    return OPENAPI_DIR / "trading" / f"{date.strftime('%Y%m%d')}.parquet"


def fetch_daily_trading(date: pd.Timestamp, force_refresh: bool = False) -> pd.DataFrame:
    """특정 일자의 KOSPI+KOSDAQ 전종목 시세+시가총액 (인덱스=종목코드)."""
    path = _trading_path(date)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    rows = []
    for market, endpoint in TRADING_ENDPOINTS.items():
        records = _get(endpoint, {"basDd": date.strftime("%Y%m%d")})
        time.sleep(REQUEST_INTERVAL)
        for r in records:
            row = {"ticker": r["ISU_CD"], "market": market}
            for src_col, dst_col in TRADING_COLUMNS.items():
                row[dst_col] = r.get(src_col)
            rows.append(row)

    result = pd.DataFrame(rows)
    if len(result):
        result = result.set_index("ticker")
        result[NUMERIC_TRADING_COLS] = result[NUMERIC_TRADING_COLS].apply(
            pd.to_numeric, errors="coerce"
        )
        result = result[~result.index.duplicated(keep="last")]
    else:
        result = pd.DataFrame(columns=["market", *NUMERIC_TRADING_COLS])
        result.index.name = "ticker"

    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path)
    return result


def fetch_base_info(force_refresh: bool = False) -> pd.DataFrame:
    """
    KOSPI+KOSDAQ 전종목 기본정보(상장일/액면가/상장주식수 등).
    basDd는 필수 파라미터인데, 당일/휴장일은 아직 반영 전이라 비어있다.
    그래서 데이터가 나올 때까지 하루씩 거슬러 올라간다. 상장일 같은 정적
    정보 위주라 스냅샷 하나로 캐싱해도 충분하다.
    """
    as_of = pd.Timestamp.today() - pd.Timedelta(days=1)
    path = OPENAPI_DIR / "base_info" / f"{as_of.strftime('%Y%m%d')}.parquet"
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    rows = []
    for market, endpoint in BASE_INFO_ENDPOINTS.items():
        for lookback in range(7):
            probe_date = as_of - pd.Timedelta(days=lookback)
            records = _get(endpoint, {"basDd": probe_date.strftime("%Y%m%d")})
            time.sleep(REQUEST_INTERVAL)
            if records:
                break
        for r in records:
            rows.append(
                {
                    "ticker": r["ISU_SRT_CD"],
                    "market": market,
                    "name": r["ISU_NM"],
                    "listed_date": r["LIST_DD"],
                    "security_group": r["SECUGRP_NM"],
                    "par_value": r["PARVAL"],
                    "listed_shares": r["LIST_SHRS"],
                }
            )

    result = pd.DataFrame(rows).set_index("ticker")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path)
    return result


def build_close_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    기업행위가 보정된 종가 패널.

    분석 코드는 이걸 써야 한다. build_panel("close", ...)은 Open API 원본 종가라
    액면분할이 -98% 수익률로 잡힌다(삼성전자 2018-05-04 등). 자세한 내용은
    src/data_loader/price_adjust.py 참고.
    """
    from src.data_loader.price_adjust import load_adjusted_close

    return load_adjusted_close(build_panel("close", dates), build_panel("listed_shares", dates))


def build_panel(column: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """캐시된 날짜별 거래 데이터를 (날짜 x 종목) 패널로 조립한다."""
    rows = {}
    for date in dates:
        path = _trading_path(date)
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
