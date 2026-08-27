"""
DART 지분공시 수집: 임원·주요주주 소유상황 변동(내부자 거래).

지금까지 쓴 정보원(가격, 재무제표, 시가총액)과 근본적으로 다른 종류의 데이터다.
앞의 것들은 전부 '시장이 이미 아는 사실'을 가공한 것이라, 어떤 조합을 해도 새로운
정보가 들어오지 않는다. 반면 내부자 거래는 **회사 사정을 가장 잘 아는 사람이 자기
돈으로 무엇을 했는가**라는, 가격에 아직 반영되지 않았을 수 있는 정보다.

핵심 필드는 sp_stock_lmp_irds_cnt(특정증권등 소유주식수 증감)다. 양수면 내부자가
샀고 음수면 팔았다는 뜻이다.

**look-ahead 방지**: rcept_dt(접수일자)를 기준으로 삼는다. 내부자는 거래를 한 뒤
며칠 내에 신고하므로 실제 거래일과 공시일이 다른데, 시장이 이 정보를 알게 되는 건
공시된 시점이다. 거래일 기준으로 쓰면 아직 공개되지 않은 정보를 쓰는 것이 된다.

**알려진 한계**: 이 API는 날짜/페이지 파라미터를 무시하고 항상 최근 2년치만 준다
(2024-08~2026-08 확인). 과거 이력을 늘릴 방법이 없어서, 인샘플/아웃오브샘플 분리가
불가능하고 표본이 얇다. 여기서 나오는 결과는 검증이 아니라 탐색으로 봐야 한다.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from src.data_loader.dart_loader import BASE_URL, load_corp_codes
from src.data_loader.env import get_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSIDER_DIR = PROJECT_ROOT / "data" / "raw" / "dart_insider"

REQUEST_INTERVAL = 0.4

NUMERIC_COLUMNS = ["shares_held", "shares_change"]

COLUMN_MAP = {
    "rcept_no": "rcept_no",
    "rcept_dt": "disclosed_date",
    "corp_code": "corp_code",
    "repror": "reporter",
    "isu_exctv_rgist_at": "is_registered_officer",
    "isu_exctv_ofcps": "position",
    "isu_main_shrholdr": "is_major_shareholder",
    "sp_stock_lmp_cnt": "shares_held",
    "sp_stock_lmp_irds_cnt": "shares_change",
}


def _path(ticker: str) -> Path:
    return INSIDER_DIR / f"{ticker}.parquet"


def _to_number(series: pd.Series) -> pd.Series:
    """'1,000' / '-500' / '-' 형태의 문자열을 숫자로. 빈값·하이픈은 NaN."""
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def fetch_insider_trades(ticker: str, corp_code: str, force_refresh: bool = False) -> pd.DataFrame:
    """한 종목의 임원·주요주주 소유변동 내역 전체."""
    path = _path(ticker)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    response = requests.get(
        f"{BASE_URL}/elestock.json",
        params={"crtfc_key": get_api_key("DART_API_KEY"), "corp_code": corp_code},
        timeout=30,
    )
    time.sleep(REQUEST_INTERVAL)
    payload = response.json()

    records = payload.get("list", []) if payload.get("status") == "000" else []
    if records:
        raw = pd.DataFrame(records)
        available = {k: v for k, v in COLUMN_MAP.items() if k in raw.columns}
        result = raw.rename(columns=available)[list(available.values())]
        result["ticker"] = ticker
        result["disclosed_date"] = pd.to_datetime(result["disclosed_date"], errors="coerce")
        for col in NUMERIC_COLUMNS:
            if col in result.columns:
                result[col] = _to_number(result[col])
        result = result.dropna(subset=["disclosed_date"])
    else:
        # 빈 결과에도 dtype을 명시한다. 안 그러면 object 컬럼이 되고, 나중에 다른
        # 종목들과 concat할 때 숫자 컬럼 전체가 object로 오염돼 nlargest 같은
        # 연산이 조용히 깨진다.
        result = pd.DataFrame(
            {
                **{c: pd.Series(dtype="float64") for c in NUMERIC_COLUMNS},
                "disclosed_date": pd.Series(dtype="datetime64[ns]"),
                **{
                    c: pd.Series(dtype="object")
                    for c in COLUMN_MAP.values()
                    if c not in NUMERIC_COLUMNS and c != "disclosed_date"
                },
                "ticker": pd.Series(dtype="object"),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path)
    return result


def load_insider_trades(tickers: list[str], verbose: bool = True) -> pd.DataFrame:
    """여러 종목의 내부자 거래를 한 테이블로 모은다."""
    mapping = load_corp_codes().drop_duplicates(subset="stock_code").set_index("stock_code")

    frames = []
    for i, ticker in enumerate(tickers):
        if ticker not in mapping.index:
            continue
        try:
            frames.append(fetch_insider_trades(ticker, mapping.loc[ticker, "corp_code"]))
        except Exception as exc:
            if verbose:
                print(f"  {ticker} 실패: {type(exc).__name__}", flush=True)

        if verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(tickers)}", flush=True)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_insider_signal(
    trades: pd.DataFrame,
    dates: pd.DatetimeIndex,
    shares_outstanding: pd.DataFrame,
    lookback: int = 120,
) -> pd.DataFrame:
    """
    (날짜 x 종목) 순매수 신호. 최근 lookback 거래일간 내부자 순매수 주식수를
    상장주식수로 나눈 값이다.

    상장주식수로 나누는 이유: 대형주는 절대 주식수가 크므로 나누지 않으면 규모
    차이만 재게 된다. 비율로 바꿔야 종목간 비교가 성립한다.

    t일 값에는 t일까지 **공시된** 건만 들어간다(rolling 윈도우가 과거만 본다).
    """
    if trades.empty:
        return pd.DataFrame(index=dates)

    daily = (
        trades.dropna(subset=["shares_change"])
        .groupby(["disclosed_date", "ticker"])["shares_change"]
        .sum()
        .unstack("ticker")
        .reindex(dates)
        .fillna(0.0)
    )

    net_bought = daily.rolling(lookback).sum()
    shares = shares_outstanding.reindex_like(net_bought)
    return (net_bought / shares.where(shares > 0)).replace([float("inf"), float("-inf")], pd.NA)
