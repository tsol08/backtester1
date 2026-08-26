"""
DART(전자공시시스템) 재무제표 수집 + 캐싱.

가장 중요한 설계 포인트는 **look-ahead bias 방지**다.
재무제표는 "회계기간 종료일"과 "실제 공시일"이 다르다. 예를 들어 삼성전자의 2023년
사업보고서(2023-12-31 기준)는 2024-03-12에야 공시됐다. 따라서 2024년 1월에
2023년 재무데이터를 알고 있었다고 가정하면 미래 정보를 쓰는 것이 된다.

DART 응답의 rcept_no 앞 8자리가 접수일자(공시일)이므로, 이를 available_date로 보존해
"그 날짜 이후에만 이 데이터를 쓸 수 있다"는 제약을 뒤 단계에서 강제할 수 있게 한다.
"""
from __future__ import annotations

import io
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.data_loader.env import get_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DART_DIR = RAW_DIR / "dart"

BASE_URL = "https://opendart.fss.or.kr/api"
REQUEST_INTERVAL = 0.5  # DART 서버 부담을 줄이기 위한 최소 요청 간격(초)
MAX_RETRIES = 4

# 보고서 코드 -> 회계기간 종료 월(분기)
REPORT_CODES = {
    "11013": 1,  # 1분기보고서
    "11012": 2,  # 반기보고서
    "11014": 3,  # 3분기보고서
    "11011": 4,  # 사업보고서
}

# IFRS 표준 계정 ID (account_nm은 회사마다 표기가 달라서 ID로 잡는다)
ACCOUNT_IDS = {
    "ifrs-full_Assets": "assets",
    "ifrs-full_Liabilities": "liabilities",
    "ifrs-full_Equity": "equity",
    "ifrs-full_Revenue": "revenue",
    "dart_OperatingIncomeLoss": "operating_income",
    "ifrs-full_ProfitLoss": "net_income",
    "ifrs-full_ProfitLossAttributableToOwnersOfParent": "net_income_controlling",
}


def _corp_code_path() -> Path:
    return DART_DIR / "corp_codes.parquet"


def load_corp_codes(force_refresh: bool = False) -> pd.DataFrame:
    """종목코드(stock_code) -> DART 고유번호(corp_code) 매핑. 상장사만."""
    path = _corp_code_path()
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    resp = requests.get(
        f"{BASE_URL}/corpCode.xml", params={"crtfc_key": get_api_key("DART_API_KEY")}, timeout=60
    )
    resp.raise_for_status()

    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    root = ET.fromstring(archive.read(archive.namelist()[0]).decode("utf-8"))

    rows = []
    for element in root.findall("list"):
        stock_code = (element.findtext("stock_code") or "").strip()
        if stock_code:  # 상장사만
            rows.append(
                {
                    "stock_code": stock_code,
                    "corp_code": element.findtext("corp_code").strip(),
                    "corp_name": element.findtext("corp_name").strip(),
                }
            )

    df = pd.DataFrame(rows).drop_duplicates(subset="stock_code", keep="last")
    DART_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def ticker_to_corp_code(ticker: str) -> str:
    mapping = load_corp_codes()
    match = mapping.loc[mapping["stock_code"] == ticker, "corp_code"]
    if match.empty:
        raise KeyError(f"DART corp_code를 찾을 수 없는 종목코드입니다: {ticker}")
    return match.iloc[0]


def _statement_path(corp_code: str, year: int, report_code: str) -> Path:
    return DART_DIR / f"fs_{corp_code}_{year}_{report_code}.parquet"


def _get_with_retry(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """DART가 연속 요청 시 연결을 끊는 경우가 있어, 지수 백오프로 재시도한다."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            time.sleep(REQUEST_INTERVAL)
            return resp
        except requests.exceptions.RequestException as exc:
            last_error = exc
            time.sleep(REQUEST_INTERVAL * (2**attempt))
    raise RuntimeError(f"DART 요청이 {MAX_RETRIES}회 실패했습니다: {type(last_error).__name__}")


def fetch_financial_statement(
    corp_code: str, year: int, report_code: str, force_refresh: bool = False
) -> pd.DataFrame:
    """
    한 기업의 한 보고서(연도+분기) 재무제표를 가져온다.

    반환 컬럼: fiscal_year, fiscal_quarter, available_date(공시일), assets, liabilities,
    equity, revenue, operating_income, net_income, net_income_controlling
    데이터가 없으면 빈 DataFrame.
    """
    path = _statement_path(corp_code, year, report_code)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    resp = _get_with_retry(
        f"{BASE_URL}/fnlttSinglAcntAll.json",
        {
            "crtfc_key": get_api_key("DART_API_KEY"),
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": report_code,
            "fs_div": "CFS",  # 연결재무제표
        },
    )
    payload = resp.json()

    if payload.get("status") != "000":
        # 013 = 조회된 데이터 없음 (해당 분기 보고서 미제출 등). 정상적인 상황.
        empty = pd.DataFrame()
        DART_DIR.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(path)
        return empty

    values: dict[str, float] = {}
    receipt_no = None
    for item in payload["list"]:
        receipt_no = receipt_no or item.get("rcept_no")
        column = ACCOUNT_IDS.get(item.get("account_id", ""))
        if column and column not in values:
            amount = str(item.get("thstrm_amount", "")).replace(",", "").strip()
            if amount and amount not in ("-", ""):
                try:
                    values[column] = float(amount)
                except ValueError:
                    pass

    if not values or receipt_no is None:
        empty = pd.DataFrame()
        DART_DIR.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(path)
        return empty

    values["fiscal_year"] = year
    values["fiscal_quarter"] = REPORT_CODES[report_code]
    # rcept_no 앞 8자리 = 접수일자(공시일). 이 날짜 전에는 이 데이터를 알 수 없다.
    values["available_date"] = pd.Timestamp(receipt_no[:8])

    df = pd.DataFrame([values])
    DART_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


BULK_BATCH_SIZE = 100  # fnlttMultiAcnt API의 1회 최대 조회 기업 수

# 대량 조회 API(fnlttMultiAcnt)는 account_id 대신 한글 계정명(account_nm)을 준다.
BULK_ACCOUNT_NAMES = {
    "자산총계": "assets",
    "부채총계": "liabilities",
    "자본총계": "equity",
    "매출액": "revenue",
    "영업이익": "operating_income",
    "당기순이익(손실)": "net_income",
    "당기순이익": "net_income",  # 회사에 따라 (손실) 표기가 없는 경우도 있다
}


def _bulk_path(year: int, report_code: str, batch_index: int) -> Path:
    return DART_DIR / f"bulk_{year}_{report_code}_{batch_index}.parquet"


def fetch_financials_bulk(
    corp_codes: list[str], year: int, report_code: str, batch_index: int, force_refresh: bool = False
) -> pd.DataFrame:
    """
    최대 100개 기업의 주요계정을 한 번에 가져온다 (fnlttMultiAcnt).
    개별 조회(fnlttSinglAcntAll) 대비 호출 수가 100분의 1로 줄어든다.
    """
    path = _bulk_path(year, report_code, batch_index)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    resp = _get_with_retry(
        f"{BASE_URL}/fnlttMultiAcnt.json",
        {
            "crtfc_key": get_api_key("DART_API_KEY"),
            "corp_code": ",".join(corp_codes),
            "bsns_year": str(year),
            "reprt_code": report_code,
        },
        timeout=60,
    )
    payload = resp.json()

    DART_DIR.mkdir(parents=True, exist_ok=True)
    if payload.get("status") != "000":
        empty = pd.DataFrame()
        empty.to_parquet(path)
        return empty

    per_corp: dict[str, dict] = {}
    for item in payload["list"]:
        if item.get("fs_div") != "CFS":  # 연결재무제표만
            continue
        corp = item["corp_code"]
        record = per_corp.setdefault(corp, {"corp_code": corp, "rcept_no": item.get("rcept_no")})
        column = BULK_ACCOUNT_NAMES.get(item.get("account_nm", "").strip())
        if column and column not in record:
            amount = str(item.get("thstrm_amount", "")).replace(",", "").strip()
            if amount and amount not in ("-", ""):
                try:
                    record[column] = float(amount)
                except ValueError:
                    pass

    rows = []
    for record in per_corp.values():
        if record.get("rcept_no"):
            record["fiscal_year"] = year
            record["fiscal_quarter"] = REPORT_CODES[report_code]
            record["available_date"] = pd.Timestamp(record["rcept_no"][:8])
            rows.append(record)

    df = pd.DataFrame(rows)
    df.to_parquet(path)
    return df


def load_fundamentals_bulk(
    tickers: list[str], start_year: int, end_year: int, verbose: bool = True
) -> pd.DataFrame:
    """여러 종목의 분기 재무데이터를 배치로 수집해 하나의 DataFrame으로 반환한다."""
    mapping = load_corp_codes()
    ticker_by_corp = {}
    corp_codes = []
    for ticker in tickers:
        match = mapping.loc[mapping["stock_code"] == ticker, "corp_code"]
        if not match.empty:
            corp_codes.append(match.iloc[0])
            ticker_by_corp[match.iloc[0]] = ticker

    batches = [
        corp_codes[i : i + BULK_BATCH_SIZE] for i in range(0, len(corp_codes), BULK_BATCH_SIZE)
    ]

    frames = []
    for year in range(start_year, end_year + 1):
        for report_code in REPORT_CODES:
            for batch_index, batch in enumerate(batches):
                df = fetch_financials_bulk(batch, year, report_code, batch_index)
                if not df.empty:
                    frames.append(df)
        if verbose:
            print(f"  DART {year}년 수집 완료", flush=True)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["ticker"] = result["corp_code"].map(ticker_by_corp)
    return result.dropna(subset=["ticker"]).sort_values(["ticker", "available_date"])


def load_fundamentals(ticker: str, start_year: int, end_year: int) -> pd.DataFrame:
    """
    한 종목의 [start_year, end_year] 분기별 재무데이터를 모아서 반환한다.
    available_date(공시일) 순으로 정렬되며, 이 컬럼이 look-ahead 방지의 기준이 된다.
    """
    corp_code = ticker_to_corp_code(ticker)

    frames = []
    for year in range(start_year, end_year + 1):
        for report_code in REPORT_CODES:
            df = fetch_financial_statement(corp_code, year, report_code)
            if not df.empty:
                frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result["ticker"] = ticker
    return result.sort_values("available_date").reset_index(drop=True)
