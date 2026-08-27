"""
DART 공시 원문(document.xml)에서 내부자 거래의 '세부변동내역'을 파싱한다.

왜 필요한가: 구조화된 API(elestock)는 공시 한 건의 순증감만 주고 **변동 사유를
알려주지 않는다**. 그런데 사유별로 정보가치가 완전히 다르다.

    01 장내매수  -> 자기 돈으로 샀다. 정보가 있다.
    02 장내매도  -> 자기 판단으로 팔았다. 정보가 있다.
    31 신규선임  -> 새로 임원이 된 사람이 **원래 갖고 있던** 주식을 처음 신고한 것.
                   매수가 아니다. 그런데 집계상으로는 거대한 '증가'로 잡힌다.
    59 자사주상여금 -> 회사가 준 것. 본인의 판단이 아니다.

elestock 기준으로 매수 10,448건 대 매도 1,144건이라는 비대칭이 나왔는데, 상당 부분이
31/33 같은 '최초 신고'가 매수로 둔갑한 결과로 보인다. 원문을 파싱하면 진짜 장내매수만
골라낼 수 있다.

원문 구조: '다. 세부변동내역' 표의 각 행이 거래 한 건이고, 사유가 텍스트가 아니라
코드로 들어있다(AUNIT="RPT_RSN" AUNITVALUE="01"). 텍스트 매칭보다 안정적이다.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.data_loader.dart_loader import BASE_URL
from src.data_loader.env import get_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILING_DIR = PROJECT_ROOT / "data" / "raw" / "dart_filing"

REQUEST_INTERVAL = 0.3

# 본인의 매매 판단이 담긴 사유. 나머지(신규선임/최초신고/상여금 등)는 보유량이
# 바뀌긴 해도 '샀다/팔았다'가 아니므로 신호에서 제외한다.
MARKET_TRADE_CODES = {
    "01",  # 장내매수
    "02",  # 장내매도
    "03",  # 장외매수
    "12",  # 장외매도
    "81",  # 시간외매매(+)
    "82",  # 시간외매매(-)
}

_ROW_SPLIT = re.compile(r"<TR\b", re.IGNORECASE)
_REASON = re.compile(r'AUNIT="RPT_RSN"\s+AUNITVALUE="(\d+)"[^>]*>([^<]*)<')
_CHANGE_DATE = re.compile(r'AUNIT="MDF_DM"\s+AUNITVALUE="(\d{8})"')
_SHARES = re.compile(r'ACODE="MDF_STK_CNT"[^>]*>([^<]*)<')
_PRICE = re.compile(r'ACODE="ACI_AMT2"[^>]*>([^<]*)<')


def _to_number(text: str) -> float:
    cleaned = text.replace(",", "").replace("−", "-").strip()
    try:
        return float(cleaned)
    except ValueError:
        return float("nan")


def parse_detail_rows(document: str) -> pd.DataFrame:
    """
    공시 원문에서 세부변동내역 행들을 뽑는다.

    증감 수량 칸에는 부호가 없고 사유 라벨에 (+)/(-)로 표시된다
    (예: '장내매수(+)', '장내매도(-)'). 라벨의 부호를 수량에 적용한다.
    """
    rows = []
    for block in _ROW_SPLIT.split(document):
        reason = _REASON.search(block)
        if not reason:
            continue  # 헤더나 합계 행

        code, label = reason.group(1), reason.group(2).strip()
        shares = _SHARES.search(block)
        if not shares:
            continue

        quantity = _to_number(shares.group(1))
        if pd.isna(quantity):
            continue
        if "(-)" in label:
            quantity = -abs(quantity)

        change_date = _CHANGE_DATE.search(block)
        price = _PRICE.search(block)

        rows.append(
            {
                "reason_code": code,
                "reason": label,
                "change_date": pd.to_datetime(change_date.group(1), errors="coerce")
                if change_date
                else pd.NaT,
                "shares_change": quantity,
                "price": _to_number(price.group(1)) if price else float("nan"),
                "is_market_trade": code in MARKET_TRADE_CODES,
            }
        )

    return pd.DataFrame(rows)


def fetch_filing_details(rcept_no: str) -> pd.DataFrame:
    """공시 한 건의 세부변동내역. 원문은 변하지 않으므로 영구 캐싱한다."""
    path = FILING_DIR / f"{rcept_no}.parquet"
    if path.exists():
        return pd.read_parquet(path)

    response = requests.get(
        f"{BASE_URL}/document.xml",
        params={"crtfc_key": get_api_key("DART_API_KEY"), "rcept_no": rcept_no},
        timeout=30,
    )
    time.sleep(REQUEST_INTERVAL)

    try:
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        document = archive.read(archive.namelist()[0]).decode("utf-8", errors="replace")
        result = parse_detail_rows(document)
    except (zipfile.BadZipFile, IndexError):
        result = pd.DataFrame()

    if result.empty:
        result = pd.DataFrame(
            {
                "reason_code": pd.Series(dtype="object"),
                "reason": pd.Series(dtype="object"),
                "change_date": pd.Series(dtype="datetime64[ns]"),
                "shares_change": pd.Series(dtype="float64"),
                "price": pd.Series(dtype="float64"),
                "is_market_trade": pd.Series(dtype="bool"),
            }
        )

    result["rcept_no"] = rcept_no
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path)
    return result


def load_filing_details(rcept_nos: list[str], verbose: bool = True) -> pd.DataFrame:
    """여러 공시의 세부변동내역을 모은다."""
    frames = []
    for i, rcept_no in enumerate(rcept_nos):
        try:
            frames.append(fetch_filing_details(rcept_no))
        except Exception as exc:
            if verbose:
                print(f"  {rcept_no} 실패: {type(exc).__name__}", flush=True)

        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rcept_nos)}", flush=True)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
