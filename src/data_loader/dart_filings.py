"""
DART 공시목록(list.json) 수집. 이벤트 기반 신호의 재료다.

**왜 이벤트인가**: 랭킹 팩터는 본질적으로 양쪽 꼬리를 다 쓴다. 그런데 한국 대형주에서
횡단면 예측력은 하위 구간에 몰려 있어서(밸류 62%, 저변동성은 대부분), 공매도를 못 하는
쪽은 정보의 절반 이상을 버리게 된다. 이벤트 공시는 "그 일이 일어난 종목만 산다"라서
구조적으로 롱온리와 맞는다 - PEAD가 살아남은 이유도 같다.

재무제표 API(fnlttMultiAcnt)와 달리 이쪽은 **날짜 구간으로 전 종목을 한 번에** 받는다.
종목별로 도는 것보다 훨씬 싸다.

**공시명에서 읽어야 하는 두 가지**:

1) 대괄호 접두어는 새 사건이 아니다.
   `[기재정정]주요사항보고서(자기주식취득결정)`은 이미 낸 공시를 고친 것이라, 원본이
   따로 목록에 있다. 이걸 세면 같은 사건을 두 번 세게 되고, 게다가 정정일을 사건일로
   쓰면 실제보다 늦은 날짜가 된다.

2) 괄호 안이 사건의 종류다.
   자기주식'취득'과 '처분'은 방향이 반대다. 회사가 자사주를 파는 것은 물량이 시장에
   나오는 일이라, 사는 것과 같이 묶으면 신호가 서로 상쇄된다.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.data_loader.dart_loader import BASE_URL
from src.data_loader.env import get_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FILING_LIST_DIR = PROJECT_ROOT / "data" / "raw" / "dart_filing_list"

REQUEST_INTERVAL = 0.3
PAGE_SIZE = 100  # list.json의 1회 최대

# 공시유형. B = 주요사항보고(자기주식·유상증자·합병 등 회사의 중대 결정)
MAJOR_REPORT = "B"

_BRACKET_PREFIX = re.compile(r"^\[[^\]]+\]")
_EVENT_KIND = re.compile(r"\(([^)]+)\)\s*$")


def _month_path(month: pd.Period) -> Path:
    return FILING_LIST_DIR / f"{month.strftime('%Y%m')}.parquet"


def fetch_filing_month(month: pd.Period, pblntf_ty: str = MAJOR_REPORT) -> pd.DataFrame:
    """한 달치 공시목록. 페이지를 끝까지 넘긴다."""
    path = _month_path(month)
    if path.exists():
        return pd.read_parquet(path)

    params = {
        "crtfc_key": get_api_key("DART_API_KEY"),
        "bgn_de": month.start_time.strftime("%Y%m%d"),
        "end_de": month.end_time.strftime("%Y%m%d"),
        "pblntf_ty": pblntf_ty,
        "page_count": PAGE_SIZE,
    }

    rows, page = [], 1
    while True:
        response = requests.get(f"{BASE_URL}/list.json", params={**params, "page_no": page}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        time.sleep(REQUEST_INTERVAL)

        if payload.get("status") != "000":
            break  # 013 = 조회된 데이터 없음. 그 달에 공시가 없는 정상 상황이다.

        rows.extend(payload.get("list", []))
        if page >= int(payload.get("total_page", 1)):
            break
        page += 1

    frame = pd.DataFrame(
        rows, columns=["corp_code", "corp_name", "stock_code", "report_nm", "rcept_no", "rcept_dt"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def load_filings(start: str, end: str, pblntf_ty: str = MAJOR_REPORT) -> pd.DataFrame:
    """
    [start, end] 구간의 공시목록.

    반환 컬럼에 더해지는 것:
      event_date  접수일(datetime). **이 날부터 시장이 안다** - look-ahead의 기준선이다.
      kind        공시명 괄호 안의 사건 종류
      is_original 대괄호 접두어가 없는 원본 공시인가
    """
    months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq="M")
    frames = [fetch_filing_month(month, pblntf_ty) for month in months]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()

    filings = pd.concat(frames, ignore_index=True)
    filings = filings[filings["stock_code"].astype(str).str.fullmatch(r"\d{6}")].copy()

    filings["event_date"] = pd.to_datetime(filings["rcept_dt"], format="%Y%m%d")
    filings["is_original"] = ~filings["report_nm"].str.startswith("[")
    filings["kind"] = (
        filings["report_nm"]
        .str.replace(_BRACKET_PREFIX, "", regex=True)
        .str.extract(_EVENT_KIND)[0]
        .fillna("")
    )
    return filings.drop_duplicates(subset="rcept_no").sort_values("event_date")


def event_dates(filings: pd.DataFrame, kinds: list[str], originals_only: bool = True) -> pd.DataFrame:
    """지정한 종류의 사건만 (ticker, event_date)로 추린다."""
    selected = filings[filings["kind"].isin(kinds)]
    if originals_only:
        selected = selected[selected["is_original"]]
    return (
        selected[["stock_code", "event_date", "kind"]]
        .rename(columns={"stock_code": "ticker"})
        .drop_duplicates()
        .sort_values("event_date")
    )


def recent_event_mask(
    events: pd.DataFrame, dates: pd.DatetimeIndex, columns: pd.Index, window_days: int
) -> pd.DataFrame:
    """
    "최근 window_days일 안에 그 사건이 있었는가" 마스크.

    사건일 **당일부터** True다. 공시 접수일이 곧 시장이 아는 날이고, 체결은 엔진이
    하루 미룬다(t+1). 여기서 미리 한 번 더 미루면 이중으로 늦춰진다.
    """
    mask = pd.DataFrame(False, index=dates, columns=columns)
    known = set(columns)

    for ticker, group in events.groupby("ticker"):
        if ticker not in known:
            continue
        column = pd.Series(False, index=dates)
        for event_day in group["event_date"]:
            window = (dates >= event_day) & (dates <= event_day + pd.Timedelta(window_days, "D"))
            column |= window
        mask[ticker] = column.values

    return mask
