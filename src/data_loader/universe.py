"""
유니버스(분석 대상 종목 집합) 구성.

두 단계로 나눈다:

1) 후보군(candidate pool): 네이버 시가총액 순위 상위 N종목.
   **알려진 한계**: 이건 '오늘' 기준 순위라 생존편향(survivorship bias)이 있다.
   과거에 컸지만 상장폐지된 종목이 빠져 있어서, 과거 성과가 낙관적으로 나올 수 있다.
   시점별 정확한 시가총액을 얻으려면 KRX 데이터가 필요한데 이 환경에서는 막혀 있다.

2) 시점별 유니버스(point-in-time): 각 날짜마다 "그 시점까지의 거래대금" 상위 K종목만 사용.
   거래대금은 그 시점의 가격·거래량만으로 계산되므로 미래 정보가 들어가지 않는다.
   이렇게 하면 "2018년엔 소형주였는데 나중에 커진 종목"이 2018년 유니버스에 잘못
   포함되는 문제를 막을 수 있다 (후보군 자체의 생존편향은 남는다).
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

NAVER_MARKET_SUM_URL = "https://finance.naver.com/sise/sise_market_sum.naver"
REQUEST_INTERVAL = 0.4
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _candidate_path() -> Path:
    return RAW_DIR / "universe_candidates.parquet"


def fetch_candidate_pool(pages: int = 8, force_refresh: bool = False) -> pd.DataFrame:
    """
    네이버 시가총액 순위에서 후보군을 수집한다 (페이지당 50종목).
    반환 컬럼: ticker, name, market_cap(억원), shares_outstanding(천주)
    """
    path = _candidate_path()
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    frames = []
    for page in range(1, pages + 1):
        resp = requests.get(
            NAVER_MARKET_SUM_URL, params={"page": page}, headers=HEADERS, timeout=20
        )
        resp.encoding = "euc-kr"
        html = resp.text

        tables = pd.read_html(io.StringIO(html))
        table = next((t for t in tables if t.shape[0] > 10 and "종목명" in t.columns), None)
        if table is None:
            continue
        table = table.dropna(subset=["종목명"]).reset_index(drop=True)

        # 종목코드는 표에 없고 링크(href)에만 있다. 코드만 따로 뽑아 표와 순서를 맞추면
        # 어긋날 수 있으므로, 앵커에서 (코드, 종목명) 쌍을 함께 뽑아 이름으로 join한다.
        pairs = re.findall(r'<a href="/item/main\.naver\?code=(\d{6})"[^>]*>([^<]+)</a>', html)
        code_by_name = pd.DataFrame(pairs, columns=["ticker", "종목명"])
        code_by_name["종목명"] = code_by_name["종목명"].str.strip()
        code_by_name = code_by_name.drop_duplicates(subset="종목명")

        merged = table.merge(code_by_name, on="종목명", how="inner")
        frames.append(
            merged[["ticker", "종목명", "시가총액", "상장주식수"]].rename(
                columns={
                    "종목명": "name",
                    "시가총액": "market_cap",
                    "상장주식수": "shares_outstanding",
                }
            )
        )
        time.sleep(REQUEST_INTERVAL)

    result = pd.concat(frames, ignore_index=True)
    # 우선주 제외: 종목코드 끝자리가 0이 아니면 우선주/신주인수권 등
    result = result[result["ticker"].str.endswith("0")]
    result = result.drop_duplicates(subset="ticker").reset_index(drop=True)

    # ETF/리츠 등 '기업'이 아닌 종목 제외: DART에 고유번호가 없으면 재무제표도 없다.
    from src.data_loader.dart_loader import load_corp_codes

    dart_tickers = set(load_corp_codes()["stock_code"])
    result = result[result["ticker"].isin(dart_tickers)].reset_index(drop=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path)
    return result


def _snapshot_path(date: pd.Timestamp) -> Path:
    return RAW_DIR / "universe_snapshots" / f"{date.strftime('%Y%m%d')}.parquet"


def build_universe_snapshots(
    dates: list[str] | pd.DatetimeIndex, top_k: int = 200, force_refresh: bool = False
) -> pd.DataFrame:
    """
    각 기준일마다 '그 시점의' 시가총액 상위 top_k 종목을 뽑아 저장한다.

    네이버 현재 시총 순위를 쓰던 방식과 결정적으로 다른 점:
    이건 해당 날짜에 실제로 관측된 시가총액이라 미래 정보가 들어가지 않는다.
    그 시점에 상장돼 있던 종목만 포함되므로, 나중에 상장폐지된 종목도 자연히 들어가
    생존편향이 크게 줄어든다.

    반환: (rebalance_date, ticker) 목록
    """
    from src.data_loader.krx_extra import load_all_market_cap

    frames = []
    for date in pd.DatetimeIndex(dates):
        path = _snapshot_path(date)
        if path.exists() and not force_refresh:
            frames.append(pd.read_parquet(path))
            continue

        snapshot = load_all_market_cap(date.strftime("%Y%m%d"))
        snapshot = snapshot[snapshot["market_cap"] > 0]
        top = snapshot.nlargest(top_k, "market_cap")

        result = pd.DataFrame(
            {
                "rebalance_date": date,
                "ticker": top.index,
                "market_cap": top["market_cap"].values,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(path)
        frames.append(result)

    return pd.concat(frames, ignore_index=True)


def point_in_time_universe(
    trading_value: pd.DataFrame, top_k: int = 100, lookback: int = 60
) -> pd.DataFrame:
    """
    각 날짜마다 최근 lookback일 평균 거래대금 상위 top_k 종목을 True로 표시한 마스크를 만든다.

    거래대금은 그 시점까지의 가격·거래량만 쓰므로 미래 정보가 들어가지 않는다.
    """
    avg_value = trading_value.rolling(lookback).mean()
    ranks = avg_value.rank(axis=1, ascending=False, method="first")
    return ranks <= top_k
