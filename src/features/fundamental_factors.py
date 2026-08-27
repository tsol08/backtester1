"""
재무제표에서 뽑는 횡단면 팩터: 밸류 / 수익성 / 자산성장.

Fama-French 5팩터의 HML, RMW, CMA에 해당한다. 임의로 고른 조합이 아니라 표준
모형이 존재를 주장하는 팩터들이고, 우리 도구는 이미 French 데이터로 이들을
검출할 수 있음을 확인했다(run_french_validation.py).

**DART 데이터의 함정 하나**: 분기보고서의 '당기순이익'은 3개월 값인데
사업보고서(Q4)는 **연간 누적**이다. 삼성전자 2024년을 보면 Q1 6.75 / Q2 9.84 /
Q3 10.10조인데 Q4가 34.45조다. 앞 셋을 더하면 26.7조이므로 Q4 단독은 7.75조다.
매출로 보면 더 분명하다 - 4분기 매출이 300.9조로 찍히는데 이건 연매출이다.

그래서 손익 항목(순이익·매출)은 Q4를 되돌려야 하고, 재무상태표 항목(자본총계·
자산총계)은 시점 잔액이라 그대로 쓴다.

**look-ahead 방지**: 모든 값이 공시일(available_date) 기준으로 일별에 실린다.
회계기간이 끝난 날이 아니라 시장이 그 숫자를 알게 된 날부터 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.earnings_surprise import quarterly_grid

QUARTERS_PER_YEAR = 4


def standalone_quarterly(gridded: pd.DataFrame, column: str) -> pd.Series:
    """
    연간 누적으로 들어온 Q4를 분기 단독값으로 되돌린다.

    Q4단독 = 연간 - (Q1 + Q2 + Q3). 앞 세 분기 중 하나라도 비면 계산할 수 없으므로
    NaN으로 둔다 - 추정해서 채우면 그 값이 어디서 왔는지 나중에 알 수 없게 된다.
    """
    values = gridded[column].astype(float)
    quarters = gridded.index.quarter if hasattr(gridded.index, "quarter") else None
    if quarters is None:
        return values

    years = pd.Series(gridded.index.year, index=gridded.index)
    quarter_no = pd.Series(quarters, index=gridded.index)

    first_three = values.where(quarter_no <= 3)
    ytd_through_q3 = first_three.groupby(years).transform("sum")
    complete = first_three.groupby(years).transform(lambda s: s.notna().sum() == 3)

    standalone = values.copy()
    is_q4 = quarter_no == 4
    standalone[is_q4] = np.where(complete[is_q4], values[is_q4] - ytd_through_q3[is_q4], np.nan)
    return standalone


def trailing_twelve_months(gridded: pd.DataFrame, column: str) -> pd.Series:
    """최근 4개 분기 합. 분기가 하나라도 비면 NaN이다."""
    standalone = standalone_quarterly(gridded, column)
    return standalone.rolling(QUARTERS_PER_YEAR, min_periods=QUARTERS_PER_YEAR).sum()


def to_daily(gridded: pd.DataFrame, values: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """
    분기값을 일별로 펼친다. 각 거래일에는 **그 날까지 공시된** 가장 최근 값만 실린다.

    SUE와 달리 표류 창을 두지 않는다. 재무 상태는 다음 공시까지 유효한 정보이지,
    발표 직후에만 의미가 있는 뉴스가 아니기 때문이다.
    """
    frame = pd.DataFrame(
        {"available_date": gridded["available_date"], "value": values}
    ).dropna()
    if frame.empty:
        return pd.Series(np.nan, index=dates)

    frame = frame.sort_values("available_date")
    query = pd.DataFrame({"date": pd.DatetimeIndex(dates)}).sort_values("date")
    merged = pd.merge_asof(
        query, frame, left_on="date", right_on="available_date", direction="backward"
    ).set_index("date")
    return merged["value"]


def build_fundamental_panels(
    fundamentals: pd.DataFrame, dates: pd.DatetimeIndex, market_cap: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    (날짜 x 종목) 팩터 패널들.

    book_to_market  자본총계 / 시가총액          - 높을수록 싸다 (HML)
    roe             최근 4분기 순이익 / 자본총계  - 높을수록 좋다 (RMW)
    asset_growth    자산총계 전년 대비 증가율     - **낮을수록** 좋다 (CMA)
    """
    equity_by_ticker, income_by_ticker, growth_by_ticker = {}, {}, {}

    for ticker, group in fundamentals.groupby("ticker"):
        gridded = quarterly_grid(group)
        if len(gridded) < QUARTERS_PER_YEAR + 1:
            continue

        equity_by_ticker[ticker] = to_daily(gridded, gridded["equity"].astype(float), dates)
        income_by_ticker[ticker] = to_daily(
            gridded, trailing_twelve_months(gridded, "net_income"), dates
        )

        assets = gridded["assets"].astype(float)
        growth = assets / assets.shift(QUARTERS_PER_YEAR) - 1
        growth_by_ticker[ticker] = to_daily(gridded, growth, dates)

    if not equity_by_ticker:
        empty = pd.DataFrame(index=dates)
        return {"book_to_market": empty, "roe": empty, "asset_growth": empty}

    equity = pd.DataFrame(equity_by_ticker, index=dates).reindex(columns=market_cap.columns)
    income = pd.DataFrame(income_by_ticker, index=dates).reindex(columns=market_cap.columns)
    growth = pd.DataFrame(growth_by_ticker, index=dates).reindex(columns=market_cap.columns)

    positive_equity = equity.where(equity > 0)  # 자본잠식 기업의 B/M은 의미가 없다
    return {
        "book_to_market": positive_equity / market_cap.where(market_cap > 0),
        "roe": income / positive_equity,
        "asset_growth": growth,
    }
