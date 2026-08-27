"""
어닝 서프라이즈(SUE)와 실적발표 후 표류(PEAD).

가설: 시장은 실적 뉴스를 발표 즉시 전부 반영하지 않는다. 놀라운 실적이 나온 종목의
주가는 발표 후 수 주에서 수 개월에 걸쳐 같은 방향으로 계속 흐른다. 전 세계에서
반복 확인된 이상현상이고, 없어지지 않는 이유로 '정보 처리 비용'이 지목된다 —
분기 재무제표를 공시일 기준으로 정렬해 서프라이즈를 계산하는 일 자체가 장벽이다.

SUE(Standardized Unexpected Earnings) 정의:

    SUE_q = (EPS_q - EPS_{q-4}) / std(EPS_q - EPS_{q-4} 의 과거 8분기)

분자는 '전년 동기 대비 이익 변화'다. 계절성이 강한 사업이 많으므로 직전 분기가
아니라 1년 전 같은 분기와 비교한다. 분모는 그 종목 고유의 이익 변동성이라,
"이 회사 기준으로 얼마나 이례적인가"로 표준화된다. 애널리스트 추정치가 필요 없어
(우리에게 없다) 이 방식을 쓴다.

**look-ahead 방지**: 모든 것이 공시일(available_date) 기준이다. 회계기간이 끝난
시점이 아니라 실제로 공시된 날부터 그 SUE를 알 수 있다.

**표류 창(drift window)**: PEAD는 발표 직후의 현상이다. 다음 분기 공시까지 60일 넘게
같은 SUE 값을 끌고 가면 '오래된 서프라이즈'까지 신호로 세게 되어 효과가 희석된다.
그래서 공시 후 일정 기간만 유효하게 두는 선택지를 제공한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_HISTORY = 8  # 표준편차 계산에 필요한 최소 분기 수


def compute_sue(fundamentals: pd.DataFrame, shares_outstanding: pd.Series | None = None) -> pd.DataFrame:
    """
    한 종목의 분기 재무데이터에서 SUE를 계산한다.

    EPS 대신 순이익을 그대로 써도 SUE는 같은 값이 된다 — 분자와 분모가 같은
    발행주식수로 나눠지므로 상쇄되기 때문이다. 다만 유상증자 등으로 주식수가 크게
    변한 종목에서는 달라지므로, 주식수가 주어지면 EPS로 계산한다.
    """
    df = fundamentals.sort_values(["fiscal_year", "fiscal_quarter"]).copy()

    earnings = df["net_income"]
    if shares_outstanding is not None:
        shares = shares_outstanding.reindex(df.index)
        earnings = earnings / shares.where(shares > 0)

    # 전년 동기 대비 변화 (4분기 전)
    surprise = earnings - earnings.shift(4)

    # 그 변화량의 과거 변동성. 현재 값을 포함하면 자기 자신으로 표준화되어
    # 극단값이 눌리므로, shift(1)로 과거만 쓴다.
    volatility = surprise.shift(1).rolling(MIN_HISTORY, min_periods=MIN_HISTORY).std()

    df["sue"] = (surprise / volatility.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return df


def to_daily_sue(
    sue_frame: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    drift_window: int | None = None,
) -> pd.Series:
    """
    분기 SUE를 일별로 펼친다. 각 거래일에는 그 날까지 **공시된** 가장 최근 SUE만 실린다.

    drift_window를 주면 공시 후 그 일수까지만 값을 유지하고 이후는 NaN이 된다.
    PEAD는 발표 직후의 현상이므로, 다음 공시까지 무한정 끌고 가면 이미 시장이
    소화한 오래된 서프라이즈까지 신호로 세게 된다.
    """
    available = sue_frame[["available_date", "sue"]].dropna(subset=["available_date", "sue"])
    if available.empty:
        return pd.Series(np.nan, index=trading_dates)

    available = available.sort_values("available_date")
    daily = pd.DataFrame({"date": pd.DatetimeIndex(trading_dates)}).sort_values("date")

    merged = pd.merge_asof(
        daily,
        available,
        left_on="date",
        right_on="available_date",
        direction="backward",  # 공시일 <= 거래일 중 가장 최근
    ).set_index("date")

    if drift_window is not None:
        elapsed = (merged.index - merged["available_date"]).dt.days
        merged.loc[elapsed > drift_window, "sue"] = np.nan

    return merged["sue"]


def build_sue_panel(
    fundamentals: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    drift_window: int | None = None,
) -> pd.DataFrame:
    """여러 종목의 (날짜 x 종목) SUE 패널."""
    columns = {}
    for ticker, group in fundamentals.groupby("ticker"):
        if len(group) < MIN_HISTORY + 4:  # 4분기 시차 + 8분기 표준편차
            continue
        columns[ticker] = to_daily_sue(compute_sue(group), trading_dates, drift_window)

    if not columns:
        return pd.DataFrame(index=trading_dates)
    return pd.DataFrame(columns, index=trading_dates)
