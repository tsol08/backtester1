"""
재무데이터(DART) -> 일별 팩터 변환.

핵심 규칙: t일에 쓸 수 있는 재무데이터는 "available_date(공시일) <= t"인 것 중
가장 최근 것뿐이다. 회계기간이 아무리 최신이어도 공시 전이면 알 수 없는 정보다.

merge_asof(direction="backward")가 정확히 이 의미를 구현한다: 각 거래일에 대해
"그 날짜 이하의 공시일 중 가장 가까운 것"을 붙인다.

밸류에이션 지표(PBR, PER)는 재무데이터와 그날의 시가총액이 함께 필요한데, 시가총액은
발행주식수가 있어야 한다. KRX 시총 API가 이 환경에서 막혀 있어, 여기서는 발행주식수
없이 계산 가능한 팩터(ROE, 영업이익률, 부채비율, 성장률)를 먼저 만든다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QUARTERLY_FLOW_COLUMNS = ["revenue", "operating_income", "net_income", "net_income_controlling"]


def _to_quarterly_flow(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    DART의 분기보고서는 손익 항목이 '해당 분기'값이지만, 사업보고서(4분기)는 연간 누적값이다.
    연간값에서 1~3분기 합을 빼서 4분기 단독 값으로 바꿔, 분기 흐름을 일관되게 만든다.
    """
    df = fundamentals.sort_values(["fiscal_year", "fiscal_quarter"]).copy()

    for year, group in df.groupby("fiscal_year"):
        annual_mask = (df["fiscal_year"] == year) & (df["fiscal_quarter"] == 4)
        quarters_1_3 = group[group["fiscal_quarter"].isin([1, 2, 3])]
        if annual_mask.any() and len(quarters_1_3) == 3:
            for col in QUARTERLY_FLOW_COLUMNS:
                if col in df.columns:
                    df.loc[annual_mask, col] = (
                        df.loc[annual_mask, col].iloc[0] - quarters_1_3[col].sum()
                    )
    return df


def build_fundamental_factors(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """분기 재무데이터에서 팩터를 계산한다 (아직 일별로 펼치기 전)."""
    df = _to_quarterly_flow(fundamentals).copy()

    # 최근 4개 분기 합계(TTM, Trailing Twelve Months) - 계절성 제거
    for col in QUARTERLY_FLOW_COLUMNS:
        if col in df.columns:
            df[f"{col}_ttm"] = df[col].rolling(4).sum()

    df["roe"] = df["net_income_ttm"] / df["equity"]
    df["operating_margin"] = df["operating_income_ttm"] / df["revenue_ttm"]
    df["net_margin"] = df["net_income_ttm"] / df["revenue_ttm"]
    df["debt_ratio"] = df["liabilities"] / df["equity"]
    df["asset_turnover"] = df["revenue_ttm"] / df["assets"]

    # 전년 동기 대비 성장률 (4분기 전과 비교)
    df["revenue_growth_yoy"] = df["revenue_ttm"].pct_change(4)
    df["operating_income_growth_yoy"] = df["operating_income_ttm"].pct_change(4)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


FACTOR_COLUMNS = [
    "roe",
    "operating_margin",
    "net_margin",
    "debt_ratio",
    "asset_turnover",
    "revenue_growth_yoy",
    "operating_income_growth_yoy",
]


def to_daily_factors(factor_df: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    분기 팩터를 일별로 펼친다. 각 거래일에는 "그 날까지 실제로 공시된" 가장 최근
    재무데이터만 반영된다 (look-ahead 방지의 핵심).
    """
    available = factor_df[["available_date"] + FACTOR_COLUMNS].dropna(subset=["available_date"])
    available = available.sort_values("available_date")

    daily = pd.DataFrame({"date": pd.DatetimeIndex(trading_dates)}).sort_values("date")

    merged = pd.merge_asof(
        daily,
        available,
        left_on="date",
        right_on="available_date",
        direction="backward",  # 공시일 <= 거래일 중 가장 최근 것
    )

    merged = merged.set_index("date")
    return merged[FACTOR_COLUMNS]
