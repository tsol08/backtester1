"""
재무데이터(DART) -> 일별 팩터 변환.

핵심 규칙: t일에 쓸 수 있는 재무데이터는 "available_date(공시일) <= t"인 것 중
가장 최근 것뿐이다. 회계기간이 아무리 최신이어도 공시 전이면 알 수 없는 정보다.

merge_asof(direction="backward")가 정확히 이 의미를 구현한다: 각 거래일에 대해
"그 날짜 이하의 공시일 중 가장 가까운 것"을 붙인다.

팩터는 두 종류다:
- 재무제표만으로 계산되는 것(ROE, 영업이익률, 부채비율, 성장률): build_fundamental_factors
- 시가총액이 함께 있어야 하는 밸류에이션(PBR/PER의 역수): build_valuation_from_fundamentals

KRX Open API는 PER/PBR을 직접 주지 않으므로(무료 서비스 목록에 없음), 밸류에이션은
DART 재무제표 + Open API 시가총액을 직접 나눠서 만든다. 분자(재무제표)는 공시일 기준
일별로 펼친 값이고 분모(시총)는 당일 값이라, 둘 다 t일에 실제로 관측 가능하다.
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

# 밸류에이션 계산에 필요한 '수준(level)' 값들. 비율이 아니라 금액이라 그 자체로는
# 종목간 비교가 안 되고, 시가총액으로 나눠야 팩터가 된다.
VALUATION_LEVEL_COLUMNS = ["equity", "net_income_ttm", "revenue_ttm"]


def to_daily_factors(
    factor_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    분기 팩터를 일별로 펼친다. 각 거래일에는 "그 날까지 실제로 공시된" 가장 최근
    재무데이터만 반영된다 (look-ahead 방지의 핵심).
    """
    columns = FACTOR_COLUMNS if columns is None else columns

    available = factor_df[["available_date"] + columns].dropna(subset=["available_date"])
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
    return merged[columns]


def build_valuation_from_fundamentals(
    equity: pd.DataFrame,
    net_income_ttm: pd.DataFrame,
    revenue_ttm: pd.DataFrame,
    market_cap: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    재무제표 수준값을 시가총액으로 나눠 밸류에이션 팩터를 만든다.
    PBR/PER 대신 그 역수를 쓰는 이유는 방향 통일과 적자 처리 때문이다:
    PER은 적자에서 음수가 되고 이익이 0에 가까우면 발산하는데, 역수인
    earnings yield는 그냥 연속적인 값이라 순위 비교에 훨씬 안정적이다.

    분자는 공시일 기준으로 펼친 값, 분모는 당일 시가총액이라 둘 다 t일에 관측 가능하다.
    """
    cap = market_cap.where(market_cap > 0)

    def _ratio(numerator: pd.DataFrame) -> pd.DataFrame:
        aligned = numerator.reindex_like(cap)
        return (aligned / cap).replace([np.inf, -np.inf], np.nan)

    return {
        # 자본총계/시총 = PBR의 역수. 자본잠식(음수)은 의미가 달라 제외한다.
        "book_to_market": _ratio(equity.where(equity > 0)),
        # 순이익/시총 = PER의 역수. 적자(음수)도 정보이므로 그대로 둔다.
        "earnings_yield": _ratio(net_income_ttm),
        # 매출/시총. 이익이 불안정한 기업에서 이익 기반 지표를 보완한다.
        "sales_to_price": _ratio(revenue_ttm.where(revenue_ttm > 0)),
    }
