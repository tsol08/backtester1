"""
거래 비용 모델: 수수료 + 슬리피지 + 시장충격비용 + 증권거래세.

시장충격비용은 "주문금액이 해당 종목의 평균거래대금 대비 얼마나 큰가(participation rate)"에
비례해서 커지되, 참여율이 커질수록 한계 충격은 체감한다고 보고 sqrt 형태로 모델링한다
(participation이 4배 커지면 충격은 2배만 커짐).

증권거래세는 **매도할 때만** 붙고, 이익이 났든 손해가 났든 매도금액에 부과된다.
수수료(0.015%)보다 한 자릿수 크기 때문에, 이걸 빼먹으면 회전율이 있는 전략의
비용이 심하게 과소평가된다. 세율이 해마다 바뀌어와서 날짜별 스케줄로 관리한다.

거래대금(trading value)은 pykrx가 항상 제공하지는 않아 종가*거래량으로 근사한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 매도금액에 부과되는 실효세율 (시행일, 세율).
#
# 코스피는 증권거래세에 농어촌특별세 0.15%가 더 붙고 코스닥은 농특세가 없는데,
# 코스닥 증권거래세율이 그만큼 높게 설정돼 있어서 **투자자가 실제로 내는 총액은
# 두 시장이 같다**. 그래서 시장 구분 없이 하나의 스케줄로 처리한다.
TRANSACTION_TAX_SCHEDULE: list[tuple[str, float]] = [
    ("1900-01-01", 0.0030),  # ~2019.6.2
    ("2019-06-03", 0.0025),
    ("2021-01-01", 0.0023),
    ("2023-01-01", 0.0020),
    ("2024-01-01", 0.0018),
    ("2025-01-01", 0.0015),
    ("2026-01-01", 0.0020),
]


def transaction_tax_rate(dates: pd.DatetimeIndex) -> pd.Series:
    """각 날짜에 적용되는 증권거래세율(매도금액 대비)."""
    schedule = pd.DataFrame(TRANSACTION_TAX_SCHEDULE, columns=["effective_date", "rate"])
    schedule["effective_date"] = pd.to_datetime(schedule["effective_date"])

    query = pd.DataFrame({"date": pd.DatetimeIndex(dates)}).sort_values("date")
    merged = pd.merge_asof(
        query, schedule, left_on="date", right_on="effective_date", direction="backward"
    )
    return merged.set_index("date")["rate"].reindex(dates)


@dataclass
class CostModel:
    commission_rate: float = 0.00015  # 편도 수수료율
    slippage_rate: float = 0.0005  # 편도 슬리피지율
    impact_coefficient: float = 0.1  # 시장충격 계수
    impact_window: int = 20  # 평균거래대금 계산 윈도우
    apply_transaction_tax: bool = True  # 매도 시 증권거래세 부과 여부

    def sell_tax_rate(self, dates: pd.DatetimeIndex) -> pd.Series:
        """매도금액에 부과할 세율. 끄면 전부 0."""
        if not self.apply_transaction_tax:
            return pd.Series(0.0, index=pd.DatetimeIndex(dates))
        return transaction_tax_rate(pd.DatetimeIndex(dates))

    def average_trading_value(self, df: pd.DataFrame) -> pd.Series:
        trading_value = df["close"] * df["volume"]
        return trading_value.rolling(self.impact_window).mean()

    def total_cost_rate(self, df: pd.DataFrame, order_value: pd.Series) -> pd.Series:
        """order_value(원화, 매매 시점의 주문금액 절댓값) 대비 총 거래비용 비율."""
        avg_value = self.average_trading_value(df)
        participation = (order_value.abs() / avg_value).clip(lower=0).fillna(0.0)
        impact = self.impact_coefficient * np.sqrt(participation)
        return self.commission_rate + self.slippage_rate + impact
