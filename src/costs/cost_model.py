"""
거래 비용 모델: 수수료 + 슬리피지 + 시장충격비용.

시장충격비용은 "주문금액이 해당 종목의 평균거래대금 대비 얼마나 큰가(participation rate)"에
비례해서 커지되, 참여율이 커질수록 한계 충격은 체감한다고 보고 sqrt 형태로 모델링한다
(participation이 4배 커지면 충격은 2배만 커짐).

거래대금(trading value)은 pykrx가 항상 제공하지는 않아 종가*거래량으로 근사한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    commission_rate: float = 0.00015  # 편도 수수료율
    slippage_rate: float = 0.0005  # 편도 슬리피지율
    impact_coefficient: float = 0.1  # 시장충격 계수
    impact_window: int = 20  # 평균거래대금 계산 윈도우

    def average_trading_value(self, df: pd.DataFrame) -> pd.Series:
        trading_value = df["close"] * df["volume"]
        return trading_value.rolling(self.impact_window).mean()

    def total_cost_rate(self, df: pd.DataFrame, order_value: pd.Series) -> pd.Series:
        """order_value(원화, 매매 시점의 주문금액 절댓값) 대비 총 거래비용 비율."""
        avg_value = self.average_trading_value(df)
        participation = (order_value.abs() / avg_value).clip(lower=0).fillna(0.0)
        impact = self.impact_coefficient * np.sqrt(participation)
        return self.commission_rate + self.slippage_rate + impact
