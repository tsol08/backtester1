"""
거래 비용 모델: 수수료 + 슬리피지 + 시장충격비용 + 증권거래세.

시장충격은 참여율(주문금액/평균거래대금)의 제곱근에 비례한다고 본다. 참여율이 4배
커지면 충격은 2배만 커진다는 뜻이다. 여기까지는 널리 쓰이는 형태다.

**계수의 단위**: 문헌의 제곱근 법칙은

    충격 = Y x (그 종목의 일간변동성) x sqrt(참여율)

꼴이고, Y는 무차원 상수로 대략 0.3~1.5 범위가 보고된다. 변동성 항이 있어야 하는
이유는, 같은 참여율이라도 잘 흔들리는 종목이 더 많이 밀리기 때문이다. 이 항을 빼고
상수 하나로 뭉뚱그리면 그 상수가 "변동성 x Y"를 대신 흡수하는데, 그 값이 얼마여야
하는지 아무도 모르게 된다. 실제로 이 프로젝트는 그 상수를 0.1로 두고 있었고,
PEAD 검정에서 판정이 그 값에 갈리는 것을 보고서야 정체를 확인했다
(experiments/log.md 2026-08-27).

일간변동성은 추정하지 않고 **잰다**. 고가/저가로 구하는 Parkinson 추정량을 쓰는데,
종가-종가 표준편차보다 효율이 높기도 하지만 여기서는 이유가 하나 더 있다:
고가/저가 비율은 액면분할 보정과 무관하다(같은 날 두 값이 같은 배율로 움직인다).
보정되지 않은 종가로 변동성을 재면 분할일이 -98%로 잡혀 오염된다.

**거래대금**은 KRX Open API 응답의 실제 거래대금(ACC_TRDVAL)을 쓴다. 종가x거래량
근사는 조정 종가와 원본 거래량을 곱하게 되어, 분할 이력이 있는 종목에서 거래대금을
수십 배 부풀린다(삼성전자 최대 52배 -> 참여율 50배 과소 -> 충격 7배 과소).

증권거래세는 **매도할 때만** 붙고, 이익이 났든 손해가 났든 매도금액에 부과된다.
수수료(0.015%)보다 한 자릿수 크기 때문에, 이걸 빼먹으면 회전율이 있는 전략의
비용이 심하게 과소평가된다. 세율이 해마다 바뀌어와서 날짜별 스케줄로 관리한다.
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


# 참여율(주문금액/평균거래대금)의 상한. 1.0은 "하루 평균 거래대금 전부를 내가 낸다"는
# 뜻이고, 그 이상은 모델이 아는 바가 없다. 상한을 씌우지 않으면 거래정지 종목
# (평균거래대금 0)에서 참여율이 무한대가 되어 백테스트 전체가 조용히 -inf가 된다.
# 다만 진짜로 못 사는 종목의 비용은 실제로는 무한대에 가깝다 - 상한은 안전장치이지
# 그 상황의 옳은 값이 아니다.
MAX_PARTICIPATION = 1.0

# 고가/저가가 없어 변동성을 잴 수 없을 때 쓰는 값. 2018~2026 시가총액 상위 200종목의
# Parkinson 일간변동성 중앙값(2.24%)이다. 추측이 아니라 이 패널에서 측정한 값이지만,
# 종목별 차이를 뭉갠 값이므로 어디까지나 대타다.
FALLBACK_VOLATILITY = 0.0224


@dataclass
class CostModel:
    commission_rate: float = 0.00015  # 편도 수수료율
    slippage_rate: float = 0.0005  # 편도 슬리피지율 (반호가 중앙값 0.034%의 1.5배)
    impact_coefficient: float = 1.0  # 제곱근 법칙의 Y. 무차원, 문헌 범위 0.3~1.5
    impact_window: int = 20  # 평균거래대금 계산 윈도우
    volatility_window: int = 60  # 일간변동성 추정 윈도우
    apply_transaction_tax: bool = True  # 매도 시 증권거래세 부과 여부

    def sell_tax_rate(self, dates: pd.DatetimeIndex) -> pd.Series:
        """매도금액에 부과할 세율. 끄면 전부 0."""
        if not self.apply_transaction_tax:
            return pd.Series(0.0, index=pd.DatetimeIndex(dates))
        return transaction_tax_rate(pd.DatetimeIndex(dates))

    def average_trading_value(self, df: pd.DataFrame) -> pd.Series:
        """
        최근 impact_window일 평균 거래대금.

        trading_value 컬럼이 있으면 그걸 쓴다. 종가x거래량 대체 경로는 **조정 종가를
        넘겨받으면 틀린다** - 조정 종가는 원본 거래량과 스케일이 다르다.

        min_periods=1인 이유: 이력이 짧으면 NaN이 되고, NaN 참여율은 아래에서 0으로
        해석되어 **거래가 공짜가 된다**. 추정치가 거칠더라도 있는 자료로 재는 편이
        "모르니까 비용 0"보다 낫다.
        """
        if "trading_value" in df.columns:
            trading_value = df["trading_value"]
        else:
            trading_value = df["close"] * df["volume"]
        return trading_value.rolling(self.impact_window, min_periods=1).mean()

    def daily_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        종목별 일간변동성. 고가/저가가 있으면 Parkinson 추정량을 쓴다.

            sigma^2 = mean( ln(고가/저가)^2 ) / (4 ln 2)

        상한가/하한가로 고가=저가인 날은 0을 기여해 약간 과소추정되지만, 윈도우
        전체로 희석된다. 고가/저가가 없으면 종가 수익률의 표준편차로 대신하는데,
        이 경로는 **보정된 종가**를 받아야 한다(원본 종가는 분할일이 -98%로 잡힌다).
        """
        if {"high", "low"}.issubset(df.columns):
            high = df["high"].where(df["high"] > 0)
            low = df["low"].where(df["low"] > 0)
            log_range = np.log(high / low)
            variance = (log_range**2).rolling(
                self.volatility_window, min_periods=self.volatility_window // 3
            ).mean() / (4 * np.log(2))
            volatility = np.sqrt(variance)
        else:
            volatility = df["close"].pct_change(fill_method=None).rolling(
                self.volatility_window, min_periods=self.volatility_window // 3
            ).std()

        return volatility.replace(0.0, np.nan).fillna(FALLBACK_VOLATILITY)

    def participation_rate(self, df: pd.DataFrame, order_value: pd.Series) -> pd.Series:
        """
        주문금액 / 평균거래대금. MAX_PARTICIPATION에서 잘린다.

        거래정지 종목은 평균거래대금이 0이라 참여율이 무한대가 된다(지주사 전환·인적분할·
        회생절차 등으로 실제로 발생한다). 상한이 없으면 그런 하루가 백테스트 전체를
        -inf로 만들면서도 예외 없이 조용히 지나간다.
        """
        avg_value = self.average_trading_value(df)
        rate = order_value.abs() / avg_value.where(avg_value > 0)
        rate = rate.replace([np.inf, -np.inf], np.nan)
        # 거래대금을 알 수 없는 날은 '거래 불가'로 본다. 실제로 주문이 없는 날이라면
        # 비용이 주문금액에 곱해지므로 결과에 영향이 없다.
        return rate.fillna(MAX_PARTICIPATION).clip(lower=0.0, upper=MAX_PARTICIPATION)

    def market_impact_rate(self, df: pd.DataFrame, order_value: pd.Series) -> pd.Series:
        """Y x 일간변동성 x sqrt(참여율)."""
        participation = self.participation_rate(df, order_value)
        return self.impact_coefficient * self.daily_volatility(df) * np.sqrt(participation)

    def total_cost_rate(self, df: pd.DataFrame, order_value: pd.Series) -> pd.Series:
        """order_value(원화, 매매 시점의 주문금액 절댓값) 대비 총 거래비용 비율."""
        return self.commission_rate + self.slippage_rate + self.market_impact_rate(df, order_value)
