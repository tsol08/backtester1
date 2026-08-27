"""
기업행위(액면분할/병합) 보정.

KRX Open API의 TDD_CLSPRC는 **보정되지 않은 원본 종가**다. 삼성전자 50:1 액면분할
당일(2018-05-04) 종가가 2,650,000원 -> 51,900원으로 찍히는데, 주주 입장에서는
주식 수가 50배로 늘어난 것이라 실제 손익은 0이다. 그대로 pct_change를 하면
-98% 수익률이 되어 수익률·변동성·모멘텀이 전부 오염된다.

보정 원리: 액면분할은 주가가 k분의 1이 되는 동시에 상장주식수가 k배가 된다.
Open API가 상장주식수(LIST_SHRS)를 같이 주므로, 주식수 변화율을 곱해주면 상쇄된다.

    보정수익률 = (종가_t / 종가_t-1) x (주식수_t / 주식수_t-1) - 1

분할이면 (1/k) x k = 1 이 되어 수익률 0. 아무 일 없으면 주식수 비율이 1이라 무영향.

다만 유상증자처럼 '주식수만 늘고 주가는 그대로'인 경우에 이 보정을 적용하면
없는 수익이 생겨버린다. 그래서 주가가 주식수 변화의 역수만큼 실제로 움직인
경우(= 분할/병합의 특징)에만 보정한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 분할 판정 기준
MIN_SHARES_CHANGE = 0.05  # 주식수가 최소 5%는 변해야 후보
MIN_RAW_MOVE = 0.30  # 원본 수익률이 이만큼 급변해야 후보 (국내 가격제한폭이 ±30%)
MAX_ADJUSTED_MOVE = 0.30  # 보정 후 수익률이 이 안으로 들어와야 분할로 인정

# 국내 주식은 가격제한폭이 ±30%라 하루에 이보다 큰 변동은 물리적으로 불가능하다.
# 보정 후에도 남아있다면 거래정지 후 재개, 종목코드 재사용 등 데이터 이슈로 본다.
IMPOSSIBLE_DAILY_MOVE = 0.50


def split_adjustment_factor(close: pd.DataFrame, listed_shares: pd.DataFrame) -> pd.DataFrame:
    """
    누적 분할 보정 계수. adjusted_close = close * factor 로 쓴다.

    반환값을 곱하면 분할 이후 구간의 가격이 분할 이전 스케일로 환산되어,
    시계열 전체에서 수익률이 연속적으로 이어진다.
    """
    price_ratio = close / close.shift(1)
    shares_ratio = listed_shares / listed_shares.shift(1)

    raw_move = (price_ratio - 1).abs()
    adjusted_move = (price_ratio * shares_ratio - 1).abs()

    is_split = (
        ((shares_ratio - 1).abs() > MIN_SHARES_CHANGE)
        & (raw_move > MIN_RAW_MOVE)
        & (adjusted_move < MAX_ADJUSTED_MOVE)
        & (adjusted_move < raw_move)
    )

    factor = shares_ratio.where(is_split, 1.0).fillna(1.0)
    return factor.cumprod()


def adjust_close(close: pd.DataFrame, listed_shares: pd.DataFrame) -> pd.DataFrame:
    """기업행위가 보정된 종가 패널."""
    return close * split_adjustment_factor(close, listed_shares)


def mask_impossible_moves(close: pd.DataFrame) -> pd.DataFrame:
    """
    보정 후에도 남은 물리적으로 불가능한 가격 점프를 NaN 처리한다.

    가격제한폭(±30%)을 크게 벗어나는 일간 변동은 실제 수익률이 아니라 데이터
    문제(거래정지 후 재개로 며칠이 건너뛰어짐, 종목코드 재사용 등)다. 그대로 두면
    소수의 가짜 급등락이 cross-sectional 순위를 통째로 왜곡한다. 조용히 0으로
    바꾸지 않고 NaN으로 두어, 해당 종목이 그 구간 분석에서 빠지게 한다.
    """
    move = (close / close.shift(1) - 1).abs()
    # 직전 가격이 없는 첫 관측치는 판단할 근거가 없으므로 그대로 둔다
    # (move가 NaN인데 비교식은 False가 되어 통째로 지워지는 걸 막는다).
    return close.where(move.isna() | (move <= IMPOSSIBLE_DAILY_MOVE))


def load_adjusted_close(close: pd.DataFrame, listed_shares: pd.DataFrame) -> pd.DataFrame:
    """보정 + 이상치 제거를 한 번에."""
    return mask_impossible_moves(adjust_close(close, listed_shares))


def adjustment_report(close: pd.DataFrame, listed_shares: pd.DataFrame) -> dict:
    """보정이 무엇을 얼마나 바꿨는지 요약 (검증/기록용)."""
    raw_extreme = ((close / close.shift(1) - 1).abs() > IMPOSSIBLE_DAILY_MOVE).sum().sum()
    adjusted = adjust_close(close, listed_shares)
    left = ((adjusted / adjusted.shift(1) - 1).abs() > IMPOSSIBLE_DAILY_MOVE).sum().sum()

    return {
        "보정 전 극단 변동": int(raw_extreme),
        "분할 보정으로 해소": int(raw_extreme - left),
        "NaN 처리 대상(잔여)": int(left),
    }
