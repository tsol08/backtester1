"""
엔진 검증용 단순 규칙기반 시그널: 이격도 z-score 평균회귀.

이격도(disparity_w = 종가/이동평균 - 1)가 자기 자신의 최근 분포 대비 많이 낮을 때
(과매도 구간) 매수 포지션을 들고, 그렇지 않으면 포지션 없음.

Phase 1의 목적은 "엔진과 비용모델이 올바르게 동작하는지" 확인하는 것이라
전략 자체의 정교함(청산 조건, 헤저시스 등)은 의도적으로 최소화했다.
"""
from __future__ import annotations

import pandas as pd


def disparity_zscore_signal(
    features: pd.DataFrame,
    window: int = 20,
    entry_z: float = -1.0,
) -> pd.Series:
    disparity_col = f"disparity_{window}"
    disparity = features[disparity_col]
    mean = disparity.rolling(window).mean()
    std = disparity.rolling(window).std()
    z = (disparity - mean) / std

    signal = (z < entry_z).astype(float)
    return signal
