"""
다요소 결합 스코어 + 히스테리시스 진입/청산.

각 피처를 자기 자신의 최근 분포 대비 z-score로 정규화한 뒤, 부호(방향)를 맞춰서
가중합한다 (모든 피처가 "높을수록 매수 신호"가 되도록 통일).

가설: "장기 추세는 살아있는데(모멘텀 60일 양호) 단기적으로 조정받았고(이격도 20일 낮음),
거래량은 평소와 비슷한(비정상적으로 튀지 않은)" 종목을 사는 눌림목 매수 전략.

청산에는 진입보다 느슨한 기준(entry_threshold > exit_threshold)을 둬서, 스코어가
경계선 근처에서 흔들릴 때마다 매매가 발생하지 않도록 한다(히스테리시스).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# {피처 컬럼: (z-score 계산에 쓸 rolling window, 부호(+1: 높을수록 매수신호, -1: 반대), 가중치)}
FEATURE_SPECS = {
    "disparity_20": (20, -1, 1.0),  # 이격도가 낮을수록(과매도) 매수 신호 -> 부호 반전
    "momentum_60": (60, 1, 1.0),  # 장기 추세가 강할수록 매수 신호
    "volume_ratio_20": (20, -1, 1.0),  # 거래량이 비정상적으로 튄 날은 페널티
}


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def composite_score(features: pd.DataFrame, specs: dict = FEATURE_SPECS) -> pd.Series:
    total = pd.Series(0.0, index=features.index)
    weight_sum = sum(weight for _, _, weight in specs.values())
    for col, (window, sign, weight) in specs.items():
        z = _rolling_zscore(features[col], window)
        total = total + sign * weight * z
    return total / weight_sum


def composite_signal(
    features: pd.DataFrame,
    entry_threshold: float = 1.0,
    exit_threshold: float = 0.0,
    specs: dict = FEATURE_SPECS,
) -> pd.Series:
    """
    entry_threshold보다 스코어가 높아지면 진입(1), exit_threshold보다 낮아지면 청산(0).
    그 사이 구간에서는 직전 상태를 그대로 유지한다(ffill로 구현).
    """
    score = composite_score(features, specs)

    raw = pd.Series(np.nan, index=score.index)
    raw[score > entry_threshold] = 1.0
    raw[score < exit_threshold] = 0.0

    return raw.ffill().fillna(0.0)
