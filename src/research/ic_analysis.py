"""
정보계수(IC, Information Coefficient) 분석.

"이 피처가 애초에 미래 수익률과 관련이 있는가?"를, 어떤 전략/파라미터도 정하지 않은
상태에서 먼저 확인하기 위한 진단 도구. 특정 시그널·임계값을 만들기 전에 쓰는 거라
과적합 위험이 거의 없다 (끼워맞출 대상 자체가 없음).

핵심 아이디어: 매일, 여러 종목에 걸쳐 "오늘의 피처값"과 "N일 뒤 실제 수익률"의
순위상관(스피어만 상관)을 구한다(cross-sectional IC). 이걸 여러 날에 걸쳐 평균 내면,
그 피처가 평균적으로 미래 수익률의 방향을 얼마나 잘 맞추는지 알 수 있다.
- IC가 뚜렷한 음수 -> "피처가 낮을수록 나중에 오른다"는 관계가 데이터에 있다는 증거
- IC가 뚜렷한 양수 -> "피처가 높을수록 나중에 오른다"는 관계
- IC가 0 근처 -> 이 피처는 (적어도 단순한 형태로는) 예측력이 없다는 증거
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """t일 종가 대비 t+horizon일 종가의 수익률. 미래 정보를 의도적으로 담고 있음
    (전략 신호가 아니라 "피처가 미래를 예측하는지" 측정하기 위한 정답값이므로)."""
    return close.shift(-horizon) / close - 1


def daily_cross_sectional_ic(
    feature_panel: pd.DataFrame, fwd_return_panel: pd.DataFrame, min_obs: int = 5
) -> pd.Series:
    """날짜별로, 그날 여러 종목에 걸친 피처값과 미래수익률의 스피어만 상관을 구한다."""
    ic_by_date = {}
    for date in feature_panel.index:
        f = feature_panel.loc[date]
        r = fwd_return_panel.loc[date]
        valid = f.notna() & r.notna()
        if valid.sum() >= min_obs:
            ic_by_date[date] = f[valid].corr(r[valid], method="spearman")
    return pd.Series(ic_by_date)


def summarize_ic(ic_series: pd.Series) -> dict:
    ic_series = ic_series.dropna()
    n = len(ic_series)
    mean_ic = ic_series.mean()
    std_ic = ic_series.std()
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 and n > 0 else 0.0
    return {
        "평균 IC": mean_ic,
        "IC 표준편차": std_ic,
        "t-stat": t_stat,
        "IC>0 비율": (ic_series > 0).mean(),
        "관측일수": n,
    }


def non_overlapping_ic(
    feature_panel: pd.DataFrame,
    fwd_return_panel: pd.DataFrame,
    horizon: int,
    min_obs: int = 5,
) -> pd.Series:
    """
    horizon 간격으로 띄엄띄엄 샘플링한 IC.

    일별 IC를 그대로 평균/t-stat 내면 심각하게 과대평가된다:
    (1) horizon일 forward return은 구간이 겹쳐서(overlapping) 인접 관측이 거의 같은
        정보를 담고, (2) 재무 팩터는 분기당 한 번만 바뀌어서 수십 일간 값이 동일하다.
    즉 "관측일수 567일"이 실제로는 독립 관측 몇십 개에 불과하다.

    horizon 간격으로만 뽑으면 forward return 구간이 겹치지 않아 이 문제가 크게 줄어든다.
    """
    ic = daily_cross_sectional_ic(feature_panel, fwd_return_panel, min_obs=min_obs)
    return ic.iloc[::horizon]


def effective_sample_note(feature_panel: pd.DataFrame) -> int:
    """팩터 값이 실제로 바뀐 횟수 (분기 팩터가 몇 번 갱신됐는지)."""
    changes = feature_panel.diff().abs().sum(axis=1) > 1e-12
    return int(changes.sum())


def pooled_ic(feature_panel: pd.DataFrame, fwd_return_panel: pd.DataFrame) -> float:
    """날짜/종목 구분 없이 전부 하나로 모아서 계산한 상관(참고용, 시계열 자기상관 때문에
    daily_cross_sectional_ic보다 통계적으로는 덜 엄밀함)."""
    f = feature_panel.stack()
    r = fwd_return_panel.stack()
    aligned = pd.concat([f, r], axis=1, keys=["feature", "fwd_return"]).dropna()
    return aligned["feature"].corr(aligned["fwd_return"], method="spearman")
