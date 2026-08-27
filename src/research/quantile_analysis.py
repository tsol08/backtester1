"""
분위수(portfolio sort) 분석: 롱온리 관점에서 신호를 평가한다.

IC는 cross-sectional 순위상관이라 "줄을 제대로 세웠는가"를 잰다. 그런데 공매도를
하지 않는다면 아래쪽 순위가 아무리 정확해도 쓸 수가 없다. 롱온리에서 실제로
중요한 건 **상위 분위가 평균보다 나은가** 하나뿐이다.

이 차이는 사소하지 않다. 신호의 예측력이 하위 구간에 몰려 있으면(= 못 오를 종목은
잘 골라내는데 오를 종목은 못 고르는 경우) IC는 좋게 나오지만 롱온리 수익은 0이다.
반대로 IC가 평범해도 최상위 분위만 유독 좋으면 롱온리로는 훌륭하다.

그래서 보는 것:
- 분위별 평균 수익률: 신호가 커질수록 수익도 커지는가(단조성)
- 상위 분위 - 유니버스 평균: 롱온리로 실제로 얻는 초과분
- 상위 분위 - 하위 분위: 신호의 전체 스프레드(참고용, 롱숏이어야 얻는 값)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def assign_quantiles(
    signal: pd.DataFrame, universe: pd.DataFrame, n_quantiles: int = 5, min_obs: int = 30
) -> pd.DataFrame:
    """
    날짜마다 유니버스 안에서 신호를 n개 분위로 나눈다 (0=최하위, n-1=최상위).

    유효 종목이 min_obs 미만인 날은 분위를 나눠도 의미가 없으므로 통째로 비운다.
    """
    masked = signal.where(universe)
    valid_counts = masked.notna().sum(axis=1)

    ranks = masked.rank(axis=1, pct=True)
    quantiles = (ranks * n_quantiles).apply(np.ceil) - 1
    quantiles = quantiles.clip(lower=0, upper=n_quantiles - 1)

    return quantiles.where(valid_counts >= min_obs, other=np.nan)


def quantile_forward_returns(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    universe: pd.DataFrame,
    n_quantiles: int = 5,
    min_obs: int = 30,
    sample_every: int | None = None,
) -> pd.DataFrame:
    """
    분위별 평균 forward return 시계열 (행=날짜, 열=분위).

    sample_every를 주면 그 간격으로만 샘플링한다. forward return은 구간이 겹치므로
    (20일 수익률을 매일 재면 인접 관측이 거의 같은 정보) 호라이즌 간격으로 띄워야
    통계가 부풀지 않는다 — IC 분석에서 쓴 것과 같은 이유다.
    """
    quantiles = assign_quantiles(signal, universe, n_quantiles, min_obs)
    aligned = fwd_returns.where(quantiles.notna())

    if sample_every:
        quantiles = quantiles.iloc[::sample_every]
        aligned = aligned.iloc[::sample_every]

    rows = {}
    for date in quantiles.index:
        q = quantiles.loc[date].dropna()
        r = aligned.loc[date]
        if q.empty:
            continue
        rows[date] = r.groupby(q).mean()

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).T
    result.index.name = "date"
    return result.reindex(columns=range(n_quantiles))


def summarize_quantiles(quantile_returns: pd.DataFrame, periods_per_year: float) -> pd.DataFrame:
    """분위별 평균 수익률과 그 통계적 신뢰도."""
    rows = []
    for q in quantile_returns.columns:
        series = quantile_returns[q].dropna()
        if series.empty:
            continue
        mean = series.mean()
        t_stat = mean / (series.std() / np.sqrt(len(series))) if series.std() > 0 else 0.0
        rows.append(
            {
                "분위": f"Q{int(q) + 1}",
                "평균수익률": mean,
                "연율화": (1 + mean) ** periods_per_year - 1,
                "t-stat": t_stat,
                "관측": len(series),
            }
        )
    return pd.DataFrame(rows).set_index("분위")


def long_only_edge(
    quantile_returns: pd.DataFrame, periods_per_year: float, top_quantile: int | None = None
) -> dict:
    """
    롱온리로 실제로 얻는 초과분: 상위 분위 - 전체 평균.

    전체 평균을 기준으로 삼는 이유는, 신호를 안 쓰고 유니버스를 통째로 동일가중
    보유하는 것이 가장 정직한 비교 대상이기 때문이다. 상위-하위 스프레드는
    공매도를 해야 얻는 값이라 롱온리 성과를 과대평가한다.
    """
    if quantile_returns.empty:
        return {}

    columns = list(quantile_returns.columns)
    top = columns[-1] if top_quantile is None else top_quantile

    universe_mean = quantile_returns.mean(axis=1)
    excess = (quantile_returns[top] - universe_mean).dropna()
    spread = (quantile_returns[top] - quantile_returns[columns[0]]).dropna()

    def _annualize(series: pd.Series) -> float:
        return (1 + series.mean()) ** periods_per_year - 1

    def _t(series: pd.Series) -> float:
        return series.mean() / (series.std() / np.sqrt(len(series))) if series.std() > 0 else 0.0

    return {
        "상위분위 초과(연율화)": _annualize(excess),
        "상위분위 초과 t-stat": _t(excess),
        "상하위 스프레드(연율화)": _annualize(spread),
        "상하위 스프레드 t-stat": _t(spread),
        "관측": len(excess),
    }


def monotonicity(quantile_returns: pd.DataFrame) -> float:
    """
    분위 번호와 평균수익률의 순위상관. 1에 가까우면 신호가 클수록 수익도 커진다.

    단조성이 없는데 상위 분위만 좋으면 우연일 가능성이 높다 — 진짜 신호라면
    보통 전 구간에 걸쳐 완만하게라도 방향성이 나타난다.
    """
    means = quantile_returns.mean()
    if means.notna().sum() < 3:
        return float("nan")
    return pd.Series(means.index, index=means.index).corr(means, method="spearman")
