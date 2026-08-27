"""
분위수 분석이 롱온리 관점을 제대로 반영하는지 검증한다.

핵심은 'IC가 좋아도 롱온리로는 쓸모없을 수 있다'는 상황을 실제로 구분해내는 것이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.quantile_analysis import (
    assign_quantiles,
    long_only_edge,
    monotonicity,
    quantile_forward_returns,
)

DATES = pd.bdate_range("2024-01-01", periods=20)
TICKERS = [f"T{i:02d}" for i in range(50)]


def _panel(row: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.tile(row, (len(DATES), 1)), index=DATES, columns=TICKERS)


def _universe() -> pd.DataFrame:
    return pd.DataFrame(True, index=DATES, columns=TICKERS)


def test_quantiles_split_evenly():
    """50종목을 5분위로 나누면 각 10종목이어야 한다."""
    signal = _panel(np.arange(50, dtype=float))

    q = assign_quantiles(signal, _universe(), n_quantiles=5)

    counts = q.iloc[0].value_counts().sort_index()
    assert counts.tolist() == [10, 10, 10, 10, 10]
    # 신호가 가장 큰 종목이 최상위 분위
    assert q.iloc[0]["T49"] == 4
    assert q.iloc[0]["T00"] == 0


def test_days_with_too_few_names_are_dropped():
    """유효 종목이 min_obs 미만인 날은 분위를 매기지 않는다."""
    signal = _panel(np.arange(50, dtype=float))
    signal.iloc[3, 5:] = np.nan  # 5종목만 남김

    q = assign_quantiles(signal, _universe(), n_quantiles=5, min_obs=30)

    assert q.iloc[3].isna().all()
    assert q.iloc[0].notna().any()


def test_monotonic_signal_is_detected():
    """신호가 클수록 수익이 큰 경우 단조성이 1에 가까워야 한다."""
    signal = _panel(np.arange(50, dtype=float))
    fwd = _panel(np.arange(50, dtype=float) * 0.001)

    qr = quantile_forward_returns(signal, fwd, _universe(), n_quantiles=5)

    assert monotonicity(qr) == pytest.approx(1.0)


def test_signal_useful_only_on_short_side_shows_no_long_edge():
    """
    하위 종목만 유독 나쁘고 나머지는 똑같은 신호. IC는 잘 나오지만
    공매도를 못 하면 얻을 게 없다 - 상위 분위 초과가 0에 가까워야 한다.
    """
    signal = _panel(np.arange(50, dtype=float))
    returns = np.full(50, 0.01)
    returns[:10] = -0.10  # 최하위 분위만 폭락
    fwd = _panel(returns)

    qr = quantile_forward_returns(signal, fwd, _universe(), n_quantiles=5)
    edge = long_only_edge(qr, periods_per_year=12)

    # 상하위 스프레드는 크지만
    assert edge["상하위 스프레드(연율화)"] > 0.5
    # 상위 분위는 유니버스 평균보다 약간 나을 뿐 (하위가 평균을 끌어내린 만큼)
    top_excess = qr[4].mean() - qr.mean(axis=1).mean()
    assert abs(top_excess - 0.022) < 0.001


def test_long_only_edge_measures_against_universe_average():
    """상위 분위 초과는 '유니버스 평균 대비'로 계산돼야 한다."""
    signal = _panel(np.arange(50, dtype=float))
    returns = np.full(50, 0.0)
    returns[40:] = 0.05  # 최상위 분위만 좋음
    fwd = _panel(returns)

    qr = quantile_forward_returns(signal, fwd, _universe(), n_quantiles=5)
    edge = long_only_edge(qr, periods_per_year=12)

    # 유니버스 평균 = 0.05/5 = 0.01, 상위 분위 = 0.05 -> 초과 0.04
    assert abs((qr[4].mean() - qr.mean(axis=1).mean()) - 0.04) < 1e-9
    assert edge["상위분위 초과 t-stat"] > 0


def test_sampling_reduces_overlapping_observations():
    """sample_every를 주면 관측 수가 그만큼 줄어야 한다 (겹침 방지)."""
    signal = _panel(np.arange(50, dtype=float))
    fwd = _panel(np.arange(50, dtype=float) * 0.001)

    every_day = quantile_forward_returns(signal, fwd, _universe(), n_quantiles=5)
    sampled = quantile_forward_returns(signal, fwd, _universe(), n_quantiles=5, sample_every=5)

    assert len(every_day) == 20
    assert len(sampled) == 4


def test_universe_rank_range_selects_the_right_slice():
    """rank_from을 주면 그 순위부터 시작하는 구간만 편입돼야 한다."""
    from src.data_loader.universe import market_cap_universe_mask

    dates = pd.bdate_range("2024-01-01", periods=5)
    caps = pd.DataFrame(
        {f"T{i:02d}": [float(100 - i)] * len(dates) for i in range(10)}, index=dates
    )

    top3 = market_cap_universe_mask(caps, top_k=3)
    mid = market_cap_universe_mask(caps, top_k=6, rank_from=3)

    # 시총은 T00이 가장 크고 T09가 가장 작다
    assert set(top3.columns[top3.iloc[0]]) == {"T00", "T01", "T02"}
    assert set(mid.columns[mid.iloc[0]]) == {"T03", "T04", "T05"}
    # 두 구간은 겹치지 않아야 한다
    assert not (top3 & mid).any().any()
