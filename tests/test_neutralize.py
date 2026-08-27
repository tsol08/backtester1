"""
중립화가 이상치에 흔들리지 않는지 검증한다.

원본값 회귀는 극단값 하나가 기울기를 좌우해서, 실제로는 거의 무관한 두 변수인데도
중립화 후 팩터가 완전히 달라질 수 있다. 내부자 순매수 신호에서 실제로 겪은 문제다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.multi_source import neutralize

DATES = pd.bdate_range("2024-01-01", periods=3)
TICKERS = [f"T{i:02d}" for i in range(40)]


def _panel(row: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([row] * len(DATES), index=DATES, columns=TICKERS)


def test_rank_neutralize_survives_outlier():
    """
    control과 순위상관이 거의 없는 팩터에, 극단값 하나를 넣어도 신호 순위가
    보존돼야 한다. 원본값 회귀는 여기서 무너진다.
    """
    rng = np.random.default_rng(0)
    control = _panel(np.arange(40, dtype=float))

    values = rng.permutation(40).astype(float)
    values[0] = 100_000.0  # 이상치 하나
    factor = _panel(values)

    ranked = neutralize(factor, control, use_ranks=True)
    raw = neutralize(factor, control, use_ranks=False)

    original_order = factor.iloc[0].rank()
    # 순위 기반은 원래 순위를 거의 그대로 유지한다
    assert ranked.iloc[0].corr(original_order, method="spearman") > 0.95
    # 원본값 회귀는 그렇지 못하다 (이상치가 기울기를 끌고 감)
    assert raw.iloc[0].corr(original_order, method="spearman") < ranked.iloc[0].corr(
        original_order, method="spearman"
    )


def test_perfectly_correlated_control_is_removed():
    """control과 순위가 완전히 같으면 잔차에 정보가 남지 않아야 한다."""
    control = _panel(np.arange(40, dtype=float))
    factor = _panel(np.arange(40, dtype=float) * 3.0)

    result = neutralize(factor, control)

    assert np.allclose(result.iloc[0].dropna().values, 0.0, atol=1e-9)


def test_skips_dates_with_too_few_observations():
    """유효 관측이 min_obs 미만인 날은 계산하지 않고 NaN으로 둔다."""
    control = _panel(np.arange(40, dtype=float))
    factor = _panel(np.arange(40, dtype=float))
    factor.iloc[1, 5:] = np.nan  # 유효 5개만 남김

    result = neutralize(factor, control, min_obs=30)

    assert result.iloc[1].isna().all()
    assert result.iloc[0].notna().any()
