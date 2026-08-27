"""
검정력 분석 도구 자체를 검증한다.

신호 주입이 의도한 강도대로 동작하지 않으면 검정력 결과 전체가 무의미하므로,
'심은 만큼 나오는가'를 먼저 확인한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.ic_analysis import daily_cross_sectional_ic
from src.research.power_analysis import detection_rate, inject_signal

DATES = pd.bdate_range("2020-01-01", periods=200)
TICKERS = [f"T{i:03d}" for i in range(200)]


def _returns(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((len(DATES), len(TICKERS))) * 0.05, index=DATES, columns=TICKERS
    )


def _universe() -> pd.DataFrame:
    return pd.DataFrame(True, index=DATES, columns=TICKERS)


@pytest.mark.parametrize("target", [0.0, 0.05, 0.15, 0.30])
def test_injected_signal_has_the_intended_ic(target):
    """심은 강도와 실제로 측정되는 IC가 일치해야 한다."""
    fwd = _returns()
    signal = inject_signal(fwd, target, np.random.default_rng(1), _universe())

    realized = daily_cross_sectional_ic(signal, fwd, min_obs=30).mean()

    assert realized == pytest.approx(target, abs=0.03)


def test_zero_signal_is_not_detected_more_than_chance():
    """
    신호가 없으면 검출률이 유의수준 근처여야 한다.
    이보다 높으면 도구가 없는 신호를 만들어내고 있다는 뜻이다.
    """
    fwd = _returns()

    result = detection_rate(
        fwd, _universe(), target_ic=0.0, horizon=20, n_trials=100, threshold=1.96
    )

    assert result["검출률"] < 0.15


def test_strong_signal_is_almost_always_detected():
    """충분히 강한 신호는 거의 항상 잡아야 한다 (도구가 멀쩡하다는 최소 조건)."""
    fwd = _returns()

    result = detection_rate(
        fwd, _universe(), target_ic=0.30, horizon=20, n_trials=100, threshold=1.96
    )

    assert result["검출률"] > 0.9


def test_detection_rate_rises_with_signal_strength():
    """신호가 강할수록 검출률이 높아져야 한다 (단조성)."""
    fwd = _returns()

    rates = [
        detection_rate(fwd, _universe(), ic, horizon=20, n_trials=80, threshold=1.96)["검출률"]
        for ic in (0.02, 0.10, 0.25)
    ]

    assert rates[0] <= rates[1] <= rates[2]


def test_universe_mask_is_respected():
    """유니버스 밖 종목에는 신호가 생기면 안 된다."""
    fwd = _returns()
    universe = _universe()
    universe.iloc[:, 100:] = False

    signal = inject_signal(fwd, 0.1, np.random.default_rng(2), universe)

    assert signal.iloc[:, 100:].isna().all().all()
    assert signal.iloc[:, :100].notna().any().any()
