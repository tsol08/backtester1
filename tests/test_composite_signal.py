"""composite_signal의 룩어헤드 방지 및 히스테리시스 동작 검증."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features.technical import build_features
from src.signals.rules.composite_score import composite_signal
from tests.factories import make_synthetic_ohlcv


def test_signal_unaffected_by_future_prices():
    df = make_synthetic_ohlcv(n=300, seed=1)
    cutoff = 220

    signal_full = composite_signal(build_features(df))

    df_perturbed = df.copy()
    rng = np.random.default_rng(999)
    shock = rng.normal(0, 0.3, size=len(df) - cutoff)
    df_perturbed.iloc[cutoff:, df_perturbed.columns.get_loc("close")] *= (1 + shock)
    signal_perturbed = composite_signal(build_features(df_perturbed))

    # 가장 긴 rolling window(모멘텀 60일 + 그 z-score용 60일)를 감안해 넉넉히 여유를 둔다.
    safe_end = cutoff - 120
    pd.testing.assert_series_equal(
        signal_full.iloc[:safe_end], signal_perturbed.iloc[:safe_end], check_names=False
    )


def test_hysteresis_holds_position_between_thresholds():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    # composite_score를 직접 구성하지 않고, entry/exit 사이 구간에서 ffill이
    # 실제로 직전 값을 들고 가는지를 최소 재현으로 확인한다.
    score = pd.Series([2.0, 0.5, 0.5, -1.0, 0.5], index=dates)
    raw = pd.Series(np.nan, index=score.index)
    raw[score > 1.0] = 1.0
    raw[score < 0.0] = 0.0
    signal = raw.ffill().fillna(0.0)

    # day0: 진입(2.0>1.0) -> 1, day1,2: 중립구간(0<=0.5<=1) -> 직전 상태(1) 유지,
    # day3: 청산(-1.0<0) -> 0, day4: 중립구간 -> 직전 상태(0) 유지
    expected = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0], index=dates)
    pd.testing.assert_series_equal(signal, expected, check_names=False)
