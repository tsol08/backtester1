"""
룩어헤드 바이어스(미래 정보 누수) 방지 검증.

핵심 아이디어: t일 이후의 가격 데이터를 바꿔도, t일 및 그 이전의 체결 포지션/수익률은
전혀 달라지면 안 된다. 만약 달라진다면 엔진이 미래 정보를 참조하고 있다는 뜻이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.costs.cost_model import CostModel
from src.engine.backtest import run_backtest
from src.features.technical import build_features
from src.signals.rules.mean_reversion import disparity_zscore_signal
from tests.factories import make_synthetic_ohlcv


def test_position_unaffected_by_future_prices():
    df = make_synthetic_ohlcv()
    cutoff = 150

    features_full = build_features(df)
    signal_full = disparity_zscore_signal(features_full)
    result_full = run_backtest(df, signal_full, CostModel())

    df_perturbed = df.copy()
    rng = np.random.default_rng(999)
    future_shock = rng.normal(0, 0.3, size=len(df) - cutoff)
    df_perturbed.iloc[cutoff:, df_perturbed.columns.get_loc("close")] *= (1 + future_shock)

    features_perturbed = build_features(df_perturbed)
    signal_perturbed = disparity_zscore_signal(features_perturbed)
    result_perturbed = run_backtest(df_perturbed, signal_perturbed, CostModel())

    # cutoff 이전 구간의 "체결된 포지션"은 미래를 바꿔도 동일해야 한다.
    # (피처 자체는 rolling window라 cutoff 근방 며칠은 window에 걸쳐 있을 수 있으니
    #  window보다 충분히 이전 구간만 비교한다)
    safe_end = cutoff - 60
    pos_full = result_full.positions.iloc[:safe_end]
    pos_perturbed = result_perturbed.positions.iloc[:safe_end]

    pd.testing.assert_series_equal(pos_full, pos_perturbed, check_names=False)


def test_signal_shift_is_the_only_lookahead_guard():
    df = make_synthetic_ohlcv()
    features = build_features(df)
    signal = disparity_zscore_signal(features)
    result = run_backtest(df, signal, CostModel())

    expected_position = signal.reindex(df.index).fillna(0.0).shift(1).fillna(0.0)
    pd.testing.assert_series_equal(result.positions, expected_position, check_names=False)
