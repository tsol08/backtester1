"""
재무데이터 일별 변환의 look-ahead bias 방지 검증.

재무제표는 회계기간 종료일과 실제 공시일이 다르다(보통 1~3개월 시차). 공시 전에
그 데이터를 알고 있었다고 가정하면 심각한 미래 정보 누수가 된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features.fundamental import FACTOR_COLUMNS, to_daily_factors


def test_factor_not_visible_before_disclosure_date():
    # 2023 회계연도 데이터가 2024-03-12에 공시된 상황
    factor_df = pd.DataFrame(
        {
            "available_date": [pd.Timestamp("2023-03-07"), pd.Timestamp("2024-03-12")],
            "roe": [0.10, 0.99],
            "operating_margin": [0.10, 0.99],
            "net_margin": [0.10, 0.99],
            "debt_ratio": [0.10, 0.99],
            "asset_turnover": [0.10, 0.99],
            "revenue_growth_yoy": [0.10, 0.99],
            "operating_income_growth_yoy": [0.10, 0.99],
        }
    )

    trading_dates = pd.to_datetime(
        ["2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13"]
    )
    daily = to_daily_factors(factor_df, trading_dates)

    # 공시 전날(03-11)까지는 직전 공시(2023-03-07)의 값 0.10만 알 수 있어야 한다.
    assert daily.loc[pd.Timestamp("2024-03-08"), "roe"] == 0.10
    assert daily.loc[pd.Timestamp("2024-03-11"), "roe"] == 0.10
    # 공시 당일부터 새 값 0.99가 반영된다.
    assert daily.loc[pd.Timestamp("2024-03-12"), "roe"] == 0.99
    assert daily.loc[pd.Timestamp("2024-03-13"), "roe"] == 0.99


def test_no_factor_before_first_disclosure():
    factor_df = pd.DataFrame(
        {
            "available_date": [pd.Timestamp("2024-03-12")],
            **{col: [0.5] for col in FACTOR_COLUMNS},
        }
    )
    trading_dates = pd.to_datetime(["2024-01-02", "2024-03-12"])
    daily = to_daily_factors(factor_df, trading_dates)

    # 첫 공시 이전에는 아무 값도 없어야 한다(NaN).
    assert np.isnan(daily.loc[pd.Timestamp("2024-01-02"), "roe"])
    assert daily.loc[pd.Timestamp("2024-03-12"), "roe"] == 0.5


def test_future_disclosure_never_leaks_backward():
    """미래 공시 값을 바꿔도 과거 거래일의 팩터는 변하지 않아야 한다."""
    base = pd.DataFrame(
        {
            "available_date": [pd.Timestamp("2023-03-07"), pd.Timestamp("2024-03-12")],
            **{col: [0.10, 0.20] for col in FACTOR_COLUMNS},
        }
    )
    perturbed = base.copy()
    perturbed.loc[1, FACTOR_COLUMNS] = 99.0  # 미래 공시값만 크게 변경

    trading_dates = pd.to_datetime(["2023-06-01", "2024-01-02", "2024-03-11"])

    daily_base = to_daily_factors(base, trading_dates)
    daily_perturbed = to_daily_factors(perturbed, trading_dates)

    pd.testing.assert_frame_equal(daily_base, daily_perturbed)
