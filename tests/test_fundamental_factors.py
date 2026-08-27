"""
재무 팩터 검증.

가장 중요한 것은 **DART의 Q4가 연간 누적**이라는 사실을 잊지 않는 것이다. 그걸
분기값으로 착각하면 4분기 순이익이 4배쯤 부풀고, 그 상태로 TTM을 만들면 수익성이
연중 내내 요동친다 - 그런데 숫자는 그럴듯해서 조용히 지나간다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.features.earnings_surprise import quarterly_grid
from src.features.fundamental_factors import (
    build_fundamental_panels,
    standalone_quarterly,
    to_daily,
    trailing_twelve_months,
)


def _filings(rows: list[tuple[int, int, dict]], ticker: str = "A") -> pd.DataFrame:
    """(연도, 분기, 값들). 공시일은 분기 종료 후 45일로 둔다."""
    records = []
    for year, quarter, values in rows:
        period_end = pd.Period(year=year, quarter=quarter, freq="Q").end_time
        records.append(
            {
                "ticker": ticker,
                "fiscal_year": year,
                "fiscal_quarter": quarter,
                "available_date": (period_end + pd.Timedelta("45D")).normalize(),
                **values,
            }
        )
    return pd.DataFrame(records)


def test_fourth_quarter_is_converted_from_annual_to_standalone():
    """
    Q4는 연간 누적으로 들어온다. 실제 삼성전자 2024년 값으로 확인한다.

    Q1 6.75 / Q2 9.84 / Q3 10.10조가 분기값이고 Q4가 34.45조인데, 이건 연간이다.
    앞 셋을 더하면 26.69조이므로 Q4 단독은 7.76조다.
    """
    rows = [
        (2024, 1, {"net_income": 6.75}),
        (2024, 2, {"net_income": 9.84}),
        (2024, 3, {"net_income": 10.10}),
        (2024, 4, {"net_income": 34.45}),
    ]

    standalone = standalone_quarterly(quarterly_grid(_filings(rows)), "net_income")

    assert standalone.iloc[3] == pytest.approx(34.45 - (6.75 + 9.84 + 10.10))
    assert standalone.iloc[:3].tolist() == [6.75, 9.84, 10.10]  # 앞 셋은 그대로


def test_trailing_twelve_months_equals_the_reported_annual_figure():
    """되돌린 4개 분기를 다시 더하면 공시된 연간값이 나와야 한다."""
    rows = [
        (2024, 1, {"net_income": 6.75}),
        (2024, 2, {"net_income": 9.84}),
        (2024, 3, {"net_income": 10.10}),
        (2024, 4, {"net_income": 34.45}),
    ]

    ttm = trailing_twelve_months(quarterly_grid(_filings(rows)), "net_income")

    assert ttm.iloc[3] == pytest.approx(34.45)


def test_incomplete_year_gives_no_standalone_q4():
    """앞 세 분기 중 하나라도 없으면 Q4 단독값을 만들지 않는다 (추정해 채우지 않는다)."""
    rows = [
        (2024, 1, {"net_income": 6.75}),
        (2024, 3, {"net_income": 10.10}),  # Q2 누락
        (2024, 4, {"net_income": 34.45}),
    ]

    standalone = standalone_quarterly(quarterly_grid(_filings(rows)), "net_income")

    assert pd.isna(standalone.iloc[-1])


def test_values_appear_only_after_the_filing_date():
    """공시일 전에는 그 값을 알 수 없다 - look-ahead 방지의 핵심."""
    rows = [(2024, 1, {"equity": 100.0}), (2024, 2, {"equity": 200.0})]
    gridded = quarterly_grid(_filings(rows))
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")

    daily = to_daily(gridded, gridded["equity"].astype(float), dates)
    q1_filed = gridded["available_date"].iloc[0]

    assert daily[daily.index < q1_filed].isna().all()
    assert daily[daily.index >= q1_filed].iloc[0] == 100.0


def test_negative_equity_does_not_produce_a_book_to_market():
    """자본잠식 기업의 B/M은 의미가 없다 (음수 B/M이 '싼 주식'으로 줄 서면 안 된다)."""
    rows = [(2023, q, {"equity": -50.0, "assets": 100.0, "net_income": 1.0}) for q in (1, 2, 3, 4)]
    rows += [(2024, q, {"equity": -50.0, "assets": 100.0, "net_income": 1.0}) for q in (1, 2, 3, 4)]
    dates = pd.date_range("2023-01-02", "2025-06-30", freq="B")
    market_cap = pd.DataFrame({"A": 1000.0}, index=dates)

    panels = build_fundamental_panels(_filings(rows), dates, market_cap)

    assert panels["book_to_market"]["A"].isna().all()


def test_panels_are_built_for_a_normal_company():
    rows = []
    for year in (2022, 2023, 2024):
        for quarter in (1, 2, 3, 4):
            income = 10.0 if quarter < 4 else 40.0  # Q4는 연간 누적
            rows.append((year, quarter, {
                "equity": 500.0 + 10 * year, "assets": 1000.0 + 20 * year, "net_income": income,
            }))
    dates = pd.date_range("2022-01-03", "2025-06-30", freq="B")
    market_cap = pd.DataFrame({"A": 2000.0}, index=dates)

    panels = build_fundamental_panels(_filings(rows), dates, market_cap)
    last = {name: panel["A"].iloc[-1] for name, panel in panels.items()}

    assert last["book_to_market"] == pytest.approx((500.0 + 10 * 2024) / 2000.0)
    assert last["roe"] == pytest.approx(40.0 / (500.0 + 10 * 2024))  # TTM = 공시된 연간값
    assert last["asset_growth"] == pytest.approx(20 / (1000.0 + 20 * 2023))
