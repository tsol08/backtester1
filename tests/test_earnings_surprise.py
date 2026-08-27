"""
SUE 계산과 look-ahead 방지를 검증한다.

재무데이터 기반 신호에서 가장 흔한 실수가 '회계기간 기준으로 값을 쓰는 것'이다.
2024년 4분기 실적은 2025년 3월에야 공시되므로, 1월에 알고 있었다고 가정하면
미래를 보는 것이 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.earnings_surprise import compute_sue, to_daily_sue

TRADING_DATES = pd.bdate_range("2021-01-01", periods=1600)


def _frame(net_income: list[float], start_year: int = 2016) -> pd.DataFrame:
    """분기 재무데이터. 공시는 회계기간 종료 45일 뒤로 둔다."""
    rows = []
    for i, value in enumerate(net_income):
        year = start_year + i // 4
        quarter = i % 4 + 1
        period_end = pd.Timestamp(year=year, month=quarter * 3, day=1) + pd.offsets.MonthEnd(0)
        rows.append(
            {
                "fiscal_year": year,
                "fiscal_quarter": quarter,
                "available_date": period_end + pd.Timedelta(45, unit="D"),
                "net_income": value,
            }
        )
    return pd.DataFrame(rows)


def _varied(n: int = 24, seed: int = 0) -> list[float]:
    """
    계절성(4분기가 큼) + 불규칙한 성장을 가진 이익 시계열.

    전년 대비 변화가 매번 달라야 표준편차가 정의된다. 변화가 일정하면 분모가 0이 되어
    SUE가 계산되지 않는데, 그건 올바른 동작이지만 다른 성질을 검증할 수는 없다.
    """
    rng = np.random.default_rng(seed)
    seasonal = np.tile([100.0, 120.0, 110.0, 250.0], n // 4 + 1)[:n]
    drift = np.cumsum(rng.normal(5.0, 20.0, n))
    return list(seasonal + drift)


def test_constant_year_over_year_change_gives_no_sue():
    """
    전년 대비 변화가 늘 같으면 '이례적'이라는 개념 자체가 성립하지 않는다.
    분모(변화의 표준편차)가 0이므로 SUE는 계산되지 않아야 한다.
    """
    linear = _frame([100.0 + 10 * i for i in range(20)])

    assert compute_sue(linear)["sue"].isna().all()


def test_seasonality_does_not_create_fake_surprise():
    """
    4분기 이익이 항상 큰 회사에서, 직전 분기와 비교하면 매년 4분기가 거대한
    서프라이즈로 보인다. 전년 동기와 비교하므로 그런 일이 없어야 한다.
    """
    result = compute_sue(_frame(_varied(28))).dropna(subset=["sue"])

    by_quarter = result.groupby("fiscal_quarter")["sue"].mean()
    # 4분기 SUE가 다른 분기보다 체계적으로 크지 않아야 한다
    assert abs(by_quarter.get(4, 0.0)) < 3.0
    assert by_quarter.abs().max() < 3.0


def test_matching_last_year_gives_near_zero_sue():
    """마지막 분기 이익이 전년 동기와 같으면 서프라이즈가 0이어야 한다."""
    values = _varied(24)
    values.append(values[-4])  # 4분기 전과 동일

    result = compute_sue(_frame(values))

    assert result["sue"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_large_beat_gives_large_positive_sue():
    """전년 동기 대비 이익이 크게 늘면 SUE가 뚜렷한 양수여야 한다."""
    values = _varied(24)
    values.append(values[-4] + 2000.0)

    result = compute_sue(_frame(values))

    assert result["sue"].iloc[-1] > 3.0


def test_large_miss_gives_large_negative_sue():
    """반대로 크게 줄면 뚜렷한 음수."""
    values = _varied(24)
    values.append(values[-4] - 2000.0)

    result = compute_sue(_frame(values))

    assert result["sue"].iloc[-1] < -3.0


def test_sue_needs_enough_history():
    """표준편차를 낼 과거가 부족하면 계산하지 않는다."""
    short = _frame(_varied(8))

    assert compute_sue(short)["sue"].isna().all()


def test_daily_sue_is_unavailable_before_disclosure():
    """공시일 이전에는 그 분기 SUE를 알 수 없어야 한다."""
    sue = compute_sue(_frame(_varied(28)))
    last = sue.dropna(subset=["sue"]).iloc[-1]

    daily = to_daily_sue(sue, TRADING_DATES)
    before = daily.loc[daily.index < last["available_date"]].dropna()

    assert not (before == last["sue"]).any()


def test_drift_window_expires_the_signal():
    """
    PEAD는 발표 직후 현상이다. drift_window를 주면 그 기간이 지난 뒤 신호가
    사라져야 한다 - 이미 시장이 소화한 오래된 서프라이즈를 계속 세지 않도록.
    """
    sue = compute_sue(_frame(_varied(28)))
    disclosure = sue.dropna(subset=["sue"])["available_date"].iloc[-1]

    windowed = to_daily_sue(sue, TRADING_DATES, drift_window=30)
    unwindowed = to_daily_sue(sue, TRADING_DATES, drift_window=None)

    stale = TRADING_DATES[TRADING_DATES > disclosure + pd.Timedelta(31, unit="D")]
    assert len(stale) > 0
    assert windowed.loc[stale].isna().all()
    assert unwindowed.loc[stale].notna().all()


def test_future_disclosure_does_not_change_past_signal():
    """나중 분기를 추가해도 그 이전 날짜의 SUE는 그대로여야 한다."""
    values = _varied(24)
    base = compute_sue(_frame(values))
    extended = compute_sue(_frame(values + [999999.0]))

    cutoff = extended["available_date"].iloc[-1]
    mask = TRADING_DATES < cutoff

    pd.testing.assert_series_equal(
        to_daily_sue(base, TRADING_DATES)[mask],
        to_daily_sue(extended, TRADING_DATES)[mask],
    )


def _quarters(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    """(연도, 분기, 순이익) 목록. 공시일은 분기말 다음달 15일로 둔다."""
    return pd.DataFrame([
        {
            "fiscal_year": y,
            "fiscal_quarter": q,
            "net_income": income,
            "available_date": pd.Timestamp(year=y + (q == 4), month=(q * 3) % 12 + 1, day=15),
        }
        for y, q, income in rows
    ])


def test_missing_quarter_is_not_silently_compared_to_the_wrong_one():
    """
    분기가 빠지면 그 자리를 NaN으로 두고, 전년 동기가 없으면 SUE를 내지 않는다.

    격자 없이 shift(4)를 쓰면 '4행 전'이 되어, 한 분기가 빠진 종목은 전년 동기가
    아니라 **3분기 전과** 비교된다. 계절성 보정이 SUE의 존재 이유인데 그게 사라진다.
    수집된 DART 데이터는 400종목 전부에 빈 분기가 있었다(중앙값 5분기).
    """
    rows = [(2015, q, 100.0) for q in (1, 2, 3, 4)]
    rows += [(2016, q, 200.0) for q in (1, 3, 4)]  # 2016 Q2 누락
    rows += [(2017, q, 300.0) for q in (1, 2, 3, 4)]

    result = compute_sue(_quarters(rows)).set_index(["fiscal_year", "fiscal_quarter"])

    # 빠진 분기는 행으로는 남되 값이 없다
    assert pd.isna(result.loc[(2016, 2), "net_income"])
    assert pd.isna(result.loc[(2016, 2), "available_date"])
    # 전년 동기가 없는 2017 Q2는 서프라이즈를 낼 수 없다
    assert pd.isna(result.loc[(2017, 2), "sue"])
    # 2017 Q1은 전년 동기(2016 Q1)가 있으므로 값이 나올 자격이 있다
    assert result.loc[(2017, 1), "net_income"] == 300.0


def test_duplicate_quarters_do_not_shift_the_year_over_year_lag():
    """
    같은 분기가 두 번 들어와도 결과가 달라지지 않는다.

    수집 데이터에 중복 행이 2,844개(63종목) 있었다. 중복이 있으면 shift(4)가 밀려서
    엉뚱한 분기와 비교되고, **나중에 데이터를 더 받으면 과거 SUE 값이 바뀐다**
    (실제로 000210의 2023-07-28 SUE가 -2.65에서 +1.21로 뒤집혔다).
    """
    rows = [(y, q, 100.0 * y + q) for y in range(2015, 2021) for q in (1, 2, 3, 4)]
    clean = compute_sue(_quarters(rows))
    with_dupes = compute_sue(_quarters(rows + rows[4:8]))

    pd.testing.assert_series_equal(clean["sue"], with_dupes["sue"])


def test_later_filings_do_not_change_past_sue():
    """
    미래 분기를 더 받아도 과거 SUE는 그대로여야 한다 - 아니면 look-ahead다.

    이건 이 프로젝트에서 실제로 발생했고, DART 수집 범위를 2024년에서 2026년으로
    늘렸을 때 과거 구간 셀 45만개 중 1만개가 바뀌면서 드러났다.
    """
    rows = [(y, q, 100.0 * y + q * 7) for y in range(2015, 2022) for q in (1, 2, 3, 4)]
    extra = [(2022, q, 999.0 * q) for q in (1, 2, 3, 4)]

    short = compute_sue(_quarters(rows))
    long = compute_sue(_quarters(rows + extra))

    pd.testing.assert_series_equal(short["sue"], long["sue"].iloc[: len(short)])
