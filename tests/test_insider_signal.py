"""
내부자 거래 신호가 공시일 이전 정보를 쓰지 않는지 검증한다.

재무제표와 같은 함정이 있다: 내부자는 거래를 한 뒤 며칠 지나서 신고한다.
실제 거래일 기준으로 신호를 만들면 아직 공개되지 않은 정보를 쓰게 되므로,
반드시 공시일(rcept_dt) 기준이어야 한다.
"""
from __future__ import annotations

import pandas as pd

from src.data_loader.dart_insider import build_insider_signal

DATES = pd.bdate_range("2024-01-01", periods=10)


def _shares(value: float = 1_000_000.0) -> pd.DataFrame:
    return pd.DataFrame({"A": [value] * len(DATES)}, index=DATES)


def _trade(date: str, change: float, ticker: str = "A") -> dict:
    return {"disclosed_date": pd.Timestamp(date), "ticker": ticker, "shares_change": change}


def test_signal_is_zero_before_disclosure():
    """공시 전에는 그 거래가 신호에 반영되면 안 된다."""
    trades = pd.DataFrame([_trade(DATES[5].strftime("%Y-%m-%d"), 10_000)])

    signal = build_insider_signal(trades, DATES, _shares(), lookback=3)

    # 공시일(index 5) 이전은 0이어야 한다
    assert (signal["A"].iloc[2:5] == 0).all()
    # 공시일부터 반영: 10,000 / 1,000,000 = 0.01
    assert signal["A"].iloc[5] == 0.01


def test_signal_uses_only_trailing_window():
    """lookback을 벗어난 과거 거래는 신호에서 빠져야 한다."""
    trades = pd.DataFrame([_trade(DATES[2].strftime("%Y-%m-%d"), 10_000)])

    signal = build_insider_signal(trades, DATES, _shares(), lookback=3)

    assert signal["A"].iloc[2] == 0.01  # 공시 당일
    assert signal["A"].iloc[4] == 0.01  # 윈도우 안 (2,3,4)
    assert signal["A"].iloc[5] == 0.0  # 윈도우 밖으로 밀려남


def test_buys_and_sells_net_out():
    """같은 날 매수/매도가 있으면 순매수로 상계돼야 한다."""
    day = DATES[4].strftime("%Y-%m-%d")
    trades = pd.DataFrame([_trade(day, 30_000), _trade(day, -10_000)])

    signal = build_insider_signal(trades, DATES, _shares(), lookback=3)

    assert signal["A"].iloc[4] == 0.02  # (30,000 - 10,000) / 1,000,000


def test_normalized_by_shares_outstanding():
    """같은 주식수라도 상장주식수가 다르면 신호 크기가 달라야 한다."""
    trades = pd.DataFrame([_trade(DATES[4].strftime("%Y-%m-%d"), 10_000)])

    small = build_insider_signal(trades, DATES, _shares(1_000_000), lookback=3)
    large = build_insider_signal(trades, DATES, _shares(10_000_000), lookback=3)

    assert small["A"].iloc[4] == 10 * large["A"].iloc[4]


def test_future_disclosure_does_not_change_past_signal():
    """나중 공시를 추가해도 그 이전 날짜의 신호는 그대로여야 한다."""
    early = pd.DataFrame([_trade(DATES[2].strftime("%Y-%m-%d"), 10_000)])
    with_future = pd.concat(
        [early, pd.DataFrame([_trade(DATES[7].strftime("%Y-%m-%d"), 999_000)])],
        ignore_index=True,
    )

    before = build_insider_signal(early, DATES, _shares(), lookback=3)
    after = build_insider_signal(with_future, DATES, _shares(), lookback=3)

    pd.testing.assert_series_equal(before["A"].iloc[:7], after["A"].iloc[:7])


def test_archive_keeps_records_that_fell_out_of_api_window():
    """
    API는 2년 롤링 윈도우라 오래된 공시가 응답에서 빠진다. 아카이브는 그걸
    보존해야 한다 - 안 그러면 아무리 오래 수집해도 표본이 2년에 고정된다.
    """
    from src.data_loader.dart_insider import merge_archive

    old = pd.DataFrame(
        {
            "rcept_no": ["A1", "A2"],
            "disclosed_date": pd.to_datetime(["2024-09-01", "2025-01-01"]),
            "shares_change": [100.0, 200.0],
        }
    )
    # 윈도우가 밀려서 A1은 빠지고 A3가 새로 들어온 응답
    fetched = pd.DataFrame(
        {
            "rcept_no": ["A2", "A3"],
            "disclosed_date": pd.to_datetime(["2025-01-01", "2026-06-01"]),
            "shares_change": [200.0, 300.0],
        }
    )

    merged = merge_archive(old, fetched)

    assert set(merged["rcept_no"]) == {"A1", "A2", "A3"}
    assert len(merged) == 3


def test_archive_prefers_corrected_disclosure():
    """접수번호가 같으면 새로 받은 값이 이긴다 (정정공시 반영)."""
    from src.data_loader.dart_insider import merge_archive

    old = pd.DataFrame(
        {
            "rcept_no": ["A1"],
            "disclosed_date": pd.to_datetime(["2025-01-01"]),
            "shares_change": [100.0],
        }
    )
    fetched = old.copy()
    fetched["shares_change"] = [150.0]

    merged = merge_archive(old, fetched)

    assert len(merged) == 1
    assert merged["shares_change"].iloc[0] == 150.0
