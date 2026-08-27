"""
공시목록 파싱과 이벤트 마스크 검증.

이벤트 신호에서 조용히 틀리는 곳은 둘이다: **같은 사건을 두 번 세는 것**(정정공시를
새 사건으로 오인)과 **사건을 알기 전에 아는 것**(접수일 이전에 마스크가 켜지는 것).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader.dart_filings import event_dates, recent_event_mask


def _filings(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """(종목코드, 접수일, 공시명)."""
    frame = pd.DataFrame(rows, columns=["stock_code", "rcept_dt", "report_nm"])
    frame["event_date"] = pd.to_datetime(frame["rcept_dt"], format="%Y%m%d")
    frame["is_original"] = ~frame["report_nm"].str.startswith("[")
    frame["kind"] = frame["report_nm"].str.replace(
        r"^\[[^\]]+\]", "", regex=True
    ).str.extract(r"\(([^)]+)\)\s*$")[0].fillna("")
    return frame


def test_corrections_are_not_counted_as_new_events():
    """
    [기재정정] 공시는 이미 낸 공시를 고친 것이라 원본이 따로 목록에 있다.

    이걸 새 사건으로 세면 같은 자사주 취득을 두 번 세게 되고, 정정일을 사건일로 쓰면
    실제보다 늦은 날짜가 된다.
    """
    filings = _filings([
        ("005930", "20240110", "주요사항보고서(자기주식취득결정)"),
        ("005930", "20240220", "[기재정정]주요사항보고서(자기주식취득결정)"),
        ("000660", "20240115", "[연장결정]주요사항보고서(자기주식취득신탁계약체결결정)"),
    ])

    events = event_dates(filings, ["자기주식취득결정", "자기주식취득신탁계약체결결정"])

    assert len(events) == 1
    assert events.iloc[0]["ticker"] == "005930"
    assert events.iloc[0]["event_date"] == pd.Timestamp("2024-01-10")


def test_acquisition_and_disposal_are_different_events():
    """자기주식 '취득'과 '처분'은 방향이 반대다. 한 바구니에 담으면 서로 상쇄된다."""
    filings = _filings([
        ("005930", "20240110", "주요사항보고서(자기주식취득결정)"),
        ("000660", "20240111", "주요사항보고서(자기주식처분결정)"),
    ])

    assert list(event_dates(filings, ["자기주식취득결정"])["ticker"]) == ["005930"]
    assert list(event_dates(filings, ["자기주식처분결정"])["ticker"]) == ["000660"]


def test_mask_is_off_before_the_filing_date():
    """접수일 이전에는 그 사건을 알 수 없다."""
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="B")
    events = pd.DataFrame({"ticker": ["005930"], "event_date": [pd.Timestamp("2024-02-01")]})

    mask = recent_event_mask(events, dates, pd.Index(["005930"]), window_days=60)

    assert not mask.loc[mask.index < "2024-02-01", "005930"].any()
    assert mask.loc["2024-02-01", "005930"]


def test_mask_expires_after_the_window():
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    events = pd.DataFrame({"ticker": ["005930"], "event_date": [pd.Timestamp("2024-02-01")]})

    mask = recent_event_mask(events, dates, pd.Index(["005930"]), window_days=60)
    live = mask.index[mask["005930"]]

    assert live.min() == pd.Timestamp("2024-02-01")
    assert live.max() <= pd.Timestamp("2024-04-01")  # 2/1 + 60일
    assert not mask.loc[mask.index > "2024-04-01", "005930"].any()


def test_overlapping_events_extend_the_window():
    """사건이 연달아 나면 창이 이어진다 - 마지막 사건 기준으로 다시 센다."""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    events = pd.DataFrame({
        "ticker": ["005930", "005930"],
        "event_date": [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-15")],
    })

    mask = recent_event_mask(events, dates, pd.Index(["005930"]), window_days=60)

    assert mask.loc["2024-03-20", "005930"]  # 두 번째 사건의 창 안
    assert mask.loc["2024-05-10", "005930"]  # 3/15 + 60일 안
    assert not mask.loc["2024-05-20", "005930"]


def test_tickers_outside_the_panel_are_ignored():
    """유니버스에 없는 종목의 사건은 마스크에 들어올 자리가 없다."""
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="B")
    events = pd.DataFrame({
        "ticker": ["005930", "999999"],
        "event_date": [pd.Timestamp("2024-02-01")] * 2,
    })

    mask = recent_event_mask(events, dates, pd.Index(["005930"]), window_days=30)

    assert list(mask.columns) == ["005930"]
