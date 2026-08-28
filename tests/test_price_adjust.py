"""기업행위 보정이 분할만 잡고 증자는 건드리지 않는지 검증한다."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader.price_adjust import (
    adjust_close,
    load_adjusted_close,
    mask_impossible_moves,
)

DATES = pd.date_range("2024-01-01", periods=5, freq="B")


def _panel(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"A": values}, index=DATES)


def test_split_is_removed_from_returns():
    """10:1 액면분할이면 주가 1/10, 주식수 10배 -> 보정 후 수익률 0이어야 한다."""
    close = _panel([1000.0, 1000.0, 100.0, 100.0, 100.0])
    shares = _panel([1_000.0, 1_000.0, 10_000.0, 10_000.0, 10_000.0])

    adjusted = adjust_close(close, shares)
    returns = adjusted["A"].pct_change(fill_method=None)

    assert returns.iloc[2] == 0.0
    # 분할 이후 구간이 분할 이전 스케일로 이어져야 한다
    assert adjusted["A"].iloc[2] == 1000.0


def test_reverse_split_is_removed():
    """1:10 주식병합(주가 10배, 주식수 1/10)도 같은 방식으로 상쇄돼야 한다."""
    close = _panel([100.0, 100.0, 1000.0, 1000.0, 1000.0])
    shares = _panel([10_000.0, 10_000.0, 1_000.0, 1_000.0, 1_000.0])

    returns = adjust_close(close, shares)["A"].pct_change(fill_method=None)

    assert returns.iloc[2] == 0.0


def test_share_issuance_is_not_adjusted():
    """
    유상증자는 주식수만 늘고 주가는 그대로다. 여기에 주식수 비율을 곱해버리면
    없는 수익이 생기므로, 보정에서 제외돼야 한다.
    """
    close = _panel([1000.0, 1000.0, 990.0, 990.0, 990.0])
    shares = _panel([1_000.0, 1_000.0, 1_200.0, 1_200.0, 1_200.0])

    returns = adjust_close(close, shares)["A"].pct_change(fill_method=None)

    # 주가가 주식수 변화의 역수만큼 움직이지 않았으므로 분할이 아니다 -> 원본 유지
    assert returns.iloc[2] == pytest.approx(-0.01)


def test_small_bonus_issue_is_adjusted():
    """
    무상증자 20%도 보정돼야 한다. **이게 실제로 빠져 있던 버그다.**

    주식수가 1.2배가 되면 주가는 1/1.2로 16.7% 빠진다. 주주는 주식이 1.2배가 되어
    손익이 0인데, 예전 판정 조건에 '원본 수익률 30% 초과 급변'이 들어 있어서
    이 정도 크기는 기업행위로 인정되지 않았다. 30%는 국내 가격제한폭에서 가져온
    값이지 기업행위의 성질이 아니다.

    실사 결과 이렇게 놓친 것이 325건(254종목)이었고 313건이 가짜 하락이었다
    (중앙값 -6.8%). experiments/log.md 2026-08-28 참조.
    """
    close = _panel([1200.0, 1200.0, 1000.0, 1000.0, 1000.0])
    shares = _panel([1_000.0, 1_000.0, 1_200.0, 1_200.0, 1_200.0])

    returns = adjust_close(close, shares)["A"].pct_change(fill_method=None)

    assert returns.iloc[2] == pytest.approx(0.0), "무상증자로 손익이 생기면 안 된다"


def test_reverse_split_with_limit_move_on_resumption_is_still_adjusted():
    """
    액면병합은 거래정지를 끼고 이뤄져서 재개일에 상한가를 치는 일이 흔하다.

    주식수 1/5, 주가는 5배가 아니라 6.4배(=5 x 1.28)로 찍힌다. 잔차 28%는 가격제한폭
    (±30%) 안이므로 실제로 있을 수 있는 하루 등락이고, 보정은 여전히 해야 한다.
    보정을 포기하면 **+540%짜리 가짜 수익률**이 그대로 남는다.

    실제 데이터에서 이런 건이 427건이었다. 한때 잔차 상한을 0.10으로 좁혔다가
    이것들이 통째로 빠지는 것을 보고 되돌렸다.
    """
    close = _panel([1000.0, 1000.0, 6400.0, 6400.0, 6400.0])
    shares = _panel([10_000.0, 10_000.0, 2_000.0, 2_000.0, 2_000.0])

    returns = adjust_close(close, shares)["A"].pct_change(fill_method=None)

    assert returns.iloc[2] == pytest.approx(0.28), "병합은 보정하되 진짜 등락은 남는다"


def test_normal_moves_are_untouched():
    """평범한 등락은 주식수가 그대로이므로 아무 영향이 없어야 한다."""
    close = _panel([1000.0, 1100.0, 1050.0, 1080.0, 1020.0])
    shares = _panel([1_000.0] * 5)

    pd.testing.assert_frame_equal(adjust_close(close, shares), close)


def test_impossible_move_becomes_nan():
    """보정으로도 설명 안 되는 물리적 불가능 점프(±30% 제한폭 초과)는 NaN 처리."""
    close = _panel([1000.0, 1000.0, 100.0, 100.0, 100.0])

    masked = mask_impossible_moves(close)

    assert np.isnan(masked["A"].iloc[2])
    assert masked["A"].iloc[0] == 1000.0


def test_split_survives_masking():
    """분할은 보정으로 해소되므로 NaN 처리 단계까지 가면 안 된다."""
    close = _panel([1000.0, 1000.0, 100.0, 100.0, 100.0])
    shares = _panel([1_000.0, 1_000.0, 10_000.0, 10_000.0, 10_000.0])

    result = load_adjusted_close(close, shares)

    assert result["A"].notna().all()


def test_cached_trading_dates_matches_pykrx_calendar():
    """
    패널에서 유도한 거래일이 pykrx 기준 달력과 일치해야 한다.

    2014년 이전은 pykrx(네이버 폴백)가 데이터를 주지 않아 패널 유도 방식을 써야 하는데,
    겹치는 구간에서 두 방식이 어긋나면 그 방식을 신뢰할 수 없다.
    """
    from src.data_loader.krx_openapi import cached_trading_dates
    from src.data_loader.krx_panel import trading_dates

    derived = cached_trading_dates("2018-01-01", "2018-12-31")
    reference = trading_dates("2018-01-01", "2018-12-31")

    assert len(derived) > 200  # 데이터가 없어서 우연히 통과하는 것 방지
    assert derived.equals(reference)
