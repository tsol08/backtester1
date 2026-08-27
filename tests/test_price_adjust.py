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
