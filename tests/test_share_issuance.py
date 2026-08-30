"""
순 발행 신호의 불변식.

**핵심은 하나다: 액면분할과 무상증자는 발행이 아니다.** 주주 지분이 그대로인데
주식 수만 k배가 되는 것이라, 안 걸러내면 50:1 분할이 '4,900% 발행'으로 잡혀
삼성전자가 최대 희석 기업이 된다. 신호가 통째로 뒤집힌다.

이 분리가 이 저장소에서 특히 중요하다. 무상증자 종목이 이후 크게 부진하는 것은
확인됐지만 원인은 발행이 아니라 '최근 급등한 소형주'라는 표지였다
(experiments/log.md 2026-08-28 (10)). 기업행위 몫을 남겨두면 그 반전 효과가
발행 신호로 위장해 들어온다.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.share_issuance import corporate_action_growth, net_issuance

DATES = pd.date_range("2020-01-01", periods=6, freq="D")
LOOKBACK = 3


def _panel(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"A": values}, index=DATES)


def test_split_is_not_counted_as_issuance():
    """10:1 액면분할: 주식수 10배, 주가 1/10. 순 발행은 0이어야 한다."""
    close = _panel([1000.0, 1000.0, 1000.0, 100.0, 100.0, 100.0])
    shares = _panel([1_000.0, 1_000.0, 1_000.0, 10_000.0, 10_000.0, 10_000.0])

    issuance = net_issuance(close, shares, LOOKBACK)

    assert issuance["A"].iloc[3] == pytest.approx(0.0, abs=1e-9)
    # 반대로 기업행위 쪽에는 잡혀야 한다
    assert corporate_action_growth(close, shares, LOOKBACK)["A"].iloc[3] == pytest.approx(9.0)


def test_bonus_issue_is_not_counted_as_issuance():
    """
    무상증자 20%: 주식수 1.2배, 주가 1/1.2. 주주 지분이 그대로라 발행이 아니다.

    작은 무상증자는 예전에 보정에서 아예 빠져 있었다(미보정 325건). 그 시절
    코드였다면 이 값이 +20% 발행으로 잡혔을 것이다.
    """
    close = _panel([1200.0, 1200.0, 1200.0, 1000.0, 1000.0, 1000.0])
    shares = _panel([1_000.0, 1_000.0, 1_000.0, 1_200.0, 1_200.0, 1_200.0])

    assert net_issuance(close, shares, LOOKBACK)["A"].iloc[3] == pytest.approx(0.0, abs=1e-9)


def test_cash_issuance_is_counted():
    """유상증자: 주식수는 늘고 주가는 그대로다. 이건 진짜 발행이다."""
    close = _panel([1000.0, 1000.0, 1000.0, 990.0, 990.0, 990.0])
    shares = _panel([1_000.0, 1_000.0, 1_000.0, 1_200.0, 1_200.0, 1_200.0])

    issuance = net_issuance(close, shares, LOOKBACK)

    assert issuance["A"].iloc[3] == pytest.approx(0.20)
    # 기업행위 쪽에는 안 잡혀야 한다
    assert corporate_action_growth(close, shares, LOOKBACK)["A"].iloc[3] == pytest.approx(0.0)


def test_buyback_shows_as_negative_issuance():
    """자사주 소각으로 주식수가 줄면 순 발행은 음수다."""
    close = _panel([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    shares = _panel([1_000.0, 1_000.0, 1_000.0, 900.0, 900.0, 900.0])

    assert net_issuance(close, shares, LOOKBACK)["A"].iloc[3] == pytest.approx(-0.10)


def test_no_change_is_zero():
    """아무 일도 없으면 0이다. 잡음이 끼면 분위가 흔들린다."""
    close = _panel([1000.0, 1010.0, 990.0, 1005.0, 1000.0, 1020.0])
    shares = _panel([1_000.0] * 6)

    issuance = net_issuance(close, shares, LOOKBACK).dropna()

    assert (issuance["A"].abs() < 1e-12).all()
