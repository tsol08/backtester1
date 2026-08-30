"""
순 발행(net share issuance) — 기업행위를 뺀 주식 수 증가율.

**주식 수가 늘어나는 이유는 둘이고, 성질이 정반대다.**

  발행    유상증자·전환사채 전환·신주인수권 행사. 회사가 자본을 조달하거나
          기존 주주가 희석된다. 문헌에서 이후 수익률과 음의 관계가 반복 보고된다.
  기업행위 액면분할·무상증자. **주주 지분이 그대로다.** 주식이 k배가 되고 주가가
          1/k배가 될 뿐이라 경제적 사건이 아니다.

둘을 안 나누고 주식수 증가율을 그대로 쓰면 50:1 액면분할이 '4,900% 발행'으로 잡혀
신호가 통째로 오염된다. 삼성전자가 최대 희석 기업이 된다.

`split_adjustment_factor`가 이미 기업행위 몫을 누적 계수로 들고 있으므로, 총증가를
그 계수 증가로 나누면 발행분만 남는다:

    순발행 = (주식수_t / 주식수_t-k) / (보정계수_t / 보정계수_t-k) - 1

이 분리가 이 프로젝트에서 특히 중요하다. **무상증자 종목이 이후 크게 부진하는 것은
확인됐지만, 그 원인은 발행이 아니라 '최근 급등한 소형주'라는 표지였다**
(experiments/log.md 2026-08-28 (10)). 기업행위 몫을 남겨두면 그 반전 효과가
발행 신호로 위장해 들어온다.
"""
from __future__ import annotations

import pandas as pd

from src.data_loader.price_adjust import split_adjustment_factor

TRADING_DAYS_PER_YEAR = 252


def net_issuance(
    close: pd.DataFrame,
    listed_shares: pd.DataFrame,
    lookback: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    (날짜 x 종목) 순 발행 비율. 높을수록 발행을 많이 한 것이다.

    lookback은 거래일 수다. 기본 252일(1년)은 이 저장소가 자산성장을 잰 구간과
    같아서 둘을 나란히 볼 수 있다.
    """
    factor = split_adjustment_factor(close, listed_shares)

    gross = listed_shares / listed_shares.shift(lookback)
    corporate_action = factor / factor.shift(lookback)
    return gross / corporate_action - 1


def corporate_action_growth(
    close: pd.DataFrame,
    listed_shares: pd.DataFrame,
    lookback: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    기업행위(분할·무상증자)로 인한 주식 수 증가율. **반증 확인용이다.**

    순 발행이 진짜 발행 효과라면 이쪽은 효과가 없어야 한다. 둘 다 효과가 있으면
    '주식 수가 늘어난 회사' 효과이지 발행 고유분이 아니다.
    """
    factor = split_adjustment_factor(close, listed_shares)
    return factor / factor.shift(lookback) - 1
