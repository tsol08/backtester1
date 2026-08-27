"""
전략 인터페이스.

**도구와 전략을 분리하는 것이 요점이다.** 이 프로젝트는 지금까지 다섯 개 가설 중
넷을 기각했다. 남은 하나도 언제든 무너질 수 있다. 전략이 죽어도 백테스터는 남아야
하고, 새 가설을 꽂을 때 배선을 다시 짜지 않아야 한다.

전략이 하는 일은 하나다: **어느 날 무엇을 얼마나 들고 있을지 정한다.** 체결 시점,
비용, 성과 계산은 전부 엔진 몫이다. 특히 t+1 체결은 전략이 관여할 수 없게 두었다 -
look-ahead는 이 프로젝트에서 가장 조용하게 발생하는 오류다.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from src.data_loader.panels import Panels


# 리밸런싱 격자의 기준점. PEAD 검정이 시작된 날이다.
REBALANCE_ANCHOR = pd.Timestamp("2018-01-02")


def periodic_schedule(
    dates: pd.DatetimeIndex, horizon: int, anchor: pd.Timestamp = REBALANCE_ANCHOR
) -> list[pd.Timestamp]:
    """
    앵커일 이후 horizon 거래일 간격의 리밸런싱 격자.

    `dates[::horizon]`로 잡으면 **패널을 어디서부터 실었느냐에 따라 격자가 통째로
    밀린다.** 백테스트는 2018년부터 세고 실제 운용은 2010년부터 세면, 같은 전략이
    서로 다른 날을 리밸런싱일이라고 말하게 된다(실제로 8월 25일과 7월 31일로
    엇갈렸다). 기준점을 데이터가 아니라 달력에 고정한다.
    """
    return list(dates[int(dates.searchsorted(anchor)) :: horizon])


@runtime_checkable
class Strategy(Protocol):
    """
    구현 순서: prepare()로 데이터를 받고, rebalance_dates()로 언제 갈아탈지 알리고,
    target_weights(date)로 그 날 목표 비중을 돌려준다.

    target_weights는 리밸런싱일이 아닌 날짜에도 답할 수 있어야 한다. 백테스트는
    격자 위에서만 부르지만, 실제 운용에서는 '오늘' 무엇을 들고 있어야 하는지
    물어보게 된다.
    """

    name: str

    def prepare(self, panels: Panels) -> None:
        """시장 데이터를 받아 신호 계산에 필요한 준비를 한다."""

    def rebalance_dates(self) -> list[pd.Timestamp]:
        """갈아탈 날짜들. 비어 있지 않은 목표를 못 내는 날도 포함한다(그날은 청산)."""

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        """그 날의 목표 비중 (종목 -> 비중). 합이 1을 넘지 않는다. 비면 현금."""


class EqualWeightUniverse:
    """
    비교 기준: 시점별 시가총액 상위 K종목 동일가중.

    롱온리 투자자가 신호를 쓰지 않을 때 실제로 할 수 있는 일이라, 가장 정직한
    비교 대상이다. 상하위 분위 스프레드는 공매도를 해야 얻는 값이라 쓰지 않는다.
    """

    name = "유니버스 동일가중"

    def __init__(self, horizon: int = 20):
        self.horizon = horizon
        self.panels: Panels | None = None

    def prepare(self, panels: Panels) -> None:
        self.panels = panels

    def rebalance_dates(self) -> list[pd.Timestamp]:
        return periodic_schedule(self.panels.dates, self.horizon)

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        pool = self.panels.universe.loc[date] & self.panels.tradeable.loc[date]
        members = pool.index[pool]
        if len(members) == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(members), index=members)


class SubsetEqualWeight:
    """
    유니버스의 **일부 집합**을 동일가중 보유하는 비교 기준.

    왜 필요한가: 전략이 고르는 후보 집합 자체가 유니버스와 다르면, 유니버스 전체를
    벤치마크로 쓸 때 '후보에 들어온 것만으로 생기는 차이'가 전략 성과로 잡힌다.

    PEAD가 그렇다. SUE를 가진 종목(3년치 연속 분기 이력이 있는 회사)이 유니버스
    평균을 연 4.65%(t 2.97) 이기는데, 그중 신규상장 저조로 설명되는 것은 1.17%p뿐이고
    나머지는 정체를 모른다. 그걸 PEAD 성과에 얹으면 우리가 검정한 적 없는 것을
    성과로 파는 셈이다. 그래서 **후보 집합 안에서** 비교한다.
    """

    def __init__(self, mask: pd.DataFrame, name: str, horizon: int = 20):
        self.mask = mask
        self.name = name
        self.horizon = horizon
        self.panels: Panels | None = None

    def prepare(self, panels: Panels) -> None:
        self.panels = panels

    def rebalance_dates(self) -> list[pd.Timestamp]:
        return periodic_schedule(self.panels.dates, self.horizon)

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        row = self.mask.loc[date] & self.panels.tradeable.loc[date]
        members = row.index[row]
        if len(members) == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(members), index=members)


def build_weight_panel(strategy: Strategy, panels: Panels) -> pd.DataFrame:
    """
    전략을 (날짜 x 종목) 목표비중 패널로 펼친다.

    각 리밸런싱일의 목표를 **다음 리밸런싱일 직전까지** 유지한다. 보유 기간을 정하는
    것이 전략이 답을 낸 날짜가 아니라 격자라는 점이 중요하다 - 신호가 있는 날만
    이어붙이면 "다음 신호가 뜰 때까지" 들고 있는 전혀 다른 전략이 된다.
    """
    schedule = strategy.rebalance_dates()
    dates = panels.dates
    weights = pd.DataFrame(0.0, index=dates, columns=panels.close.columns)

    for i, start in enumerate(schedule):
        target = strategy.target_weights(start)
        if target.empty:
            continue
        window = dates >= start
        if i + 1 < len(schedule):
            window &= dates < schedule[i + 1]
        weights.loc[window, target.index] = target.values

    return weights
