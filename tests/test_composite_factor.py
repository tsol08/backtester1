"""결합 팩터가 손계산과 맞는지 검증한다."""
from __future__ import annotations

import pandas as pd
import pytest

from src.research.composite_factor import combine, cross_sectional_rank


@pytest.fixture
def universe() -> pd.DataFrame:
    return pd.DataFrame(
        True, index=pd.to_datetime(["2024-01-02", "2024-01-03"]), columns=["A", "B", "C", "D"]
    )


def test_rank_is_centered_and_ordered(universe):
    """백분위 순위가 -1~1로 중심화되고, 원래 크기 순서를 보존해야 한다."""
    panel = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0], [40.0, 30.0, 20.0, 10.0]],
        index=universe.index,
        columns=universe.columns,
    )

    result = cross_sectional_rank(panel, universe)

    # 4종목이면 백분위는 0.25/0.5/0.75/1.0 -> 중심화하면 -0.5/0/0.5/1.0
    expected_first = pd.Series([-0.5, 0.0, 0.5, 1.0], index=universe.columns, name=universe.index[0])
    pd.testing.assert_series_equal(result.iloc[0], expected_first)

    # 순서가 뒤집힌 날은 결과도 뒤집혀야 한다
    assert result.iloc[1].tolist() == [1.0, 0.5, 0.0, -0.5]


def test_universe_members_only_are_ranked():
    """유니버스 밖 종목은 순위 계산에서 빠져야 한다 (편입 종목끼리만 비교)."""
    dates = pd.to_datetime(["2024-01-02"])
    panel = pd.DataFrame([[10.0, 20.0, 30.0, 40.0]], index=dates, columns=["A", "B", "C", "D"])
    universe = pd.DataFrame([[True, True, False, False]], index=dates, columns=["A", "B", "C", "D"])

    result = cross_sectional_rank(panel, universe)

    assert result.loc[dates[0], ["C", "D"]].isna().all()
    # 남은 2종목만으로 다시 순위 -> 0.5/1.0 -> 중심화 0.0/1.0
    assert result.loc[dates[0], "A"] == 0.0
    assert result.loc[dates[0], "B"] == 1.0


def test_sign_flips_direction(universe):
    """sign=-1이면 '낮을수록 유리'가 되어 부호가 뒤집혀야 한다."""
    panel = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]],
        index=universe.index,
        columns=universe.columns,
    )

    positive = combine({"f": panel}, universe, signs={"f": 1})
    negative = combine({"f": panel}, universe, signs={"f": -1})

    pd.testing.assert_frame_equal(negative, -positive)


def test_combine_averages_components(universe):
    """두 팩터 결합은 각각을 순위화한 뒤의 평균이어야 한다."""
    rising = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0]] * 2, index=universe.index, columns=universe.columns
    )
    falling = pd.DataFrame(
        [[40.0, 30.0, 20.0, 10.0]] * 2, index=universe.index, columns=universe.columns
    )

    result = combine({"a": rising, "b": falling}, universe, signs={"a": 1, "b": 1})

    # 정확히 반대 순서인 두 팩터를 같은 방향으로 더하면 서로 상쇄된다.
    # 백분위 -0.5/0/0.5/1.0 과 1.0/0.5/0/-0.5 의 평균 = 0.25 전부
    assert (result.iloc[0] == 0.25).all()


def test_combine_preserves_index_and_columns(universe):
    """결합 결과의 인덱스/컬럼이 입력과 같아야 한다 (groupby 후 순서 뒤바뀜 방지)."""
    panel = pd.DataFrame(
        [[10.0, 20.0, 30.0, 40.0]] * 2, index=universe.index, columns=universe.columns
    )

    result = combine({"a": panel, "b": panel}, universe, signs={"a": 1, "b": -1})

    assert result.index.equals(universe.index)
    assert result.columns.equals(universe.columns)
