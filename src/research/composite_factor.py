"""
서로 독립적인 팩터를 결합한다.

IC 진단에서 나온 근거:
- 반전 계열(momentum/disparity)끼리는 순위상관 0.7~0.83으로 사실상 같은 팩터다.
- 반면 변동성은 반전 계열과 상관 -0.02~0.12로 거의 독립이다.

상관이 낮은 두 신호를 합치면, 각각의 노이즈가 부분적으로 상쇄되면서 결합 신호의
신호대잡음비가 개별보다 좋아질 수 있다. 상관이 높은 것끼리 합치면 얻는 게 거의 없다.
그래서 '반전 계열 내부는 평균내서 하나로 합치고, 그것과 변동성을 결합'하는 구조다.

가중치는 동일가중으로 고정한다. 인샘플에서 가중치를 탐색하면 그 결과는 더 이상
검증이 아니라 과적합이 되고, 아웃오브샘플 검증의 의미도 사라진다.
"""
from __future__ import annotations

import pandas as pd


def cross_sectional_rank(panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """
    날짜마다 유니버스 안에서 백분위 순위로 바꾼 뒤 -1~1로 중심화한다.

    표준화(z-score) 대신 순위를 쓰는 이유: 팩터마다 분포 모양이 크게 다르고
    (변동성은 오른쪽으로 긴 꼬리, 모멘텀은 대칭에 가깝다) 이상치도 있는데,
    순위로 바꾸면 분포에 상관없이 같은 척도가 되어 그대로 더할 수 있다.
    IC 자체가 순위상관이므로 척도를 순위로 맞추는 게 일관적이기도 하다.
    """
    masked = panel.where(universe)
    ranks = masked.rank(axis=1, pct=True)
    return (ranks - 0.5) * 2


def combine(
    components: dict[str, pd.DataFrame], universe: pd.DataFrame, signs: dict[str, int]
) -> pd.DataFrame:
    """
    여러 팩터를 동일가중으로 결합한다.

    signs는 각 팩터의 방향이다(+1이면 높을수록 유리, -1이면 낮을수록 유리).
    부호를 곱해 '높을수록 기대수익이 높다'로 방향을 통일한 뒤 평균낸다. 이렇게 하면
    결합 신호의 IC는 양수로 나와야 정상이라, 해석이 헷갈리지 않는다.
    """
    normalized = [
        cross_sectional_rank(panel, universe) * signs[name]
        for name, panel in components.items()
    ]
    stacked = pd.concat(normalized)
    return stacked.groupby(stacked.index).mean()
