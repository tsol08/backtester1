"""
합성 스코어의 불변식.

이 가설(experiments/plan_value_pead_composite.md)은 기각됐지만 테스트는 남긴다.
**중립값 채우기가 이 검정의 결론을 만든 장치**라서, 이게 조용히 깨지면 로그에
적힌 판정이 무슨 계산에서 나왔는지 알 수 없게 된다.

특히 첫 번째가 핵심이다: 아무도 SUE가 없는 날 합성이 밸류 단독과 **정확히** 같지
않으면, 'PEAD 성분이 공백에서 아무 일도 하지 않는다'는 전제가 깨지고 FC1 비교
자체가 성립하지 않는다.
"""
import numpy as np
import pandas as pd
import pytest

from src.strategy.composite import NEUTRAL, ALLOWED_COMPONENTS, CompositeStrategy, composite_score


@pytest.fixture
def frame():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    tickers = ["A", "B", "C", "D"]
    return dates, tickers


def test_no_sue_anywhere_makes_composite_identical_to_value_only(frame):
    """SUE가 하나도 없는 날, 합성 순위는 밸류 단독 순위와 정확히 같아야 한다."""
    dates, tickers = frame
    bm = pd.DataFrame([[0.1, 0.4, 0.2, 0.3]] * 3, index=dates, columns=tickers)
    sue = pd.DataFrame(np.nan, index=dates, columns=tickers)
    covered = pd.DataFrame(True, index=dates, columns=tickers)

    combined = composite_score(bm, sue, covered)
    value_only = composite_score(bm, None, covered)

    # 값 자체는 다를 수 있어도(0.5*z + 0.5*0.5) **순위**가 같아야 한다.
    pd.testing.assert_frame_equal(
        combined.rank(axis=1), value_only.rank(axis=1), check_dtype=False
    )


def test_stock_without_sue_is_neutral_not_missing(frame):
    """유효 SUE가 없는 후보는 결측이 아니라 중립(0.5)이다. 빠지면 후보에서 사라진다."""
    dates, tickers = frame
    bm = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]] * 3, index=dates, columns=tickers)
    sue = pd.DataFrame(np.nan, index=dates, columns=tickers)
    sue["A"] = 5.0  # A만 서프라이즈가 있다
    covered = pd.DataFrame(True, index=dates, columns=tickers)

    score = composite_score(bm, sue, covered)

    assert score.notna().all().all(), "SUE가 없다고 후보에서 탈락하면 안 된다"
    # B는 z_value 0.5, z_pead 중립 0.5 -> 0.5
    assert score.loc[dates[0], "B"] == pytest.approx(0.5 * 0.5 + 0.5 * NEUTRAL)


def test_score_outside_candidate_set_is_missing(frame):
    """후보 집합(B/M 보유) 밖은 스코어가 없어야 한다. 벤치마크와 같은 집합이어야 한다."""
    dates, tickers = frame
    bm = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]] * 3, index=dates, columns=tickers)
    covered = pd.DataFrame(True, index=dates, columns=tickers)
    covered["D"] = False

    score = composite_score(bm, None, covered)

    assert score["D"].isna().all()
    assert score[["A", "B", "C"]].notna().all().all()


def test_good_surprise_outranks_no_surprise_at_equal_value():
    """
    밸류가 같으면 서프라이즈가 좋은 쪽이 위로 가야 한다. 부호가 뒤집히면 신호가 아니다.

    종목을 넉넉히 둔다. rank(pct=True)의 범위가 [1/n, 1]이라 n이 아주 작으면 최하위
    순위조차 중립값 0.5 위로 올라와 이 불변식이 성립하지 않는다(n=2면 0.5로 동률).
    실제 신호일의 SUE 보유 종목은 224개라 그 영역이 아니다.
    """
    dates = pd.date_range("2020-01-01", periods=1, freq="D")
    tickers = [f"T{i:02d}" for i in range(20)]
    bm = pd.DataFrame(0.2, index=dates, columns=tickers)
    covered = pd.DataFrame(True, index=dates, columns=tickers)

    sue = pd.DataFrame(np.nan, index=dates, columns=tickers)
    sue.loc[:, tickers[:10]] = np.linspace(-5.0, 5.0, 10)  # 절반만 서프라이즈가 있다

    score = composite_score(bm, sue, covered).loc[dates[0]]
    best, worst, silent = tickers[9], tickers[0], tickers[15]

    assert score[best] > score[silent] > score[worst], "좋은 것 > 무소식 > 나쁜 것"


def test_unregistered_component_combination_is_refused():
    """사전 등록된 두 구성만 허용한다. 조합을 늘리는 순간 변형 스캔이 된다."""
    for components in (("pead",), ("value", "pead", "lowvol"), ()):
        with pytest.raises(ValueError, match="사전 등록"):
            CompositeStrategy(components=components)

    for components in ALLOWED_COMPONENTS:
        CompositeStrategy(components=components)  # 등록된 것은 통과해야 한다
