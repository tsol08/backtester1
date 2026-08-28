"""
밸류 + PEAD 합성 스코어 전략.

**사전 등록: experiments/plan_value_pead_composite.md (결과를 보기 전에 커밋됨).**
아래 값은 전부 그 문서에 먼저 적힌 것이고 성과를 보고 고른 것이 하나도 없다.
바꾸면 그 순간 다른 가설이 되므로, 바꿀 거라면 사전 등록부터 다시 해야 한다.

왜 이 둘인가. 이 저장소에서 롱온리 초과가 양수로 측정된 신호는 정확히 둘이다:

    밸류(B/M)   +4.77%/년 (t 1.33)  약하지만 **항상 켜져 있다**
    PEAD(SUE)   +9.55%/년 (t 2.69)  강하지만 **리밸런싱의 58%에만 있다**

그리고 PEAD가 죽은 원인은 신호 강도가 아니라 공백이었다. 회전율의 79%가
'신호 <-> 유휴' 전환에서 나왔고 그 전환은 신호 노출을 전혀 주지 않았다.

**가설: 포트폴리오를 갈아타는 대신 스코어 단계에서 합치면 항상 켜져 있고
전환이 사라진다.**

스코어:

    z_value = 유니버스 내 book_to_market 의 횡단면 백분위 순위 [0, 1]
    z_pead  = 유니버스 내 SUE 의 횡단면 백분위 순위 [0, 1]
              **유효 SUE가 없으면 0.5 (중립)**
    score   = 0.5 * z_value + 0.5 * z_pead

설계에서 명시해둘 것 셋:

1. **결측을 0.5로 채우는 것이 이 가설의 장치다.** 최근 서프라이즈가 없는 종목은
   데이터가 빠진 게 아니라 'PEAD 관점에서 의견 없음'이다. 이것 때문에 공백
   구간에 스코어가 자동으로 밸류 단독으로 수렴하고 포트폴리오가 통째로 갈리지 않는다.

2. **가중치 0.5/0.5 고정, 최적화하지 않는다.** 동일가중은 선택이 아니라 자유도를
   0으로 만드는 장치다.

3. **이 구성은 밸류 쪽으로 기울어 있고, 알고 시작한다.** z_pead는 대부분의 종목이
   0.5에 몰려 횡단면 분산이 z_value보다 작다. 희소한 정보에 비례해서 작은 가중이
   실리는 것은 올바른 처리다. 분산을 맞춰 PEAD를 증폭하면 1~3차 구현이 실패한
   집중도·회전율 문제가 그대로 돌아온다.

`components`는 두 가지만 허용한다. ("value", "pead")가 가설이고, ("value",)는
사전 등록된 반증 확인 FC1의 기준선이다 - **합성이 밸류 단독을 이기지 못하면
PEAD 성분이 아무것도 더하지 않은 것이므로 t가 얼마든 기각**이다. 둘은 같은 후보
집합과 같은 벤치마크를 쓰므로 그대로 비교된다.

## 결과: 기각 (2026-08-28)

    주 판정  |t| 1.14 < 2.50                    미달
    FC1      합성 +3.83% < 밸류 단독 +4.73%      실패  <- 여기서 갈렸다
    FC2      하위분위 -7.65% (음수)              통과
    FC3      회전율 3.38 > 상한 3.0              실패

**PEAD 성분이 한 일은 회전율 3배였다.**

              밸류 단독   합성
    연회전율     1.08     3.38   (3.1배)
    총비용       4.8%    15.0%   (3.1배)
    초과수익    +4.73%   +3.83%   (0.81배)

메커니즘 주장은 절반만 맞았다. 포트폴리오가 55↔300종목으로 통째로 갈리는 일은
없어졌지만(PEAD 2차 회전율 5.58 → 3.38), SUE 순위가 분기마다 바뀌면서 상위분위
구성원이 계속 교체됐다. 비용을 3배 내고 수익을 덜 받았다.

**이것으로 PEAD를 닫았다. 5차는 없다** - 등록 문서에 그렇게 적었다.
이 파일을 남겨두는 이유는 결론을 만든 계산이 무엇이었는지 남기기 위해서다.
"""
from __future__ import annotations

import pandas as pd

from config import settings
from src.data_loader.dart_loader import load_corp_codes, load_fundamentals_bulk
from src.data_loader.panels import Panels
from src.features.earnings_surprise import build_sue_panel
from src.features.fundamental_factors import build_fundamental_panels
from src.research.quantile_analysis import assign_quantiles
from src.strategy.base import SubsetEqualWeight, periodic_schedule

DART_START_YEAR = 2015  # DART가 그 이전 데이터를 주지 않는다
DRIFT_WINDOW = 60  # 공시 후 며칠까지 SUE를 신호로 볼 것인가 (PEAD 등록값)
MIN_CROSS_SECTION = 30  # 이보다 적으면 분위를 나눠도 의미가 없다
NEUTRAL = 0.5  # 유효 SUE가 없는 종목의 z_pead

ALLOWED_COMPONENTS = {
    ("value", "pead"): "밸류+PEAD 합성",
    ("value",): "밸류 단독 (반증 확인 FC1의 기준선)",
}


def composite_score(
    book_to_market: pd.DataFrame,
    sue: pd.DataFrame | None,
    covered: pd.DataFrame,
) -> pd.DataFrame:
    """
    합성 스코어. **결측 SUE를 중립값으로 채우는 것이 이 가설의 장치라 따로 뺐다.**

    covered(= 후보 집합) 안에서만 순위를 매긴다. sue가 None이면 밸류 단독이다.

    유효 SUE가 없는 후보에 0.5를 주면 그 종목은 PEAD 관점에서 '의견 없음'이 되고,
    아무도 SUE가 없는 날은 스코어가 밸류 단독과 **정확히 같아진다.** 그 덕에 공백
    구간에 포트폴리오가 통째로 갈리지 않는다 - PEAD 1~3차를 죽인 것이 그 전환이었다.
    """
    z_value = book_to_market.where(covered).rank(axis=1, pct=True)
    if sue is None:
        return z_value.where(covered)

    # 알고 있어야 할 성질: rank(pct=True)의 범위는 [1/n, 1]이라 0에 닿지 않는다.
    # 그래서 SUE 보유 종목이 n개일 때 중립값 0.5는 그 분포의 정확한 중앙값이
    # 아니라 1/(2n)만큼 아래에 있다. 신호일 실측 n이 224라 편차가 0.002 수준이고
    # 판정에 영향이 없지만, n이 아주 작은 날에는 '나쁜 서프라이즈'가 '무소식'보다
    # 위로 갈 수 있다.
    z_pead = sue.where(covered).rank(axis=1, pct=True)
    z_pead = z_pead.where(~covered | z_pead.notna(), NEUTRAL)
    return (0.5 * z_value + 0.5 * z_pead).where(covered)


class CompositeStrategy:
    def __init__(
        self,
        components: tuple[str, ...] = ("value", "pead"),
        horizon: int = settings.FORWARD_HORIZON,
        n_quantiles: int = settings.N_QUANTILES,
        drift_window: int = DRIFT_WINDOW,
    ):
        if components not in ALLOWED_COMPONENTS:
            raise ValueError(
                f"사전 등록된 구성이 아니다: {components!r}."
                f" 허용: {sorted(ALLOWED_COMPONENTS)}"
            )
        self.components = components
        self.name = ALLOWED_COMPONENTS[components]
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.drift_window = drift_window
        self.panels: Panels | None = None
        self.score: pd.DataFrame | None = None
        self.top_quantile: pd.DataFrame | None = None
        self.covered: pd.DataFrame | None = None  # 후보 집합 = B/M이 계산되는 종목
        self.pead_coverage: pd.Series | None = None

    def prepare(self, panels: Panels) -> None:
        self.panels = panels

        tickers = [t for t in panels.members if t in set(load_corp_codes()["stock_code"])]
        fundamentals = load_fundamentals_bulk(
            tickers, DART_START_YEAR, panels.dates[-1].year, verbose=False
        )

        book_to_market = build_fundamental_panels(
            fundamentals, panels.dates, panels.market_cap
        )["book_to_market"].reindex_like(panels.close)

        # 후보 집합은 이 전략이 실제로 고를 수 있는 종목이다. 스코어가 B/M을
        # 요구하므로 B/M이 없는 종목은 애초에 후보가 아니다. 벤치마크도 이것이다
        # - 전략이 고르는 집합과 벤치마크가 다르면 '후보에 든 것만으로 생기는
        # 차이'가 성과로 잡힌다(CLAUDE.md 규율 5).
        self.covered = book_to_market.notna() & panels.universe

        sue = None
        if "pead" in self.components:
            sue = build_sue_panel(
                fundamentals, panels.dates, drift_window=self.drift_window
            ).reindex_like(panels.close)
            self.pead_coverage = sue.where(self.covered).notna().sum(axis=1)

        self.score = composite_score(book_to_market, sue, self.covered)
        quantiles = assign_quantiles(
            self.score, panels.universe, self.n_quantiles, MIN_CROSS_SECTION
        )
        self.top_quantile = quantiles == self.n_quantiles - 1
        self.quantiles = quantiles

    def rebalance_dates(self) -> list[pd.Timestamp]:
        return periodic_schedule(self.panels.dates, self.horizon)

    def benchmark(self) -> SubsetEqualWeight:
        """비교 기준은 유니버스 전체가 아니라 **B/M이 계산되는 종목 동일가중**이다."""
        return SubsetEqualWeight(self.covered, "후보 집합(B/M 보유) 동일가중", self.horizon)

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        tradeable = self.panels.tradeable.loc[date]
        picks = self.top_quantile.loc[date] & tradeable

        if not picks.any():
            # 밸류는 항상 켜져 있어 여기 오지 않는 것이 정상이다. 그래도 분위를
            # 나눌 종목이 30개가 안 되는 날은 있을 수 있으므로, 그 경우 후보
            # 집합을 든다. 전량 청산 후 재매수하면 커버리지 아티팩트만으로
            # 왕복 비용과 거래세를 문다.
            picks = self.covered.loc[date] & tradeable
        members = picks.index[picks]
        if len(members) == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(members), index=members)
