"""
실적발표 후 표류(PEAD) 전략.

가설과 검정 결과는 experiments/log.md에 있다. 요약하면: 어닝 서프라이즈(SUE) 상위
분위가 20일 뒤까지 유니버스 평균을 앞선다. 비용 차감 후 연 5.49% 초과, t 2.15로
사전에 정한 기준(|t| > 1.96)을 넘었다. **확정은 아니다** - 비겹침 관측 50개,
전반부가 약하고, 진짜 아웃오브샘플이 없다.

**여기 있는 값들은 검정 전에 정해진 것이고 성과를 보고 고른 것이 아니다.**
바꾸면 그 순간 다른 가설이 되므로, 바꿀 거라면 사전 등록부터 다시 해야 한다.
이 프로젝트는 이미 40건 넘게 검정해서 우연히 t 3이 나오는 게 정상인 상태다.

신호가 비는 계절이 있다는 점이 이 전략의 성격을 결정한다. SUE는 공시 후 60일만
유효한데, 한국 공시 일정 탓에 2월에는 유효한 SUE를 가진 종목이 거의 없다.
리밸런싱 시점의 58%에서만 분위가 성립한다. **그동안 무엇을 들고 있는가가 이 전략의
성패를 가른다** - 신호 자체보다 그쪽이 크다.

  idle="persist"  (기본) 직전 구성을 그대로 유지한다.
  idle="covered"  후보 집합 동일가중으로 갈아탄다.
  idle="cash"     현금. 분위 분석을 글자 그대로 옮긴 형태.

**2026-08-28 3차 사전 등록: persist.** covered로 대기하던 2차는 비용 차감 후
연 1.93%(t 0.93)로 미달이었다. 회전율을 쪼개보니:

    신호->신호 (분위 교체)  39회  평균 24%  연 1.09회  20%
    신호->유휴             22회  평균 82%  연 2.08회  38%
    유휴->신호             23회  평균 84%  연 2.23회  41%

**회전율의 79%가 55종목 <-> 300종목 전환이고, 그 전환은 신호 노출을 전혀 주지
않는다.** 순수 마찰이다.

틸트(후보 집합을 늘 들고 상위분위만 초과가중)를 먼저 떠올렸지만 답이 아니다.
초과가중 폭을 줄이면 총수익과 회전율이 **같은 비율로** 줄어서 비율이 보존된다.
고쳐야 할 것은 집중도가 아니라 전환 자체다.

유휴 구간에 오래된 분위를 들고 있는 대가는 작다. 표류 창 진단에서 90일 창의
롱온리 초과가 2.3%(t 0.49), 제한 없음이 0.5%(t 0.12)로 사실상 중립이었다 -
오래된 서프라이즈는 돈을 잃게 하지 않고 그냥 아무것도 아니다.

  판정 **|t| > 2.39** (Bonferroni 3건, 같은 가설의 세 번째 구현)
  벤치마크는 2차와 같이 후보 집합 동일가중. 신호 정의·표류 창 60일·5분위·
  20일 격자·유니버스 상위 500위는 전부 동일하다.
"""
from __future__ import annotations

import pandas as pd

from config import settings
from src.data_loader.dart_loader import load_corp_codes, load_fundamentals_bulk
from src.data_loader.panels import Panels
from src.features.earnings_surprise import build_sue_panel
from src.research.quantile_analysis import assign_quantiles
from src.strategy.base import SubsetEqualWeight, periodic_schedule

DART_START_YEAR = 2015  # DART가 그 이전 데이터를 주지 않는다
DRIFT_WINDOW = 60  # 공시 후 며칠까지 신호로 볼 것인가
MIN_CROSS_SECTION = 30  # 이보다 적으면 분위를 나눠도 의미가 없다


class PeadStrategy:
    name = "PEAD (어닝 서프라이즈 상위분위)"

    def __init__(
        self,
        horizon: int = settings.FORWARD_HORIZON,
        n_quantiles: int = settings.N_QUANTILES,
        drift_window: int = DRIFT_WINDOW,
        idle: str = "persist",
    ):
        if idle not in ("persist", "covered", "cash"):
            raise ValueError(f"idle은 persist/covered/cash 중 하나여야 한다: {idle!r}")
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.drift_window = drift_window
        self.idle = idle
        self.panels: Panels | None = None
        self.top_quantile: pd.DataFrame | None = None
        self.covered: pd.DataFrame | None = None  # 그날 유효한 SUE를 가진 종목
        self.coverage: pd.Series | None = None

    def prepare(self, panels: Panels) -> None:
        self.panels = panels

        tickers = [t for t in panels.members if t in set(load_corp_codes()["stock_code"])]
        fundamentals = load_fundamentals_bulk(
            tickers, DART_START_YEAR, panels.dates[-1].year, verbose=False
        )
        sue = build_sue_panel(fundamentals, panels.dates, drift_window=self.drift_window)
        signal = sue.reindex_like(panels.close).where(panels.universe)
        self.coverage = signal.notna().sum(axis=1)

        # 후보 집합은 "표류 창 안에 있는가"가 아니라 "SUE를 계산할 이력이 있는가"로
        # 잡는다. 표류 창으로 잡으면 분기마다 구성원이 통째로 갈려서, 비교 기준
        # 주제에 연 회전율 6.3회에 비용을 자본의 34%나 내게 된다. 둘의 수익률 차이는
        # 연 -0.83%(t -0.97)로 무의미하다.
        history = build_sue_panel(fundamentals, panels.dates, drift_window=None)
        self.covered = history.reindex_like(panels.close).notna() & panels.universe

        quantiles = assign_quantiles(signal, panels.universe, self.n_quantiles, MIN_CROSS_SECTION)
        self.top_quantile = quantiles == self.n_quantiles - 1

    def rebalance_dates(self) -> list[pd.Timestamp]:
        return periodic_schedule(self.panels.dates, self.horizon)

    def benchmark(self) -> SubsetEqualWeight:
        """
        비교 기준은 유니버스 전체가 아니라 **SUE를 가진 종목 동일가중**이다.

        SUE 보유 집단이 유니버스를 연 4.65%(t 2.97) 이기는데, 그건 어닝 서프라이즈가
        아니라 '3년치 연속 분기 이력이 있는 회사'라는 조건이 만드는 것이다. 신규상장
        저조로 설명되는 몫이 1.17%p뿐이고 나머지는 정체를 모른다. 검정한 적 없는 것을
        성과에 얹지 않기 위해 후보 집합 안에서 비교한다(experiments/log.md 2026-08-28).
        """
        return SubsetEqualWeight(self.covered, "SUE 보유 종목 동일가중", self.horizon)

    def signal_available(self, date: pd.Timestamp) -> bool:
        """그 날 분위를 나눌 만큼 유효 SUE가 있었는가."""
        return bool(self.top_quantile.loc[date].any())

    def _last_signalled_on_or_before(self, date: pd.Timestamp) -> pd.Timestamp | None:
        """직전에 분위가 성립했던 리밸런싱일. 전부 과거이므로 look-ahead가 없다."""
        past = [d for d in self.rebalance_dates() if d <= date and self.signal_available(d)]
        return past[-1] if past else None

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        tradeable = self.panels.tradeable.loc[date]
        picks = self.top_quantile.loc[date] & tradeable

        if not picks.any():
            if self.idle == "cash":
                return pd.Series(dtype=float)
            if self.idle == "persist":
                # 직전 구성을 그대로 유지한다. 상장폐지·거래정지된 종목만 빠진다.
                last = self._last_signalled_on_or_before(date)
                if last is not None:
                    held = self.top_quantile.loc[last] & tradeable
                    if held.any():
                        picks = held
            if not picks.any():
                # 아직 신호가 한 번도 없었으면 후보 집합으로 시작한다
                picks = self.covered.loc[date] & tradeable

        members = picks.index[picks]
        if len(members) == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(members), index=members)

    def latest_signal_date(self) -> pd.Timestamp | None:
        """분위가 성립한 마지막 날. 신호가 조용히 과거에서 멈췄는지 확인하는 용도."""
        available = self.top_quantile.any(axis=1)
        return available[available].index[-1] if available.any() else None
