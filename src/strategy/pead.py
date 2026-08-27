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
유효한데, 한국 공시 일정 탓에 2월에는 유니버스 200종목 중 1종목만 유효한 SUE를
가진다. 리밸런싱 시점의 47%에서만 분위가 성립한다. 그동안 무엇을 들고 있을지가
`hold_universe_when_idle`이고, 두 선택지 모두 검정 전에 등록했다:

  True  (기본) 유니버스 동일가중으로 대기. 실제로 굴린다면 이쪽이다.
  False        현금으로 대기. 분위 분석을 글자 그대로 옮긴 형태.
"""
from __future__ import annotations

import pandas as pd

from config import settings
from src.data_loader.dart_loader import load_corp_codes, load_fundamentals_bulk
from src.data_loader.panels import Panels
from src.features.earnings_surprise import build_sue_panel
from src.research.quantile_analysis import assign_quantiles
from src.strategy.base import periodic_schedule

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
        hold_universe_when_idle: bool = True,
    ):
        self.horizon = horizon
        self.n_quantiles = n_quantiles
        self.drift_window = drift_window
        self.hold_universe_when_idle = hold_universe_when_idle
        self.panels: Panels | None = None
        self.top_quantile: pd.DataFrame | None = None
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

        quantiles = assign_quantiles(signal, panels.universe, self.n_quantiles, MIN_CROSS_SECTION)
        self.top_quantile = quantiles == self.n_quantiles - 1

    def rebalance_dates(self) -> list[pd.Timestamp]:
        return periodic_schedule(self.panels.dates, self.horizon)

    def signal_available(self, date: pd.Timestamp) -> bool:
        """그 날 분위를 나눌 만큼 유효 SUE가 있었는가."""
        return bool(self.top_quantile.loc[date].any())

    def target_weights(self, date: pd.Timestamp) -> pd.Series:
        tradeable = self.panels.tradeable.loc[date]
        picks = self.top_quantile.loc[date] & tradeable

        if not picks.any():
            if not self.hold_universe_when_idle:
                return pd.Series(dtype=float)
            picks = self.panels.universe.loc[date] & tradeable

        members = picks.index[picks]
        if len(members) == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(members), index=members)

    def latest_signal_date(self) -> pd.Timestamp | None:
        """분위가 성립한 마지막 날. 신호가 조용히 과거에서 멈췄는지 확인하는 용도."""
        available = self.top_quantile.any(axis=1)
        return available[available].index[-1] if available.any() else None
