"""
분석과 백테스트가 함께 쓰는 시장 데이터 묶음.

패널을 낱개로 들고 다니면 두 가지가 반복해서 어긋난다.

1) **보정 종가와 원본 거래량을 섞어 쓴다.** 조정 종가는 분할 배율만큼 스케일이
   달라서, 거래대금을 종가x거래량으로 근사하면 분할 이력이 있는 종목에서 수십 배
   부풀려진다(삼성전자 최대 52배). 실제 거래대금 컬럼이 있는데도 그랬다.
2) **비용 모델에 필요한 컬럼을 빠뜨린다.** 고가/저가가 없으면 변동성이 상수로
   대체되고, 그 사실이 아무 데도 드러나지 않는다.

그래서 한 덩어리로 싣고 다니고, 종목별 프레임도 여기서 만들어 넘긴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import settings
from src.data_loader.krx_openapi import build_panels, cached_trading_dates
from src.data_loader.price_adjust import load_adjusted_close
from src.data_loader.universe import market_cap_universe_mask

RAW_COLUMNS = ["close", "listed_shares", "volume", "trading_value", "high", "low", "market_cap"]

# 비용 모델이 종목별로 받아야 하는 컬럼. close는 보정본, 나머지는 원본이다.
COST_COLUMNS = ["close", "volume", "trading_value", "high", "low"]


@dataclass
class Panels:
    close: pd.DataFrame  # 액면분할 보정됨
    volume: pd.DataFrame
    trading_value: pd.DataFrame
    high: pd.DataFrame  # 원본. 고가/저가 비율은 분할 보정과 무관하다
    low: pd.DataFrame
    market_cap: pd.DataFrame
    universe: pd.DataFrame  # 시점별 시가총액 상위 K종목 마스크

    @classmethod
    def load(
        cls,
        start: str = settings.DATA_START,
        end: str | None = None,
        top_k: int = settings.UNIVERSE_TOP_K,
    ) -> "Panels":
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        dates = cached_trading_dates(start, end)
        if len(dates) == 0:
            raise FileNotFoundError(
                f"{start} ~ {end} 구간에 수집된 가격 데이터가 없다."
                " scripts/collect_openapi_panel.py를 먼저 실행할 것."
            )

        raw = build_panels(RAW_COLUMNS, dates)
        return cls(
            close=load_adjusted_close(raw["close"], raw["listed_shares"]),
            volume=raw["volume"],
            trading_value=raw["trading_value"],
            high=raw["high"],
            low=raw["low"],
            market_cap=raw["market_cap"],
            universe=market_cap_universe_mask(raw["market_cap"], top_k=top_k),
        )

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def members(self) -> list[str]:
        """한 번이라도 유니버스에 편입된 종목. 고정 티커 목록이 아니라 시점별 결과다."""
        return sorted(self.universe.columns[self.universe.any(axis=0)])

    @property
    def tradeable(self) -> pd.DataFrame:
        """그날 종가가 있는 종목. 거래정지·상장폐지 구간을 걸러낸다."""
        return self.close.notna()

    def price_frames(self, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """비용 모델이 쓰는 종목별 프레임. 필요한 컬럼을 전부 담아서 넘긴다."""
        panels = {name: getattr(self, name) for name in COST_COLUMNS}
        return {
            ticker: pd.DataFrame({name: panel[ticker] for name, panel in panels.items()})
            for ticker in (tickers if tickers is not None else self.members)
        }

    def slice(self, start: str | pd.Timestamp, end: str | pd.Timestamp | None = None) -> "Panels":
        """평가 구간만 잘라낸다. 워밍업 구간을 포함해 실은 뒤 자를 때 쓴다."""
        window = self.dates >= pd.Timestamp(start)
        if end is not None:
            window &= self.dates <= pd.Timestamp(end)
        dates = self.dates[window]
        return Panels(
            **{
                name: getattr(self, name).loc[dates]
                for name in ("close", "volume", "trading_value", "high", "low", "market_cap", "universe")
            }
        )
