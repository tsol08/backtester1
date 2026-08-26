"""
펀더멘털 팩터(DART) vs 가격 파생 팩터의 IC 비교.

"가격 하나만 보는 것보다 재무정보를 섞으면 정말 나은가?"를 전략을 만들기 전에
데이터로 먼저 확인한다. 인샘플 기간만 사용.

펀더멘털 팩터는 공시일(available_date) 이후에만 사용 가능하도록 처리돼 있어
look-ahead bias가 없다 (tests/test_fundamental_lookahead.py 참고).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import phase1
from src.data_loader.dart_loader import load_fundamentals
from src.data_loader.krx_loader import load_ohlcv
from src.features.fundamental import FACTOR_COLUMNS, build_fundamental_factors, to_daily_factors
from src.features.technical import build_features
from src.research.ic_analysis import (
    effective_sample_note,
    forward_return,
    non_overlapping_ic,
    summarize_ic,
)

PRICE_FACTORS = ["disparity_20", "momentum_60", "volatility_20", "volume_ratio_20"]
HORIZONS = [20, 60]


def main() -> None:
    price_factor_panels: dict[str, dict[str, pd.Series]] = {f: {} for f in PRICE_FACTORS}
    fundamental_panels: dict[str, dict[str, pd.Series]] = {f: {} for f in FACTOR_COLUMNS}
    close_by_ticker: dict[str, pd.Series] = {}

    for ticker, name in phase1.PORTFOLIO_TICKERS.items():
        print(f"준비 중: {name} ({ticker})")
        prices = load_ohlcv(ticker, phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END)
        close_by_ticker[ticker] = prices["close"]

        technical = build_features(prices)
        for factor in PRICE_FACTORS:
            price_factor_panels[factor][ticker] = technical[factor]

        fundamentals = load_fundamentals(ticker, 2016, 2024)
        factor_df = build_fundamental_factors(fundamentals)
        daily = to_daily_factors(factor_df, prices.index)
        for factor in FACTOR_COLUMNS:
            fundamental_panels[factor][ticker] = daily[factor]

    for horizon in HORIZONS:
        print(f"\n===== forward return horizon = {horizon}일 =====")
        fwd_returns = pd.DataFrame(
            {t: forward_return(close, horizon) for t, close in close_by_ticker.items()}
        )

        rows = []
        for group_label, panels in [("가격", price_factor_panels), ("펀더멘털", fundamental_panels)]:
            for factor, series_by_ticker in panels.items():
                panel = pd.DataFrame(series_by_ticker)
                ic = non_overlapping_ic(panel, fwd_returns, horizon)
                summary = summarize_ic(ic)
                summary["구분"] = group_label
                summary["factor"] = factor
                summary["팩터변경횟수"] = effective_sample_note(panel)
                rows.append(summary)

        table = pd.DataFrame(rows).set_index(["구분", "factor"])
        table = table[["평균 IC", "t-stat", "IC>0 비율", "관측일수", "팩터변경횟수"]]
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(table.sort_values("t-stat", key=abs, ascending=False))
        print("  (관측일수 = 겹치지 않게 샘플링한 독립 관측 수)")


if __name__ == "__main__":
    main()
