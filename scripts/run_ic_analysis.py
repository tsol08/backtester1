"""
Phase 2 전략(평균회귀 가설)을 계속 밀고 갈지 결정하기 전에, 원본 피처들이 애초에
미래 수익률과 관련이 있는지부터 확인하는 진단 스크립트. 인샘플 기간만 사용.

전략/파라미터를 전혀 정하지 않은 상태에서 하는 진단이라 과적합 위험이 낮다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import phase1
from src.data_loader.krx_loader import load_ohlcv
from src.features.technical import build_features
from src.research.ic_analysis import daily_cross_sectional_ic, forward_return, pooled_ic, summarize_ic

FEATURE_COLUMNS = [
    "disparity_20",
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "volatility_20",
    "volume_ratio_20",
]
HORIZONS = [5, 10, 20]


def main() -> None:
    features_by_ticker = {}
    close_by_ticker = {}

    for ticker in phase1.PORTFOLIO_TICKERS:
        df = load_ohlcv(ticker, phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END)
        features_by_ticker[ticker] = build_features(df)
        close_by_ticker[ticker] = df["close"]

    for horizon in HORIZONS:
        print(f"\n===== forward return horizon = {horizon}일 =====")
        fwd_return_panel = pd.DataFrame(
            {t: forward_return(close, horizon) for t, close in close_by_ticker.items()}
        )

        rows = []
        for feature_col in FEATURE_COLUMNS:
            feature_panel = pd.DataFrame(
                {t: features_by_ticker[t][feature_col] for t in phase1.PORTFOLIO_TICKERS}
            )

            ic_series = daily_cross_sectional_ic(feature_panel, fwd_return_panel)
            summary = summarize_ic(ic_series)
            summary["pooled_IC"] = pooled_ic(feature_panel, fwd_return_panel)
            summary["feature"] = feature_col
            rows.append(summary)

        result = pd.DataFrame(rows).set_index("feature")
        result = result[["평균 IC", "pooled_IC", "t-stat", "IC>0 비율", "관측일수"]]
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(result)


if __name__ == "__main__":
    main()
