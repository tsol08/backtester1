"""
Phase 1 엔진 검증 스크립트.

005930(삼성전자), 000660(SK하이닉스)에 대해 이격도 z-score 평균회귀 시그널로
백테스트를 돌려서 데이터로더/피처/비용모델/엔진/리포팅이 엔드투엔드로 잘 동작하는지 확인한다.
인샘플/아웃오브샘플 구간을 나눠서 각각 출력한다 (이 단계에서는 파라미터 튜닝을 하지 않으므로
과적합 이슈는 없지만, Phase 2부터는 인샘플에서만 파라미터를 정하는 규율을 지킨다).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import phase1
from src.costs.cost_model import CostModel
from src.data_loader.krx_loader import load_ohlcv
from src.engine.backtest import run_backtest
from src.features.technical import build_features
from src.reporting.metrics import summarize
from src.reporting.plots import plot_equity_curve
from src.signals.rules.mean_reversion import disparity_zscore_signal


def run_for_ticker(ticker: str, name: str) -> None:
    print(f"\n===== {name} ({ticker}) =====")

    df = load_ohlcv(ticker, phase1.FULL_START, phase1.FULL_END)
    features = build_features(df)
    signal = disparity_zscore_signal(features, window=20, entry_z=-1.0)

    cost_model = CostModel()

    for label, start, end in [
        ("인샘플", phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END),
        ("아웃오브샘플", phase1.OUT_OF_SAMPLE_START, phase1.OUT_OF_SAMPLE_END),
    ]:
        period_df = df.loc[start:end]
        period_signal = signal.loc[start:end]

        result = run_backtest(period_df, period_signal, cost_model, phase1.INITIAL_CAPITAL)
        metrics = summarize(result)

        print(f"--- {label} ({start} ~ {end}) ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:,.4f}")
            else:
                print(f"  {k}: {v}")

        out_path = PROJECT_ROOT / "experiments" / "plots" / f"{ticker}_{label}_equity.png"
        plot_equity_curve(result.equity_curve, f"{name} {label} equity curve", out_path)


def main() -> None:
    for ticker, name in phase1.TICKERS.items():
        run_for_ticker(ticker, name)


if __name__ == "__main__":
    main()
