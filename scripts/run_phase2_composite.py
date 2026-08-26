"""
Phase 2-A 검증: 다요소 결합 스코어 + 히스테리시스 청산 신호를,
Phase 1의 단일 지표(이격도 z-score, 청산 없음) 신호와 같은 종목/기간에서 비교한다.

포트폴리오 결합(2개 종목 동시 보유)은 아직 없음 — 여기서는 종목별로 독립적인
단일 종목 백테스트만 돌려서 "신호 자체가 개선됐는가"만 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import phase1
from src.costs.cost_model import CostModel
from src.data_loader.krx_loader import load_ohlcv
from src.engine.backtest import run_backtest
from src.features.technical import build_features
from src.reporting.metrics import summarize
from src.signals.rules.composite_score import composite_signal
from src.signals.rules.mean_reversion import disparity_zscore_signal


def compare_for_ticker(ticker: str, name: str) -> None:
    print(f"\n===== {name} ({ticker}) =====")

    df = load_ohlcv(ticker, phase1.FULL_START, phase1.FULL_END)
    features = build_features(df)

    signals = {
        "Phase1 (단일지표, 청산없음)": disparity_zscore_signal(features, window=20, entry_z=-1.0),
        "Phase2 (다요소+히스테리시스)": composite_signal(features, entry_threshold=1.0, exit_threshold=0.0),
    }

    cost_model = CostModel()

    for label, start, end in [
        ("인샘플", phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END),
        ("아웃오브샘플", phase1.OUT_OF_SAMPLE_START, phase1.OUT_OF_SAMPLE_END),
    ]:
        print(f"--- {label} ({start} ~ {end}) ---")
        period_df = df.loc[start:end]

        for sig_label, signal in signals.items():
            period_signal = signal.loc[start:end]
            result = run_backtest(period_df, period_signal, cost_model, phase1.INITIAL_CAPITAL)
            metrics = summarize(result)
            print(f"  [{sig_label}]")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:,.4f}")
                else:
                    print(f"    {k}: {v}")


def main() -> None:
    for ticker, name in phase1.TICKERS.items():
        compare_for_ticker(ticker, name)


if __name__ == "__main__":
    main()
