"""
Phase 2-B 검증: composite_signal + 히스테리시스로 10종목을 동시에 다루는
포트폴리오 백테스트를 실행한다. 매일 활성(포지션>0) 종목에 동일가중 배분.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import phase1
from src.costs.cost_model import CostModel
from src.data_loader.krx_loader import load_ohlcv
from src.features.technical import build_features
from src.portfolio.portfolio_engine import run_portfolio_backtest
from src.reporting.metrics import summarize
from src.reporting.plots import plot_equity_curve
from src.signals.rules.composite_score import composite_signal


def main() -> None:
    price_by_ticker = {}
    signal_by_ticker = {}

    for ticker, name in phase1.PORTFOLIO_TICKERS.items():
        print(f"로딩 중: {name} ({ticker})")
        df = load_ohlcv(ticker, phase1.FULL_START, phase1.FULL_END)
        features = build_features(df)
        price_by_ticker[ticker] = df
        signal_by_ticker[ticker] = composite_signal(features, entry_threshold=1.0, exit_threshold=0.0)

    cost_model = CostModel()
    n_tickers = len(phase1.PORTFOLIO_TICKERS)

    weighting_modes = {
        "동일가중(활성종목간)": None,
        f"고정비중(1/{n_tickers})": 1.0 / n_tickers,
        "고정비중(1/3, 캡100%)": 1.0 / 3,
        "고정비중(1/2, 캡100%)": 1.0 / 2,
    }

    for label, start, end in [
        ("인샘플", phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END),
        ("아웃오브샘플", phase1.OUT_OF_SAMPLE_START, phase1.OUT_OF_SAMPLE_END),
    ]:
        period_prices = {t: df.loc[start:end] for t, df in price_by_ticker.items()}
        period_signals = {t: s.loc[start:end] for t, s in signal_by_ticker.items()}

        print(f"\n--- {label} ({start} ~ {end}) ---")

        for mode_label, position_size in weighting_modes.items():
            result = run_portfolio_backtest(
                period_prices, period_signals, cost_model, phase1.INITIAL_CAPITAL, position_size
            )
            metrics = summarize(result)
            avg_active = (result.weights > 0).sum(axis=1).mean()
            avg_invested = result.weights.sum(axis=1).mean()

            print(f"  [{mode_label}]")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:,.4f}")
                else:
                    print(f"    {k}: {v}")
            print(f"    평균 동시보유종목수: {avg_active:.2f}")
            print(f"    평균 투자비중(현금 제외): {avg_invested:.2%}")

            safe_mode_label = mode_label.replace("/", "_")
            out_path = (
                PROJECT_ROOT / "experiments" / "plots" / f"portfolio_{label}_{safe_mode_label}_equity.png"
            )
            plot_equity_curve(
                result.equity_curve, f"포트폴리오({n_tickers}종목, {mode_label}) {label}", out_path
            )


if __name__ == "__main__":
    main()
