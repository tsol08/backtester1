"""
확대된 유니버스(수백 종목)에서 가격 팩터 vs 펀더멘털 팩터의 IC를 비교한다.

앞선 10종목 분석에서는 표본이 너무 적어 어떤 팩터도 유의성을 확인할 수 없었다.
여기서는 종목 수를 크게 늘려 cross-sectional IC의 노이즈를 줄인다.

방법론상 지키는 것:
- 시점별 유니버스: 각 날짜마다 최근 거래대금 상위 K종목만 사용(미래 정보 없음)
- 펀더멘털은 공시일 이후에만 사용
- IC는 겹치지 않게(non-overlapping) 샘플링해 자기상관에 의한 t-stat 과대평가를 방지
- 다중검정을 감안해 임계값을 보수적으로 본다

남아있는 한계: 후보군을 '현재' 시가총액 상위에서 뽑았기 때문에 생존편향이 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scipy import stats

from config import phase1
from src.data_loader.dart_loader import load_fundamentals_bulk
from src.data_loader.krx_loader import load_ohlcv
from src.data_loader.universe import fetch_candidate_pool, point_in_time_universe
from src.features.fundamental import FACTOR_COLUMNS, build_fundamental_factors, to_daily_factors
from src.features.technical import build_features
from src.research.ic_analysis import forward_return, non_overlapping_ic, summarize_ic

PRICE_FACTORS = ["disparity_20", "momentum_60", "volatility_20", "volume_ratio_20"]
HORIZONS = [20, 60]
TOP_K = 100  # 시점별 유니버스 크기


def main() -> None:
    candidates = fetch_candidate_pool()
    tickers = candidates["ticker"].tolist()

    print("가격/기술적 팩터 준비 중...")
    close_by_ticker: dict[str, pd.Series] = {}
    trading_value_by_ticker: dict[str, pd.Series] = {}
    price_panels: dict[str, dict[str, pd.Series]] = {f: {} for f in PRICE_FACTORS}

    for ticker in tickers:
        try:
            prices = load_ohlcv(ticker, phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END)
        except Exception:
            continue
        if len(prices) < 300:
            continue

        close_by_ticker[ticker] = prices["close"]
        trading_value_by_ticker[ticker] = prices["close"] * prices["volume"]

        technical = build_features(prices)
        for factor in PRICE_FACTORS:
            price_panels[factor][ticker] = technical[factor]

    print(f"  가격 데이터 확보: {len(close_by_ticker)}종목")

    print("펀더멘털 팩터 준비 중...")
    raw_fundamentals = load_fundamentals_bulk(
        list(close_by_ticker.keys()), 2016, 2024, verbose=False
    )

    trading_value = pd.DataFrame(trading_value_by_ticker)
    universe_mask = point_in_time_universe(trading_value, top_k=TOP_K)

    fundamental_panels: dict[str, dict[str, pd.Series]] = {f: {} for f in FACTOR_COLUMNS}
    for ticker, group in raw_fundamentals.groupby("ticker"):
        if ticker not in close_by_ticker:
            continue
        factor_df = build_fundamental_factors(group)
        daily = to_daily_factors(factor_df, close_by_ticker[ticker].index)
        for factor in FACTOR_COLUMNS:
            fundamental_panels[factor][ticker] = daily[factor]

    print(f"  펀더멘털 확보: {raw_fundamentals['ticker'].nunique()}종목")
    print(f"  시점별 유니버스 평균 종목 수: {universe_mask.sum(axis=1).mean():.0f}")

    for horizon in HORIZONS:
        print(f"\n===== forward return horizon = {horizon}일 =====")
        fwd_returns = pd.DataFrame(
            {t: forward_return(c, horizon) for t, c in close_by_ticker.items()}
        )
        fwd_returns = fwd_returns.where(universe_mask.reindex_like(fwd_returns).fillna(False))

        rows = []
        for group_label, panels in [("가격", price_panels), ("펀더멘털", fundamental_panels)]:
            for factor, series_by_ticker in panels.items():
                panel = pd.DataFrame(series_by_ticker)
                panel = panel.where(universe_mask.reindex_like(panel).fillna(False))
                ic = non_overlapping_ic(panel, fwd_returns, horizon, min_obs=20)
                summary = summarize_ic(ic)
                summary["구분"] = group_label
                summary["factor"] = factor
                rows.append(summary)

        table = pd.DataFrame(rows).set_index(["구분", "factor"])
        table = table[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]]
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(table.sort_values("t-stat", key=abs, ascending=False))

        n_tests = len(rows)
        bonferroni_threshold = stats.norm.ppf(1 - 0.05 / (2 * n_tests))
        print(
            f"  (팩터 {n_tests}개 동시 검정 -> Bonferroni 보정 시 |t| > "
            f"{bonferroni_threshold:.2f} 는 넘어야 유의하다고 볼 수 있음)"
        )


if __name__ == "__main__":
    main()
