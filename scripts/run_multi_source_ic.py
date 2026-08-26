"""
여러 정보원의 팩터들이 실제로 미래 수익률을 예측하는지 한 번에 비교한다.

이 프로젝트에서 반복해서 확인한 교훈 두 가지를 그대로 반영한다:
1) 전략을 만들기 전에 팩터의 예측력부터 데이터로 확인한다 (임의로 정한 가중치로
   전략을 만들고 성과를 보는 순서는 무엇이 문제인지 알려주지 못한다)
2) 일별 IC를 그대로 평균내면 t-stat이 심하게 부풀려진다. 겹치지 않게 샘플링하고,
   여러 팩터를 동시에 검정한다는 점(다중검정)도 감안해서 본다

유니버스는 각 시점의 실제 시가총액 상위 종목(point-in-time)으로 제한한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scipy import stats

from config import phase1
from src.data_loader.krx_panel import build_panel, trading_dates
from src.features.multi_source import (
    build_excess_return_factors,
    build_price_factors,
    build_shorting_factors,
    build_size_factor,
    build_valuation_factors,
)
from src.research.ic_analysis import forward_return, non_overlapping_ic, summarize_ic

HORIZONS = [20, 60]
TOP_K = 200
MIN_OBS = 30


def load_universe_mask(dates: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """분기 스냅샷(시점별 시총 상위)을 일별 마스크로 펼친다."""
    snapshot_dir = PROJECT_ROOT / "data" / "raw" / "universe_snapshots"
    frames = [pd.read_parquet(p) for p in sorted(snapshot_dir.glob("*.parquet"))]
    snapshots = pd.concat(frames)

    mask = pd.DataFrame(False, index=dates, columns=columns)
    rebalance_dates = sorted(snapshots["rebalance_date"].unique())

    for i, rebalance in enumerate(rebalance_dates):
        members = snapshots.loc[snapshots["rebalance_date"] == rebalance, "ticker"]
        members = [t for t in members if t in mask.columns]
        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else dates.max()
        window = (mask.index >= rebalance) & (mask.index < end)
        mask.loc[window, members] = True

    return mask


def main() -> None:
    dates = trading_dates(phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END)
    print(f"인샘플 거래일 {len(dates)}일", flush=True)

    close = build_panel("ohlcv", "close", dates)
    volume = build_panel("ohlcv", "volume", dates)
    print(f"가격 패널: {close.shape[0]}일 x {close.shape[1]}종목", flush=True)

    factors: dict[str, pd.DataFrame] = {}
    groups: dict[str, str] = {}

    for name, panel in build_price_factors(close, volume).items():
        factors[name] = panel
        groups[name] = "가격"

    per = build_panel("valuation", "per", dates)
    pbr = build_panel("valuation", "pbr", dates)
    div_yield = build_panel("valuation", "div_yield", dates)
    if len(per):
        for name, panel in build_valuation_factors(per, pbr, div_yield).items():
            factors[name] = panel
            groups[name] = "밸류에이션"

    market_cap = build_panel("cap", "market_cap", dates)
    if len(market_cap):
        for name, panel in build_size_factor(market_cap).items():
            factors[name] = panel
            groups[name] = "규모"

    short_ratio = build_panel("shorting", "short_ratio", dates)
    if len(short_ratio):
        for name, panel in build_shorting_factors(short_ratio).items():
            factors[name] = panel
            groups[name] = "공매도"

    index_panel = build_panel("ohlcv", "close", dates)
    if len(index_panel):
        # KOSPI 지수는 별도 캐시에서 읽는다
        index_path = PROJECT_ROOT / "data" / "raw" / "index" / "1001.parquet"
        if index_path.exists():
            index_close = pd.read_parquet(index_path)["close"]
            for name, panel in build_excess_return_factors(close, index_close).items():
                factors[name] = panel
                groups[name] = "초과수익"

    print(f"팩터 {len(factors)}개 준비 완료", flush=True)

    universe = load_universe_mask(dates, close.columns)
    print(f"시점별 유니버스 평균 종목 수: {universe.sum(axis=1).mean():.0f}", flush=True)

    for horizon in HORIZONS:
        print(f"\n===== forward return horizon = {horizon}일 =====")
        fwd = forward_return(close, horizon) if isinstance(close, pd.Series) else close.pct_change(horizon).shift(-horizon)
        fwd = fwd.where(universe)

        rows = []
        for name, panel in factors.items():
            aligned = panel.reindex_like(fwd).where(universe)
            ic = non_overlapping_ic(aligned, fwd, horizon, min_obs=MIN_OBS)
            summary = summarize_ic(ic)
            summary["구분"] = groups[name]
            summary["factor"] = name
            rows.append(summary)

        table = pd.DataFrame(rows).set_index(["구분", "factor"])
        table = table[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]]
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(table.sort_values("t-stat", key=abs, ascending=False).to_string())

        threshold = stats.norm.ppf(1 - 0.05 / (2 * len(rows)))
        print(f"  팩터 {len(rows)}개 동시검정 -> Bonferroni 보정 임계값 |t| > {threshold:.2f}")


if __name__ == "__main__":
    main()
