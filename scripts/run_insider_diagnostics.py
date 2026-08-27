"""
내부자(임원) 순매수 신호가 진짜인지 깨보는 진단.

run_insider_ic.py에서 '임원만 120일' 신호가 20일 호라이즌에서 t 3.12, IC>0 비율
78%로 나왔다. 이 프로젝트에서 반복해서 겪은 패턴이 있다 — 반전 팩터도, 결합 팩터도,
저변동성도 처음엔 그럴듯했다가 파보면 무너졌다. 같은 잣대를 여기에도 댄다.

확인하는 것:
1) 구간을 반으로 갈라도 유지되는가 (전체 t-stat이 한쪽 구간의 산물은 아닌가)
2) 규모/변동성의 대리변수는 아닌가 (내부자 매수가 잦은 게 그냥 소형주 특성이라면,
   우리가 본 건 내부자 정보가 아니라 이미 아는 소형주 효과다)
3) 매수와 매도 중 무엇이 신호를 만드는가 (매수에는 스톡옵션 행사·주식상여가 섞여
   있어 노이즈가 크고, 매도는 대부분 자발적 판단이라 성격이 다르다)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader.dart_insider import build_insider_signal, load_insider_trades
from src.data_loader.krx_openapi import build_close_panel, build_panel
from src.data_loader.krx_panel import trading_dates
from src.data_loader.universe import fetch_candidate_pool, market_cap_universe_mask
from src.features.multi_source import build_price_factors, build_size_factor, neutralize
from src.research.ic_analysis import daily_cross_sectional_ic, summarize_ic

TOP_K = 200
MIN_OBS = 30
HORIZON = 20
LOOKBACK = 120

START = "2024-09-01"
END = "2026-08-26"


def sampled_ic(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    return daily_cross_sectional_ic(signal.reindex_like(fwd), fwd, min_obs=MIN_OBS).iloc[::HORIZON]


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def summarize(label: str, ic: pd.Series) -> dict:
    return {"구분": label, **summarize_ic(ic)}


def main() -> None:
    dates = trading_dates(START, END)
    close = build_close_panel(dates)
    volume = build_panel("volume", dates)
    market_cap = build_panel("market_cap", dates)
    shares = build_panel("listed_shares", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

    trades = load_insider_trades(fetch_candidate_pool()["ticker"].tolist(), verbose=False)
    executives = trades[trades["is_major_shareholder"].fillna("-").str.strip() == "-"]

    signal = build_insider_signal(executives, dates, shares, lookback=LOOKBACK)
    fwd = close.pct_change(HORIZON).shift(-HORIZON).where(universe)
    aligned = signal.reindex_like(fwd).where(universe).astype(float)

    print("=" * 70)
    print("1) 구간을 반으로 갈라도 유지되는가")
    print("=" * 70)

    midpoint = len(dates) // 2
    rows = [summarize("전체", sampled_ic(aligned, fwd))]
    for label, window in [("전반부", dates[:midpoint]), ("후반부", dates[midpoint:])]:
        rows.append(summarize(label, sampled_ic(aligned.loc[window], fwd.loc[window])))

    show(
        "[임원 순매수 120일, horizon 20일]",
        pd.DataFrame(rows).set_index("구분")[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]],
    )

    print("\n" + "=" * 70)
    print("2) 규모/변동성의 대리변수는 아닌가")
    print("=" * 70)

    log_cap = build_size_factor(market_cap)["log_market_cap"]
    volatility = build_price_factors(close, volume)["volatility_60"]

    rows = [summarize("원본", sampled_ic(aligned, fwd))]
    for label, control in [("규모중립화", log_cap), ("변동성중립화", volatility)]:
        residual = neutralize(aligned, control.where(universe))
        rows.append(summarize(label, sampled_ic(residual, fwd)))

    show(
        "[임원 순매수 120일, horizon 20일]",
        pd.DataFrame(rows).set_index("구분")[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]],
    )

    for label, control in [("로그시총", log_cap), ("변동성", volatility)]:
        corr = aligned.corrwith(control.where(universe), axis=1, method="spearman").dropna().mean()
        print(f"  내부자 신호 vs {label} 평균 순위상관: {corr:.2f}")

    print("\n" + "=" * 70)
    print("3) 매수와 매도 중 무엇이 신호를 만드는가")
    print("=" * 70)

    buys = executives[executives["shares_change"] > 0]
    sells = executives[executives["shares_change"] < 0]

    rows = [summarize("순매수(매수-매도)", sampled_ic(aligned, fwd))]
    for label, subset, sign in [("매수만", buys, 1), ("매도만", sells, -1)]:
        part = build_insider_signal(subset, dates, shares, lookback=LOOKBACK)
        part = (part * sign).reindex_like(fwd).where(universe).astype(float)
        rows.append(summarize(label, sampled_ic(part, fwd)))

    show(
        "[방향 통일: 값이 클수록 매수 우세, horizon 20일]",
        pd.DataFrame(rows).set_index("구분")[["평균 IC", "t-stat", "IC>0 비율", "관측일수"]],
    )
    print(f"  (매수 {len(buys)}건 / 매도 {len(sells)}건)")


if __name__ == "__main__":
    main()
