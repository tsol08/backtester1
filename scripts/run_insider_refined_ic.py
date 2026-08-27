"""
사유별로 정제한 내부자 신호의 예측력을 재측정한다.

앞선 분석(run_insider_ic.py)은 구조화 API(elestock)의 순증감을 그대로 썼다.
그런데 그 숫자에는 사유가 섞여 있다:

    01 장내매수  -> 자기 돈으로 샀다 (정보 있음)
    02 장내매도  -> 자기 판단으로 팔았다 (정보 있음)
    31 신규선임  -> 새 임원이 원래 갖고 있던 주식을 처음 신고 (매수가 아님)
    59 자사주상여금 -> 회사가 준 것 (본인 판단 아님)

매수 10,448건 대 매도 1,144건이라는 비대칭 자체가 오염의 징후였다. 원문을 파싱해
진짜 매매만 남기면, 신호가 강해질지 약해질지가 이 팩터의 성격을 말해준다.
강해지면 '내부자 판단'이 진짜 원인이라는 증거고, 약해지면 우리가 본 게
매매가 아니라 다른 무언가(예: 신규 선임이 잦은 회사의 특성)였다는 뜻이다.

비교 대상:
- elestock 원본  : 사유 구분 없는 순증감 (앞선 분석과 동일)
- 매매만         : 장내/장외/시간외 매매만
- 장내매수만     : 가장 순수한 형태
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader.dart_filing import load_filing_details
from src.data_loader.dart_insider import build_insider_signal, load_insider_trades
from src.data_loader.krx_openapi import build_close_panel, build_panel
from src.data_loader.krx_panel import trading_dates
from src.data_loader.universe import fetch_candidate_pool, market_cap_universe_mask
from src.research.ic_analysis import daily_cross_sectional_ic, summarize_ic
from src.research.quantile_analysis import (
    long_only_edge,
    monotonicity,
    quantile_forward_returns,
    summarize_quantiles,
)

TOP_K = 200
MIN_OBS = 30
HORIZONS = [20, 60]
LOOKBACK = 120

START = "2024-09-01"
END = "2026-08-26"


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def main() -> None:
    dates = trading_dates(START, END)
    close = build_close_panel(dates)
    market_cap = build_panel("market_cap", dates)
    shares = build_panel("listed_shares", dates)
    universe = market_cap_universe_mask(market_cap, top_k=TOP_K)

    trades = load_insider_trades(fetch_candidate_pool()["ticker"].tolist(), verbose=False)
    executives = trades[trades["is_major_shareholder"].fillna("-").str.strip() == "-"]

    print(f"임원 공시 {len(executives)}건", flush=True)

    details = load_filing_details(executives["rcept_no"].dropna().unique().tolist(), verbose=False)
    print(f"원문 파싱: {len(details)}행 / {details['rcept_no'].nunique()}건", flush=True)

    breakdown = (
        details.groupby(["reason_code", "reason"])
        .agg(건수=("shares_change", "size"), 주식수=("shares_change", "sum"))
        .sort_values("건수", ascending=False)
    )
    show("[사유별 분포]", breakdown.head(12))

    # 공시일(rcept_dt)을 붙인다. 변동일이 아니라 공시일 기준이어야 look-ahead가 없다.
    disclosed = executives[["rcept_no", "disclosed_date", "ticker"]].drop_duplicates("rcept_no")
    detailed = details.merge(disclosed, on="rcept_no", how="inner")

    signals = {
        "elestock 원본": build_insider_signal(executives, dates, shares, lookback=LOOKBACK),
        "장내매매 순매수": build_insider_signal(
            detailed[detailed["is_open_market"]], dates, shares, lookback=LOOKBACK
        ),
        "장외포함 순매수": build_insider_signal(
            detailed[detailed["is_market_trade"]], dates, shares, lookback=LOOKBACK
        ),
        "장내매수만": build_insider_signal(
            detailed[detailed["reason_code"] == "01"], dates, shares, lookback=LOOKBACK
        ),
    }

    for horizon in HORIZONS:
        fwd = close.pct_change(horizon).shift(-horizon).where(universe)

        rows = []
        for name, panel in signals.items():
            aligned = panel.reindex_like(fwd).where(universe).astype(float)
            ic = daily_cross_sectional_ic(aligned, fwd, min_obs=MIN_OBS).iloc[::horizon]

            summary = summarize_ic(ic)
            summary["신호"] = name
            summary["0아닌종목"] = (aligned.notna() & (aligned != 0)).sum(axis=1).mean()
            rows.append(summary)

        show(
            f"[forward return {horizon}일]",
            pd.DataFrame(rows).set_index("신호")[
                ["평균 IC", "t-stat", "IC>0 비율", "관측일수", "0아닌종목"]
            ],
        )

    # 공매도를 하지 않으므로 IC보다 이쪽이 실제로 중요하다:
    # 상위 분위가 유니버스 평균보다 나은가.
    print("\n" + "=" * 70)
    print("롱온리 관점: 상위 분위가 유니버스 평균보다 나은가")
    print("=" * 70)

    fwd20 = close.pct_change(20).shift(-20)
    for name, panel in signals.items():
        qr = quantile_forward_returns(
            panel.reindex_like(fwd20), fwd20, universe, n_quantiles=5, sample_every=20
        )
        if qr.empty:
            print(f"\n[{name}] 분위 구성 불가")
            continue

        show(f"[{name}] 분위별 20일 수익률", summarize_quantiles(qr, 252 / 20))
        edge = long_only_edge(qr, 252 / 20)
        print(f"  단조성: {monotonicity(qr):.2f}")
        print(
            f"  롱온리 초과(상위분위 - 유니버스평균): 연 {edge['상위분위 초과(연율화)']:.2%}"
            f" (t {edge['상위분위 초과 t-stat']:.2f})"
        )

    print(
        "\n주의: 표본 구간은 여전히 2년(비겹침 관측 18개 안팎)이라 아웃오브샘플 검증이 아니다. "
        "여기서 보는 것은 '정제가 신호를 강화하는가'라는 방향성이다."
    )


if __name__ == "__main__":
    main()
