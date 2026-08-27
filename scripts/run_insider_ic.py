"""
내부자 거래(임원·주요주주 소유변동)의 예측력을 검증한다.

지금까지 시도한 정보원 — 가격, 거래량, 재무제표, 시가총액 — 은 전부 '시장이 이미
아는 사실'의 가공이었다. 어떤 조합을 해도 새 정보가 들어오지 않으니, 찾아낸 신호가
얇았던 게 이상한 일이 아니다. 내부자 거래는 성격이 다르다: 회사 사정을 가장 잘 아는
사람이 자기 돈을 걸고 무엇을 했는지가 담긴다.

신호: 최근 lookback 거래일간 내부자 순매수 주식수 / 상장주식수.
공시일(rcept_dt) 기준으로만 집계하므로, 실제 거래 시점과 공시 시점의 시차 때문에
미래를 보는 일은 없다(tests/test_insider_signal.py에서 검증).

**이 분석의 한계를 먼저 밝힌다.** DART 지분공시 API가 최근 2년치만 제공해서
(날짜 파라미터를 무시함) 인샘플/아웃오브샘플 분리가 불가능하다. 비겹침 관측이
20일 호라이즌 기준 20여 개에 불과하다. 여기서 t-stat이 크게 나와도 그건 '검증됨'이
아니라 '더 볼 가치가 있음' 정도로만 읽어야 한다.
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
from src.research.ic_analysis import daily_cross_sectional_ic, summarize_ic

TOP_K = 200
MIN_OBS = 30
HORIZONS = [20, 60]
LOOKBACKS = [60, 120]

# 지분공시 API가 커버하는 구간. 앞쪽 여유는 rolling 윈도우 워밍업용.
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

    print(f"거래일 {len(dates)}일 ({START} ~ {END})", flush=True)
    print(f"시점별 유니버스: 일별 {universe.sum(axis=1).mean():.0f}종목", flush=True)

    tickers = fetch_candidate_pool()["ticker"].tolist()
    trades = load_insider_trades(tickers, verbose=False)
    print(f"내부자 거래: {len(trades)}건 / {trades['ticker'].nunique()}종목", flush=True)

    buy = trades[trades["shares_change"] > 0]["shares_change"].sum()
    sell = -trades[trades["shares_change"] < 0]["shares_change"].sum()
    print(f"  매수 {buy:,.0f}주 / 매도 {sell:,.0f}주", flush=True)

    # 임원과 10%이상 대주주는 성격이 전혀 다른 거래다. 임원 매수는 회사 사정을 아는
    # 사람의 판단이지만, 대주주 지분 변동은 경영권·상속·블록딜 같은 구조적 사건인
    # 경우가 많다. 실제로 상위 10건이 전체 매수량의 39%를 차지할 만큼 대형 건에
    # 쏠려 있어서, 섞어놓으면 소수의 지배구조 이벤트가 신호를 삼킨다.
    is_executive = trades["is_major_shareholder"].fillna("-").str.strip() == "-"
    groups = {
        "전체": trades,
        "임원만": trades[is_executive],
        "주요주주만": trades[~is_executive],
    }

    signals = {
        f"{label} {lb}일": build_insider_signal(subset, dates, shares, lookback=lb)
        for label, subset in groups.items()
        for lb in LOOKBACKS
    }

    for horizon in HORIZONS:
        fwd = close.pct_change(horizon).shift(-horizon).where(universe)

        rows = []
        for name, panel in signals.items():
            aligned = panel.reindex_like(fwd).where(universe).astype(float)

            # 내부자 거래가 아예 없는 종목은 0이 되는데, 이걸 '중립'으로 두면
            # 대다수가 0인 날에는 순위가 사실상 무의미해진다. 실제로 신호가 있는
            # 종목이 얼마나 되는지 같이 본다.
            nonzero = (aligned != 0).sum(axis=1)

            ic = daily_cross_sectional_ic(aligned, fwd, min_obs=MIN_OBS).iloc[::horizon]
            summary = summarize_ic(ic)
            summary["신호"] = name
            summary["일평균 유효종목"] = nonzero.mean()
            rows.append(summary)

        show(
            f"[forward return {horizon}일]",
            pd.DataFrame(rows).set_index("신호")[
                ["평균 IC", "t-stat", "IC>0 비율", "관측일수", "일평균 유효종목"]
            ],
        )

    print(
        "\n주의: 지분공시 API가 최근 2년만 제공해 인샘플/아웃오브샘플 분리가 불가능하다. "
        "관측 수가 적어 이 결과는 검증이 아니라 탐색이다."
    )


if __name__ == "__main__":
    main()
