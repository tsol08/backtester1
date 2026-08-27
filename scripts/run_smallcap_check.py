"""
시가총액 구간을 바꿔가며 롱온리 관점에서 신호를 재본다.

지금까지 모든 분석을 시총 상위 200종목에서 했다. 그런데 거기는 기관 경쟁이 가장
치열하고 애널리스트 커버리지가 가장 두꺼운 영역이라, 이론적으로 우위가 남아있을
가능성이 가장 낮은 자리다. 반대로 중소형 구간은 대형 펀드가 용량 제약(사려는 순간
가격이 튄다) 때문에 구조적으로 들어오지 못한다.

데이터는 이미 있다 — 패널이 3,024종목이고 우리가 상위 200으로 걸러왔을 뿐이다.

**다중검정을 의식해서** 신호는 새로 만들지 않고 이미 검증해본 것만 쓴다. 구간만
바꾼다. 구간 3개 x 신호 4개 = 12개 검정이므로, Bonferroni 임계값 |t| > 2.87을
넘지 못하면 유의하다고 보지 않는다. 여기서 뭔가 나와도 그건 '이 구간을 더 볼
가치가 있다'는 뜻이지 검증된 게 아니다.

롱온리이므로 IC가 아니라 **상위 분위가 유니버스 평균보다 나은가**를 본다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scipy import stats

from config import phase1
from src.data_loader.krx_openapi import build_close_panel, build_panel
from src.data_loader.krx_panel import trading_dates
from src.data_loader.universe import market_cap_universe_mask
from src.features.multi_source import build_price_factors, build_size_factor
from src.research.quantile_analysis import long_only_edge, monotonicity, quantile_forward_returns

HORIZON = 20
N_QUANTILES = 5
PERIODS_PER_YEAR = 252 / HORIZON
WARMUP_DAYS = 400

# (라벨, 시작순위, 끝순위)
SEGMENTS = [
    ("대형 1~200위", 0, 200),
    ("중형 201~500위", 200, 500),
    ("소형 501~1000위", 500, 1000),
]


def build_signals(close: pd.DataFrame, volume: pd.DataFrame, market_cap: pd.DataFrame) -> dict:
    """이미 분석해본 팩터들. 방향은 전부 '클수록 유리'로 통일한다."""
    price = build_price_factors(close, volume)
    return {
        "저변동성": -price["volatility_60"],
        "반전(60일)": -price["momentum_60"],
        "모멘텀(12-1)": price["momentum_12_1"],
        "소형주(역시총)": -build_size_factor(market_cap)["log_market_cap"],
    }


def main() -> None:
    for period_label, start, end in [
        ("인샘플", phase1.IN_SAMPLE_START, phase1.IN_SAMPLE_END),
        ("아웃오브샘플", phase1.OUT_OF_SAMPLE_START, phase1.OUT_OF_SAMPLE_END),
    ]:
        warmup = (pd.Timestamp(start) - pd.Timedelta(WARMUP_DAYS, unit="D")).strftime("%Y-%m-%d")
        dates = trading_dates(warmup, end)
        eval_dates = dates[dates >= pd.Timestamp(start)]

        close = build_close_panel(dates)
        volume = build_panel("volume", dates)
        market_cap = build_panel("market_cap", dates)

        signals = build_signals(close, volume, market_cap)
        fwd = close.pct_change(HORIZON).shift(-HORIZON).loc[eval_dates]

        print("\n" + "=" * 78)
        print(f"{period_label} ({start} ~ {end})")
        print("=" * 78)

        rows = []
        for seg_label, rank_from, top_k in SEGMENTS:
            universe = market_cap_universe_mask(
                market_cap, top_k=top_k, rank_from=rank_from
            ).loc[eval_dates]

            for sig_label, panel in signals.items():
                qr = quantile_forward_returns(
                    panel.loc[eval_dates],
                    fwd,
                    universe,
                    n_quantiles=N_QUANTILES,
                    sample_every=HORIZON,
                )
                if qr.empty:
                    continue

                edge = long_only_edge(qr, PERIODS_PER_YEAR)
                rows.append(
                    {
                        "구간": seg_label,
                        "신호": sig_label,
                        "상위분위(연)": (1 + qr[qr.columns[-1]].mean()) ** PERIODS_PER_YEAR - 1,
                        "유니버스평균(연)": (1 + qr.mean(axis=1).mean()) ** PERIODS_PER_YEAR - 1,
                        "롱온리초과(연)": edge["상위분위 초과(연율화)"],
                        "t-stat": edge["상위분위 초과 t-stat"],
                        "단조성": monotonicity(qr),
                    }
                )

        table = pd.DataFrame(rows).set_index(["구간", "신호"])
        with pd.option_context("display.float_format", "{:.4f}".format):
            print(table.to_string())

    threshold = stats.norm.ppf(1 - 0.05 / (2 * len(SEGMENTS) * 4))
    print(f"\n구간 {len(SEGMENTS)}개 x 신호 4개 = {len(SEGMENTS) * 4}개 동시검정")
    print(f"-> Bonferroni 임계값 |t| > {threshold:.2f} 를 넘어야 유의하다고 볼 수 있다.")
    print("비용은 아직 반영 전이다. 중소형은 거래대금이 작아 시장충격이 커지므로,")
    print("여기서 초과가 나오더라도 비용 차감 후 남는지는 별도로 확인해야 한다.")


if __name__ == "__main__":
    main()
