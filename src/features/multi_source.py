"""
여러 정보원에서 팩터 패널을 만든다.

정보원별로 성격이 다르고, 서로 다른 종류의 질문에 답한다:
- 가격/거래량: 최근 어떻게 움직였나 (기술적)
- 밸류에이션: 이익·자산 대비 싼가 (가치)
- 재무제표: 실제로 돈을 잘 버나 (퀄리티/성장)
- 수급: 외국인·기관이 사는가 (주체별 행동)
- 공매도: 하락에 베팅하는 자금이 있나

모든 팩터는 (날짜 x 종목) 패널이며, t일 값은 t일까지 관측 가능한 정보만 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_log(panel: pd.DataFrame) -> pd.DataFrame:
    """0/음수는 NaN 처리 후 로그 (시가총액처럼 규모 차이가 큰 값에 사용)."""
    return np.log(panel.where(panel > 0))


def build_price_factors(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """가격/거래량 기반 기술적 팩터."""
    factors: dict[str, pd.DataFrame] = {}

    for window in (20, 60):
        ma = close.rolling(window).mean()
        factors[f"disparity_{window}"] = close / ma - 1
        factors[f"momentum_{window}"] = close.pct_change(window)

    log_return = np.log(close / close.shift(1))
    factors["volatility_60"] = log_return.rolling(60).std()

    # 최근 1개월을 제외한 12개월 모멘텀. 단기 반전 효과를 제거한 형태로,
    # 해외 팩터 연구에서 표준적으로 쓰이는 정의다.
    factors["momentum_12_1"] = close.shift(20) / close.shift(250) - 1

    volume_ma = volume.rolling(60).mean()
    factors["volume_ratio_60"] = volume / volume_ma

    return factors


def build_valuation_factors(
    per: pd.DataFrame, pbr: pd.DataFrame, div_yield: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    가치 팩터. PER/PBR은 '낮을수록 싸다'이므로 역수를 취해 '높을수록 매력적'으로 방향을
    통일한다(earnings yield, book-to-market). 0 이하(적자 등)는 NaN 처리한다.
    """
    return {
        "earnings_yield": (1 / per.where(per > 0)),
        "book_to_market": (1 / pbr.where(pbr > 0)),
        "dividend_yield": div_yield.where(div_yield >= 0),
    }


def build_size_factor(market_cap: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """규모 팩터. 로그 시가총액 (작을수록 소형주)."""
    return {"log_market_cap": _safe_log(market_cap)}


def build_shorting_factors(short_ratio: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    공매도 팩터. 거래대금 대비 공매도 비중과, 그 비중의 최근 변화량.
    수준 자체보다 '갑자기 늘었는가'가 더 정보가 있을 수 있어 둘 다 만든다.
    """
    return {
        "short_ratio_20": short_ratio.rolling(20).mean(),
        "short_ratio_change": short_ratio.rolling(5).mean() - short_ratio.rolling(60).mean(),
    }


def build_excess_return_factors(
    close: pd.DataFrame, index_close: pd.Series
) -> dict[str, pd.DataFrame]:
    """
    시장 대비 초과수익. 개별 종목 수익률에서 시장 수익률을 빼면, 시장 전체가 움직여서
    생긴 부분을 걷어내고 그 종목 고유의 움직임만 남는다.
    """
    stock_return = close.pct_change()
    market_return = index_close.reindex(close.index).pct_change()
    excess = stock_return.sub(market_return, axis=0)

    return {
        "excess_return_20": excess.rolling(20).sum(),
        "excess_return_60": excess.rolling(60).sum(),
    }


def neutralize_by_size(factor: pd.DataFrame, log_market_cap: pd.DataFrame) -> pd.DataFrame:
    """
    팩터에서 규모(시가총액) 효과를 제거한다.

    많은 팩터가 사실은 '소형주라서' 생기는 효과를 반영할 뿐일 수 있다. 각 날짜마다
    팩터를 시가총액에 회귀시키고 잔차만 남기면, 규모로 설명되지 않는 고유 정보만 남는다.
    """
    result = pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)

    for date in factor.index:
        y = factor.loc[date]
        x = log_market_cap.loc[date] if date in log_market_cap.index else None
        if x is None:
            continue

        valid = y.notna() & x.notna()
        if valid.sum() < 30:
            continue

        y_valid = y[valid]
        x_valid = x[valid]
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
        result.loc[date, valid[valid].index] = y_valid - (slope * x_valid + intercept)

    return result
