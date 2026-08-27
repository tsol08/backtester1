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
    시장 대비 초과수익. 개별 종목 수익률에서 시장 수익률을 뺀다.

    **cross-sectional 팩터로는 쓰지 말 것.** 실측 결과 momentum_20과 순위상관이
    정확히 1.00, momentum_60과 0.99로 나왔다. 당연한 결과다 — 시장수익률은 그날
    모든 종목에 대해 같은 값이라, 빼봐야 전 종목이 같은 상수만큼 이동할 뿐이고
    종목간 '순위'는 전혀 바뀌지 않는다. IC는 순위상관이므로 결과가 동일하다.

    시장조정이 의미를 갖는 건 시계열 신호("이 종목이 시장보다 나은가")를 볼 때이지,
    같은 날 종목들을 줄세우는 cross-sectional 분석에서는 아니다. IC 검정에 넣으면
    독립 팩터가 하나 늘어난 것처럼 보여서 다중검정 보정만 부당하게 엄격해진다.
    """
    stock_return = close.pct_change()
    market_return = index_close.reindex(close.index).pct_change()
    excess = stock_return.sub(market_return, axis=0)

    return {
        "excess_return_20": excess.rolling(20).sum(),
        "excess_return_60": excess.rolling(60).sum(),
    }


def neutralize(
    factor: pd.DataFrame, control: pd.DataFrame, min_obs: int = 30, use_ranks: bool = True
) -> pd.DataFrame:
    """
    팩터에서 control로 설명되는 부분을 제거하고 잔차만 남긴다.

    "이 팩터가 사실은 다른 것의 대리변수 아닌가"를 확인하는 도구다. 각 날짜마다
    팩터를 control에 회귀시키고 잔차를 취하면, control로 설명되지 않는 고유 정보만
    남는다. 잔차의 IC가 원본과 비슷하면 그 팩터는 독립적인 정보를 담고 있는 것이고,
    확 줄어들면 사실상 control을 다르게 잰 것이었다는 뜻이다.

    예: 변동성을 시가총액으로 중립화 -> '소형주 효과'인지 확인
        변동성을 베타로 중립화       -> '저베타(시장방어) 효과'인지 확인

    use_ranks=True(기본)면 회귀 전에 양쪽을 그날의 백분위 순위로 바꾼다. 원본값에
    그냥 회귀하면 **이상치 하나가 기울기를 통째로 좌우**한다. 내부자 순매수처럼
    소수 종목에 극단적으로 쏠린 팩터에서는, 실제 순위상관이 0.08에 불과한데도
    중립화 후 IC 부호가 뒤집히는 일이 실제로 벌어졌다. 게다가 우리가 재는 IC 자체가
    순위상관이므로, 중립화도 순위 기준으로 해야 앞뒤가 맞는다.
    """
    if use_ranks:
        factor = factor.rank(axis=1, pct=True)
        control = control.rank(axis=1, pct=True)

    result = pd.DataFrame(index=factor.index, columns=factor.columns, dtype=float)

    for date in factor.index:
        y = factor.loc[date]
        x = control.loc[date] if date in control.index else None
        if x is None:
            continue

        valid = y.notna() & x.notna()
        if valid.sum() < min_obs:
            continue

        y_valid = y[valid]
        x_valid = x[valid]
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
        result.loc[date, valid[valid].index] = y_valid - (slope * x_valid + intercept)

    return result


def neutralize_by_size(factor: pd.DataFrame, log_market_cap: pd.DataFrame) -> pd.DataFrame:
    """팩터에서 규모(시가총액) 효과를 제거한다. neutralize의 규모 전용 별칭."""
    return neutralize(factor, log_market_cap)


def rolling_beta(
    close: pd.DataFrame, market_index: pd.Series, window: int = 60
) -> pd.DataFrame:
    """
    종목별 시장 베타 (rolling 공분산 / 시장 분산).

    저변동성 팩터가 사실은 '저베타 = 시장이 빠질 때 덜 빠짐'에 불과한지 확인하는 데 쓴다.
    변동성이 낮은 종목은 대개 베타도 낮으므로, 둘을 구분하지 않으면 cross-sectional
    알파를 찾은 게 아니라 시장 방향에 베팅한 것을 알파로 착각할 수 있다.
    """
    stock_return = close.pct_change()
    market_return = market_index.reindex(close.index).pct_change()

    covariance = stock_return.rolling(window).cov(market_return)
    market_variance = market_return.rolling(window).var()
    return covariance.div(market_variance, axis=0)
