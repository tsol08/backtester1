"""
검정력 분석: 우리 도구가 '있는 신호'를 찾아낼 수 있는가.

지금까지 이 백테스터로 한 일은 전부 기각이었다 — 반전, 저변동성, 내부자, 소형주
넷 다 아니라고 판정했다. 그런데 여기엔 논리적 구멍이 있다.

    모든 것을 기각하는 도구는 정확한 것일 수도 있고,
    애초에 아무것도 못 잡는 도구일 수도 있다.

둘을 구분하려면 **강도를 아는 신호를 일부러 심어놓고** 찾아내는지 보면 된다.
실제 한국 주가 데이터의 미래수익률에 알려진 IC만큼 상관된 인공 신호를 만들어,
우리 파이프라인(비겹침 샘플링 -> IC -> t-stat)에 그대로 통과시킨다.

여기서 나오는 답이 결론을 좌우한다:
- 심어놓은 IC 0.03을 대부분 검출한다면 -> 도구는 정상이고, 한국 시장에서 못 찾은
  것은 진짜로 없어서다.
- IC 0.03을 놓친다면 -> 우리 표본(비겹침 관측 20~60개)으로는 그 크기의 신호를
  애초에 검출할 수 없었다는 뜻이다. 즉 "없다"가 아니라 "못 본다"였다.

후자라면 지금까지의 기각 판정은 '증거의 부재'일 뿐 '부재의 증거'가 아니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.ic_analysis import daily_cross_sectional_ic, summarize_ic


def inject_signal(
    fwd_returns: pd.DataFrame,
    target_ic: float,
    rng: np.random.Generator,
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    미래수익률과 target_ic만큼 순위상관된 인공 신호를 만든다.

    날짜마다 '미래수익률의 표준화된 순위'와 '순수 잡음'을 target_ic 비율로 섞는다.
    상관계수 r로 두 변수를 합성하는 표준적인 방법이다:
        signal = r * z(정답) + sqrt(1 - r^2) * noise

    실제 수익률 분포와 결측 패턴을 그대로 쓰기 때문에, 인공 데이터가 아니라
    '우리가 실제로 다루는 조건'에서의 검정력을 측정하게 된다.
    """
    masked = fwd_returns.where(universe) if universe is not None else fwd_returns

    ranks = masked.rank(axis=1, pct=True)
    truth = pd.DataFrame(
        # 백분위를 표준정규로 바꿔야 선형 합성이 의도한 상관을 만든다
        np.sqrt(2) * np.vectorize(_inverse_erf)(2 * ranks.to_numpy() - 1),
        index=ranks.index,
        columns=ranks.columns,
    )

    noise = pd.DataFrame(
        rng.standard_normal(masked.shape), index=masked.index, columns=masked.columns
    )

    signal = target_ic * truth + np.sqrt(max(0.0, 1 - target_ic**2)) * noise
    return signal.where(masked.notna())


def _inverse_erf(x: float) -> float:
    """scipy 없이 쓰기 위한 역오차함수 근사 (Winitzki)."""
    if x <= -1:
        return -np.inf
    if x >= 1:
        return np.inf
    a = 0.147
    ln = np.log(1 - x**2)
    term = 2 / (np.pi * a) + ln / 2
    return np.sign(x) * np.sqrt(np.sqrt(term**2 - ln / a) - term)


def detection_rate(
    fwd_returns: pd.DataFrame,
    universe: pd.DataFrame,
    target_ic: float,
    horizon: int,
    n_trials: int = 200,
    threshold: float = 1.96,
    min_obs: int = 30,
    seed: int = 0,
) -> dict:
    """
    target_ic 크기의 신호를 몇 번이나 검출하는지 (= 검정력).

    threshold는 유의성 기준 t값이다. 1.96은 단일 검정 5% 수준이고, 다중검정
    보정을 감안하려면 2.9 안팎을 쓴다 — 실제로 우리가 팩터를 판정할 때 쓴 기준이다.
    """
    rng = np.random.default_rng(seed)

    t_stats = []
    realized_ics = []
    for _ in range(n_trials):
        signal = inject_signal(fwd_returns, target_ic, rng, universe)
        ic = daily_cross_sectional_ic(signal, fwd_returns.where(universe), min_obs=min_obs)
        ic = ic.iloc[::horizon]
        if ic.dropna().empty:
            continue

        summary = summarize_ic(ic)
        t_stats.append(summary["t-stat"])
        realized_ics.append(summary["평균 IC"])

    t_stats = np.array(t_stats)
    return {
        "목표 IC": target_ic,
        "실현 IC(평균)": float(np.mean(realized_ics)) if len(realized_ics) else np.nan,
        "t-stat 중앙값": float(np.median(t_stats)) if len(t_stats) else np.nan,
        "검출률": float(np.mean(t_stats > threshold)) if len(t_stats) else np.nan,
        "시행": len(t_stats),
    }


def minimum_detectable_ic(
    fwd_returns: pd.DataFrame,
    universe: pd.DataFrame,
    horizon: int,
    candidates: list[float],
    target_power: float = 0.8,
    **kwargs,
) -> pd.DataFrame:
    """여러 신호 강도에 대한 검출률 표. target_power를 넘는 최소 IC를 찾는 데 쓴다."""
    rows = [detection_rate(fwd_returns, universe, ic, horizon, **kwargs) for ic in candidates]
    table = pd.DataFrame(rows).set_index("목표 IC")
    table["검정력 달성"] = table["검출률"] >= target_power
    return table
