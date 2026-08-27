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
from scipy.special import erfinv

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

    ranks = masked.rank(axis=1, pct=True).to_numpy()
    # 백분위를 표준정규로 바꿔야 선형 합성이 의도한 상관을 만든다.
    # 양 끝(0, 1)은 무한대가 되므로 살짝 안으로 밀어넣는다.
    ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
    truth = np.sqrt(2) * erfinv(2 * ranks - 1)

    noise = rng.standard_normal(masked.shape)
    signal = target_ic * truth + np.sqrt(max(0.0, 1 - target_ic**2)) * noise

    return pd.DataFrame(signal, index=masked.index, columns=masked.columns).where(masked.notna())


def _trim_to_universe(
    fwd_returns: pd.DataFrame, universe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    유니버스에 한 번이라도 포함된 종목만 남긴다.

    패널은 3,000종목인데 유니버스는 200종목뿐이라, 그대로 두면 시뮬레이션마다
    쓰이지도 않을 종목의 난수를 생성하고 순위를 매기게 된다. 결과는 같고 시간만
    십수 배 든다.
    """
    used = universe.any(axis=0)
    used = used[used].index
    return fwd_returns[used], universe[used]


def detection_rate(
    fwd_returns: pd.DataFrame,
    universe: pd.DataFrame,
    target_ic: float,
    horizon: int,
    n_trials: int = 200,
    thresholds: tuple[float, ...] = (1.96,),
    min_obs: int = 30,
    seed: int = 0,
) -> dict:
    """
    target_ic 크기의 신호를 몇 번이나 검출하는지 (= 검정력).

    thresholds는 유의성 기준 t값들이다. 1.96은 단일 검정 5% 수준이고, 다중검정
    보정을 감안하려면 2.9 안팎을 쓴다 — 실제로 우리가 팩터를 판정할 때 쓴 기준이다.
    여러 기준을 한 번에 받는 이유는, 같은 시뮬레이션 결과에 임계값만 달리 적용하면
    되는데 기준마다 전체를 다시 돌리는 것이 순전한 낭비이기 때문이다.
    """
    fwd_returns, universe = _trim_to_universe(fwd_returns, universe)

    # 어차피 horizon 간격으로만 쓸 것이므로 미리 솎아낸다. 매일 IC를 구한 뒤
    # 20일마다 골라내면 필요한 계산의 20배를 하게 된다.
    #
    # non_overlapping_ic는 '유효한 날' 기준으로 간격을 두는 반면 여기서는 달력
    # 기준으로 솎아내므로 뽑히는 날짜가 조금 다르다. 하지만 검정력을 좌우하는 것은
    # 독립 관측의 개수이고 그것은 동일하다.
    fwd_returns = fwd_returns.iloc[::horizon]
    universe = universe.iloc[::horizon]
    masked_fwd = fwd_returns.where(universe)

    rng = np.random.default_rng(seed)

    t_stats = []
    realized_ics = []
    for _ in range(n_trials):
        signal = inject_signal(fwd_returns, target_ic, rng, universe)
        ic = daily_cross_sectional_ic(signal, masked_fwd, min_obs=min_obs)
        if ic.dropna().empty:
            continue

        summary = summarize_ic(ic)
        t_stats.append(summary["t-stat"])
        realized_ics.append(summary["평균 IC"])

    t_stats = np.array(t_stats)
    result = {
        "목표 IC": target_ic,
        "실현 IC(평균)": float(np.mean(realized_ics)) if len(realized_ics) else np.nan,
        "t-stat 중앙값": float(np.median(t_stats)) if len(t_stats) else np.nan,
        "시행": len(t_stats),
    }
    for threshold in thresholds:
        label = f"검출률(t>{threshold})"
        result[label] = float(np.mean(t_stats > threshold)) if len(t_stats) else np.nan
    return result


def minimum_detectable_ic(
    fwd_returns: pd.DataFrame,
    universe: pd.DataFrame,
    horizon: int,
    candidates: list[float],
    **kwargs,
) -> pd.DataFrame:
    """여러 신호 강도에 대한 검출률 표. 검정력 80%를 넘는 최소 IC를 찾는 데 쓴다."""
    rows = [detection_rate(fwd_returns, universe, ic, horizon, **kwargs) for ic in candidates]
    return pd.DataFrame(rows).set_index("목표 IC")
