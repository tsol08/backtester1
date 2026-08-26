"""
기술적/변동성/거래량 피처.

모든 피처는 t일 시점까지의 데이터(과거 값들의 rolling 통계)로만 계산되므로
그 자체로 미래 정보 누수는 없다. 다만 "시그널로 사용해 t일에 바로 체결"하면
누수가 되므로, 실제 체결은 engine 쪽에서 반드시 한 칸(shift) 밀어서 처리한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MA_WINDOWS = (5, 20, 60)
DEFAULT_MOMENTUM_WINDOWS = (5, 20, 60)
DEFAULT_VOL_WINDOWS = (20, 60)
DEFAULT_VOLUME_WINDOWS = (20,)


def add_moving_average_features(df: pd.DataFrame, windows=DEFAULT_MA_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        ma = out["close"].rolling(w).mean()
        out[f"ma_{w}"] = ma
        out[f"disparity_{w}"] = out["close"] / ma - 1
    return out


def add_momentum_features(df: pd.DataFrame, windows=DEFAULT_MOMENTUM_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"momentum_{w}"] = out["close"].pct_change(w)
    return out


def add_volatility_features(df: pd.DataFrame, windows=DEFAULT_VOL_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    log_return = np.log(out["close"] / out["close"].shift(1))
    for w in windows:
        out[f"volatility_{w}"] = log_return.rolling(w).std() * np.sqrt(252)
    return out


def add_volume_features(df: pd.DataFrame, windows=DEFAULT_VOLUME_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        volume_ma = out["volume"].rolling(w).mean()
        out[f"volume_ratio_{w}"] = out["volume"] / volume_ma
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(add_moving_average_features)
        .pipe(add_momentum_features)
        .pipe(add_volatility_features)
        .pipe(add_volume_features)
    )
