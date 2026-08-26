"""테스트용 합성 OHLCV 데이터 생성."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    log_returns = rng.normal(0, 0.015, size=n)
    close = 10_000 * np.exp(np.cumsum(log_returns))
    volume = rng.integers(100_000, 1_000_000, size=n)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    return df
