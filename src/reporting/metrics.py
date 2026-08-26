"""성과 지표: CAGR, 샤프비율, MDD, 승률 등."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / periods_per_year
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    n = len(equity_curve)
    if n < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n / periods_per_year
    return float(total_return ** (1 / years) - 1)


def win_rate(returns: pd.Series, trades: pd.Series) -> float:
    traded_days = trades.abs() > 1e-9
    day_returns = returns[traded_days]
    if len(day_returns) == 0:
        return 0.0
    return float((day_returns > 0).mean())


def summarize(result) -> dict:
    return {
        "CAGR": cagr(result.equity_curve),
        "Sharpe": sharpe_ratio(result.returns),
        "MDD": max_drawdown(result.equity_curve),
        "총비용(원)": float(result.cost_paid.sum()),
        "거래횟수": int((result.trades.abs() > 1e-9).sum()),
        "최종자산": float(result.equity_curve.iloc[-1]),
    }
