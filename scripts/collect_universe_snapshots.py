"""
분기별 시점 유니버스 스냅샷을 수집한다.

각 분기 시작일의 '실제 그 시점' 시가총액 상위 종목을 뽑아둔다. 이렇게 하면
과거 백테스트에서 그 시점에 실제로 대형주였던 종목만 쓰게 되어, 현재 시총 순위를
쓸 때 생기는 생존편향과 look-ahead가 크게 줄어든다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_loader.universe import build_universe_snapshots

START = "2017-01-01"
END = "2024-12-31"
TOP_K = 200


def main() -> None:
    # 각 분기의 첫 영업일 근처를 리밸런스 기준일로 사용
    quarter_starts = pd.date_range(START, END, freq="QS")
    print(f"리밸런스 기준일 {len(quarter_starts)}개 ({START} ~ {END}), 상위 {TOP_K}종목")

    snapshots = build_universe_snapshots(quarter_starts, top_k=TOP_K)

    print(f"\n수집 완료: {len(snapshots)}행")
    print(f"고유 종목 수(전 기간 합집합): {snapshots['ticker'].nunique()}")

    per_date = snapshots.groupby("rebalance_date").size()
    print(f"기준일당 종목 수: 최소 {per_date.min()}, 최대 {per_date.max()}")

    # 유니버스가 시간에 따라 얼마나 바뀌는지 (교체율)
    by_date = {d: set(g["ticker"]) for d, g in snapshots.groupby("rebalance_date")}
    dates = sorted(by_date)
    turnovers = [
        len(by_date[b] - by_date[a]) / len(by_date[a]) for a, b in zip(dates, dates[1:])
    ]
    print(f"분기별 유니버스 교체율 평균: {sum(turnovers) / len(turnovers):.1%}")


if __name__ == "__main__":
    main()
