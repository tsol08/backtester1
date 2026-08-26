"""후보군 전체의 가격 데이터를 수집해 캐시에 채워넣는다 (오래 걸리므로 별도 실행)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.krx_loader import load_ohlcv
from src.data_loader.universe import fetch_candidate_pool

START = "2016-01-01"  # TTM/성장률 계산 여유분 포함
END = "2024-12-31"


def main() -> None:
    candidates = fetch_candidate_pool()
    total = len(candidates)
    ok = 0
    failed = []

    for i, row in candidates.iterrows():
        ticker = row["ticker"]
        try:
            df = load_ohlcv(ticker, START, END)
            if len(df) > 0:
                ok += 1
            else:
                failed.append((ticker, row["name"], "빈 데이터"))
        except Exception as exc:
            failed.append((ticker, row["name"], type(exc).__name__))

        if (i + 1) % 25 == 0:
            print(f"  진행 {i + 1}/{total} (성공 {ok}, 실패 {len(failed)})", flush=True)

    print(f"\n완료: 성공 {ok} / 전체 {total}")
    if failed:
        print(f"실패 {len(failed)}건:")
        for ticker, name, reason in failed[:20]:
            print(f"  {ticker} {name}: {reason}")


if __name__ == "__main__":
    main()
