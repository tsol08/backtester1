"""후보군 전체의 DART 재무데이터를 배치로 수집해 캐시에 채워넣는다."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.dart_loader import load_fundamentals_bulk
from src.data_loader.universe import fetch_candidate_pool

START_YEAR = 2016  # TTM(4분기 누적) + 전년동기 성장률 계산에 여유분 필요
END_YEAR = 2024


def main() -> None:
    tickers = fetch_candidate_pool()["ticker"].tolist()
    print(f"대상: {len(tickers)}종목, {START_YEAR}~{END_YEAR}년")

    df = load_fundamentals_bulk(tickers, START_YEAR, END_YEAR)

    print(f"\n수집 완료: {len(df)}행, {df['ticker'].nunique()}종목")
    coverage = df.groupby("ticker").size().describe()
    print(f"종목당 분기 수 - 중앙값 {coverage['50%']:.0f}, 최소 {coverage['min']:.0f}, 최대 {coverage['max']:.0f}")

    for col in ["assets", "liabilities", "equity", "revenue", "operating_income", "net_income"]:
        if col in df.columns:
            print(f"  {col}: 결측 {df[col].isna().mean():.1%}")
        else:
            print(f"  {col}: 컬럼 없음")


if __name__ == "__main__":
    main()
