"""공시 원문 파서가 사유별 거래를 정확히 뽑는지 검증한다."""
from __future__ import annotations

import pandas as pd

from src.data_loader.dart_filing import parse_detail_rows

# 실제 공시 원문(삼성전자 20240828000001)에서 가져온 구조.
BUY_ROW = """
<TR ACOPY="Y" ADELETE="N">
<TU ALIGN="CENTER" WIDTH="75" AUNIT="RPT_RSN" AUNITVALUE="01">장내매수(+)</TU>
<TU ALIGN="CENTER" WIDTH="112" AUNIT="MDF_DM" AUNITVALUE="20240820">2024년 08월 20일</TU>
<TU ALIGN="CENTER" WIDTH="115" AUNIT="STR_KND" AUNITVALUE="1">보통주</TU>
<TE ALIGN="RIGHT" WIDTH="74" ACODE="BFR_STK_CNT">20,000</TE>
<TE ALIGN="RIGHT" WIDTH="73" ACODE="MDF_STK_CNT">1,000</TE>
<TE ALIGN="RIGHT" WIDTH="73" ACODE="AFR_STK_CNT">21,000</TE>
<TE ALIGN="RIGHT" WIDTH="121" ACODE="ACI_AMT2">79,100</TE>
</TR>
"""

SELL_ROW = """
<TR ACOPY="Y" ADELETE="N">
<TU ALIGN="CENTER" AUNIT="RPT_RSN" AUNITVALUE="02">장내매도(-)</TU>
<TU ALIGN="CENTER" AUNIT="MDF_DM" AUNITVALUE="20250310">2025년 03월 10일</TU>
<TE ALIGN="RIGHT" ACODE="MDF_STK_CNT">5,000</TE>
<TE ALIGN="RIGHT" ACODE="ACI_AMT2">55,300</TE>
</TR>
"""

NEW_APPOINTMENT_ROW = """
<TR ACOPY="Y" ADELETE="N">
<TU ALIGN="CENTER" AUNIT="RPT_RSN" AUNITVALUE="31">신규선임(+)</TU>
<TU ALIGN="CENTER" AUNIT="MDF_DM" AUNITVALUE="20250320">2025년 03월 20일</TU>
<TE ALIGN="RIGHT" ACODE="MDF_STK_CNT">300,000</TE>
<TE ALIGN="RIGHT" ACODE="ACI_AMT2">-</TE>
</TR>
"""

TOTAL_ROW = """
<TR ACOPY="N" ADELETE="N">
<TD COLSPAN="3" ALIGN="CENTER">합  계</TD>
<TE ALIGN="RIGHT" ACODE="MDF_STK_CNT">6,000</TE>
</TR>
"""


def test_buy_row_is_positive():
    result = parse_detail_rows(BUY_ROW)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["reason_code"] == "01"
    assert row["shares_change"] == 1000.0
    assert row["price"] == 79100.0
    assert row["change_date"] == pd.Timestamp("2024-08-20")
    assert row["is_market_trade"]


def test_sell_row_becomes_negative():
    """수량 칸에는 부호가 없고 사유 라벨의 (-)로만 방향을 알 수 있다."""
    result = parse_detail_rows(SELL_ROW)

    assert result.iloc[0]["shares_change"] == -5000.0
    assert result.iloc[0]["is_market_trade"]


def test_new_appointment_is_not_a_trade():
    """
    신규선임은 새 임원이 원래 갖고 있던 주식을 처음 신고한 것이다.
    보유량은 늘지만 '샀다'가 아니므로 매매로 분류되면 안 된다.
    """
    result = parse_detail_rows(NEW_APPOINTMENT_ROW)

    assert result.iloc[0]["shares_change"] == 300_000.0
    assert not result.iloc[0]["is_market_trade"]


def test_total_row_is_skipped():
    """합계 행에는 보고사유가 없으므로 거래로 세면 안 된다."""
    assert parse_detail_rows(TOTAL_ROW).empty


def test_multiple_rows_are_all_parsed():
    document = BUY_ROW + SELL_ROW + NEW_APPOINTMENT_ROW + TOTAL_ROW

    result = parse_detail_rows(document)

    assert len(result) == 3
    assert result["shares_change"].tolist() == [1000.0, -5000.0, 300_000.0]
    # 매매만 추리면 순매수는 1,000 - 5,000 = -4,000주
    trades = result[result["is_market_trade"]]
    assert trades["shares_change"].sum() == -4000.0


def test_missing_price_does_not_break_row():
    """단가가 '-'인 행도 수량은 살려야 한다."""
    result = parse_detail_rows(NEW_APPOINTMENT_ROW)

    assert pd.isna(result.iloc[0]["price"])
    assert result.iloc[0]["shares_change"] == 300_000.0
