"""
도구 검증: 우리 분위 분석기가 '답이 알려진' 팩터를 검출하는가.

이 프로젝트는 네 개 팩터를 기각했다. 검정력 분석으로 도구가 IC 0.05를 100%
검출한다는 것은 확인했지만, 그건 우리가 만든 인공 신호에 대한 것이다. 실제 시장에
존재하는 것으로 수십 년간 문서화된 팩터를 잡아내는지는 별개 문제다.

모멘텀·규모·가치는 미국 데이터에서 1927년부터 관측된 프리미엄이 있다. 우리
quantile_analysis를 그대로 통과시켜 다음을 본다:

- 방향과 단조성이 문헌과 일치하는가
- 롱온리 초과가 유의하게 나오는가
- **한국 표본(관측 61개)으로 잘랐을 때도 여전히 보이는가**

마지막이 핵심이다. 100년치로는 보이는데 5년치로는 안 보인다면, 한국에서의 기각은
"신호가 없다"가 아니라 "이 표본으로는 못 본다"였을 수 있다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader.french_loader import load_quantile_returns
from src.research.quantile_analysis import long_only_edge, monotonicity, summarize_quantiles

PERIODS_PER_YEAR = 12

# 문헌상 알려진 방향 (상위 분위가 유리한가)
EXPECTED = {
    "momentum": ("모멘텀", "상위(과거 상승) 우세"),
    "size": ("규모", "하위(소형주) 우세"),
    "value": ("가치", "상위(저PBR) 우세"),
}

# 한국 인샘플과 같은 관측 수. 5년치 월별이면 60개.
KOREA_SAMPLE = 61


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())


def report(label: str, qr: pd.DataFrame) -> dict:
    edge = long_only_edge(qr, PERIODS_PER_YEAR)
    return {
        "구간": label,
        "관측(개월)": len(qr),
        "상위분위(연)": (1 + qr[qr.columns[-1]].mean()) ** PERIODS_PER_YEAR - 1,
        "하위분위(연)": (1 + qr[qr.columns[0]].mean()) ** PERIODS_PER_YEAR - 1,
        "롱온리초과(연)": edge["상위분위 초과(연율화)"],
        "t-stat": edge["상위분위 초과 t-stat"],
        "단조성": monotonicity(qr),
    }


def main() -> None:
    print("=" * 78)
    print("1) 전체 기간 - 알려진 프리미엄이 검출되는가")
    print("=" * 78)

    panels = {}
    rows = []
    for name, (korean, expectation) in EXPECTED.items():
        qr = load_quantile_returns(name)
        panels[name] = qr
        row = report(f"{korean} (전체)", qr)
        row["문헌상 기대"] = expectation
        rows.append(row)

    table = pd.DataFrame(rows).set_index("구간")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table.to_string())

    print("\n" + "=" * 78)
    print(f"2) 한국 표본 크기({KOREA_SAMPLE}개월)로 잘라도 보이는가")
    print("=" * 78)
    print("전체 기간을 겹치지 않는 구간으로 나눠, 각 구간에서 검출되는 비율을 센다.")

    rows = []
    for name, (korean, _) in EXPECTED.items():
        qr = panels[name]
        n_windows = len(qr) // KOREA_SAMPLE

        detected_5pct = 0
        detected_bonf = 0
        t_values = []
        for i in range(n_windows):
            window = qr.iloc[i * KOREA_SAMPLE : (i + 1) * KOREA_SAMPLE]
            edge = long_only_edge(window, PERIODS_PER_YEAR)
            t = edge["상위분위 초과 t-stat"]
            # 규모는 하위 분위가 유리하므로 부호를 뒤집어 평가
            if name == "size":
                t = -t
            t_values.append(t)
            detected_5pct += t > 1.96
            detected_bonf += t > 2.87

        rows.append(
            {
                "팩터": korean,
                "구간수": n_windows,
                "t 중앙값": float(np.median(t_values)),
                "검출률(t>1.96)": detected_5pct / n_windows,
                "검출률(t>2.87)": detected_bonf / n_windows,
            }
        )

    show("[5년 단위 구간별 검출률]", pd.DataFrame(rows).set_index("팩터"))

    print(
        "\n읽는 법: 100년치로는 뚜렷한 팩터가 5년 구간에서는 얼마나 자주 보이는가."
        "\n이 비율이 낮다면, 한국 5년 표본에서 아무것도 못 찾은 것은 시장의 문제가"
        "\n아니라 표본 길이의 문제다."
    )


if __name__ == "__main__":
    main()
