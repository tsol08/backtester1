"""
Ken French 데이터 라이브러리: 미국 장기 분위 포트폴리오 수익률.

왜 필요한가. 우리 검정력 분석 결과 이 프로젝트의 검출 하한은 IC 0.03~0.04였다
(비겹침 관측 61개 기준). 효율적 시장에서 실제로 존재한다고 알려진 팩터들이 대개
IC 0.02~0.05이므로, **약한 쪽 절반은 애초에 볼 수 없는 조건**에서 작업해온 것이다.
관측 수를 늘리는 것이 유일한 해법이고, 미국 데이터는 1927년부터 있다(월별 1,190개).

여기서 하려는 것은 새 알파 찾기가 아니라 **도구 검증**이다. 모멘텀·규모·가치는
수십 년간 문서화된 프리미엄이 있다. 우리 분위 분석기가 그것들을 제대로 검출하면
도구가 정상이라는 독립적 증거가 되고, 검출하지 못하면 한국에서의 기각 판정들을
다시 봐야 한다.

파일 형식: 헤더 주석 뒤에 '월별 가치가중' -> '월별 동일가중' -> '연별' 순으로
여러 표가 이어 붙어 있다. 결측은 -99.99 / -999로 표시된다.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRENCH_DIR = PROJECT_ROOT / "data" / "raw" / "french"

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# 라벨 -> (파일명, 최저분위가 팩터의 '낮은' 쪽인가)
DATASETS = {
    "momentum": "10_Portfolios_Prior_12_2_CSV.zip",
    "size": "Portfolios_Formed_on_ME_CSV.zip",
    "value": "Portfolios_Formed_on_BE-ME_CSV.zip",
}

MISSING_VALUES = (-99.99, -999.0)


def _download(name: str) -> str:
    """원문 CSV 텍스트. 한 번 받으면 디스크에 캐싱한다."""
    path = FRENCH_DIR / f"{name}.csv"
    if path.exists():
        return path.read_text(encoding="latin-1")

    response = requests.get(BASE_URL + DATASETS[name], timeout=60)
    response.raise_for_status()

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    text = archive.read(archive.namelist()[0]).decode("latin-1")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="latin-1")
    return text


def _decile_columns(header: list[str]) -> list[int]:
    """
    10분위 컬럼의 위치.

    규모/가치 파일에는 10분위 외에 3분위(Lo 30/Med 40/Hi 30)와 5분위(Lo 20/Qnt 2...)
    표가 같은 행에 함께 들어있다. 컬럼 개수로 자르면 어느 표를 집었는지 알 수 없으므로
    이름으로 고른다: 'Lo 10' ~ 'Hi 10'.
    """
    if "Lo 10" in header and "Hi 10" in header:
        start, end = header.index("Lo 10"), header.index("Hi 10")
        return list(range(start, end + 1))
    return list(range(len(header)))  # 모멘텀 파일처럼 10분위만 있는 경우


def _split_sections(text: str) -> list[tuple[str, list[str], list[str]]]:
    """
    (섹션 제목, 데이터 줄들) 목록.

    섹션 제목은 숫자로 시작하지 않는 줄이고, 그 다음 줄이 컬럼 헤더다.
    파일 앞부분의 설명 문단도 같은 조건에 걸리므로, 데이터가 실제로 뒤따르는
    것만 남긴다.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str], list[str]]] = []

    current_title: str | None = None
    header: list[str] = []
    rows: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped[0].isdigit():
            if current_title is not None:
                rows.append(stripped)
        elif stripped.startswith(","):
            header = [c.strip() for c in stripped.split(",")[1:]]
        else:
            if current_title is not None and rows:
                sections.append((current_title, header, rows))
            current_title, rows = stripped, []

    if current_title is not None and rows:
        sections.append((current_title, header, rows))

    return sections


def load_quantile_returns(name: str, weighting: str = "value") -> pd.DataFrame:
    """
    월별 10분위 수익률 (행=월말, 열=분위 0~9, 값=소수 수익률).

    분위 0이 팩터값이 가장 낮은 쪽, 9가 가장 높은 쪽이다. 우리 quantile_analysis가
    쓰는 규약(마지막 열이 상위 분위)과 같다.
    """
    text = _download(name)
    sections = _split_sections(text)

    keyword = "Value Weight" if weighting == "value" else "Equal Weight"
    monthly = [
        (header, rows)
        for title, header, rows in sections
        if keyword in title and "Monthly" in title and "Annual" not in title
    ]
    if not monthly:
        raise ValueError(f"{name}: '{keyword} ... Monthly' 섹션을 찾지 못했습니다")

    header, rows = monthly[0]
    keep = _decile_columns(header)

    records = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        period = parts[0]
        if len(period) != 6:  # 연별 표(4자리)가 섞여 들어오는 것 방지
            continue
        values = [float(parts[1 + i]) for i in keep]
        records.append([period] + values)

    frame = pd.DataFrame(records)
    frame[0] = pd.to_datetime(frame[0], format="%Y%m") + pd.offsets.MonthEnd(0)
    frame = frame.set_index(0)
    frame.index.name = "date"
    frame.columns = range(len(frame.columns))

    # 결측 표시값을 NaN으로, 퍼센트를 소수로
    frame = frame.mask(frame.isin(MISSING_VALUES))
    return frame / 100.0
