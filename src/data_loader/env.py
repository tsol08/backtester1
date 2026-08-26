"""
API 키 로딩.

키는 프로젝트 루트의 *.env 파일에 두고 git에는 절대 올리지 않는다(.gitignore 처리).
키 값 자체는 로그/에러메시지에 찍지 않는다.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> dict[str, str]:
    """프로젝트 루트의 모든 *.env 파일에서 KEY=VALUE를 읽어온다."""
    env: dict[str, str] = {}
    for path in sorted(PROJECT_ROOT.glob("*.env")):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def get_api_key(name: str) -> str:
    """환경변수 -> *.env 순으로 키를 찾는다. 없으면 안내 메시지와 함께 에러."""
    value = os.environ.get(name) or load_env().get(name)
    if not value:
        raise RuntimeError(
            f"{name}를 찾을 수 없습니다. 프로젝트 루트의 .env 파일에 "
            f"'{name}=발급받은키' 형식으로 추가해주세요."
        )
    return value


def import_pykrx_stock():
    """
    pykrx의 stock 모듈을 익명 모드로 import한다.

    설치된 pykrx는 `pykrx.website.comm.webio` 모듈이 import되는 순간
    (즉 `from pykrx import stock`만 해도) KRX_ID/KRX_PW 환경변수가 있으면
    **무조건 실제 KRX 로그인 POST 요청을 보낸다** (webio.py 상단의
    `_session = build_krx_session()`). 이건 OHLCV처럼 로그인이 전혀
    필요없는 데이터를 캐시에서 읽기만 할 때도 마찬가지로 일어난다.

    문제는 이 로그인이 실패해도 예외를 던지지 않고 내부에서 메시지만
    찍고 넘어간다는 것이다. 그래서 "로그인 실패 시 익명 재시도"로는
    이 자동 로그인 요청 자체를 막을 수 없다 — 예외가 안 나니 재시도
    분기를 탈 일이 없다. 즉 KRX_ID/KRX_PW가 환경변수에 있는 한, pykrx를
    import할 때마다 KRX 서버로 로그인 요청이 계속 나간다.

    KRX 계정이 "자동화된 비정상 대량조회"로 일시 차단된 상태이므로,
    이런 반복적인 자동 로그인 시도 자체가 문제다. 그래서 아예 환경변수를
    비우고 나서 import한다 — pykrx가 로그인을 시도할 대상 자체가 없게
    만드는 것이다. OHLCV 같은 기본 데이터는 로그인 없이도 네이버 폴백으로
    받아진다. 로그인이 실제로 필요한 함수(get_market_fundamental 등)는
    지금 쓰지 않는다 (밴 해제 또는 KRX Open API 승인 전까지 보류).
    """
    os.environ.pop("KRX_ID", None)
    os.environ.pop("KRX_PW", None)
    import sys

    for name in list(sys.modules):
        if name == "pykrx" or name.startswith("pykrx."):
            del sys.modules[name]

    from pykrx import stock

    return stock
