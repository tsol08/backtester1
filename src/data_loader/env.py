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
    pykrx의 stock 모듈을 안전하게 import한다.

    pykrx는 KRX_ID/KRX_PW 환경변수가 있으면 **import 시점에 즉시 로그인을 시도**하고,
    실패 시 예외를 그대로 던진다. 그런데 KRX는 짧은 시간에 요청이 몰리면 "자동화된
    비정상 대량조회"로 보고 계정을 일시 차단하는데(이용약관에 명시), 차단된 상태에서는
    로그인 자체가 매번 실패해서 단순히 import만 해도 프로그램이 죽어버린다(캐시된
    데이터를 읽기만 하려던 경우도 포함해서).

    그래서 자격증명으로 한 번 시도해보고 실패하면, 환경변수를 지우고 익명 모드로
    재시도한다. 어차피 OHLCV 같은 기본 데이터는 로그인 없이도 받아진다(네이버 폴백).
    로그인이 실제로 필요한 함수(get_market_fundamental 등)를 호출하면 그때 가서
    명확한 에러가 나는 게, 여기서 통째로 죽는 것보다 낫다.
    """
    ensure_krx_credentials()
    try:
        from pykrx import stock

        return stock
    except Exception:
        os.environ.pop("KRX_ID", None)
        os.environ.pop("KRX_PW", None)
        import sys

        for name in list(sys.modules):
            if name == "pykrx" or name.startswith("pykrx."):
                del sys.modules[name]

        from pykrx import stock

        return stock


def ensure_krx_credentials() -> bool:
    """
    pykrx는 KRX 웹사이트 로그인이 있어야 시가총액/PER/PBR/수급/공매도/지수 API를 쓸 수 있고,
    로그인 정보를 KRX_ID / KRX_PW 환경변수에서 읽는다. .env에 저장된 값을 환경변수로 옮겨준다.

    pykrx 모듈을 import 하기 전에 호출해야 한다.
    """
    env = load_env()
    found = True
    for name in ("KRX_ID", "KRX_PW"):
        value = os.environ.get(name) or env.get(name)
        if value:
            os.environ[name] = value
        else:
            found = False
    return found
