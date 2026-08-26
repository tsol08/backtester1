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
