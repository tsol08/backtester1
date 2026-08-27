"""
분석 전반의 공통 설정.

유니버스를 여기에 티커 목록으로 박아두지 않는다. 고정 목록은 '오늘 살아남은 종목'이
되어 생존편향을 만들기 때문이다. 대신 매 시점의 실제 시가총액으로 유니버스를 구성한다
(src/data_loader/universe.py의 market_cap_universe_mask).
"""

# 수집된 패널의 실제 범위. KRX Open API는 2010년부터 제공한다.
DATA_START = "2010-01-01"

# 시점별 유니버스 크기 (분기마다 그 시점 시가총액 상위 N종목)
UNIVERSE_TOP_K = 200

# 팩터 평가 기본값
FORWARD_HORIZON = 20  # 거래일. IC와 분위 수익률을 재는 미래 구간
N_QUANTILES = 5

# 팩터 계산에 필요한 과거 구간. 평가 시작일부터 데이터를 로드하면 rolling(60) 등이
# 초반에 전부 NaN이 되어 관측이 통째로 날아간다.
WARMUP_DAYS = 400

INITIAL_CAPITAL = 100_000_000
