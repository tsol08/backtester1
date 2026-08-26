# backtester1 — 다요소 퀀트 백테스터 (한국 주식)

가격 추세 하나만 보던 이전 백테스터를 넘어, 기술적/변동성 지표를 비롯한 여러 정보를
종합해 시그널을 만들고 이후 ML 모델로 확장하는 것을 목표로 하는 백테스트 프로젝트입니다.

## 원칙

- **과적합 방지**: 인샘플/아웃오브샘플 기간 분리, ML은 워크포워드 검증, 매 단계 look-ahead
  bias(미래정보 누수) 여부 확인
- **현실적 비용**: 수수료 + 슬리피지 + 시장충격비용(거래대금/평균거래대금 비율 기반)
- **실험 기록**: `experiments/log.md`에 시도한 것과 채택/기각 이유를 기록
- 데이터 소스: [pykrx](https://github.com/sharebook-kr/pykrx)

## 로드맵

1. **Phase 1**: 데이터 파이프라인 + 기본 피처(이동평균/모멘텀/변동성) + 비용모델 반영 백테스트 엔진
2. **Phase 2**: 다중 피처 결합(규칙기반 스코어링/랭킹) + 포트폴리오 구성
3. **Phase 3**: ML 시그널 (walk-forward validation)
4. **Phase 4**: 재무/섹터/거시 데이터 통합
5. **Phase 5**: 실험 로그 체계화 + 최종 리포팅/대시보드

## 폴더 구조

```
data/{raw,processed}/     # raw/processed는 재생성 가능한 캐시 — git에는 포함 안 함
src/
  data_loader/             # pykrx 래퍼, 캐싱
  features/                # 기술적/변동성 지표
  costs/                   # 수수료/슬리피지/시장충격 비용 모델
  signals/{rules,ml}/      # 규칙기반 / ML 시그널
  engine/                  # 백테스트 엔진
  portfolio/               # 포트폴리오 결합
  reporting/                # 성과 지표/시각화
experiments/log.md          # 실험 로그
tests/                       # look-ahead bias 체크 등
config/                       # 백테스트 설정
```
