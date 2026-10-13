# Metro Insight

**날씨·공휴일·과거 이용 패턴 기반 지하철 이용량 예측**

과거 지하철 이용 패턴에 시간·요일·공휴일·날씨 정보를 단계적으로 추가하며 특정 역·시간대의 승차 인원을 예측하고, 각 정보가 예측 성능에 얼마나 기여하는지 분석한다. 최종 모델로 사용자가 선택한 역·날짜·시간의 예상 이용량을 보여주는 서비스를 구현한다.

> **범위**: 승차 인원만 예측한다. 하차 인원과 혼잡도는 다루지 않는다.

| 모델 | 사용 데이터 |
| --- | --- |
| A | 과거 이용 패턴 |
| B | 과거 이용 패턴 + 시간/요일/공휴일 |
| C | 과거 이용 패턴 + 시간/요일/공휴일 + 날씨 (서비스 기본 모델) |

* 예측 대상: 승차 인원 (`boarding`)
* 데이터: 서울 지하철 1~8호선, 2024-01-01 ~ 2026-06-30 (912일)
* 데이터 분할: 시간 순서 기준 Train 70% / Validation 15% / Test 15%
* 평가 지표: MAE, RMSE

기획 내용은 [docs/project-plan.md](docs/project-plan.md), 데이터셋·출처는 [notebooks/데이터_출처.md](notebooks/데이터_출처.md) 를 참고한다.

## 진행 상황

- [x] 1~8호선 노선도 화면 및 역 선택 (PySide6 데스크톱 앱)
- [x] 데이터 확보 (지하철 이용량, 공휴일, 날씨)
- [x] 전처리 파이프라인 → `data/processed/{train,valid,test}.csv.gz`
- [ ] A/B/C 모델 학습 및 평가
- [ ] 예측 표시 화면

## 실행

[Anaconda](https://www.anaconda.com/download) 또는 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 필요. 모든 명령은 프로젝트 루트에서 실행한다.

```bash
conda env create -f environment.yml
conda activate metro-insight
metro-insight              # 또는 python -m metro_insight.app
```

Python 3.12와 `requirements.txt` 라이브러리가 설치되고, `metro_insight` 패키지가 editable 모드로 설치된다.

앱: 마우스 휠로 확대·축소, 드래그로 이동, 역 클릭으로 선택. 노선도 배치는 실제 역 좌표에서 자동 계산한다 (`app/schematic_layout.py`). 노선도 데이터를 다시 만들려면 `python -m metro_insight.app.subway_data` (인터넷 연결 필요).

## 데이터 파이프라인

학습 데이터는 `notebooks/` 의 스크립트로 만든다. 지하철 원본 CSV는 저장소에 없으므로, 원본 폴더 위치를 `METRO_DATA_DIR` 환경변수로 지정한다.

```bash
python notebooks/fetch_weather.py    # 기상청 ASOS(서울 108) → data/processed/weather_시간대별.csv
python notebooks/preprocess.py       # 원본 합치기·공휴일·날씨·과거값 → data/processed/{train,valid,test}.csv.gz
```

* `fetch_weather.py` — 기상청 API허브 인증키를 환경변수 `KMA_API_KEY` 로 넣는다. 원본(`data/raw/weather/`)이 이미 있으면 키 없이 변환·검증만 돌아간다.
* `preprocess.py` — 공휴일은 `holidays` 패키지, 날씨는 `fetch_weather.py` 결과를 쓴다.

## 현재 날씨 조회

[기상청 API허브](https://apihub.kma.go.kr) 지상관측 시간자료(서울 108)에서 최근 정시 날씨를 모델 Feature 형태로 가져온다. 인증키는 `.env.production` 의 `KMA_APIHUB_AUTH_KEY` 에 넣는다 (키 발급: "지상관측 > 종관기상관측(ASOS) > 시간자료" 활용신청).

```bash
cp .env.example .env.production   # KMA_APIHUB_AUTH_KEY 값 입력
python -m metro_insight.data.weather
# {"date": "2026-10-08", "hour": 9, "temperature": 15.9, "rainfall": 0, "snowfall": 0}
```

`rainfall`은 강수 여부(눈 포함), `snowfall`은 눈 여부다.

## 개발

```bash
pytest          # 테스트
ruff check .    # 코드 검사 (--fix 로 자동 수정)
```

## 프로젝트 구조

```text
metro-insight/
├── docs/
│   └── project-plan.md          # 프로젝트 기획서
├── notebooks/                   # 데이터 수집·전처리 스크립트
│   ├── fetch_weather.py         # 기상청 날씨 수집
│   ├── preprocess.py            # 원본 합치기 → train/valid/test.csv.gz
│   └── 데이터_출처.md            # 데이터셋 링크·기간
├── data/
│   ├── raw/weather/             # 기상청 원본 (월별)
│   └── processed/               # 전처리 결과 (train/valid/test + 샘플)
├── src/
│   └── metro_insight/           # 설치 가능한 패키지 (import metro_insight)
│       ├── app/                 # PySide6 데스크톱 앱
│       └── data/weather.py      # 서울 현재 날씨 조회
├── tests/                       # pytest 테스트
├── pyproject.toml               # 패키지 정의 및 의존성
├── requirements.txt             # 재현용 버전 고정 목록
└── environment.yml              # conda 환경 정의
```

## 데이터 출처

* 지하철 승하차 인원: [공공데이터포털](https://www.data.go.kr) 서울교통공사 역별 일별 시간대별 승하차인원 (15048032 · 15099330) · [서울 열린데이터광장](https://data.seoul.go.kr) OA-12921
* 날씨: [기상청 API허브](https://apihub.kma.go.kr) 지상관측 시간자료 (ASOS, 서울 108)
* 공휴일: [`holidays`](https://pypi.org/project/holidays/) 패키지 (한국)
* 노선도 역 좌표·순서: [OpenStreetMap](https://www.openstreetmap.org/copyright) © OpenStreetMap contributors, ODbL
