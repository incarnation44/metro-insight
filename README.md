# Metro Insight

**날씨·공휴일·과거 이용 패턴 기반 지하철 이용량 예측**

과거 지하철 이용 패턴에 시간·요일·공휴일·날씨 정보를 단계적으로 추가하며 특정 역·시간대의 승차 인원을 예측하고, 각 정보가 예측 성능에 얼마나 기여하는지 분석한다. 같은 정보를 쓰는 머신러닝 모델과 시계열 딥러닝 모델(LSTM, GRU)의 성능도 비교한다. 사용자가 역·날짜·시간을 선택하면 A~E 모델의 예상 이용량과 성능을 한 화면에서 비교해 보여준다.

> **범위**: 승차 인원만 예측한다. 하차 인원과 혼잡도는 다루지 않는다.

| 모델 | 구분 | 사용 데이터 |
| --- | --- | --- |
| A | 머신러닝 | 과거 이용 패턴 |
| B | 머신러닝 | 과거 이용 패턴 + 시간/요일/공휴일 |
| C | 머신러닝 | 과거 이용 패턴 + 시간/요일/공휴일 + 날씨 |
| D | 딥러닝 (LSTM) | 과거 이용량 시계열 + 시간/요일/공휴일 + 날씨 |
| E | 딥러닝 (GRU) | D와 동일 |

* 예측 대상: 승차 인원 (`boarding`)
* 데이터: 서울 지하철 1~8호선, 2024-01-01 ~ 2026-06-30 (912일)
* 데이터 분할: 시간 순서 기준 Train 70% / Validation 15% / Test 15%
* 평가 지표: MAE, RMSE

기획 내용은 [docs/project-plan.md](docs/project-plan.md), 데이터셋·출처는 [notebooks/데이터_출처.md](notebooks/데이터_출처.md) 를 참고한다.

## 진행 상황

- [x] 1~8호선 노선도 화면 및 역 선택 (PySide6 데스크톱 앱)
- [x] 역 정보 패널: 역 검색, 예측 조건 입력 (기본값: 지금 시각·서울 현재 날씨, 공휴일은 직접 체크)
- [x] 데이터 확보 (지하철 이용량, 공휴일, 날씨)
- [x] 전처리 파이프라인 → `data/processed/{train,valid,test}.csv.gz`
- [ ] A/B/C 모델 학습 (머신러닝)
- [ ] D/E 모델 학습 (딥러닝 LSTM, GRU)
- [x] 예측 표시 화면
- [x] 모델 평가 지표 (MAE, RMSE) 및 분석 화면: 예측하기 → A~E 예상 이용량·MAE·RMSE (모델 연결 전에는 `-`)

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

* 마우스 휠로 확대·축소, 드래그로 이동, 역 클릭으로 선택
* 확대할수록 더 많은 역 이름이 더 크게 표시된다
* 공식 노선도처럼 가로·세로·45° 선으로 그리며, 배치는 실제 역 좌표에서 자동 계산한다 (`app/schematic_layout.py`)
* 역을 선택하면 오른쪽에서 역 정보 패널이 열린다. 헤더의 "역 정보" 버튼으로 열고 닫는다
  * 역 이름 검색: 입력한 글자로 시작하는 역을 찾는다 (`강` → 강남, 강남구청, ...). `Ctrl+F`로 바로 검색, `↓`·`Enter`로 선택
  * 예측 조건: 날짜, 호선, 역, 시, 요일, 주말·공휴일 여부, 기온, 강수·눈 여부
  * 기본값은 지금이다. 요일·주말 여부는 날짜에서 계산하고, 날씨는 서울 현재 날씨를 받아 채운다 (아래 인증키 필요, 없으면 직접 입력). "현재로" 버튼으로 되돌린다
  * 공휴일 여부는 아직 계산하지 않는다. 기본값은 체크 해제이며 직접 체크한다
  * "예측하기"를 누르면 아래에 조건, A~E 모델 예상 승차 인원, Test 데이터 기준 MAE·RMSE가 나온다. MAE가 가장 낮은 모델을 굵게 표시한다
  * 학습한 모델은 `model/predictor.py`의 `Predictor`(`predict`, `scores`)를 구현해 `StationPanel(predictor=...)`로 연결한다. 연결 전에는 값 대신 `-`와 안내 문구가 나온다

### WSL에서 한글 입력

WSL(WSLg) 창에는 Windows의 한/영 입력기가 적용되지 않아, 역 검색창에 한글이 입력되지 않는다. WSL 안에 ibus 한글 입력기를 설치한다 (Windows에서 직접 실행할 때는 필요 없다).

```bash
sudo apt install -y ibus ibus-hangul
gsettings set org.freedesktop.ibus.general preload-engines "['hangul']"   # 한 번만
gsettings set org.freedesktop.ibus.general engines-order "['hangul']"

# WSL을 켤 때마다 입력기를 실행한다 (~/.bashrc 에 넣어 두면 편하다)
pgrep -x ibus-daemon > /dev/null || ibus-daemon -drx
```

* ibus가 설치되어 있으면 앱이 `QT_IM_MODULE=ibus`를 자동으로 지정한다 (이미 지정한 값이 있으면 그대로 둔다)
* 한/영 전환: `Shift+Space` (또는 한/영 키). Windows 입력기는 영문 상태로 둔다
* 그래도 안 되면 `ibus engine` 으로 현재 엔진이 `hangul` 인지 확인하고, 아니면 `ibus engine hangul` 을 실행한다

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
│       │   ├── __main__.py      # 실행 진입점
│       │   ├── main_window.py   # 메인 윈도우 (헤더 + 노선도)
│       │   ├── station_map.py   # 노선도 그리기, 확대·축소, 역 선택
│       │   ├── map_overlays.py  # 노선도 위 카드 (범례, 선택한 역, 확대 버튼)
│       │   ├── schematic_layout.py  # 노선도 배치 계산 (가로·세로·45° 선)
│       │   ├── subway_data.py   # 역·노선 데이터 로드 / OpenStreetMap에서 생성
│       │   ├── resources/       # subway_network.json
│       │   └── styles/          # QSS 스타일, 아이콘
│       ├── data/                # 모델용 외부 데이터
│       │   └── weather.py       # 서울 날씨 (기상청 API허브)
│       └── model/               # 이용량 예측 모델
│           ├── predictor.py     # 예측 조건(PredictionInput)과 화면-모델 인터페이스(Predictor)
│           └── evaluation.py    # 평가 지표 (MAE, RMSE)
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
