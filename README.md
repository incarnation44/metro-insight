# Metro Insight

**날씨·공휴일·과거 이용 패턴 기반 지하철 이용량 예측**

과거 지하철 이용 패턴에 시간·요일·공휴일·날씨 정보를 단계적으로 추가하며 특정 역·시간대의 승차 인원을 예측하고, 각 정보가 예측 성능에 얼마나 기여하는지 분석한다. 같은 정보를 쓰는 머신러닝 모델과 시계열 딥러닝 모델(LSTM, GRU)의 성능도 비교한다. 사용자가 역·날짜·시간을 선택하면 A~E 모델의 예상 이용량과 성능을 한 화면에서 비교해 보여준다.

| 모델 | 구분 | 사용 데이터 |
| --- | --- | --- |
| A | 머신러닝 | 과거 이용 패턴 |
| B | 머신러닝 | 과거 이용 패턴 + 시간/요일/공휴일 |
| C | 머신러닝 | 과거 이용 패턴 + 시간/요일/공휴일 + 날씨 |
| D | 딥러닝 (LSTM) | 과거 이용량 시계열 + 시간/요일/공휴일 + 날씨 |
| E | 딥러닝 (GRU) | D와 동일 |

* 예측 대상: 승차 인원(`boarding`)
* 데이터 분할: 시간 순서 기준 Train 70% / Validation 15% / Test 15%
* 평가 지표: MAE, RMSE

자세한 기획 내용(데이터 구성, Feature 설계, 데이터 누수 방지, 진행 단계 등)은 [docs/project-plan.md](docs/project-plan.md)를 참고한다.

## 진행 상황

- [x] 1~8호선 노선도 화면 및 역 선택 (PySide6 데스크톱 앱)
- [x] 역 정보 패널: 역 검색, 예측 조건 입력 (기본값: 지금 시각·서울 현재 날씨, 공휴일은 직접 체크)
- [ ] 데이터 확보 (지하철 이용량, 공휴일, 날씨)
- [ ] 전처리 및 Feature Engineering
- [ ] A/B/C 모델 학습 (머신러닝)
- [ ] D/E 모델 학습 (딥러닝 LSTM, GRU)
- [x] 모델 평가 지표 (MAE, RMSE) 및 분석 화면: 예측하기 → A~E 예상 이용량·MAE·RMSE (모델 연결 전에는 `-`)

## 빠른 시작

[Anaconda](https://www.anaconda.com/download) 또는 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)가 필요하다. 모든 명령은 프로젝트 루트에서 실행한다.

```bash
conda env create -f environment.yml
conda activate metro-insight
metro-insight              # 또는 python -m metro_insight.app
```

## 설치

```bash
conda env create -f environment.yml
conda activate metro-insight
```

Python 3.12와 `requirements.txt`의 라이브러리가 설치되고, `metro_insight` 패키지가 editable 모드로 설치된다. `src/metro_insight/` 코드 수정은 재설치 없이 바로 반영된다.

| 파일 | 역할 |
| --- | --- |
| `pyproject.toml` | 패키지 정의 및 의존성 범위 (기준) |
| `requirements.txt` | 재현용 버전 고정 목록 |
| `environment.yml` | conda 환경 정의 (위 두 파일을 설치) |

## 실행

### 앱 실행

```bash
metro-insight              # 또는 python -m metro_insight.app
```

PySide6 데스크톱 창에 1~8호선 노선도가 열린다.

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

### 노선도 데이터 갱신

노선도는 OpenStreetMap에서 만든 `src/metro_insight/app/resources/subway_network.json`을 사용한다. 개통·역 이름 변경 등으로 갱신이 필요하면 다시 생성한다 (인터넷 연결 필요).

```bash
python -m metro_insight.app.subway_data
```

### 서울 현재 날씨 조회

[기상청 API허브](https://apihub.kma.go.kr) 지상관측 시간자료(서울 108번 관측소)에서 가장 최근 정시의 날씨를 모델 Feature 형태로 가져온다. 인증키 발급 후 "지상관측 > 종관기상관측(ASOS) > 시간자료" 활용신청이 필요하다.

```bash
cp .env.example .env.production   # KMA_APIHUB_AUTH_KEY 값 입력
python -m metro_insight.data.weather
# {"date": "2026-10-08", "hour": 9, "temperature": 15.9, "rainfall": 0, "snowfall": 0}
```

`rainfall`은 강수 여부(눈 포함), `snowfall`은 눈 여부다. 판단 규칙은 `data/weather.py` 상단 설명을 참고한다.

## 개발

### 테스트 및 코드 검사

```bash
pytest
ruff check .        # --fix 로 자동 수정
```

### 라이브러리 추가·변경

1. `pyproject.toml`의 `dependencies`(개발용은 `[dev]`)에 추가
2. `pip install -e ".[dev]"`로 설치
3. 설치된 버전을 `requirements.txt`에 `==`로 고정

변경 사항을 받은 쪽에서는 아래 명령으로 환경을 갱신한다.

```bash
conda env update -f environment.yml --prune
```

### 환경 삭제

```bash
conda deactivate
conda env remove -n metro-insight
```

## 프로젝트 구조

```text
metro-insight/
├── docs/
│   └── project-plan.md          # 프로젝트 기획서
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
├── .env.example                 # 인증키 설정 예시 (.env.production 으로 복사)
├── pyproject.toml               # 패키지 정의 및 의존성
├── requirements.txt             # 버전 고정 목록
└── environment.yml              # conda 환경 정의
```

## 문제 해결

| 증상 | 해결 |
| --- | --- |
| WSL2에서 창이 뜨지 않음 | WSLg가 필요하다 (Windows 11 기본 포함) |
| 한글이 깨짐 | `sudo apt install fonts-nanum` |
| `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` | `sudo apt install libxcb-cursor0` |

## 데이터 출처

* 노선도 역 좌표·순서: [OpenStreetMap](https://www.openstreetmap.org/copyright) © OpenStreetMap contributors, ODbL
* 날씨: [기상청 API허브](https://apihub.kma.go.kr) 지상관측 시간자료
