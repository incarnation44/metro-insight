# Metro Insight 다른 기기 분배 · 세팅 및 AI 전달 지시서

> 이 문서는 `metro-insight` 프로젝트 전체를 다른 PC(Windows / WSL / macOS / Linux)로 옮기거나, 다른 AI에게 초기 세팅·검증을 맡길 때 그대로 전달하는 경로 독립 가이드입니다.

---

## 1. 다른 AI에게 그대로 복붙할 지시서

아래 블록을 새 기기의 AI(또는 터미널 세션)에 그대로 붙여넣어 실행을 지시하세요. 압축을 푼 폴더 경로만 알려주면 됩니다.

```text
내가 전달한 `metro-insight` 프로젝트 폴더를 이 기기 환경에 맞게 세팅하고 동작을 검증해줘.

[중요 제약]
1. 이전 기기의 절대 경로(`C:\metro-insight`, `C:\Users\ildoc\...`, `D:\...`)를 가정하지 말고, 현재 프로젝트 루트(`REPO_ROOT`)를 기준으로 상대 경로만 사용해.
2. `.env.production` 같은 비밀 파일은 보안상 배포 ZIP에서 제외했으니 새로 읽거나 생성을 강요하지 말고, `.env.example`을 안내해. (실시간 날씨가 없으면 앱이 자동으로 평년값 대체 모드로 동작한다.)
3. `data/`, `models/` 안에 포함된 `.gitignore` 대상 대용량 파일들은 삭제하거나 이동하지 말고 그대로 제자리에 유지해.

[수행 순서]
1. 현재 폴더(`REPO_ROOT`)에서 `git status`와 핵심 비추적 파일 존재 여부를 먼저 확인해:
   - `data/raw/subway/*.csv` (3개)
   - `data/processed/train.csv.gz`, `valid.csv.gz`, `test.csv.gz`
   - `models/cal_A.joblib`, `cal_B.joblib`, `cal_C.joblib`, `sequence_D.pt`, `sequence_E.pt`, `cal_profiles.npz`, `cal_profiles_eval.npz`, `boarding_history.npz`
   - `models/_exp/` (9개 실험 디렉터리)
2. 이 기기의 Python / Conda 실행 경로를 직접 탐색해 (`conda`, `anaconda3`, `miniconda3`, 또는 Python 3.12 `venv`).
   - Conda가 있으면 `environment.yml`로 `metro-insight` 환경을 만들거나(`conda env create -f environment.yml`), 이미 있으면 갱신 후 `pip install -e . --no-deps`가 살아있는지 확인해.
   - Conda가 없으면 Python 3.12 가상환경(`.venv`)을 만들고 `pip install -r requirements.txt` 후 `pip install -e .`를 설치해.
   - `pyproject.toml`의 패키지 데이터(`layout_cache.json`, `tokens.json`, `modern_gray.qss`, `subway_network.json`)가 import 경로에서 읽히는지 확인해.
3. 아래 4가지 검증을 모두 실행하고 실제 출력 결과를 보고해:
   - `python -m pytest -q` (기대: 전체 통과, 1,103 passed)
   - `python -m ruff check src tests` (기대: All checks passed)
   - `python notebooks/verify_manifest.py` (기대: 전부 통과: 9개 실행)
   - 오프스크린 앱 기동 확인 (`QT_QPA_PLATFORM=offscreen python -c "from PySide6.QtWidgets import QApplication; from metro_insight.app.main_window import MainWindow; app = QApplication([]); w = MainWindow(); print(w.windowTitle())"`)
4. 이 기기에서 사용자가 앱을 바로 실행할 수 있는 정확한 한 줄 실행 명령(가상환경 절대 경로 포함)을 안내해.
```

---

## 2. 배포 ZIP 구조 및 `.gitignore` 파일 분배 규칙

단일 통합 ZIP(`metro-insight_전체프로젝트_2026-11-15.zip`)을 원하는 위치에 풀면 `metro-insight/` 루트 아래에 Git 저장소와 비추적 필수 파일이 한 번에 복원됩니다.

### 2-1. ZIP에 포함된 `.gitignore` 비추적 파일 목록과 위치

만약 `git clone`으로 코드만 먼저 받은 뒤 별도로 파일을 복사해야 한다면, 아래 상대 경로를 그대로 맞춰 배치해야 합니다.

| 구분 | 상대 경로 (`REPO_ROOT` 기준) | 용도 | 없으면 발생하는 현상 |
| --- | --- | --- | --- |
| 지하철 원본 CSV | `data/raw/subway/*.csv` (3개) | 전처리 원본 (`preprocess.py` 입력) | 전처리 재실행 불가 (외부 경로 사용 시 `METRO_DATA_DIR` 환경변수로 지정 가능) |
| 전처리 데이터셋 | `data/processed/train.csv.gz`<br>`data/processed/valid.csv.gz`<br>`data/processed/test.csv.gz` | 모델 학습·8-cut 평가·가드 테스트 입력 | 평가·학습 스크립트 및 데이터 가드 관련 테스트 실패 |
| 서빙 모델·이력 | `models/cal_A.joblib`, `cal_B.joblib`, `cal_C.joblib`<br>`models/sequence_D.pt`, `sequence_E.pt`<br>`models/cal_profiles.npz`, `cal_profiles_eval.npz`<br>`models/boarding_history.npz` | 앱 실시간 예측(A~E) 및 질의 도구 | 앱에서 예측값 대신 `-` 표시 |
| 실험 매니페스트·산출물 | `models/_exp/20261114-P0-R0-01/` 등 9개 디렉터리 + `access.jsonl` + `legacy_b_baseline.json` | `verify_manifest.py` 검증, 불확실성·평가 프로토콜 테스트 | `verify_manifest.py` 9/9 미실행 및 관련 테스트 검증 생략/실패 |

### 2-2. 보안·위생상 ZIP에서 제외한 파일

| 제외 항목 | 사유 | 새 기기에서의 처리 방법 |
| --- | --- | --- |
| `.env.production` | 기상청 API 실제 인증키(`KMA_APIHUB_AUTH_KEY`) 포함 | 필요 시 `.env.example`을 `.env.production`으로 복사해 본인 인증키 입력. 없어도 앱은 평년값 대체 모드로 정상 동작 |
| `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `*.egg-info/`, `.vscode/`, `reports/` | 로컬 실행 캐시 및 기기별 편집기 설정 | 새 기기에서 설치·실행 시 자동 생성 |

---

## 3. 기기별 경로·환경 차이 대응

### Windows (PowerShell에서 `conda`가 인식되지 않을 때)
- Conda 설치 경로가 기기마다 다를 수 있으므로 아래 위치 중 존재하는 경로를 찾습니다.
  - `%USERPROFILE%\anaconda3\Scripts\conda.exe`
  - `%USERPROFILE%\miniconda3\Scripts\conda.exe`
  - `C:\ProgramData\anaconda3\Scripts\conda.exe`
- 등록 없이 바로 실행하려면 가상환경 안의 실행 파일을 직접 호출합니다.
  - `<CONDA_PREFIX>\envs\metro-insight\Scripts\metro-insight.exe`
- 앞으로 `conda activate metro-insight`를 바로 쓰려면 1회 등록 후 터미널을 다시 엽니다.
  - `<경로>\conda.exe init powershell`

### 원천 CSV 폴더를 다른 드라이브에 둘 때 (`METRO_DATA_DIR`)
- `data/raw/subway/` 대신 외장 디스크나 공용 폴더에 원본 CSV를 둔 경우, 전처리 실행 전에 환경변수를 지정합니다.
  - PowerShell: `$env:METRO_DATA_DIR = "<원본_CSV_폴더>"`
  - bash/zsh: `export METRO_DATA_DIR="<원본_CSV_폴더>"`

### WSL / Linux에서 실행할 때
- WSL(WSLg)에서 한글 입력이 필요하면 `README.md`의 `ibus` + `ibus-hangul` 설정을 따릅니다.
- GUI 없이 테스트만 돌릴 때는 `QT_QPA_PLATFORM=offscreen` 환경변수를 지정합니다.

---

## 4. 배포 완료 확인 체크리스트

새 기기에서 세팅이 끝났는지 아래 4가지로 최종 판정합니다.

1. `git status -s` — 수정·누락 파일 없음 (깨끗한 상태)
2. `python -m pytest -q` — `1103 passed`
3. `python -m ruff check src tests` — `All checks passed!`
4. `python notebooks/verify_manifest.py` — `전부 통과: 9개 실행`
5. `metro-insight` (또는 `python -m metro_insight.app`) — 노선도·분석·질의 3탭 전환 및 라이트/다크 테마 전환 정상 동작
