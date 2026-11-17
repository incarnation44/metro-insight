# Metro Insight 이관 패키지 · 복원 지시서

> 최종 갱신: 2026-11-17 · 배포본: `metro-insight_전체프로젝트_2026-11-17.zip` (약 430 MB)
> 패키지에 `.git`이 함께 들어 있어, 복원 후 `git log -1 --oneline`으로 기준 커밋을 바로 확인할 수 있습니다.
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
4. `.git` 폴더가 함께 들어 있다. `git log`·`git status`를 그대로 쓸 수 있어야 하고, 압축을 푼 직후 `git status`는 깨끗해야 한다.

[수행 순서]
1. 현재 폴더(`REPO_ROOT`)에서 `git status`와 핵심 비추적 파일 존재 여부를 먼저 확인해:
   - `data/raw/subway/*.csv` (샘플 제외 3개)
   - `data/processed/train.csv.gz`, `valid.csv.gz`, `test.csv.gz`
   - `models/cal_A.joblib`, `cal_B.joblib`, `cal_C.joblib`, `sequence_D.pt`, `sequence_E.pt`, `cal_profiles.npz`, `cal_profiles_eval.npz`, `boarding_history.npz`
   - `models/_exp/` (9개 실험 디렉터리)
2. 이 기기의 Python / Conda 실행 경로를 직접 탐색해 (`conda`, `anaconda3`, `miniconda3`, 또는 Python 3.12 `venv`).
   - Conda가 있으면 `environment.yml`로 `metro-insight` 환경을 만들거나(`conda env create -f environment.yml`), 이미 있으면 갱신 후 `pip install -e . --no-deps`가 살아있는지 확인해.
   - Conda가 없으면 Python 3.12 가상환경(`.venv`)을 만들고 `pip install -r requirements.txt` 후 `pip install -e .`를 설치해.
   - `pyproject.toml`의 패키지 데이터(`layout_cache.json`, `tokens.json`, `modern_gray.qss`, `subway_network.json`)가 import 경로에서 읽히는지 확인해.
3. 아래 4가지를 모두 실행하고 실제 출력 결과를 보고해:
   - `python -m pytest -q` (기대: 전체 통과, 1,103 passed)
   - `python -m ruff check src tests` (기대: All checks passed!)
   - `python notebooks/verify_manifest.py` (기대: `전부 통과: 9개 실행`)
   - 오프스크린 앱 기동 확인 (`QT_QPA_PLATFORM=offscreen python -c "from PySide6.QtWidgets import QApplication; from metro_insight.app.main_window import MainWindow; app = QApplication([]); w = MainWindow(); print(w.windowTitle())"`)
4. 한 번에 검사하려면 저장소에 들어 있는 스크립트를 써도 된다:
   - `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_bundle.ps1 -ZipPath "<배포 ZIP 경로>"`
   - 이 스크립트는 임시 폴더에 ZIP을 다시 풀어 위 3번 항목과 필수 파일 존재를 한 번에 판정한다.
5. 이 기기에서 사용자가 앱을 바로 실행할 수 있는 정확한 한 줄 실행 명령(가상환경 절대 경로 포함)을 안내해.
```

---

## 2. 배포 패키지 구성

단일 통합 ZIP(`metro-insight_전체프로젝트_2026-11-17.zip`, 약 430 MB)을 원하는 위치에 풀면 `metro-insight/` 루트 아래에 Git 저장소와 비추적 필수 파일이 한 번에 복원됩니다.
압축을 풀면 `metro-insight/` 폴더 하나가 나오고, 그 안이 곧 `REPO_ROOT` 입니다.

무결성 값은 ZIP과 같은 폴더의 `metro-insight_전체프로젝트_2026-11-17.sha256.txt`에 있습니다.

PowerShell:
```powershell
Get-FileHash ".\metro-insight_전체프로젝트_2026-11-17.zip" -Algorithm SHA256
Get-Content ".\metro-insight_전체프로젝트_2026-11-17.sha256.txt"
```

bash:
```bash
sha256sum metro-insight_전체프로젝트_2026-11-17.zip
cat metro-insight_전체프로젝트_2026-11-17.sha256.txt
```

### 2-1. 포함 항목

| 구분 | 상대 경로 (`REPO_ROOT` 기준) | 용도 | 없으면 발생하는 현상 |
| --- | --- | --- | --- |
| Git 저장소 | `.git/` | `git log`·`git status`·커밋 이력 | 이력 확인·커밋 불가 (코드는 실행됨) |
| 지하철 원본 CSV | `data/raw/subway/*.csv` (샘플 제외 3개) | 전처리 원본 (`preprocess.py` 입력) | 전처리 재실행 불가 (외부 경로 사용 시 `METRO_DATA_DIR` 환경변수로 지정 가능) |
| 기상청 원본 | `data/raw/weather/*.txt` (2024-01 ~ 2026-10) | 전처리·재현 | 날씨 CSV 재생성 불가 |
| 전처리 데이터셋 | `data/processed/train.csv.gz`<br>`data/processed/valid.csv.gz`<br>`data/processed/test.csv.gz` | 모델 학습·8-cut 평가·가드 테스트 입력 | 평가·학습 스크립트 및 데이터 가드 관련 테스트 실패 |
| 서빙 모델·이력 | `models/cal_A.joblib`, `cal_B.joblib`, `cal_C.joblib`<br>`models/sequence_D.pt`, `sequence_E.pt`<br>`models/cal_profiles.npz`, `cal_profiles_eval.npz`<br>`models/boarding_history.npz` | 앱 실시간 예측(A~E) 및 질의 도구 | 앱에서 예측값 대신 `-` 표시 |
| 구 모델·성적표 | `models/model_A/B/C.joblib`<br>`models/metrics_calendar.json`, `metrics_sequence.json`, `metrics_timeseries.json`<br>`models/backtest_182_366.json` | 분석 탭 지표·백테스트 근거 | 분석 화면 수치 공백 |
| 실험 매니페스트·산출물 | `models/_exp/20261114-P0-R0-01/` 등 9개 디렉터리<br>`models/_exp/access.jsonl`, `legacy_b_baseline.json`<br>`models/_probe/grid_results.json` | `verify_manifest.py` 검증, 불확실성·평가 프로토콜 테스트 | `verify_manifest.py` 9/9 미실행 및 관련 테스트 검증 생략/실패 |
| 문서·그림 | `docs/` 전체 (모델 카드 5장, 실험 기록, 계획서, 이미지) | 보고·인수인계 | 문서 링크 깨짐 |
| 이관 도구 | `scripts/build_bundle.ps1`, `scripts/verify_bundle.ps1` | 패키지 재생성·복원 훈련 | 다음 배포 때 손으로 압축해야 함 |

### 2-2. 보안·위생상 제외한 파일

| 제외 항목 | 사유 | 새 기기에서의 처리 방법 |
| --- | --- | --- |
| `.env.production` | 기상청 API 실제 인증키(`KMA_APIHUB_AUTH_KEY`) 포함 | 필요 시 `.env.example`을 `.env.production`으로 복사해 본인 인증키 입력. 없어도 앱은 평년값 대체 모드로 정상 동작 |
| `.env`, `.env.local` | 기기별 비밀값 | 새 기기에서 직접 만든다 |
| `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `*.egg-info/`, `.venv/`, `.vscode/`, `reports/`, `var/`, `*.log` | 로컬 실행 캐시·기기별 편집기 설정·실행 로그 | 새 기기에서 설치·실행 시 자동 생성 |

### 2-3. 이 기기에서 패키지를 다시 만들 때 (재빌드)

코드나 산출물이 바뀌면 아래 한 줄로 ZIP과 SHA-256을 새로 만든다. 결과는 기본적으로 `~/Desktop/metroinsight 자료`에 떨어진다.

PowerShell:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\metro-insight\scripts\build_bundle.ps1
```

- 날짜·출력 위치를 바꾸려면: `-Stamp 2026-11-20 -OutDir D:\배포`
- 스크립트가 하는 일: `.env.production`·캐시·`.vscode` 제외 → `metro-insight/` 루트로 ZIP 생성 → 엔트리 수 확인 → SHA-256 파일 기록.
- 빌드 전에 `git status`가 깨끗한지, `pytest`가 통과하는지 먼저 확인한다.

### 2-4. 복원 훈련 (새 기기 가기 전 이 기기에서 미리 검사)

ZIP을 임시 폴더에 실제로 풀어 `git status`·필수 파일·테스트·오프스크린 기동까지 확인한다.

PowerShell:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File D:\metro-insight\scripts\verify_bundle.ps1 `
  -ZipPath "C:\Users\user1\Desktop\metroinsight 자료\metro-insight_전체프로젝트_2026-11-17.zip"
```

- 통과하면 마지막에 `전체 PASS — 복원 훈련 통과`가 찍히고 종료 코드는 0.
- 실패 항목은 `FAIL`로 표시되고, 종료 코드는 1.
- 푼 폴더를 남기려면 `-Keep`, 테스트를 건너뛰고 파일만 보려면 `-SkipTests`.
- 검사 중에는 `PYTHONPATH=<푼 폴더>\src`와 `QT_QPA_PLATFORM=offscreen`을 임시로 잡는다. 이 기기에 이미 설치된 패키지가 아니라 **ZIP 안의 코드**를 검사하기 위한 것이다.

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
- `scripts/*.ps1`은 Windows PowerShell 전용입니다. Linux에서는 같은 순서를 손으로 실행합니다(압축 해제 → `pip install -e .` → pytest).

### 한글 파일명이 깨질 때 (Windows)
- ZIP 안 원본 CSV 3개는 한글 파일명입니다. Windows 기본 압축 해제 도구 대신 7-Zip이나 탐색기(UTF-8 지원)로 풀면 깨지지 않습니다.
- 이미 깨졌다면 파일명만 바꾸지 말고 `data/raw/subway/`에 있는 CSV 3개를 그대로 다시 풀어 넣습니다. 전처리는 폴더 안 CSV를 이름이 아니라 내용으로 읽습니다.

---

## 4. 복원 완료 판정 체크리스트

새 기기에서 세팅이 끝났는지 아래 5가지로 최종 판정합니다. (괄호는 2026-11-16 이 기기에서 측정한 값)

1. `git status -s` — 수정·누락 파일 없음 (깨끗한 상태)
2. `python -m pytest -q` — `1,103 passed`
3. `python -m ruff check src tests` — `All checks passed!`
4. `python notebooks/verify_manifest.py` — `전부 통과: 9개 실행`
5. `metro-insight` (또는 `python -m metro_insight.app`) — 노선도·분석·질의 3탭 전환 및 라이트/다크 테마 전환 정상 동작

`scripts/verify_bundle.ps1`은 위 1~4번과 필수 파일 존재를 한 번에 판정합니다.
앱 화면(5번)은 사람이 직접 띄워 확인합니다.

---

## 5. 손으로 복원할 때 (ZIP 없이 `git clone`만 한 경우)

```powershell
git clone git@github.com:incarnation44/metro-insight.git
```

코드·문서·샘플만 받아진다. 아래 4묶음을 원래 상대 경로에 그대로 넣어야 앱과 검증이 완전히 동작한다.

| 묶음 | 상대 경로 |
| --- | --- |
| 지하철 원본 CSV 3개 | `data/raw/subway/` |
| 전처리 데이터셋 3개 | `data/processed/train.csv.gz`, `valid.csv.gz`, `test.csv.gz` |
| 서빙 모델 8개 | `models/` (`cal_A/B/C.joblib`, `sequence_D/E.pt`, `cal_profiles.npz`, `cal_profiles_eval.npz`, `boarding_history.npz`) |
| 실험 산출물 | `models/_exp/` (9개 디렉터리 + `access.jsonl` + `legacy_b_baseline.json`) |

넣은 뒤 `scripts/verify_bundle.ps1 -SkipTests`로 파일 존재만 먼저 확인하고, 이어서 전체 검사를 돌린다.

---

## 6. 자주 막히는 곳

| 증상 | 원인 | 조치 |
| --- | --- | --- |
| 앱 예측값이 전부 `-` | `models/` 산출물 누락 | `models/*.joblib`, `*.pt`, `*.npz` 위치 확인 |
| `verify_manifest.py`가 9개 미만 실행 | `models/_exp/` 누락 | 실험 디렉터리 9개 복원 |
| 데이터 가드 테스트 실패 | `data/processed/*.csv.gz` 누락 | 3개 파일 복원 |
| 앱이 `ModuleNotFoundError: metro_insight` | 패키지 미설치 | `pip install -e . --no-deps` |
| 날씨가 늘 평년값 | `.env.production` 없음(정상) | 인증키를 넣거나 그대로 사용 |
| 한글 CSV 파일명 깨짐 | 압축 해제 도구의 코드페이지 | 7-Zip 등 UTF-8 도구로 재해제 |