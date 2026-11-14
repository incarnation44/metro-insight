"""평가용 Test 파일 직접 읽기 금지 정책 테스트.

평가용 Test 파일은 승인 경로(DataGuard)로만 열어야 한다. 코드가 ``pd.read_csv`` 로 파일을
직접 열면 승인·기록 없이 성능을 볼 수 있게 되므로, 소스 트리 전체를 훑어 새 직접 접근이
생기지 않았는지 확인한다.

기존에 이미 직접 읽는 코드는 파일 단위로 허용 목록에 둔다. 허용 목록 밖에서 새 직접 읽기가
나오면 실패한다. 허용 목록을 늘리는 변경은 정책 위반이므로 여기서만 관리한다.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("src", "notebooks")
TEST_FILE_TOKENS = ("test.csv", "test.csv.gz")

# 정책 이전부터 파일 전체를 직접 읽는 코드 — 파일 경로 → 허용 이유.
GRANDFATHERED: dict[str, str] = {
    "notebooks/train_abc.py": "레거시 A/B/C 학습 스크립트 — 프로토콜 밖, 수정 금지",
    "notebooks/plot_results.py": "기존 결과 그림 — 승인 없이 읽는 스크립트, 수정 금지",
    "notebooks/train_calendar.py": "서빙용 전체 이력 프로파일 생성",
    "notebooks/evaluate_calendar.py": "기존 달력 모델 평가 스크립트",
    "notebooks/build_history_cache.py": "서빙 이력 캐시 생성(전 기간 필요)",
    "notebooks/make_samples.py": "샘플 파일 생성기(원본에서 36행 발췌)",
    "notebooks/train_sequence.py": "시퀀스 모델 학습 — lag 확인용 전체 기간",
    "src/metro_insight/model/predictor.py": "서빙 예측기 — 재학습·프로파일 갱신 경로",
}

READ_FUNCS = {"read_csv", "read_table", "read_parquet", "read_pickle"}
OPEN_FUNCS = {"open", "read_bytes"}


def _string_constants(node: ast.AST) -> list[str]:
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _module_constants(tree: ast.AST) -> dict[str, str]:
    """모듈 수준에서 문자열 상수에 묶인 이름 — 읽기 인자에서 이름을 값으로 바꿔 본다."""
    found: dict[str, str] = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(node.value, ast.Constant):
            continue
        if isinstance(node.value.value, str):
            found[target.id] = node.value.value
    return found


def _joined_source(node: ast.AST, constants: dict[str, str]) -> str:
    """에프 문자열(f-string) 조각·상수 이름까지 이어 붙인 대략적인 원문 — 파일명이 조각나 있어도 잡는다."""
    parts = _string_constants(node)
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and inner.id in constants:
            parts.append(constants[inner.id])
    return " ".join(parts)


def direct_test_reads(root: Path) -> list[tuple[str, int, str]]:
    """(상대경로, 줄번호, 근거) 목록. 문자열 어디에든 test.csv 가 있고 읽기 호출이면 위반."""
    hits: list[tuple[str, int, str]] = []
    for folder in SCAN_DIRS:
        for path in sorted((root / folder).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(source)
            except SyntaxError:  # pragma: no cover - 구문 오류는 다른 테스트가 잡는다
                continue
            constants = _module_constants(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name not in READ_FUNCS | OPEN_FUNCS:
                    continue
                text = _joined_source(node, constants)
                if any(token in text for token in TEST_FILE_TOKENS):
                    hits.append((rel, int(node.lineno), f"{name}({text[:80]})"))
    return hits


def _allowlisted(rel: str) -> bool:
    return rel in GRANDFATHERED


def test_no_new_direct_test_reads() -> None:
    hits = direct_test_reads(ROOT)
    violations = [hit for hit in hits if not _allowlisted(hit[0])]
    assert not violations, (
        "Test 파일을 직접 읽는 코드가 추가됐다 — DataGuard.open_split 을 쓴다: "
        + ", ".join(f"{rel}:{line}" for rel, line, _ in violations)
    )


def test_grandfathered_entries_still_exist() -> None:
    """허용 목록이 실제 파일을 가리키는지 확인한다(죽은 허용 항목 방지)."""
    for rel in GRANDFATHERED:
        assert (ROOT / rel).exists(), f"허용 목록의 파일이 없다: {rel}"


def test_allowlist_is_not_a_catch_all() -> None:
    """허용 목록이 디렉터리·패턴이 아니라 개별 파일이어야 한다."""
    for rel in GRANDFATHERED:
        assert rel.endswith(".py"), f"허용 목록은 파일 단위여야 한다: {rel}"
        assert "*" not in rel and "?" not in rel, f"와일드카드는 금지: {rel}"


def test_detector_catches_direct_read(tmp_path: Path) -> None:
    """탐지기 자체가 동작하는지 합성 파일로 확인한다(정책 테스트가 무의미하게 통과하지 않게)."""
    pkg = tmp_path / "notebooks"
    pkg.mkdir()
    (pkg / "bad_direct.py").write_text(
        "import pandas as pd\n"
        "DATA = 'data/processed'\n"
        "df = pd.read_csv(DATA + '/test.csv.gz')\n",
        encoding="utf-8",
    )
    (pkg / "bad_usecols.py").write_text(
        "import pandas as pd\n"
        "def load(path):\n"
        "    return pd.read_csv(path, usecols=['date', 'boarding'], compression='gzip')\n"
        "df = load('data/processed/test.csv.gz')\n",
        encoding="utf-8",
    )
    (pkg / "ok_guard.py").write_text(
        "from metro_insight.data.guard import DataGuard\n"
        "def load(guard):\n"
        "    return guard.open_split('test_dev', purpose='score')\n",
        encoding="utf-8",
    )
    hits = direct_test_reads(tmp_path)
    flagged = {rel for rel, _, _ in hits}
    assert "notebooks/bad_direct.py" in flagged
    assert "notebooks/bad_usecols.py" not in flagged  # 경로가 인자라 문자열에 없다
    assert "notebooks/ok_guard.py" not in flagged


def test_plain_read_in_helper_still_suspicious(tmp_path: Path) -> None:
    """파일명이 상수로 분리돼도 같은 파일 안에서 조립되면 잡는다."""
    pkg = tmp_path / "src"
    pkg.mkdir()
    (pkg / "leak.py").write_text(
        "import pandas as pd\n"
        "TEST = 'data/processed/test.csv.gz'\n"
        "def load():\n"
        "    return pd.read_csv(TEST)\n",
        encoding="utf-8",
    )
    hits = direct_test_reads(tmp_path)
    assert any(rel == "src/leak.py" for rel, _, _ in hits)