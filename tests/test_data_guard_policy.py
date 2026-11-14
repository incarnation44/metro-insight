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


_SCOPE_TYPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _own_nodes(scope: ast.AST) -> list[ast.AST]:
    """한 범위에 직접 속한 노드 — 안쪽 함수(클로저)는 따로 계산하므로 내려가지 않는다."""
    found: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        found.append(node)
        if isinstance(node, _SCOPE_TYPES):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return found


def _substituted_names(node: ast.AST) -> set[int]:
    """``표[key]`` 로 값을 꺼내는 데 쓰인 이름 — 표는 여러 값이 섞인 상자라 경로로 보지 않는다."""
    skip: set[int] = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Subscript):
            for value in ast.walk(inner.value):
                if isinstance(value, ast.Name):
                    skip.add(id(value))
    return skip


def _name_texts(node: ast.AST, bindings: dict[str, list[str]]) -> list[str]:
    """식에 나오는 이름에 묶인 문자열 조각 — 변수에 담긴 경로도 따라간다."""
    skip = _substituted_names(node)
    found: list[str] = []
    for inner in ast.walk(node):
        if isinstance(inner, ast.Name) and id(inner) not in skip:
            found.extend(bindings.get(inner.id, []))
    return found


def _joined_source(node: ast.AST, bindings: dict[str, list[str]], limit: int = 300) -> str:
    """호출 인자에 담긴 문자열 조각과 그 이름에 묶인 값 — 파일명이 조각나 있어도 잡는다.

    이름을 거듭 따라가면 문자열이 계속 길어지므로 상한을 둔다(검사에는 앞부분으로 충분하다).
    """
    text = " ".join([*_string_constants(node), *_name_texts(node, bindings)])
    return text[:limit]


def _mentions_test_file(text: str) -> bool:
    """경로에 test 파일이 지목됐는지 — 'test' 조각과 csv 확장자 조각이 따로 있어도 잡는다."""
    if any(token in text for token in TEST_FILE_TOKENS):
        return True
    words = text.split()
    return any(word == "test" for word in words) and any(".csv" in word for word in words)


def _bindings(scope: ast.AST, base: dict[str, list[str]] | None = None,
              rounds: int = 4) -> dict[str, list[str]]:
    """범위 안의 이름 → 그 이름에 흘러들어온 문자열 조각.

    대입·주석 대입·대입식(walrus)·for 순회 대상을 따라가며 값을 합친다. 다른 이름을 거치는
    대입도 잡히도록 값이 늘지 않을 때까지 되풀이한다. base 를 주면 바깥 범위(모듈)의 값도
    함께 보므로 함수 안에서 모듈 상수를 읽어 만든 경로도 따라간다. 표에서 꺼낸 값
    (``표[key]``)은 어느 값인지 알 수 없어 따라가지 않는다.
    """
    bound: dict[str, list[str]] = {name: list(texts) for name, texts in (base or {}).items()}

    def add(target: ast.AST, text: str) -> bool:
        if not text:
            return False
        values = bound.setdefault(target.id, [])
        if text in values:
            return False
        values.append(text)
        return True

    nodes = _own_nodes(scope)
    for _ in range(rounds):
        changed = False
        for node in nodes:
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr,
                                     ast.For, ast.AsyncFor, ast.comprehension)):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                if node.value is None:
                    continue
                text = _joined_source(node.value, bound)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            else:
                text = _joined_source(node.iter, bound)
                targets = [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        changed |= add(name, text)
        if not changed:
            break
    return bound


def _scope_bindings(tree: ast.AST) -> list[tuple[list[ast.AST], dict[str, list[str]]]]:
    """범위별 (그 범위의 노드, 그 범위에서 유효한 이름 묶음) — 모듈 값은 안쪽 범위에서도 보인다."""
    module = _bindings(tree)
    result: list[tuple[list[ast.AST], dict[str, list[str]]]] = []
    pending: list[ast.AST] = [tree]
    while pending:
        scope = pending.pop()
        local = module if scope is tree else _bindings(scope, base=module)
        result.append((_own_nodes(scope), local))
        found = [node for node in _own_nodes(scope) if isinstance(node, _SCOPE_TYPES)]
        pending.extend(found)
    return result


def direct_test_reads(root: Path) -> list[tuple[str, int, str]]:
    """(상대경로, 줄번호, 근거) 목록. 문자열 어디에든 test.csv 가 있고 읽기 호출이면 위반."""
    hits: list[tuple[str, int, str]] = []
    for folder in SCAN_DIRS:
        for path in sorted((root / folder).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            source = path.read_text(encoding="utf-8-sig", errors="replace")
            try:
                tree = ast.parse(source)
            except SyntaxError:  # pragma: no cover - 구문 오류는 다른 테스트가 잡는다
                continue
            for nodes, bindings in _scope_bindings(tree):
                for node in nodes:
                    if not isinstance(node, ast.Call):
                        continue
                    func = node.func
                    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                    if name not in READ_FUNCS | OPEN_FUNCS:
                        continue
                    text = _joined_source(node, bindings)
                    if _mentions_test_file(text):
                        hits.append((rel, int(node.lineno), f"{name}({text[:80]})"))
    return sorted(hits)


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


def test_detector_follows_paths_held_in_variables(tmp_path: Path) -> None:
    """경로를 변수에 담아 넘기는 우회도 잡는다 — 예전에는 변수 이름만 보여 빠져나갔다."""
    pkg = tmp_path / "notebooks"
    pkg.mkdir()
    (pkg / "via_local.py").write_text(
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "DATA = Path('data/processed')\n"
        "def load():\n"
        "    target = DATA / 'test.csv.gz'\n"
        "    return pd.read_csv(target)\n",
        encoding="utf-8",
    )
    (pkg / "via_chain.py").write_text(
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "DATA = Path('data/processed')\n"
        "TARGET = DATA / 'test.csv.gz'\n"
        "def load():\n"
        "    path = TARGET\n"
        "    with open(path, 'rb') as f:\n"
        "        return f.read()\n",
        encoding="utf-8",
    )
    (pkg / "via_loop.py").write_text(
        "import pandas as pd\n"
        "DATA = 'data/processed'\n"
        "def load():\n"
        "    frames = []\n"
        "    for split in ('train', 'valid', 'test'):\n"
        "        frames.append(pd.read_csv(DATA + '/' + split + '.csv.gz', compression='gzip'))\n"
        "    return frames\n",
        encoding="utf-8",
    )
    hits = direct_test_reads(tmp_path)
    flagged = {rel for rel, _, _ in hits}
    assert "notebooks/via_local.py" in flagged
    assert "notebooks/via_chain.py" in flagged
    assert "notebooks/via_loop.py" in flagged


def test_detector_keeps_split_tables_out_of_sight(tmp_path: Path) -> None:
    """split 이름을 표로 조회해 여는 승인 경로(DataGuard 형태)는 위반으로 잡지 않는다."""
    pkg = tmp_path / "src"
    pkg.mkdir()
    (pkg / "approved.py").write_text(
        "import pandas as pd\n"
        "SPLIT_FILES = {'train': 'data/processed/train.csv.gz',\n"
        "               'test_dev': 'data/processed/test.csv.gz'}\n"
        "class Loader:\n"
        "    def open_split(self, name):\n"
        "        rel = SPLIT_FILES[name]\n"
        "        path = self.root / rel\n"
        "        return pd.read_csv(path, compression='gzip')\n",
        encoding="utf-8",
    )
    hits = direct_test_reads(tmp_path)
    assert not hits