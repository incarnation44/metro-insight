"""평가용 Test 파일 직접 읽기 금지 정책 테스트.

평가용 Test 파일은 승인 경로(DataGuard)로만 열어야 한다. 코드가 ``pd.read_csv`` 로 파일을
직접 열면 승인·기록 없이 성능을 볼 수 있게 되므로, 소스 트리 전체를 훑어 새 직접 접근이
생기지 않았는지 확인한다.

탐지는 호출 이름 하나에 기대지 않는다. 두 형태를 모두 본다.
    1) 읽기 지점: ``read_csv`` · ``open`` 같은 실제 읽기 호출의 인자에 test 경로가 흐르는 경우.
    2) 전달 지점: 그런 읽기를 감싼 함수(``cf.load_processed`` · ``load_split`` 등)에 test 경로나
       test 분할 이름이 인자로 흐르는 호출. 호출 이름만 보면 이 경로가 통째로 빠져나간다.
경로 상수는 이름·표(딕셔너리) 값·for 순회 대상·f-string 조각을 따라간다. 표에서 값을 꺼낼 때
키가 상수면 그 값만, 키가 변수면 어느 값인지 확정할 수 없으므로 읽기 지점에서만 근거로 쓴다.

기존에 이미 직접 읽는 코드는 파일 단위로 허용 목록에 둔다. 허용 목록 밖에서 새 직접 읽기가
나오면 실패한다. 허용 목록을 늘리는 변경은 정책 위반이므로 여기서만 관리한다. 허용 목록과
탐지 결과가 어긋나도 실패한다(목록 확대·축소를 둘 다 잡는다).
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
# 승인 접근자 이름 — 이 함수 안에서 표로 경로를 만드는 것은 승인 경로 그 자체다.
GUARD_ACCESSOR = "open_split"

_SCOPE_TYPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


# ───────────────────────── AST 도우미 ─────────────────────────
def _string_constants(node: ast.AST) -> list[str]:
    """식 안의 문자열 상수 — 딕셔너리 리터럴 안쪽은 표 값이라 따로 센다."""
    found: list[str] = []
    stack: list[ast.AST] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Dict):
            continue
        if isinstance(cur, ast.Constant) and isinstance(cur.value, str):
            found.append(cur.value)
        stack.extend(ast.iter_child_nodes(cur))
    return found


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


def _subscript_names(node: ast.AST) -> set[int]:
    """``표[key]`` 로 값을 꺼내는 데 쓰인 이름 — 표는 여러 값이 섞인 상자라 경로로 보지 않는다."""
    skip: set[int] = set()
    for inner in ast.walk(node):
        if isinstance(inner, ast.Subscript):
            for value in ast.walk(inner.value):
                if isinstance(value, ast.Name):
                    skip.add(id(value))
    return skip


def _table_values(scope: ast.AST, base: dict[str, dict[str, str]] | None = None
                  ) -> dict[str, dict[str, str]]:
    """이름 → {상수 키: 그 값의 문자열 조각}. 딕셔너리 리터럴로 만든 표만 모은다."""
    tables = {name: dict(values) for name, values in (base or {}).items()}
    for node in _own_nodes(scope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        table: dict[str, str] = {}
        for key, item in zip(value.keys, value.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                text = " ".join(_string_constants(item))
                if text:
                    table[key.value] = text
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                tables.setdefault(target.id, {}).update(table)
    return tables


def _name_texts(node: ast.AST, bindings: dict[str, list[str]],
                tables: dict[str, dict[str, str]], follow_tables: bool) -> list[str]:
    """식에 나오는 이름·표 조회에 흘러든 문자열 조각.

    상수 키 조회(``표["test"]``)는 그 값만 본다. 키가 변수일 때는 어느 값인지 확정할 수 없어
    follow_tables 일 때만 표의 모든 값을 후보로 넣는다(읽기 지점 판정에만 쓴다).
    """
    skip = _subscript_names(node)
    found: list[str] = []
    for inner in ast.walk(node):
        if isinstance(inner, ast.Subscript) and isinstance(inner.value, ast.Name):
            table = tables.get(inner.value.id)
            key = inner.slice
            if table and isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value in table:
                    found.append(table[key.value])
            elif table and follow_tables:
                found.extend(dict.fromkeys(table.values()))
        if isinstance(inner, ast.Name) and id(inner) not in skip:
            found.extend(bindings.get(inner.id, []))
    return found


def _joined_source(node: ast.AST, bindings: dict[str, list[str]],
                   tables: dict[str, dict[str, str]], *, follow_tables: bool,
                   limit: int = 400) -> str:
    """호출 인자에 담긴 문자열 조각과 그 이름에 묶인 값 — 파일명이 조각나 있어도 잡는다."""
    text = " ".join([*_string_constants(node), *_name_texts(node, bindings, tables, follow_tables)])
    return text[:limit]


def _mentions_test_file(text: str) -> bool:
    """경로에 test 파일이 지목됐는지 — 'test' 조각과 csv 확장자 조각이 따로 있어도 잡는다."""
    if any(token in text for token in TEST_FILE_TOKENS):
        return True
    words = text.split()
    return any(word == "test" for word in words) and any(".csv" in word for word in words)


def _mentions_test_split(text: str) -> bool:
    """분할 이름 'test' 가 그대로 흘렀는지 — 감싼 함수의 인자 판정에 쓴다."""
    return any(word == "test" for word in text.split())


def _bindings(scope: ast.AST, tables: dict[str, dict[str, str]],
              base: dict[str, list[str]] | None = None, rounds: int = 4,
              *, follow_tables: bool = False) -> dict[str, list[str]]:
    """범위 안의 이름 → 그 이름에 흘러들어온 문자열 조각.

    대입·주석 대입·대입식(walrus)·for 순회 대상을 따라가며 값을 합친다. 다른 이름을 거치는
    대입도 잡히도록 값이 늘지 않을 때까지 되풀이한다. base 를 주면 바깥 범위(모듈)의 값도
    함께 본다. 표 조회는 follow_tables 일 때만 값으로 펼친다.
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
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                if node.value is None:
                    continue
                text = _joined_source(node.value, bound, tables, follow_tables=follow_tables)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                text = _joined_source(node.iter, bound, tables, follow_tables=follow_tables)
                targets = [node.target]
            else:
                continue
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        changed |= add(name, text)
        if not changed:
            break
    return bound


def _scopes(tree: ast.AST, module_strict: dict[str, list[str]], module_full: dict[str, list[str]],
            module_tables: dict[str, dict[str, str]]):
    """범위별 (범위, 직접 노드, 엄격 묶음, 표 펼친 묶음, 표) — 모듈 값은 안쪽 범위에서도 보인다."""
    pending: list[ast.AST] = [tree]
    while pending:
        scope = pending.pop()
        if scope is tree:
            yield scope, _own_nodes(scope), module_strict, module_full, module_tables
        else:
            tables = _table_values(scope, base=module_tables)
            yield (
                scope,
                _own_nodes(scope),
                _bindings(scope, tables, base=module_strict),
                _bindings(scope, tables, base=module_full, follow_tables=True),
                tables,
            )
        pending.extend(node for node in _own_nodes(scope) if isinstance(node, _SCOPE_TYPES))


# ───────────────────────── 읽기를 감싼 함수 찾기 ─────────────────────────
def _callee_name(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _byte_only_open(node: ast.Call) -> bool:
    """``open(path, 'rb')`` — 행을 보지 않는 바이트 접근(해시 계산 등)으로 본다."""
    if len(node.args) < 2:
        return False
    mode = node.args[1]
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) \
        and "b" in mode.value and "+" not in mode.value


def _reader_names(files: dict[Path, ast.AST]) -> set[str]:
    """인자를 실제 읽기 호출로 넘기는 함수 이름 — 이름이 아니라 몸통이 근거다."""
    names: set[str] = set()
    for tree in files.values():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args,
                                          *node.args.kwonlyargs)}
            if not params:
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                name = _callee_name(inner)
                if name in READ_FUNCS:
                    pass
                elif name in OPEN_FUNCS:
                    if name == "open" and _byte_only_open(inner):
                        continue
                else:
                    continue
                used = {item.id for item in ast.walk(inner) if isinstance(item, ast.Name)}
                if used & params:
                    names.add(node.name)
    return names


# ───────────────────────── 탐지 ─────────────────────────
def direct_test_reads(root: Path) -> list[tuple[str, int, str]]:
    """(상대경로, 줄번호, 근거) 목록 — 읽기 지점과 감싼 함수 전달 지점을 모두 본다."""
    files: dict[Path, ast.AST] = {}
    for folder in SCAN_DIRS:
        for path in sorted((root / folder).rglob("*.py")):
            source = path.read_text(encoding="utf-8-sig", errors="replace")
            try:
                files[path] = ast.parse(source)
            except SyntaxError:  # pragma: no cover - 구문 오류는 다른 테스트가 잡는다
                continue

    readers = _reader_names(files)
    hits: list[tuple[str, int, str]] = []
    for path, tree in files.items():
        rel = path.relative_to(root).as_posix()
        module_tables = _table_values(tree)
        module_strict = _bindings(tree, module_tables)
        module_full = _bindings(tree, module_tables, follow_tables=True)
        for scope, nodes, strict, full, tables in _scopes(
            tree, module_strict, module_full, module_tables
        ):
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and scope.name == GUARD_ACCESSOR:
                continue
            for node in nodes:
                if not isinstance(node, ast.Call):
                    continue
                name = _callee_name(node)
                if name in READ_FUNCS | OPEN_FUNCS:
                    text = _joined_source(node, full, tables, follow_tables=True)
                    if _mentions_test_file(text):
                        hits.append((rel, int(node.lineno), f"읽기 호출 {name}({text[:60]})"))
                elif name in readers:
                    literal = " ".join(_string_constants(node))
                    named = " ".join(_name_texts(node, strict, tables, follow_tables=False))
                    if _mentions_test_file(literal) or _mentions_test_split(named):
                        hits.append((rel, int(node.lineno), f"{name}() 경유"))
    return sorted(set(hits))


def _allowlisted(rel: str) -> bool:
    return rel in GRANDFATHERED


# ───────────────────────── 저장소 정책 ─────────────────────────
def test_no_new_direct_test_reads() -> None:
    hits = direct_test_reads(ROOT)
    violations = [hit for hit in hits if not _allowlisted(hit[0])]
    assert not violations, (
        "Test 파일을 직접 읽는 코드가 추가됐다 — DataGuard.open_split 을 쓴다: "
        + ", ".join(f"{rel}:{line}" for rel, line, _ in violations)
    )


def test_detected_files_match_allowlist() -> None:
    """탐지 히트 파일 집합이 허용 목록과 정확히 같아야 한다(확대·축소 둘 다 감지).

    이 검사는 감싼 함수 경유 탐지(``cf.load_processed`` · ``load_split``)가 켜진 뒤에 성립한다.
    그 전에는 train_calendar · evaluate_calendar · train_sequence 세 항목이 히트 0 인 죽은
    항목이었다. 지금은 셋 다 잡히며(래퍼 호출 지점), 목록에서 빠지면 이 검사가 실패한다.
    """
    hits = {rel for rel, _, _ in direct_test_reads(ROOT)}
    allowed = set(GRANDFATHERED)
    assert hits == allowed, (
        f"허용 목록에 없는데 탐지된 파일: {sorted(hits - allowed)} · "
        f"허용 목록에 있는데 탐지되지 않는 파일: {sorted(allowed - hits)}"
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


# ───────────────────────── 탐지기 자체 검사 ─────────────────────────
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
    # 경로를 인자로 넘기는 래퍼도 잡는다 — 호출 이름만 보던 예전 판정은 이 우회를 놓쳤다
    assert "notebooks/bad_usecols.py" in flagged
    assert "notebooks/ok_guard.py" not in flagged


def test_detector_catches_wrapper_with_path_constant(tmp_path: Path) -> None:
    """다른 모듈 함수(래퍼)에 test 경로 상수를 넘기는 호출도 잡는다."""
    pkg = tmp_path / "notebooks"
    pkg.mkdir()
    (pkg / "bad_wrapper.py").write_text(
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "DATA = Path('data/processed')\n"
        "def load_processed(path, columns=None):\n"
        "    return pd.read_csv(path, usecols=columns, compression='gzip')\n"
        "def main():\n"
        "    return load_processed(DATA / 'test.csv.gz')\n",
        encoding="utf-8",
    )
    hits = direct_test_reads(tmp_path)
    assert any(rel == "notebooks/bad_wrapper.py" for rel, _, _ in hits)


def test_detector_catches_split_name_through_loop(tmp_path: Path) -> None:
    """분할 이름을 도는 순회 변수가 래퍼로 흐르는 형태도 잡는다."""
    pkg = tmp_path / "notebooks"
    pkg.mkdir()
    (pkg / "bad_split_loop.py").write_text(
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "DATA = Path('data/processed')\n"
        "def load_split(name):\n"
        "    return pd.read_csv(DATA / f'{name}.csv.gz')\n"
        "def main():\n"
        "    return {name: load_split(name) for name in ('train', 'valid', 'test')}\n",
        encoding="utf-8",
    )
    hits = direct_test_reads(tmp_path)
    assert any(rel == "notebooks/bad_split_loop.py" for rel, _, _ in hits)


def test_detector_reads_table_entry_by_constant_key(tmp_path: Path) -> None:
    """표에서 상수 키로 꺼낸 test 경로도 잡는다."""
    pkg = tmp_path / "notebooks"
    pkg.mkdir()
    (pkg / "bad_table.py").write_text(
        "import pandas as pd\n"
        "SPLIT_FILES = {'train': 'data/processed/train.csv.gz',\n"
        "               'test_dev': 'data/processed/test.csv.gz'}\n"
        "df = pd.read_csv(SPLIT_FILES['test_dev'])\n",
        encoding="utf-8",
    )
    hits = direct_test_reads(tmp_path)
    assert any(rel == "notebooks/bad_table.py" for rel, _, _ in hits)


def test_hash_only_helper_is_not_a_data_read(tmp_path: Path) -> None:
    """바이트 해시만 계산하는 도우미는 데이터 읽기로 세지 않는다(행을 보지 않는다)."""
    pkg = tmp_path / "src"
    pkg.mkdir()
    (pkg / "digest_only.py").write_text(
        "import hashlib\n"
        "def digest(path):\n"
        "    h = hashlib.sha256()\n"
        "    with open(path, 'rb') as f:\n"
        "        h.update(f.read())\n"
        "    return h.hexdigest()\n"
        "print(digest('data/processed/test.csv.gz'))\n",
        encoding="utf-8",
    )
    hits = direct_test_reads(tmp_path)
    assert not hits


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