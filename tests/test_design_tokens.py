"""디자인 토큰(tokens.json) 검증.

이 파일은 토큰이 (1) 스키마를 지키고 (2) WCAG AA 대비를 넘고
(3) 기존 QSS·스타일 모듈과 어긋나지 않는지를 본다. 화면 동작은 건드리지 않는다.
"""
import json
import re
from pathlib import Path

import pytest

from metro_insight.app import styles as styles_pkg
from metro_insight.app.styles import qss
from metro_insight.app.styles import tokens as tk

# QSS 의 6자리 색 리터럴. rgba(...) 처럼 알파가 섞인 표기는 대상이 아니다(아래 테스트 주석 참고).
_HEX_LITERAL = re.compile(r"#[0-9a-fA-F]{6}\b")
_HEX_FULL = re.compile(r"^#[0-9a-fA-F]{6}$")


def _tokens():
    return tk.load_tokens()


def _light():
    return _tokens()["modes"]["light"]


def _dark():
    return _tokens()["modes"]["dark"]


def _with_patch(mutate):
    """원본 토큰을 복제해 변형한 뒤 임시 파일로 저장한다(로더 검증용)."""
    data = json.loads(json.dumps(_tokens()))
    mutate(data)
    return data


def test_schema_identifier_and_modes():
    data = _tokens()
    assert data["schema"] == "metro-tokens/1"
    assert set(data["modes"]) == {"light", "dark"}


def test_required_color_keys_present_in_both_modes():
    for mode in tk.MODES:
        color = _tokens()["modes"][mode]["color"]
        missing = [key for key in tk.REQUIRED_COLOR_KEYS if key not in color]
        assert not missing, f"{mode} 모드에 필수 색 토큰이 없습니다: {missing}"


def test_light_and_dark_share_key_sets():
    light = _light()
    dark = _dark()
    for group in tk.GROUPS:
        assert set(light[group]) == set(dark[group]), f"{group} 키 집합이 모드마다 다릅니다"


def test_light_body_text_meets_wcag_aa():
    light = _light()["color"]
    for background in ("bg", "surface"):
        ratio = tk.contrast_ratio(light["text-1"], light[background])
        assert ratio >= tk.AA_NORMAL_TEXT, f"light text-1/{background} = {ratio:.3f}:1"


def test_dark_body_text_meets_wcag_aa():
    dark = _dark()["color"]
    for background in ("bg", "surface"):
        ratio = tk.contrast_ratio(dark["text-1"], dark[background])
        assert ratio >= tk.AA_NORMAL_TEXT, f"dark text-1/{background} = {ratio:.3f}:1"


@pytest.mark.parametrize("mode", tk.MODES)
def test_secondary_text_meets_wcag_aa(mode):
    color = _tokens()["modes"][mode]["color"]
    for name in ("text-2", "text-3"):
        for background in ("bg", "surface"):
            ratio = tk.contrast_ratio(color[name], color[background])
            assert ratio >= tk.AA_NORMAL_TEXT, f"{mode} {name}/{background} = {ratio:.3f}:1"


@pytest.mark.parametrize("mode", tk.MODES)
def test_accent_text_on_accent_meets_wcag_aa(mode):
    color = _tokens()["modes"][mode]["color"]
    ratio = tk.contrast_ratio(color["accent-text"], color["accent"])
    assert ratio >= tk.AA_NORMAL_TEXT, f"{mode} accent-text/accent = {ratio:.3f}:1"


def test_text3_improved_and_legacy_value_kept():
    """3차 회색은 AA 미달 값(#9aa0a8)을 쓰지 않고, 그 값은 근거로 보존한다."""
    light = _light()["color"]
    legacy = styles_pkg.LIGHT_PALETTE[styles_pkg.QPalette.ColorRole.PlaceholderText]
    assert light["text-3-legacy"].lower() == legacy.lower()
    assert light["text-3"].lower() != light["text-3-legacy"].lower()
    assert tk.contrast_ratio(light["text-3-legacy"], light["surface"]) < tk.AA_NORMAL_TEXT
    assert tk.contrast_ratio(light["text-3"], light["surface"]) >= tk.AA_NORMAL_TEXT


def test_contrast_ratio_black_on_white_is_21():
    ratio = tk.contrast_ratio("#000000", "#ffffff")
    assert abs(ratio - 21.0) <= 0.01, f"검정/흰색 = {ratio}"


def test_contrast_ratio_same_color_is_1():
    for color in ("#ffffff", "#1f2328", "#374151"):
        ratio = tk.contrast_ratio(color, color)
        assert abs(ratio - 1.0) <= 0.01, f"{color} vs 자기 자신 = {ratio}"


def test_contrast_ratio_is_symmetric():
    forward = tk.contrast_ratio("#1f2328", "#f4f5f7")
    backward = tk.contrast_ratio("#f4f5f7", "#1f2328")
    assert abs(forward - backward) < 1e-12


def test_qss_hex_literals_all_exist_in_tokens():
    """modern_gray.qss 의 모든 #rrggbb 이 토큰 값 집합에 있어야 한다.

    알파가 섞인 rgba(255, 255, 255, 210) 한 건은 6자리 리터럴이 아니라 대상이 아니다.
    """
    token_values = set()
    for mode in _tokens()["modes"].values():
        token_values |= {str(value).lower() for value in mode["color"].values()}

    qss_path = Path(styles_pkg.__file__).with_name("modern_gray.qss")
    qss = qss_path.read_text(encoding="utf-8")
    literals = {match.lower() for match in _HEX_LITERAL.findall(qss)}

    missing = sorted(literals - token_values)
    assert not missing, f"{qss_path.name} 에서 토큰으로 표현되지 않는 색 {len(missing)}건: {missing}"


def test_font_family_matches_styles_module():
    expected = list(styles_pkg.FONT_FAMILIES)
    for mode in tk.MODES:
        assert _tokens()["modes"][mode]["font"]["family"] == expected, f"{mode} font.family 불일치"


def test_body_size_matches_styles_module():
    for mode in tk.MODES:
        assert _tokens()["modes"][mode]["font"]["size-body"] == styles_pkg.FONT_POINT_SIZE


def test_space_scale_follows_4px_grid():
    for mode in tk.MODES:
        space = _tokens()["modes"][mode]["space"]
        assert space["unit"] == 4, f"{mode} space.unit = {space['unit']}"
        off_grid = [value for value in space["scale"] if value % space["unit"] != 0]
        assert not off_grid, f"{mode} space.scale 이 4px 격자 밖: {off_grid}"


def test_light_palette_colors_absorbed():
    """기존 QPalette 값을 바꾸지 않고 흡수했는지(라이트 모드)."""
    light_values = {str(value).lower() for value in _light()["color"].values()}
    palette_values = {str(value).lower() for value in styles_pkg.LIGHT_PALETTE.values()}
    missing = sorted(palette_values - light_values)
    assert not missing, f"라이트 토큰에 없는 기존 팔레트 색: {missing}"


def test_check_contrast_covers_body_secondary_accent():
    expected = {
        f"{name}/{background}"
        for name in ("text-1", "text-2", "text-3", "accent")
        for background in ("bg", "surface")
    } | {"accent-text/accent"}
    for mode in tk.MODES:
        assert set(tk.check_contrast(mode)) == expected


def test_qss_color_lookup():
    assert tk.qss_color("text-2", "light") == _light()["color"]["text-2"].lower()
    assert tk.qss_color("color.text-2", "light") == tk.qss_color("text-2", "light")
    assert tk.qss_color("accent", "dark") == _dark()["color"]["accent"].lower()
    with pytest.raises(KeyError):
        tk.qss_color("no-such-token", "light")
    with pytest.raises(KeyError):
        tk.resolve("sepia")


def test_loader_rejects_wrong_schema(tmp_path):
    def mutate(data):
        data["schema"] = "metro-tokens/0"

    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(_with_patch(mutate)), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        tk.load_tokens(path)


def test_loader_rejects_missing_mode_and_bad_color(tmp_path):
    def drop_dark(data):
        del data["modes"]["dark"]

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(_with_patch(drop_dark)), encoding="utf-8")
    with pytest.raises(ValueError, match="모드"):
        tk.load_tokens(missing)

    def bad_color(data):
        data["modes"]["light"]["color"]["bg"] = "rgb(244, 245, 247)"

    invalid = tmp_path / "bad.json"
    invalid.write_text(json.dumps(_with_patch(bad_color)), encoding="utf-8")
    with pytest.raises(ValueError, match=r"#rrggbb"):
        tk.load_tokens(invalid)


# ---------- QSS 가 실제로 쓰는 색 조합 ----------


def _qss_blocks() -> list[tuple[str, dict[str, str]]]:
    """렌더된 QSS 를 (선택자, 선언 묶음) 목록으로 나눈다."""
    text = qss.load("modern_gray")
    blocks: list[tuple[str, dict[str, str]]] = []
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", text):
        selector = " ".join(match.group(1).split())
        declarations: dict[str, str] = {}
        for item in match.group(2).split(";"):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            declarations[key.strip().lower()] = value.strip()
        blocks.append((selector, declarations))
    return blocks


def test_qss_text_on_its_own_background_meets_wcag_aa():
    """같은 규칙에서 글자색과 배경색을 함께 적은 곳은 그 조합으로 대비를 잰다.

    표찰·배지·버튼처럼 배경을 직접 칠하는 글자가 대상이다. transparent·rgba 배경은 뒤가
    비쳐 보여 조합을 알 수 없으므로 여기서 재지 않는다.
    """
    checked: list[str] = []
    for selector, declarations in _qss_blocks():
        color = declarations.get("color")
        background = declarations.get("background")
        if not (color and background):
            continue
        if not (_HEX_FULL.match(color) and _HEX_FULL.match(background)):
            continue
        ratio = tk.contrast_ratio(color, background)
        checked.append(selector)
        assert ratio >= tk.AA_NORMAL_TEXT, f"{selector}: {color} / {background} = {ratio:.3f}:1"

    assert len(checked) >= 12, f"함께 잰 색 조합이 너무 적습니다: {checked}"
    assert any("gapTag" in selector for selector in checked), "예측 데이터 없음 표찰이 검사에서 빠졌다"
    assert any("modelCardRank" in selector for selector in checked)


def test_qss_plain_text_meets_wcag_aa_on_page_backgrounds():
    """배경을 적지 않은 규칙은 페이지 바탕(카드·배경) 위에서 읽히는지 본다."""
    light = _light()["color"]
    pages = {"surface": light["surface"], "bg": light["bg"]}
    problems: list[str] = []
    for selector, declarations in _qss_blocks():
        value = declarations.get("color")
        if not value or not _HEX_FULL.match(value) or "background" in declarations:
            continue
        if value.lower() == light["accent-text"].lower():
            continue  # 호선 배지 글자: 배경을 노선색으로 코드가 지정한다 (배지 전용 검사가 따로 있다)
        for name, page in pages.items():
            ratio = tk.contrast_ratio(value, page)
            if ratio < tk.AA_NORMAL_TEXT:
                problems.append(f"{selector}: {value} / {name} = {ratio:.3f}:1")
    assert not problems, problems