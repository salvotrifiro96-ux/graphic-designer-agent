import json

import pytest

from agent.common import (
    FORMAT_TO_SIZE,
    clean_list,
    clean_str,
    extract_json,
    normalize_hex,
    section,
)


class TestFormatMapping:
    def test_three_formats(self):
        assert set(FORMAT_TO_SIZE) == {"square", "portrait", "landscape"}

    def test_sizes_are_openai_valid(self):
        # OpenAI gpt-image-1 supporta ESATTAMENTE queste 3 size
        assert FORMAT_TO_SIZE["square"] == "1024x1024"
        assert FORMAT_TO_SIZE["portrait"] == "1024x1536"
        assert FORMAT_TO_SIZE["landscape"] == "1536x1024"


class TestExtractJson:
    def test_array(self):
        assert extract_json('[{"a": 1}]') == [{"a": 1}]

    def test_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_strips_fence(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_invalid_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("not json")


class TestSection:
    def test_empty_returns_empty(self):
        assert section("L", "") == ""
        assert section("L", "   ") == ""

    def test_renders(self):
        out = section("Target", "imprenditori 35-55")
        assert "## Target" in out
        assert "imprenditori 35-55" in out


class TestCleanStr:
    @pytest.mark.parametrize("value,expected", [
        (None, ""),
        ("", ""),
        ("  hi  ", "hi"),
        (3, "3"),
    ])
    def test_clean(self, value, expected):
        assert clean_str(value) == expected


class TestCleanList:
    def test_none(self):
        assert clean_list(None) == ()

    def test_string(self):
        assert clean_list("#000") == ("#000",)

    def test_empty_string(self):
        assert clean_list("") == ()

    def test_list_filters(self):
        assert clean_list(["a", "", "  ", "b"]) == ("a", "b")


class TestNormalizeHex:
    def test_passthrough_6(self):
        assert normalize_hex("#FACC15") == "#facc15"

    def test_no_hash(self):
        assert normalize_hex("facc15") == "#facc15"

    def test_short_form_expanded(self):
        assert normalize_hex("#f0a") == "#ff00aa"

    def test_invalid_returns_empty(self):
        assert normalize_hex("not-a-hex") == ""
        assert normalize_hex("#zzzzzz") == ""
        assert normalize_hex("#12345") == ""

    def test_empty(self):
        assert normalize_hex("") == ""
        assert normalize_hex("   ") == ""
