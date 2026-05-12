from agent.refs import ReferenceDescription, _parse_one, merge_descriptions


def test_parses_full_vision_response():
    raw = (
        '{"composition": "rule of thirds, hero left",'
        ' "palette": "muted greens with cream, #5f6f52 / #f4f1e8",'
        ' "typography": "no text, image-only",'
        ' "mood": "calm, natural, premium",'
        ' "notable_elements": "soft natural light, grain visible"}'
    )
    desc = _parse_one(raw)
    assert isinstance(desc, ReferenceDescription)
    assert "rule of thirds" in desc.composition
    assert "#5f6f52" in desc.palette


def test_parses_with_code_fence():
    raw = (
        '```json\n'
        '{"composition": "c", "palette": "p", "typography": "t",'
        ' "mood": "m", "notable_elements": "n"}'
        '\n```'
    )
    desc = _parse_one(raw)
    assert desc.composition == "c"
    assert desc.mood == "m"


def test_parses_array_falls_back_to_first():
    raw = '[{"composition": "first", "palette": "p", "typography": "t", "mood": "m", "notable_elements": "n"}]'
    desc = _parse_one(raw)
    assert desc.composition == "first"


def test_missing_fields_become_empty():
    raw = '{"composition": "only"}'
    desc = _parse_one(raw)
    assert desc.composition == "only"
    assert desc.palette == ""


def test_merge_empty_list():
    assert merge_descriptions([]) == ""


def test_merge_multiple():
    a = ReferenceDescription("ca", "pa", "ta", "ma", "na")
    b = ReferenceDescription("cb", "pb", "tb", "mb", "nb")
    out = merge_descriptions([a, b])
    assert "Reference #1" in out
    assert "Reference #2" in out
    assert "ca" in out and "cb" in out
