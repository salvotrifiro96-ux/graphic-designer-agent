from agent.brief import DesignBrief, _parse_items


def test_parses_full_item():
    items = [{
        "concept": "Split prima/dopo",
        "composition": "Vertical divider, dark left warm right",
        "palette_hex": ["#000000", "#FACC15"],
        "typography": "Inter Black uppercase",
        "mood": ["energico", "diretto"],
        "text_elements": ["+€3.500/MESE", "ZERO ADS"],
        "image_prompt": "A split-screen advertising image " * 30,
        "rationale": "pattern interrupt for feed scrolling",
    }]
    out = _parse_items(items)
    assert len(out) == 1
    b = out[0]
    assert isinstance(b, DesignBrief)
    assert b.palette_hex == ("#000000", "#FACC15")
    assert b.mood == ("energico", "diretto")
    assert b.text_elements == ("+€3.500/MESE", "ZERO ADS")


def test_drops_without_prompt():
    items = [{"concept": "x"}]
    assert _parse_items(items) == []


def test_drops_without_concept():
    items = [{"image_prompt": "x"}]
    assert _parse_items(items) == []


def test_empty_palette_ok():
    items = [{
        "concept": "x",
        "image_prompt": "y",
    }]
    out = _parse_items(items)
    assert len(out) == 1
    assert out[0].palette_hex == ()
    assert out[0].mood == ()
