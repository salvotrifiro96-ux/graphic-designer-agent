"""Shared helpers per il graphic-designer-agent.

Convenzioni del team agenti Leone:
- modello Claude unico definito qui (CLAUDE_MODEL)
- modello OpenAI image qui (IMAGE_MODEL)
- estrazione JSON tollerante a code fences
- rendering condizionale delle sezioni del prompt
- mapping formati commerciali -> size accettate da gpt-image-1
"""
from __future__ import annotations

import json
import re
from typing import Literal

# Allineato agli altri agenti del team.
CLAUDE_MODEL = "claude-sonnet-4-6"

# gpt-image-1 e` il modello image-gen disponibile via /v1/images. Accetta
# solo TRE size: square, portrait, landscape.
IMAGE_MODEL = "gpt-image-1"

# Aspect ratio commerciali piu` comuni mappati alle 3 size native gpt-image-1.
# L'operatore croppa eventualmente in post per arrivare al ratio Meta esatto.
Format = Literal["square", "portrait", "landscape"]

FORMAT_TO_SIZE: dict[Format, str] = {
    "square": "1024x1024",      # 1:1   — Meta feed, hero centrale
    "portrait": "1024x1536",    # 2:3   — Story, Reel, Pinterest, 4:5 Meta
    "landscape": "1536x1024",   # 3:2   — Landing hero, banner, 16:9
}

FORMAT_LABELS: dict[Format, str] = {
    "square": "Quadrato 1:1 (Meta feed, hero centrale)",
    "portrait": "Verticale 2:3 (Story, Reel, Pinterest, 4:5 Meta)",
    "landscape": "Orizzontale 3:2 (Landing hero, banner, 16:9)",
}

# Quality e` la leva principale costo/qualita`. Lasciamo le 3 di gpt-image-1:
#   low    ~ $0.011 per immagine 1024
#   medium ~ $0.042
#   high   ~ $0.17
# Default medium: ottimo trade-off per draft + iterazioni.
Quality = Literal["low", "medium", "high"]

QUALITY_LABELS: dict[Quality, str] = {
    "low": "Low (~$0.01) — solo draft veloci",
    "medium": "Medium (~$0.04) — default iterazione",
    "high": "High (~$0.17) — finale per produzione",
}


def extract_json(raw: str) -> list[dict] | dict:
    """Estrae JSON da una risposta Claude tollerando code fences opzionali."""
    raw = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    return json.loads(raw)


def section(label: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"\n## {label}\n{body}\n"


def clean_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(clean_str(v) for v in value if clean_str(v))
    return ()


HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def normalize_hex(value: str) -> str:
    """Normalizza un hex color a '#rrggbb' lower. Ritorna '' se non valido."""
    if not value:
        return ""
    v = value.strip()
    m = HEX_RE.match(v)
    if not m:
        return ""
    core = m.group(1).lower()
    if len(core) == 3:
        core = "".join(c * 2 for c in core)
    return f"#{core}"
