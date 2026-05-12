"""OpenAI gpt-image-1 renderer.

Prende l'image_prompt prodotto dall'art director (`brief.py`) e produce i
PNG bytes finali via /v1/images.

API quirks da ricordare:
  - le size valide sono SOLO: 1024x1024, 1024x1536, 1536x1024
  - quality: 'low' | 'medium' | 'high'
  - background: 'transparent' opzionale (utile per landing image, NON per ad)
  - response: b64_json (decodificato in bytes)
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

from openai import OpenAI

from agent.common import FORMAT_TO_SIZE, IMAGE_MODEL, Format, Quality


@dataclass
class RenderedImage:
    """Risultato del rendering: bytes PNG + size effettiva."""

    image_bytes: bytes = field(repr=False)
    size: str
    quality: Quality
    mime: str = "image/png"


def render(
    *,
    api_key: str,
    image_prompt: str,
    fmt: Format,
    quality: Quality = "medium",
    background: str | None = None,
) -> RenderedImage:
    """Chiama gpt-image-1 e ritorna i bytes PNG dell'immagine generata.

    Args:
        image_prompt: il prompt esteso prodotto dall'art director (200-400
            parole in inglese).
        fmt: 'square' | 'portrait' | 'landscape', mappato a una size accettata.
        quality: leva costo/qualita`. Default medium (~$0.04 per immagine).
        background: 'transparent' opzionale (sconsigliato per ad, OK per
            landing su sfondo brand).
    """
    if not image_prompt.strip():
        raise ValueError("image_prompt vuoto")
    if fmt not in FORMAT_TO_SIZE:
        raise ValueError(f"fmt deve essere in {list(FORMAT_TO_SIZE)}")

    size = FORMAT_TO_SIZE[fmt]

    client = OpenAI(api_key=api_key)
    kwargs: dict = {
        "model": IMAGE_MODEL,
        "prompt": image_prompt,
        "size": size,
        "quality": quality,
        "n": 1,
    }
    if background == "transparent":
        kwargs["background"] = "transparent"

    result = client.images.generate(**kwargs)
    b64 = result.data[0].b64_json
    if not b64:
        raise RuntimeError("gpt-image-1 ha ritornato b64_json vuoto")
    return RenderedImage(
        image_bytes=base64.b64decode(b64),
        size=size,
        quality=quality,
    )
