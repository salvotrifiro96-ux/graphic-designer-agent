"""Modifica immagine esistente.

Due strategie complementari:

1) **Rebrief**: l'operatore da` un feedback testuale, Claude rifa` il design
   brief (`brief.regenerate_brief`), e poi gpt-image-1 genera UN NUOVO PNG
   ex novo. La maggior parte dei casi: feedback strutturali, cambio
   composizione, palette, mood.

2) **Image-edit**: per modifiche localizzate (es. "cambia il colore della
   maglia in rosso", "togli l'oggetto in alto a destra"), passiamo l'immagine
   ORIGINALE a /v1/images/edits con un prompt corto. Piu` veloce e meno
   distruttivo del rebrief, ma puo` solo modificare quello che vede — non
   re-impostare un design completamente diverso.

L'app espone la #1 come modalita` principale. La #2 e` opt-in tramite checkbox
"Mantieni immagine, modifica solo localmente".
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from openai import OpenAI

from agent.common import FORMAT_TO_SIZE, IMAGE_MODEL, Format, Quality
from agent.render import RenderedImage


@dataclass
class EditedImage:
    image_bytes: bytes
    size: str
    quality: Quality
    edit_prompt: str
    mime: str = "image/png"


def edit_local(
    *,
    api_key: str,
    source_image_bytes: bytes,
    edit_prompt: str,
    fmt: Format,
    quality: Quality = "medium",
) -> EditedImage:
    """Modifica localizzata di una immagine esistente via /v1/images/edits.

    NON usa maschera: gpt-image-1 capisce la modifica dal prompt + immagine.
    Per modifiche radicali usa invece il `rebrief` + render completo.
    """
    if not source_image_bytes:
        raise ValueError("source_image_bytes vuoto")
    if not edit_prompt.strip():
        raise ValueError("edit_prompt vuoto")
    if fmt not in FORMAT_TO_SIZE:
        raise ValueError(f"fmt deve essere in {list(FORMAT_TO_SIZE)}")

    client = OpenAI(api_key=api_key)
    # Il client SDK accetta un file-like con .name per inferire il content type.
    src = io.BytesIO(source_image_bytes)
    src.name = "source.png"

    result = client.images.edit(
        model=IMAGE_MODEL,
        image=src,
        prompt=edit_prompt,
        size=FORMAT_TO_SIZE[fmt],
        quality=quality,
        n=1,
    )
    b64 = result.data[0].b64_json
    if not b64:
        raise RuntimeError("gpt-image-1 (edit) ha ritornato b64_json vuoto")
    return EditedImage(
        image_bytes=base64.b64decode(b64),
        size=FORMAT_TO_SIZE[fmt],
        quality=quality,
        edit_prompt=edit_prompt,
    )


def to_rendered(edit: EditedImage) -> RenderedImage:
    """Converte un EditedImage in un RenderedImage cosi` l'UI tratta tutto
    in modo uniforme."""
    return RenderedImage(
        image_bytes=edit.image_bytes,
        size=edit.size,
        quality=edit.quality,
    )
