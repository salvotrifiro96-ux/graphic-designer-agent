"""Reference image vision — Claude descrive le immagini di reference uploadate.

L'operatore puo` caricare 1-3 immagini come "look & feel guide". Le mandiamo
a Claude in modalita` vision e gli chiediamo una descrizione strutturata
(composizione, palette, mood, tipografia, dettagli grafici). Quella descrizione
viene poi inserita nel design brief come "Visual references" per nutrire
l'image_prompt di gpt-image-1.

Perche` cosi`:
  - gpt-image-1 NON e` un modello image-to-image trasparente: passargli
    direttamente le reference via /images/edit modificherebbe quelle
    invece di partire pulito.
  - una descrizione testuale ben fatta cattura cio` che conta (struttura,
    palette, mood) senza vincolare gpt-image-1 a riprodurre dettagli
    irrilevanti.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Iterable

from anthropic import Anthropic

from agent.common import CLAUDE_MODEL, clean_str


@dataclass(frozen=True)
class ReferenceDescription:
    """Descrizione strutturata di una reference image."""

    composition: str
    palette: str
    typography: str
    mood: str
    notable_elements: str


_VISION_SYSTEM = (
    "Sei un art director senior. Ricevi una reference image che l'operatore "
    "vuole usare come ispirazione (NON da copiare) per generare un nuovo "
    "visual. Descrivi cio` che conta per replicare il *feel* dell'immagine:\n"
    "  - COMPOSITION: layout, regola dei terzi, focal point, gerarchia\n"
    "  - PALETTE: colori dominanti (incluso hex approssimativo se identificabile)\n"
    "    e relazione cromatica (complementari, monocromatici, ad alto contrasto)\n"
    "  - TYPOGRAPHY: se c'e` testo, descrivi famiglia (serif/sans/display),\n"
    "    peso, dimensione relativa, eventuale text-on-image style\n"
    "  - MOOD: 3-5 aggettivi (es. minimalista, energico, lussuoso, urgente)\n"
    "  - NOTABLE ELEMENTS: iconografia, texture, fotografia vs illustrazione,\n"
    "    grain, lighting, presenza di figure umane (eta`, etnia, espressione)\n\n"
    "Risposta in inglese (gpt-image-1 lavora meglio in inglese), 4-6 frasi\n"
    "complessive divise per campo. NIENTE prosa decorativa, niente 'questa\n"
    "immagine mostra...': parti dritto col contenuto. Output JSON con i 5 campi:\n"
    '  {"composition": "...", "palette": "...", "typography": "...",\n'
    '   "mood": "...", "notable_elements": "..."}\n'
)


def _b64_data_url(image_bytes: bytes, media_type: str) -> str:
    """Codifica una immagine per il content block image dell'API Anthropic."""
    return base64.b64encode(image_bytes).decode("utf-8")


def describe_one(
    *,
    api_key: str,
    image_bytes: bytes,
    media_type: str = "image/jpeg",
) -> ReferenceDescription:
    """Manda UNA reference image a Claude e ritorna la descrizione strutturata."""
    if not image_bytes:
        raise ValueError("image_bytes vuoto")

    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=900,
        system=_VISION_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": _b64_data_url(image_bytes, media_type),
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Descrivi questa reference come da istruzioni. "
                            "Risposta SOLO JSON."
                        ),
                    },
                ],
            }
        ],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return _parse_one(text)


def _parse_one(text: str) -> ReferenceDescription:
    """Estrae il JSON e costruisce il dataclass."""
    from agent.common import extract_json

    data = extract_json(text)
    if isinstance(data, list):
        # se Claude restituisce un array (capita raramente), prendiamo il primo
        data = data[0] if data else {}
    if not isinstance(data, dict):
        raise ValueError("Vision response non e` un oggetto JSON")
    return ReferenceDescription(
        composition=clean_str(data.get("composition")),
        palette=clean_str(data.get("palette")),
        typography=clean_str(data.get("typography")),
        mood=clean_str(data.get("mood")),
        notable_elements=clean_str(data.get("notable_elements")),
    )


def describe_many(
    *,
    api_key: str,
    images: Iterable[tuple[bytes, str]],
) -> list[ReferenceDescription]:
    """Convenience: descrive piu` reference, ritorna nello stesso ordine.

    `images` e` un iterable di (image_bytes, media_type).
    """
    out: list[ReferenceDescription] = []
    for image_bytes, media_type in images:
        out.append(
            describe_one(
                api_key=api_key,
                image_bytes=image_bytes,
                media_type=media_type,
            )
        )
    return out


def merge_descriptions(descs: list[ReferenceDescription]) -> str:
    """Concatena descrizioni multiple in un blocco testuale per il brief prompt."""
    if not descs:
        return ""
    parts: list[str] = []
    for i, d in enumerate(descs, 1):
        parts.append(
            f"Reference #{i}:\n"
            f"  Composition: {d.composition}\n"
            f"  Palette: {d.palette}\n"
            f"  Typography: {d.typography}\n"
            f"  Mood: {d.mood}\n"
            f"  Notable elements: {d.notable_elements}"
        )
    return "\n\n".join(parts)
