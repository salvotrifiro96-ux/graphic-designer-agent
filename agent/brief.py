"""Art Director (Claude) — produce un design brief umano-leggibile + un
image_prompt rigoroso da passare a gpt-image-1.

Filosofia: l'agente non sputa direttamente un prompt OpenAI. Prima ragiona
come un art director vero: legge il context, decide la composizione, sceglie
la palette in funzione di brand + reference, ragiona sulla tipografia, e
infine traduce tutto in un image_prompt molto dettagliato. Cosi` l'operatore
vede *cosa* ha pensato l'agente prima di vedere l'immagine — esattamente
come farebbe un designer umano in un brief.

Use case supportati:
  - "visual_ad":  Meta/Instagram/TikTok ad design, con composizione tipica
                  da advertising (split-screen, callout, big-text, ecc.)
  - "landing":    immagine evocativa per landing page, piu` editoriale,
                  meno testo, mood-driven
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from anthropic import Anthropic

from agent.common import (
    CLAUDE_MODEL,
    Format,
    clean_list,
    clean_str,
    extract_json,
    section,
)

UseCase = Literal["visual_ad", "landing"]

TextMode = Literal["none", "headline", "auto"]
"""
- none:     l'immagine non contiene testo rendered. Il copy vivra` fuori.
- headline: solo UNA headline dominante in image.
- auto:     full advertising design (headline + sub + callout + CTA bar).
"""

MIN_VARIANTS = 1
MAX_VARIANTS = 4


@dataclass(frozen=True)
class DesignBrief:
    """Brief prodotto dall'Art Director per UNA variante.

    Campi:
      - concept: 1-2 frasi sul concept (in italiano, leggibile dall'operatore)
      - composition: descrizione composizione (split-screen, hero+callouts...)
      - palette_hex: lista di colori hex usati (dominante + accent)
      - typography: famiglia/peso tipografico, solo se TextMode != none
      - mood: 3-5 aggettivi che descrivono il mood
      - text_elements: lista delle stringhe di testo da rendering
                       (headline, sub, callout, CTA). Vuota se TextMode=none.
      - image_prompt: prompt finale (inglese, 200-400 parole) per gpt-image-1.
                      Questo e` cio` che mandiamo all'API image-gen.
      - rationale: perche` questa composizione funziona per il target/use case
    """

    concept: str
    composition: str
    palette_hex: tuple[str, ...]
    typography: str
    mood: tuple[str, ...]
    text_elements: tuple[str, ...]
    image_prompt: str
    rationale: str


# ── System prompts ────────────────────────────────────────────────


_SHARED_RULES = (
    "## REGOLE INVALICABILI per image_prompt\n"
    "1. Lingua: INGLESE (gpt-image-1 e` molto piu` affidabile in inglese).\n"
    "2. Lunghezza: 200-400 parole. Vague prompts = output banale.\n"
    "3. Dettagli OBBLIGATORI da nominare esplicitamente nel prompt:\n"
    "     - composition (layout spaziale, regola dei terzi, focal point)\n"
    "     - palette (hex codes esatti se forniti, descrizione della relazione\n"
    "       cromatica — complementare, ad alto contrasto, monocromatico)\n"
    "     - lighting/grading (dark moody / warm golden / clean studio / ecc.)\n"
    "     - texture/medium (photographic, editorial photo, 3D render, flat\n"
    "       illustration, mixed media)\n"
    "     - any text element MUST be quoted exactly inline in the prompt\n"
    "4. Personaggi: se compaiono persone, specifica eta`/etnia/espressione\n"
    "   realistica. NIENTE 'plastic stock-photo smile'. Mai dire 'beautiful',\n"
    "   'happy' generici — descrivi la micro-expression reale.\n"
    "5. NIENTE logos di brand reali. Riserva uno slot vuoto bottom-right per\n"
    "   logo da inserire in post-produzione (descrivi solo la posizione, non\n"
    "   tentare di renderizzare un logo specifico).\n"
)


def _visual_ad_system(text_mode: TextMode) -> str:
    if text_mode == "none":
        text_policy = (
            "TEXT POLICY: NIENTE testo rendered in image. Solo composizione, "
            "palette, iconografia, personaggi. Il copy vivra` come testo Meta "
            "fuori dall'immagine."
        )
    elif text_mode == "headline":
        text_policy = (
            "TEXT POLICY: SOLO UNA headline rendered in image (3-5 parole, "
            "uppercase bold, top of frame OR center). NIENTE sub, callout, "
            "CTA bar. Una sola text element."
        )
    else:  # auto
        text_policy = (
            "TEXT POLICY: full advertising design. Headline dominante (3-6 "
            "parole), eventuale sub-headline (1 linea), 2-3 callout (1-3 "
            "parole l'uno, dentro pill o badge), CTA bar in basso (verbo "
            "azione di 2-4 parole). Tutti gli elementi di testo elencati "
            "vanno scritti tra virgolette ESATTAMENTE nel image_prompt."
        )

    return (
        "Sei un Art Director senior di advertising performance, specializzato in "
        "Meta/Instagram/TikTok ads italiani. Produci design brief in stile "
        "campagna direct-response — non stock photo con headline appiccicata.\n\n"
        f"{text_policy}\n\n"
        "## COMPOSIZIONI tipiche (scegli UNA per variante, no doppioni nelle "
        "varianti di una stessa richiesta)\n"
        "  A. SPLIT-SCREEN prima/dopo: divider verticale al centro, lato sx\n"
        "     scena 'before' (dark, frustrato, caotico), lato dx 'after'\n"
        "     (warm, ordinato, sereno). Centro: icona che collega.\n"
        "  B. HERO + CALLOUTS: persona italiana 30-50 centrata, 3-4 callout\n"
        "     fluttuanti con icona + label (es. '+€3.500/MESE', 'ZERO ADS').\n"
        "  C. BIG-TEXT DOMINANT: 60% frame e` headline gigantesca bold, 40%\n"
        "     foto contestuale. Stile billboard/poster.\n"
        "  D. NUMBER SPOTLIGHT: un numero enorme (es. '+247%') come hero,\n"
        "     supporto fotografico o iconografico ai lati.\n"
        "  E. PRODUCT-IN-CONTEXT: il prodotto/servizio in uso reale, fotografato\n"
        "     come editoriale, con UNA didascalia/headline in basso.\n\n"
        f"{_SHARED_RULES}\n"
        "## OUTPUT\n"
        "Rispondi SOLO con un array JSON, niente prosa, niente fences. Ogni\n"
        "elemento e` un design brief con questi campi (TUTTI obbligatori):\n"
        '  {"concept":        "1-2 frasi italiano leggibile",\n'
        '   "composition":    "etichetta composizione + dettaglio (es.\n'
        '                      Split-screen prima/dopo, divider centrale)",\n'
        '   "palette_hex":    ["#xxxxxx", "#yyyyyy", ...],\n'
        '   "typography":     "famiglia/peso (vuoto se TextMode=none)",\n'
        '   "mood":           ["aggettivo1", "aggettivo2", ...],\n'
        '   "text_elements":  ["esatto testo 1", "esatto testo 2", ...],\n'
        '   "image_prompt":   "stringa inglese, 200-400 parole",\n'
        '   "rationale":      "max 200 char, perche` funziona su target"}\n'
    )


def _landing_system(text_mode: TextMode) -> str:
    if text_mode == "none":
        text_policy = (
            "TEXT POLICY: NIENTE testo. Solo immagine evocativa, mood-driven, "
            "editoriale. Il copy della landing vive fuori dall'immagine."
        )
    elif text_mode == "headline":
        text_policy = (
            "TEXT POLICY: SOLO una micro-headline o claim discreto, integrato "
            "tipograficamente nell'immagine (NON sovrimpresso a forza). Stile "
            "magazine cover, non Meta ad."
        )
    else:  # auto
        text_policy = (
            "TEXT POLICY: testo discreto integrato editorialmente (headline + "
            "eventuale tagline). Mai stile ad-design aggressivo: una landing "
            "image deve essere evocativa, non promozionale."
        )

    return (
        "Sei un Art Director senior specializzato in landing page editoriali per "
        "info-prodotti e brand premium. Produci design brief per immagini hero "
        "che evocano l'outcome desiderato del target — non promo, non ad.\n"
        "Tono visivo: editoriale, fotografico, cinema-style, magazine cover.\n\n"
        f"{text_policy}\n\n"
        "## STILI tipici (scegli UNO per variante, no doppioni)\n"
        "  A. CINEMATIC HERO: persona/scena con grading da film, depth of field\n"
        "     basso, golden hour o blue hour. Composizione regola dei terzi.\n"
        "  B. EDITORIAL STILL: dettaglio prodotto/oggetto + texture naturale,\n"
        "     stile fotografia di rivista architettura/lifestyle.\n"
        "  C. ABSTRACT MOOD: composizione astratta o semi-astratta che evoca\n"
        "     l'outcome (es. orizzonte aperto, mani che costruiscono, luce\n"
        "     attraverso una finestra).\n"
        "  D. ARCHITECTURAL FRAME: ambiente costruito (studio, casa, ufficio)\n"
        "     fotografato da angolo cinematografico, con figura umana piccola\n"
        "     ma significativa.\n"
        "  E. SYMBOLIC SCENE: una scena con un simbolo forte (es. una porta\n"
        "     aperta, una scala, una mano tesa), trattata fotograficamente.\n\n"
        f"{_SHARED_RULES}\n"
        "## OUTPUT\n"
        "Rispondi SOLO con un array JSON. Schema identico a quello del modo\n"
        "visual_ad (concept, composition, palette_hex, typography, mood,\n"
        "text_elements, image_prompt, rationale).\n"
    )


# ── Prompt building + Claude call ─────────────────────────────────


def _build_user_prompt(
    *,
    use_case: UseCase,
    fmt: Format,
    brief: str,
    promise: str,
    target_audience: str,
    brand_voice: str,
    brand_visual: str,
    palette_hex: tuple[str, ...],
    references_blob: str,
    style_notes: str,
    text_mode: TextMode,
    n_variants: int,
    extra_instructions: str,
) -> str:
    use_case_label = (
        "Visual Ad (Meta/Instagram/TikTok)" if use_case == "visual_ad"
        else "Landing page hero image"
    )
    parts: list[str] = [
        section("Use case", use_case_label),
        section("Formato target", f"{fmt} (output size mappato a gpt-image-1)"),
        section("Text mode", text_mode),
        section("Target audience", target_audience),
        section("Brand voice (verbale)", brand_voice),
        section(
            "Brand visual identity (descrizione: stile, materiali, riferimenti)",
            brand_visual,
        ),
        section(
            "Palette colori brand (hex)",
            ", ".join(palette_hex) if palette_hex else "",
        ),
        section(
            "Brief di cosa l'immagine deve COMUNICARE",
            brief,
        ),
        section(
            "Promessa/headline a cui ancorarsi (per text_elements)",
            promise,
        ),
        section(
            "Reference images (descrizioni vision-extracted)",
            references_blob,
        ),
        section(
            "Note stilistiche operatore (vincoli aggiuntivi)",
            style_notes,
        ),
        section("Istruzioni extra", extra_instructions),
        f"\n## Task\nProduci esattamente {n_variants} design brief. Ogni\n"
        f"variante usa una composizione/stile DIVERSO (no doppioni). Restituisci\n"
        f"SOLO un array JSON.\n",
    ]
    return "".join(p for p in parts if p)


def _call_claude(
    *, api_key: str, system: str, user_prompt: str
) -> list[dict]:
    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    parsed = extract_json(text)
    if not isinstance(parsed, list):
        raise ValueError("Risposta Claude non e` un array JSON")
    return parsed


def _parse_items(items: list[dict]) -> list[DesignBrief]:
    out: list[DesignBrief] = []
    for it in items:
        prompt = clean_str(it.get("image_prompt"))
        concept = clean_str(it.get("concept"))
        if not prompt or not concept:
            continue
        out.append(
            DesignBrief(
                concept=concept,
                composition=clean_str(it.get("composition")),
                palette_hex=clean_list(it.get("palette_hex")),
                typography=clean_str(it.get("typography")),
                mood=clean_list(it.get("mood")),
                text_elements=clean_list(it.get("text_elements")),
                image_prompt=prompt,
                rationale=clean_str(it.get("rationale")),
            )
        )
    return out


def write_briefs(
    *,
    api_key: str,
    use_case: UseCase,
    fmt: Format,
    brief: str,
    promise: str = "",
    target_audience: str = "",
    brand_voice: str = "",
    brand_visual: str = "",
    palette_hex: tuple[str, ...] = (),
    references_blob: str = "",
    style_notes: str = "",
    text_mode: TextMode = "auto",
    n_variants: int = 2,
    extra_instructions: str = "",
) -> list[DesignBrief]:
    """Genera N design brief diversi per la stessa richiesta operatore."""
    if not brief.strip():
        raise ValueError("`brief` (cosa deve comunicare l'immagine) e` obbligatorio")
    if n_variants < MIN_VARIANTS or n_variants > MAX_VARIANTS:
        raise ValueError(
            f"n_variants in [{MIN_VARIANTS}, {MAX_VARIANTS}], ricevuto {n_variants}"
        )

    system = (
        _visual_ad_system(text_mode) if use_case == "visual_ad"
        else _landing_system(text_mode)
    )
    user_prompt = _build_user_prompt(
        use_case=use_case,
        fmt=fmt,
        brief=brief,
        promise=promise,
        target_audience=target_audience,
        brand_voice=brand_voice,
        brand_visual=brand_visual,
        palette_hex=palette_hex,
        references_blob=references_blob,
        style_notes=style_notes,
        text_mode=text_mode,
        n_variants=n_variants,
        extra_instructions=extra_instructions,
    )
    items = _call_claude(api_key=api_key, system=system, user_prompt=user_prompt)
    return _parse_items(items)


def regenerate_brief(
    *,
    api_key: str,
    use_case: UseCase,
    fmt: Format,
    original: DesignBrief,
    feedback: str,
    brief: str,
    promise: str = "",
    target_audience: str = "",
    brand_voice: str = "",
    brand_visual: str = "",
    palette_hex: tuple[str, ...] = (),
    references_blob: str = "",
    style_notes: str = "",
    text_mode: TextMode = "auto",
) -> DesignBrief:
    """Rigenera UN singolo brief con feedback operatore."""
    if not feedback.strip():
        raise ValueError("feedback obbligatorio per rigenerare")

    original_block = (
        f"  Concept: {original.concept}\n"
        f"  Composition: {original.composition}\n"
        f"  Palette: {', '.join(original.palette_hex)}\n"
        f"  Typography: {original.typography}\n"
        f"  Mood: {', '.join(original.mood)}\n"
        f"  Text elements: {' / '.join(original.text_elements)}\n"
        f"  Image prompt (originale):\n{original.image_prompt}"
    )
    instructions = (
        "Stai rivedendo UNA singola variante. Versione originale:\n"
        f"{original_block}\n\n"
        f"Feedback operatore:\n  {feedback.strip()}\n\n"
        "Restituisci un array JSON con UN SOLO elemento — la nuova variante,\n"
        "che incorpora il feedback ma resta sullo stesso target/use case."
    )

    system = (
        _visual_ad_system(text_mode) if use_case == "visual_ad"
        else _landing_system(text_mode)
    )
    user_prompt = _build_user_prompt(
        use_case=use_case,
        fmt=fmt,
        brief=brief,
        promise=promise,
        target_audience=target_audience,
        brand_voice=brand_voice,
        brand_visual="",
        palette_hex=palette_hex,
        references_blob=references_blob,
        style_notes="",
        text_mode=text_mode,
        n_variants=1,
        extra_instructions=instructions,
    )
    items = _call_claude(api_key=api_key, system=system, user_prompt=user_prompt)
    briefs = _parse_items(items)
    if not briefs:
        raise ValueError("Rigenerazione non ha prodotto risultati validi")
    return briefs[0]
