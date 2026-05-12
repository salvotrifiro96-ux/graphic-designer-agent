"""Graphic Designer Agent — Streamlit UI.

Due tab principali:
  - Visual Ads (Meta/Instagram/TikTok ad design)
  - Landing image (immagini evocative editoriali)

Flow comune per ogni richiesta:
  1) operatore compila brief + reference + palette + formato
  2) Claude vision descrive eventuali reference uploadate
  3) Claude (Art Director) produce N design brief leggibili + image_prompt
  4) gpt-image-1 genera i PNG
  5) operatore puo` rigenerare con feedback (modalita`: rebrief completo
     oppure edit locale dell'immagine)
"""
from __future__ import annotations

import os
import traceback
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent import brief as brief_mod
from agent import edit as edit_mod
from agent import refs as refs_mod
from agent import render as render_mod
from agent.common import (
    FORMAT_LABELS,
    FORMAT_TO_SIZE,
    QUALITY_LABELS,
    Format,
    Quality,
    normalize_hex,
)


load_dotenv()


def _secret(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except (FileNotFoundError, AttributeError):
        return default


ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
OPENAI_API_KEY = _secret("OPENAI_API_KEY")
APP_PASSWORD = _secret("APP_PASSWORD")

st.set_page_config(page_title="Graphic Designer Agent", layout="wide", page_icon="🎨")


# ── Password gate ──────────────────────────────────────────────────
def _password_gate() -> None:
    if not APP_PASSWORD:
        return
    if st.session_state.get("authed"):
        return
    st.title("🎨 Graphic Designer Agent")
    pw = st.text_input("Password", type="password", key="pw_input")
    if st.button("Entra"):
        if pw == APP_PASSWORD:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Password errata")
    st.stop()


_password_gate()


# ── Session state ──────────────────────────────────────────────────
# Salviamo per ogni tab i "result items": ciascuno e` un dict con
#  - brief: DesignBrief
#  - image: RenderedImage | None
#  - last_inputs: dict (per regenerate)
DEFAULT_STATE: dict[str, Any] = {
    "ads_items": [],
    "ads_inputs": None,
    "landing_items": [],
    "landing_inputs": None,
    "ref_descriptions": [],   # descrizioni vision delle reference correnti
    "ref_blob": "",           # blob testuale unito (input a Claude AD)
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset_tab(prefix: str) -> None:
    for k in DEFAULT_STATE:
        if k.startswith(prefix):
            st.session_state[k] = (
                [] if isinstance(DEFAULT_STATE[k], list) else DEFAULT_STATE[k]
            )


# ── Sidebar (brand, palette, reference, quality) ───────────────────


def _sidebar() -> dict[str, Any]:
    st.sidebar.header("🎨 Identita` di brand")
    if not ANTHROPIC_API_KEY:
        st.sidebar.error("Manca `ANTHROPIC_API_KEY`")
    if not OPENAI_API_KEY:
        st.sidebar.error("Manca `OPENAI_API_KEY`")

    target = st.sidebar.text_area(
        "Target audience",
        height=80,
        placeholder="Es. imprenditori 35-55 in cerca di sistemi automatici",
        key="_sb_target",
    )
    brand_voice = st.sidebar.text_area(
        "Brand voice (verbale)",
        height=70,
        placeholder="Es. diretto, pragmatico, italiano",
        key="_sb_voice",
    )
    brand_visual = st.sidebar.text_area(
        "Brand visual identity (descrizione)",
        height=110,
        placeholder=(
            "Es. estetica editoriale, palette nero+ocra, fotografia\n"
            "documentaristica, no rendering 3D, tipografia bold sans-serif"
        ),
        key="_sb_visual",
    )

    st.sidebar.subheader("Palette colori")
    use_palette = st.sidebar.checkbox(
        "Vincola la palette ai colori brand",
        value=st.session_state.get("_sb_use_palette", True),
        key="_sb_use_palette",
        help=(
            "Se attivo, i colori scelti qui sotto vengono passati all'art "
            "director come palette da rispettare. Se disattivo, lasci\n"
            "scegliere liberamente al modello."
        ),
    )
    c1, c2, c3 = st.sidebar.columns(3)
    primary_raw = c1.color_picker(
        "Primario",
        value=st.session_state.get("_sb_primary", "#000000"),
        key="_sb_primary",
    )
    accent_raw = c2.color_picker(
        "Accent",
        value=st.session_state.get("_sb_accent", "#facc15"),
        key="_sb_accent",
    )
    third_raw = c3.color_picker(
        "Terzo",
        value=st.session_state.get("_sb_third", "#ffffff"),
        key="_sb_third",
    )
    palette: list[str] = []
    if use_palette:
        for raw in (primary_raw, accent_raw, third_raw):
            normalized = normalize_hex(raw)
            if normalized:
                palette.append(normalized)

    st.sidebar.subheader("Reference images (opzionali, max 3)")
    uploaded = st.sidebar.file_uploader(
        "Trascina qui le ispirazioni",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="_sb_refs",
    )
    if uploaded and len(uploaded) > 3:
        st.sidebar.warning("Uso solo le prime 3 reference.")
        uploaded = uploaded[:3]

    if uploaded:
        if st.sidebar.button(
            "👁 Analizza references",
            use_container_width=True,
            type="primary",
        ):
            with st.sidebar:
                with st.spinner("Vision in corso…"):
                    try:
                        images = [
                            (f.getvalue(), f.type or "image/jpeg")
                            for f in uploaded
                        ]
                        descs = refs_mod.describe_many(
                            api_key=ANTHROPIC_API_KEY,
                            images=images,
                        )
                        st.session_state.ref_descriptions = descs
                        st.session_state.ref_blob = refs_mod.merge_descriptions(descs)
                        st.success(f"{len(descs)} reference analizzate.")
                    except Exception as e:
                        st.error(f"Vision fallita: {e}")
    if st.session_state.ref_descriptions:
        with st.sidebar.expander(
            f"✓ {len(st.session_state.ref_descriptions)} reference analizzate"
        ):
            for i, d in enumerate(st.session_state.ref_descriptions, 1):
                st.markdown(f"**#{i}** _{d.mood}_")
                st.caption(f"Palette: {d.palette}")
        if st.sidebar.button("🗑 Svuota reference", use_container_width=True):
            st.session_state.ref_descriptions = []
            st.session_state.ref_blob = ""
            st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Default rendering")
    quality: Quality = st.sidebar.selectbox(  # type: ignore[assignment]
        "Quality (impatta costo OpenAI)",
        options=list(QUALITY_LABELS),
        format_func=lambda q: QUALITY_LABELS[q],
        index=1,  # medium
        key="_sb_quality",
    )

    if st.sidebar.button("🔄 Reset totale", use_container_width=True):
        for k in list(DEFAULT_STATE):
            st.session_state[k] = (
                [] if isinstance(DEFAULT_STATE[k], list) else DEFAULT_STATE[k]
            )
        st.rerun()

    return {
        "target_audience": (target or "").strip(),
        "brand_voice": (brand_voice or "").strip(),
        "brand_visual": (brand_visual or "").strip(),
        "palette_hex": tuple(palette),
        "quality": quality,
    }


# ── Shared rendering helpers ───────────────────────────────────────


def _render_brief_card(brief: brief_mod.DesignBrief) -> None:
    """Renderizza il design brief (la parte 'umana' del designer)."""
    st.markdown(f"**Concept:** {brief.concept}")
    st.markdown(f"**Composizione:** {brief.composition}")
    if brief.palette_hex:
        swatches = "".join(
            f'<span style="display:inline-block; width:18px; height:18px; '
            f'background:{h}; margin-right:3px; border:1px solid #ccc; '
            f'border-radius:3px; vertical-align:middle;"></span>'
            for h in brief.palette_hex
        )
        st.markdown(
            f"**Palette:** {swatches} "
            f"<span style='color:#666; font-size:0.85rem;'>"
            f"{', '.join(brief.palette_hex)}</span>",
            unsafe_allow_html=True,
        )
    if brief.typography:
        st.markdown(f"**Tipografia:** {brief.typography}")
    if brief.mood:
        st.caption("Mood: " + ", ".join(brief.mood))
    if brief.text_elements:
        st.markdown(
            "**Testo nell'immagine:** "
            + " · ".join(f'«{t}»' for t in brief.text_elements)
        )
    with st.expander("📝 Image prompt (cosa va a OpenAI)"):
        st.code(brief.image_prompt, language=None)
    if brief.rationale:
        st.caption(f"Razionale: {brief.rationale}")


def _trigger_render(
    *,
    brief: brief_mod.DesignBrief,
    fmt: Format,
    quality: Quality,
) -> render_mod.RenderedImage | None:
    """Chiama gpt-image-1 e ritorna RenderedImage, o None su errore (mostrato)."""
    try:
        with st.spinner("Generazione immagine (10-30s)…"):
            return render_mod.render(
                api_key=OPENAI_API_KEY,
                image_prompt=brief.image_prompt,
                fmt=fmt,
                quality=quality,
            )
    except Exception as e:
        st.error(f"Render fallito: {e}")
        st.caption(traceback.format_exc())
        return None


# ── Tab Visual Ads ─────────────────────────────────────────────────


TEXT_MODE_LABELS = {
    "auto": "Auto — full ad design (headline + sub + callout + CTA bar)",
    "headline": "Headline — solo una headline grande",
    "none": "None — nessun testo nell'immagine",
}


def _render_visual_ads_tab(sidebar: dict[str, Any]) -> None:
    st.subheader("📣 Visual Ads — Meta, Instagram, TikTok")
    st.caption(
        "Composizioni da advertising performance: split-screen prima/dopo, "
        "hero+callouts, big-text, number-spotlight."
    )

    with st.form("ads_form"):
        cols = st.columns([2, 1, 1])
        fmt: Format = cols[0].selectbox(  # type: ignore[assignment]
            "Formato",
            options=list(FORMAT_LABELS),
            format_func=lambda f: FORMAT_LABELS[f],
            index=0,  # square
            key="ads_fmt",
        )
        n_variants = cols[1].slider(
            "Varianti",
            min_value=brief_mod.MIN_VARIANTS,
            max_value=brief_mod.MAX_VARIANTS,
            value=2,
            key="ads_n",
        )
        text_mode: brief_mod.TextMode = cols[2].selectbox(  # type: ignore[assignment]
            "Testo nell'immagine",
            options=list(TEXT_MODE_LABELS),
            format_func=lambda m: TEXT_MODE_LABELS[m],
            index=0,
            key="ads_textmode",
        )

        promise = st.text_input(
            "Promessa/headline da renderizzare nella visual (se text_mode != none)",
            placeholder="Es. 5 NUOVI CLIENTI / MESE IN 90 GIORNI",
        )
        brief_input = st.text_area(
            "📥 Brief — cosa deve COMUNICARE l'immagine",
            height=200,
            placeholder=(
                "Carica tutto: cosa vendiamo, a chi parla, qual e` il pain del\n"
                "target, qual e` il dream outcome, su quale leva emotiva\n"
                "vogliamo battere (urgency, autorita`, scarcity, sociale)."
            ),
        )
        style = st.text_area(
            "Note stilistiche (vincoli aggiuntivi, opzionale)",
            height=80,
            placeholder=(
                "Es. niente illustrazioni 3D, vogliamo foto realistiche; "
                "evita persone sopra i 60; tonalita` calde."
            ),
        )
        extra = st.text_input(
            "Istruzioni extra (opzionale)",
            placeholder="Es. una variante deve fare 'PRIMA/DOPO'",
        )

        submitted = st.form_submit_button(
            "✨ Genera design brief",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not brief_input.strip():
            st.error("Il **brief** e` obbligatorio.")
            return
        if not ANTHROPIC_API_KEY or not OPENAI_API_KEY:
            st.error("Mancano API key (ANTHROPIC_API_KEY o OPENAI_API_KEY).")
            return

        with st.spinner(f"Art Director ragiona ({n_variants} brief)…"):
            try:
                briefs = brief_mod.write_briefs(
                    api_key=ANTHROPIC_API_KEY,
                    use_case="visual_ad",
                    fmt=fmt,
                    brief=brief_input,
                    promise=promise,
                    target_audience=sidebar["target_audience"],
                    brand_voice=sidebar["brand_voice"],
                    brand_visual=sidebar["brand_visual"],
                    palette_hex=sidebar["palette_hex"],
                    references_blob=st.session_state.ref_blob,
                    style_notes=style,
                    text_mode=text_mode,
                    n_variants=n_variants,
                    extra_instructions=extra,
                )
            except Exception as e:
                st.error(f"Generazione brief fallita: {e}")
                st.caption(traceback.format_exc())
                return

        # Render immagini in sequenza (gpt-image-1 e` lento)
        items = []
        for b in briefs:
            img = _trigger_render(brief=b, fmt=fmt, quality=sidebar["quality"])
            items.append({"brief": b, "image": img})

        st.session_state.ads_items = items
        st.session_state.ads_inputs = {
            "use_case": "visual_ad",
            "fmt": fmt,
            "text_mode": text_mode,
            "brief": brief_input,
            "promise": promise,
            "style": style,
            "extra": extra,
            "target_audience": sidebar["target_audience"],
            "brand_voice": sidebar["brand_voice"],
            "brand_visual": sidebar["brand_visual"],
            "palette_hex": sidebar["palette_hex"],
            "references_blob": st.session_state.ref_blob,
            "quality": sidebar["quality"],
        }
        st.rerun()

    if st.session_state.ads_items:
        st.divider()
        _render_results_panel(
            tab_prefix="ads",
            items=st.session_state.ads_items,
            inputs=st.session_state.ads_inputs or {},
        )


# ── Tab Landing image ──────────────────────────────────────────────


def _render_landing_tab(sidebar: dict[str, Any]) -> None:
    st.subheader("🏞 Landing image — hero editoriali evocative")
    st.caption(
        "Immagini per landing page: editoriali, fotografiche, cinematic. "
        "Non sembrano ads — evocano l'outcome del target."
    )

    with st.form("landing_form"):
        cols = st.columns([2, 1, 1])
        fmt: Format = cols[0].selectbox(  # type: ignore[assignment]
            "Formato",
            options=list(FORMAT_LABELS),
            format_func=lambda f: FORMAT_LABELS[f],
            index=2,  # landscape default per landing
            key="landing_fmt",
        )
        n_variants = cols[1].slider(
            "Varianti",
            min_value=brief_mod.MIN_VARIANTS,
            max_value=brief_mod.MAX_VARIANTS,
            value=2,
            key="landing_n",
        )
        text_mode: brief_mod.TextMode = cols[2].selectbox(  # type: ignore[assignment]
            "Testo nell'immagine",
            options=list(TEXT_MODE_LABELS),
            format_func=lambda m: TEXT_MODE_LABELS[m],
            index=2,  # none di default per landing
            key="landing_textmode",
        )

        brief_input = st.text_area(
            "📥 Brief — cosa l'immagine deve EVOCARE",
            height=200,
            placeholder=(
                "Es. una sensazione di liberta` post-lavoro intellettuale,\n"
                "una persona che gestisce il proprio tempo, una scrivania\n"
                "in penombra che si illumina."
            ),
        )
        style = st.text_area(
            "Note stilistiche (opzionale)",
            height=80,
            placeholder=(
                "Es. fotografia naturale, golden hour, niente persone in primo "
                "piano, mood riflessivo."
            ),
        )
        extra = st.text_input(
            "Istruzioni extra (opzionale)",
            placeholder="Es. una variante deve evocare 'casa', un'altra 'studio'",
        )

        submitted = st.form_submit_button(
            "✨ Genera design brief",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not brief_input.strip():
            st.error("Il **brief** e` obbligatorio.")
            return
        if not ANTHROPIC_API_KEY or not OPENAI_API_KEY:
            st.error("Mancano API key.")
            return

        with st.spinner(f"Art Director ragiona ({n_variants} brief)…"):
            try:
                briefs = brief_mod.write_briefs(
                    api_key=ANTHROPIC_API_KEY,
                    use_case="landing",
                    fmt=fmt,
                    brief=brief_input,
                    target_audience=sidebar["target_audience"],
                    brand_voice=sidebar["brand_voice"],
                    brand_visual=sidebar["brand_visual"],
                    palette_hex=sidebar["palette_hex"],
                    references_blob=st.session_state.ref_blob,
                    style_notes=style,
                    text_mode=text_mode,
                    n_variants=n_variants,
                    extra_instructions=extra,
                )
            except Exception as e:
                st.error(f"Generazione brief fallita: {e}")
                st.caption(traceback.format_exc())
                return

        items = []
        for b in briefs:
            img = _trigger_render(brief=b, fmt=fmt, quality=sidebar["quality"])
            items.append({"brief": b, "image": img})

        st.session_state.landing_items = items
        st.session_state.landing_inputs = {
            "use_case": "landing",
            "fmt": fmt,
            "text_mode": text_mode,
            "brief": brief_input,
            "promise": "",
            "style": style,
            "extra": extra,
            "target_audience": sidebar["target_audience"],
            "brand_voice": sidebar["brand_voice"],
            "brand_visual": sidebar["brand_visual"],
            "palette_hex": sidebar["palette_hex"],
            "references_blob": st.session_state.ref_blob,
            "quality": sidebar["quality"],
        }
        st.rerun()

    if st.session_state.landing_items:
        st.divider()
        _render_results_panel(
            tab_prefix="landing",
            items=st.session_state.landing_items,
            inputs=st.session_state.landing_inputs or {},
        )


# ── Pannello risultati condiviso ───────────────────────────────────


def _render_results_panel(
    *,
    tab_prefix: str,
    items: list[dict],
    inputs: dict[str, Any],
) -> None:
    cols = st.columns([1, 1, 4])
    if cols[0].button(
        "⬅️ Nuovo brief", key=f"{tab_prefix}_new"
    ):
        _reset_tab(f"{tab_prefix}_")
        st.rerun()
    if cols[1].button(
        "🔁 Rigenera tutte le immagini",
        key=f"{tab_prefix}_regen_all",
        help="Riusa gli stessi brief, rifa solo le immagini con gpt-image-1.",
    ):
        for it in items:
            it["image"] = _trigger_render(
                brief=it["brief"],
                fmt=inputs["fmt"],
                quality=inputs["quality"],
            )
        st.rerun()

    for i, item in enumerate(items):
        brief = item["brief"]
        image = item["image"]
        with st.container(border=True):
            st.markdown(f"### Variante #{i + 1}")
            left, right = st.columns([3, 2])
            with left:
                if image is not None:
                    st.image(
                        image.image_bytes,
                        use_container_width=True,
                        caption=f"{image.size} · quality={image.quality}",
                    )
                    st.download_button(
                        "⬇️ Scarica PNG",
                        data=image.image_bytes,
                        file_name=f"{tab_prefix}_var{i + 1}.png",
                        mime="image/png",
                        key=f"{tab_prefix}_dl_{i}",
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "Immagine non generata (errore al render). Usa "
                        "'Re-render immagine' qui sotto."
                    )
            with right:
                _render_brief_card(brief)

            _render_edit_box(
                tab_prefix=tab_prefix,
                index=i,
                item=item,
                inputs=inputs,
            )


def _render_edit_box(
    *,
    tab_prefix: str,
    index: int,
    item: dict,
    inputs: dict[str, Any],
) -> None:
    """Box di modifica per una variante. Tre azioni:
    A) Re-render — riusa stesso image_prompt, rifa solo il PNG.
    B) Rebrief — Claude rifa il design brief con feedback, poi nuovo PNG.
    C) Edit locale — feedback testuale + /v1/images/edits sull'immagine attuale.
    """
    with st.expander("🛠 Modifica questa variante", expanded=False):
        action = st.radio(
            "Modalita`",
            options=["rebrief", "edit_locale", "rerender"],
            format_func=lambda a: {
                "rebrief": "🪄 Rebrief — Claude ripensa il design, OpenAI rigenera",
                "edit_locale": "✏️ Edit locale — modifica l'immagine corrente (piu` veloce)",
                "rerender": "🎲 Re-render — stesso brief, immagine nuova",
            }[a],
            key=f"{tab_prefix}_action_{index}",
            horizontal=False,
        )

        if action == "rerender":
            if st.button(
                "🎲 Re-render",
                key=f"{tab_prefix}_btn_rerender_{index}",
                type="primary",
            ):
                item["image"] = _trigger_render(
                    brief=item["brief"],
                    fmt=inputs["fmt"],
                    quality=inputs["quality"],
                )
                st.rerun()
            return

        # rebrief / edit_locale richiedono entrambi un feedback
        feedback = st.text_area(
            "Feedback per il designer",
            placeholder={
                "rebrief": (
                    "Es. 'cambia la composizione, voglio un primo piano della\n"
                    "persona invece dello split-screen', 'usa palette piu` calda'"
                ),
                "edit_locale": (
                    "Es. 'cambia il colore della maglia in rosso',\n"
                    "'togli l'oggetto in alto a destra', 'rendi piu` luminoso'"
                ),
            }[action],
            key=f"{tab_prefix}_fb_{index}",
            height=100,
        )

        if action == "rebrief":
            if st.button(
                "🪄 Rebrief + render",
                key=f"{tab_prefix}_btn_rebrief_{index}",
                disabled=not feedback.strip(),
                type="primary",
            ):
                try:
                    with st.spinner("Claude rivede il brief…"):
                        new_brief = brief_mod.regenerate_brief(
                            api_key=ANTHROPIC_API_KEY,
                            use_case=inputs["use_case"],
                            fmt=inputs["fmt"],
                            original=item["brief"],
                            feedback=feedback,
                            brief=inputs["brief"],
                            promise=inputs.get("promise", ""),
                            target_audience=inputs["target_audience"],
                            brand_voice=inputs["brand_voice"],
                            brand_visual=inputs["brand_visual"],
                            palette_hex=inputs["palette_hex"],
                            references_blob=inputs["references_blob"],
                            style_notes=inputs.get("style", ""),
                            text_mode=inputs["text_mode"],
                        )
                    item["brief"] = new_brief
                    item["image"] = _trigger_render(
                        brief=new_brief,
                        fmt=inputs["fmt"],
                        quality=inputs["quality"],
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Rebrief fallito: {e}")

        elif action == "edit_locale":
            if item["image"] is None:
                st.info("Serve un'immagine corrente per l'edit locale.")
                return
            if st.button(
                "✏️ Applica edit locale",
                key=f"{tab_prefix}_btn_edit_{index}",
                disabled=not feedback.strip(),
                type="primary",
            ):
                try:
                    with st.spinner("Edit gpt-image-1…"):
                        edited = edit_mod.edit_local(
                            api_key=OPENAI_API_KEY,
                            source_image_bytes=item["image"].image_bytes,
                            edit_prompt=feedback,
                            fmt=inputs["fmt"],
                            quality=inputs["quality"],
                        )
                    item["image"] = edit_mod.to_rendered(edited)
                    st.rerun()
                except Exception as e:
                    st.error(f"Edit fallito: {e}")


# ── Top-level rendering ────────────────────────────────────────────


def _main() -> None:
    sidebar = _sidebar()

    st.title("🎨 Graphic Designer Agent")
    st.caption(
        "Un art director virtuale che pensa **prima** di disegnare: legge "
        "il brief, decide composizione/palette/tipografia, scrive un image "
        "prompt rigoroso, poi gpt-image-1 produce il PNG finale. "
        "Modificalo con feedback testuali finche` non ti convince."
    )

    tab_ads, tab_landing = st.tabs(
        ["📣 Visual Ads", "🏞 Landing image"]
    )
    with tab_ads:
        _render_visual_ads_tab(sidebar)
    with tab_landing:
        _render_landing_tab(sidebar)


_main()
