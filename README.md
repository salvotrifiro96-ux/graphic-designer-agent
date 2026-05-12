# Graphic Designer Agent

Agente Streamlit del team marketing Leone. Un **art director virtuale** che:

1. Legge il brief operatore (cosa l'immagine deve comunicare/evocare)
2. Ragiona come un designer umano: sceglie composizione, palette, tipografia, mood
3. Produce un **design brief leggibile** + un **image_prompt** rigoroso (200-400 parole, inglese)
4. Manda l'image_prompt a **gpt-image-1** (OpenAI) → ottieni PNG finale
5. Itera con feedback testuale (rebrief, edit locale o re-render)

Due modalità:

- **📣 Visual Ads** — composizioni da advertising performance (split-screen prima/dopo, hero+callouts, big-text, number-spotlight) per Meta / Instagram / TikTok
- **🏞 Landing image** — hero editoriali evocative per landing page (cinematic, editorial still, abstract mood, architectural frame, symbolic scene)

## Setup locale

```bash
cd graphic-designer-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edita .env con ANTHROPIC_API_KEY + OPENAI_API_KEY
streamlit run app.py
```

Apre su http://localhost:8501. Password gate: `faraone.92`.

## Setup Streamlit Cloud

1. Connetti il repo a Streamlit Cloud
2. Settings → Secrets:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   OPENAI_API_KEY = "sk-proj-..."
   APP_PASSWORD = "faraone.92"
   ```

## Costi indicativi (gpt-image-1)

| Quality | Costo per immagine 1024 | Quando usarlo |
|---|---|---|
| low | ~$0.01 | Brainstorm rapido, draft |
| medium | ~$0.04 | **Default**, iterazioni |
| high | ~$0.17 | Finale per produzione |

Una sessione tipica di lavoro produce 2-4 brief × 1-3 re-render = $0.20-$1 a richiesta.

## Test

```bash
pytest
```

31 unit test puri (parsing, normalizzazione hex, format mapping). Niente API calls.

## Struttura

```
app.py                 → Streamlit (2 tab + sidebar brand identity)
agent/
  common.py            → modello, format mapping, JSON parsing, hex normalize
  refs.py              → Claude vision: descrive reference image uploadate
  brief.py             → Art Director Claude: design brief + image_prompt
  render.py            → OpenAI gpt-image-1: PNG bytes
  edit.py              → /v1/images/edits per edit locale di un'immagine
tests/                 → pytest, niente API
```

## Pattern condivisi col team

- Sidebar globale: target audience + brand voice + brand visual + palette
- Password gate, `claude-sonnet-4-6` per la parte testuale, gpt-image-1 per la parte visiva
- Reference upload (max 3 immagini): Claude vision le descrive e nutre il design brief
- Funzione `regenerate_brief(...)` come negli altri agenti (promise-writer, copywriter)

## Tre modi di iterare su una variante

1. **🪄 Rebrief** — Claude ripensa la composizione/palette/etc dato il feedback, poi nuovo render. Per modifiche strutturali ("cambia composizione", "togli il testo", "palette piu` calda").
2. **✏️ Edit locale** — /v1/images/edits modifica l'immagine attuale. Per modifiche localizzate ("cambia il colore della maglia in rosso", "togli l'oggetto in alto a destra"). Piu` veloce e meno distruttivo.
3. **🎲 Re-render** — stesso image_prompt, immagine diversa. Per quando il brief è buono ma l'output specifico non convince.
