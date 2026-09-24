"""Shared note-screening and LinkedIn-drafting logic, used by both the
local polling script (bot.py) and the Vercel webhook (api/webhook.py) so
the two never drift out of sync.
"""

from pathlib import Path

import requests
from google.genai import types

MIN_NOTE_LENGTH = 25
SKILL_FILE = Path(__file__).resolve().parent / "meera_voice_skill.md"


def load_skill_text() -> str:
    return SKILL_FILE.read_text(encoding="utf-8")


def send_telegram_message(token: str, chat_id: int, text: str) -> None:
    api = f"https://api.telegram.org/bot{token}"
    for i in range(0, len(text), 4000):
        requests.post(
            f"{api}/sendMessage",
            data={"chat_id": chat_id, "text": text[i:i + 4000]},
            timeout=20,
        )


def is_note_substantive(client, note: str) -> tuple[bool, str]:
    """Cheap screening pass: does this fragment have enough in it to
    become a post, or is it noise? Returns (yes/no, one-line reason)."""
    if len(note.strip()) < MIN_NOTE_LENGTH:
        return False, "too short to contain a real claim or observation"

    prompt = f"""You are screening raw notes for whether they are worth
turning into a LinkedIn post for a skincare-brand founder whose posts are
technical, specific, and always contain a concrete fact, number, or
first-hand observation (not vague wellness talk).

Reject (answer NO) anything that is operational noise rather than content,
even if it's a full sentence. This includes, for example:
- Greetings or filler with no claim in them ("Hey", "There", "Love you")
- Logistics / admin: vendor names, phone numbers, meeting times/places
  ("vendor number is XX and meet him on XX for XX")
- To-do reminders, internal scheduling, or anything meant for a person,
  not an audience

Only answer YES if the note contains an actual observation, claim, number,
or story that a reader outside the company would learn something from.

Note:
---
{note}
---

Reply with exactly two lines:
DECISION: YES or NO
REASON: one short sentence why
"""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    text = response.text or ""
    lines = text.splitlines()
    decision = bool(lines) and "YES" in lines[0].upper()
    reason_line = next((l for l in lines if l.upper().startswith("REASON")), "")
    reason = reason_line.split(":", 1)[-1].strip() if reason_line else "no reason returned"
    return decision, reason


def _extract_citations(response) -> list[dict]:
    """Pull the real web sources Gemini's search grounding actually used,
    so we never present a source that wasn't genuinely retrieved."""
    citations: list[dict] = []
    try:
        candidates = response.candidates or []
        for candidate in candidates:
            metadata = getattr(candidate, "grounding_metadata", None)
            if not metadata:
                continue
            for chunk in getattr(metadata, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                if web and getattr(web, "uri", None):
                    citations.append({
                        "title": getattr(web, "title", None) or web.uri,
                        "uri": web.uri,
                    })
    except Exception:  # noqa: BLE001
        pass

    # de-dupe while preserving order
    seen = set()
    unique = []
    for c in citations:
        if c["uri"] not in seen:
            seen.add(c["uri"])
            unique.append(c)
    return unique


def draft_linkedin_post(client, search_tool, note: str, skill_text: str) -> dict:
    """Returns {"post": str, "citations": [{"title", "uri"}, ...]}."""
    prompt = f"""{skill_text}

---

Using ONLY the voice rules above, draft ONE LinkedIn post based on the raw
note below from Meera.

Before writing, use Google Search to look for current news, data, or
industry/regulatory updates relevant to the note's topic (skincare
actives, formulation science, Indian D2C/consumer trends, or cosmetic
regulation), and weave in ONE such item naturally if you find a genuinely
relevant one, the way she references real studies and sources in her
existing posts.

Hard rules against hallucination:
- Never state a statistic, study finding, regulation, or news event unless
  it came from an actual search result you retrieved just now.
- Never invent a source, study name, publication, or number. If you are
  not certain a fact is real and retrieved, leave it out entirely.
- If search turns up nothing genuinely relevant to this note's topic,
  write the post from the note alone with no external claim added. A
  post with no news angle is far better than one with a fabricated one.
- Do not invent specific numbers or details that aren't in the note
  either. If the note says "pH dropped by about 0.4 units" without giving
  the exact before/after values, write it exactly that way -- do not
  invent a starting or ending pH value to make it sound more precise.
  Only use numbers that are either directly from the note or directly
  from a real search result.

Raw note from Meera:
---
{note}
---

Output ONLY the finished LinkedIn post text (no preamble, no explanation,
no markdown formatting, no headers). End it signed "Meera" on its own line,
matching her sign-off style.
"""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(tools=[search_tool]),
    )
    return {
        "post": (response.text or "").strip(),
        "citations": _extract_citations(response),
    }


def format_citations_message(citations: list[dict]) -> str:
    if not citations:
        return "Sources: none — this draft is based only on your note, no external data was found or used."
    lines = ["Sources used for this draft:"]
    for c in citations:
        lines.append(f"- {c['title']}: {c['uri']}")
    return "\n".join(lines)
