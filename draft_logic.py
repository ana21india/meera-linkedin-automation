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

Note:
---
{note}
---

Reply with exactly two lines:
DECISION: YES or NO
REASON: one short sentence why
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = response.text or ""
    lines = text.splitlines()
    decision = bool(lines) and "YES" in lines[0].upper()
    reason_line = next((l for l in lines if l.upper().startswith("REASON")), "")
    reason = reason_line.split(":", 1)[-1].strip() if reason_line else "no reason returned"
    return decision, reason


def draft_linkedin_post(client, search_tool, note: str, skill_text: str) -> str:
    prompt = f"""{skill_text}

---

Using ONLY the voice rules above, draft ONE LinkedIn post based on the raw
note below from Meera. Before writing, use Google Search to find one
current, real, specific news item, industry data point, or regulatory
update relevant to the note's topic (skincare actives, formulation
science, Indian D2C/consumer trends, or cosmetic regulation) and weave it
in naturally, the way she references real studies and sources in her
existing posts. Do not fabricate a source — if you can't find a genuinely
relevant one, skip the news angle rather than inventing one.

Raw note from Meera:
---
{note}
---

Output ONLY the finished LinkedIn post text (no preamble, no explanation,
no markdown formatting, no headers). End it signed "Meera" on its own line,
matching her sign-off style.
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(tools=[search_tool]),
    )
    return (response.text or "").strip()
