"""
Meera / Skinstinct LinkedIn draft automation.

Flow (Trigger -> Input -> Context -> Processing -> AI -> Output):
  Trigger    : new message arrives in Meera's Telegram bot chat
  Input      : the raw note text, fetched via Telegram's getUpdates
  Context    : meera_voice_skill.md (her style rules)
  Processing : a cheap Gemini call decides if the note is substantive
               enough to become a post at all (the "Cut")
  AI         : Gemini (with Google Search grounding for a news angle)
               drafts the LinkedIn post in her voice
  Output     : draft is sent back to her in Telegram AND saved to
               automation/drafts/ as a text file

Run this with: python bot.py
It polls Telegram every POLL_SECONDS and keeps running until you stop it
(Ctrl+C), or close the window/terminal it's running in.
"""

import json
import os
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent
SKILL_FILE = BASE_DIR / "meera_voice_skill.md"
DRAFTS_DIR = BASE_DIR / "drafts"
STATE_FILE = BASE_DIR / "state.json"
POLL_SECONDS = 30
MIN_NOTE_LENGTH = 25  # notes shorter than this are almost never substantive

load_dotenv(BASE_DIR / ".env")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"last_update_id": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_telegram_updates(offset: int) -> list:
    resp = requests.get(
        f"{TELEGRAM_API}/getUpdates",
        params={"offset": offset, "timeout": 10},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("result", [])


def send_telegram_message(chat_id: int, text: str) -> None:
    # Telegram messages are capped at 4096 chars; split long drafts.
    for i in range(0, len(text), 4000):
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            data={"chat_id": chat_id, "text": text[i:i + 4000]},
            timeout=20,
        )


def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    keep = "".join(c if c.isalnum() else "-" for c in text.lower())
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-")[:max_len] or "note"


gemini_client = genai.Client(api_key=GEMINI_API_KEY)
search_tool = types.Tool(google_search=types.GoogleSearch())


def is_note_substantive(note: str) -> tuple[bool, str]:
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
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    text = response.text or ""
    decision = "YES" in text.splitlines()[0].upper() if text else False
    reason_line = next((l for l in text.splitlines() if l.upper().startswith("REASON")), "")
    reason = reason_line.split(":", 1)[-1].strip() if reason_line else "no reason returned"
    return decision, reason


def draft_linkedin_post(note: str, skill_text: str) -> str:
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
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(tools=[search_tool]),
    )
    return (response.text or "").strip()


def process_note(chat_id: int, note: str, skill_text: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] New note: {note[:60]!r}")

    substantive, reason = is_note_substantive(note)
    if not substantive:
        print(f"  -> skipped ({reason})")
        send_telegram_message(
            chat_id,
            f"Skipped this note for a draft: {reason}\n\nNote was: \"{note}\"",
        )
        return

    print("  -> drafting...")
    draft = draft_linkedin_post(note, skill_text)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = DRAFTS_DIR / f"{timestamp}-{slugify(note)}.txt"
    filename.write_text(draft, encoding="utf-8")
    print(f"  -> saved {filename.name}")

    send_telegram_message(chat_id, f"Draft ready:\n\n{draft}")


def main() -> None:
    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        raise SystemExit(
            "Missing keys. Copy .env.example to .env in the automation "
            "folder and paste your Telegram bot token and Gemini API key."
        )

    if not SKILL_FILE.exists():
        raise SystemExit(f"Can't find skill file at {SKILL_FILE}")

    skill_text = SKILL_FILE.read_text(encoding="utf-8")
    DRAFTS_DIR.mkdir(exist_ok=True)
    state = load_state()

    print("Meera LinkedIn draft bot running. Waiting for Telegram notes...")
    print("(Press Ctrl+C to stop)")

    while True:
        try:
            updates = get_telegram_updates(state["last_update_id"] + 1)
            for update in updates:
                state["last_update_id"] = update["update_id"]
                message = update.get("message") or update.get("channel_post")
                if not message or "text" not in message:
                    continue
                chat_id = message["chat"]["id"]
                note = message["text"]
                try:
                    process_note(chat_id, note, skill_text)
                except Exception as exc:  # noqa: BLE001
                    print(f"  -> error processing note: {exc}")
                    send_telegram_message(
                        chat_id, f"Something went wrong drafting this note: {exc}"
                    )
                save_state(state)
        except requests.RequestException as exc:
            print(f"Telegram polling error: {exc}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
