"""
Meera / Skinstinct LinkedIn draft automation -- LOCAL DEV / TESTING VERSION.

This polls Telegram in a loop, which only works while this script is kept
running on your own machine. The production version deployed to Vercel is
api/webhook.py -- it does the same drafting (via the shared draft_logic.py
module) but gets notes pushed to it instead of polling, since Vercel can't
run a permanent background process.

Flow (Trigger -> Input -> Context -> Processing -> AI -> Output):
  Trigger    : new message arrives in Meera's Telegram channel
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

from draft_logic import (
    draft_linkedin_post,
    format_citations_message,
    is_note_substantive,
    send_telegram_message,
)

BASE_DIR = Path(__file__).resolve().parent
SKILL_FILE = BASE_DIR / "meera_voice_skill.md"
DRAFTS_DIR = BASE_DIR / "drafts"
STATE_FILE = BASE_DIR / "state.json"
POLL_SECONDS = 30

load_dotenv(BASE_DIR / ".env")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


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


def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    keep = "".join(c if c.isalnum() else "-" for c in text.lower())
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-")[:max_len] or "note"


def process_note(chat_id: int, note: str, skill_text: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] New note: {note[:60]!r}")

    substantive, reason, score = is_note_substantive(gemini_client, note)
    if not substantive:
        print(f"  -> skipped (score {score}/10: {reason})")
        send_telegram_message(
            TELEGRAM_BOT_TOKEN,
            chat_id,
            f"Skipped this note for a draft (score {score}/10): {reason}\n\nNote was: \"{note}\"",
        )
        return

    print(f"  -> drafting (score {score}/10)...")
    result = draft_linkedin_post(gemini_client, note, skill_text)
    citations_text = format_citations_message(result["citations"])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = DRAFTS_DIR / f"{timestamp}-{slugify(note)}.txt"
    filename.write_text(f"{result['post']}\n\n{citations_text}", encoding="utf-8")
    print(f"  -> saved {filename.name}")

    send_telegram_message(
        TELEGRAM_BOT_TOKEN, chat_id, f"Draft ready (score {score}/10):\n\n{result['post']}"
    )
    send_telegram_message(TELEGRAM_BOT_TOKEN, chat_id, citations_text)


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
                        TELEGRAM_BOT_TOKEN, chat_id, f"Something went wrong drafting this note: {exc}"
                    )
                save_state(state)
        except requests.RequestException as exc:
            print(f"Telegram polling error: {exc}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
