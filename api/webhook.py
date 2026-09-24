"""
Vercel serverless endpoint that replaces bot.py's polling loop.

Telegram calls this URL directly the moment a note is posted in the
channel ("push" instead of "poll"), which is what makes this deployable
on Vercel at all -- serverless functions can't run an infinite loop.

Setup after deploying:
  1. In the Vercel project settings, add env vars:
       TELEGRAM_BOT_TOKEN
       GEMINI_API_KEY
       TELEGRAM_WEBHOOK_SECRET   (any random string you make up)
  2. Point Telegram at this URL (one-time, from your own machine):
       https://api.telegram.org/bot<TOKEN>/setWebhook
         ?url=https://<your-vercel-domain>/api/webhook
         &secret_token=<same random string>
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai

from draft_logic import (
    download_telegram_file,
    draft_linkedin_post,
    format_citations_message,
    is_note_substantive,
    load_skill_text,
    send_telegram_message,
    transcribe_voice_note,
)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)
_skill_text = load_skill_text()


def _handle_note(chat_id: int, note: str) -> None:
    try:
        substantive, reason, score = is_note_substantive(_gemini_client, note)
        if not substantive:
            send_telegram_message(
                TELEGRAM_BOT_TOKEN,
                chat_id,
                f"Skipped this note for a draft (score {score}/10): {reason}\n\nNote was: \"{note}\"",
            )
            return

        result = draft_linkedin_post(_gemini_client, note, _skill_text)
        send_telegram_message(
            TELEGRAM_BOT_TOKEN, chat_id, f"Draft ready (score {score}/10):\n\n{result['post']}"
        )
        send_telegram_message(
            TELEGRAM_BOT_TOKEN, chat_id, format_citations_message(result["citations"])
        )
    except Exception as exc:  # noqa: BLE001
        send_telegram_message(
            TELEGRAM_BOT_TOKEN, chat_id, f"Something went wrong drafting this note: {exc}"
        )


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Meera LinkedIn draft webhook is running.")

    def do_POST(self):
        expected_secret = WEBHOOK_SECRET.strip()
        if expected_secret:
            got = self.headers.get("X-Telegram-Bot-Api-Secret-Token", "").strip()
            if got != expected_secret:
                self.send_response(401)
                self.end_headers()
                return

        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b"{}"

        # Ack Telegram immediately so it doesn't time out / retry while we
        # call Gemini; we keep running after this until the function returns.
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

        try:
            update = json.loads(body or b"{}")
        except json.JSONDecodeError:
            return

        message = update.get("message") or update.get("channel_post")
        if not message:
            return
        chat_id = message["chat"]["id"]

        if "text" in message:
            _handle_note(chat_id, message["text"])
            return

        voice = message.get("voice") or message.get("audio")
        if voice:
            try:
                audio_bytes = download_telegram_file(TELEGRAM_BOT_TOKEN, voice["file_id"])
                mime_type = voice.get("mime_type", "audio/ogg")
                note = transcribe_voice_note(_gemini_client, audio_bytes, mime_type)
                if not note:
                    send_telegram_message(
                        TELEGRAM_BOT_TOKEN, chat_id, "Couldn't transcribe that voice note (empty result)."
                    )
                    return
                send_telegram_message(TELEGRAM_BOT_TOKEN, chat_id, f"Transcribed: \"{note}\"")
                _handle_note(chat_id, note)
            except Exception as exc:  # noqa: BLE001
                send_telegram_message(
                    TELEGRAM_BOT_TOKEN, chat_id, f"Something went wrong transcribing that voice note: {exc}"
                )
