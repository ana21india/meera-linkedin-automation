"""Shared note-screening and LinkedIn-drafting logic, used by both the
local polling script (bot.py) and the Vercel webhook (api/webhook.py) so
the two never drift out of sync.
"""

import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

import requests
from google.genai import types

MIN_NOTE_LENGTH = 25
SKILL_FILE = Path(__file__).resolve().parent / "meera_voice_skill.md"

SCORE_THRESHOLD = 7.0  # weighted score must be STRICTLY ABOVE this to draft
MAX_NEWS_AGE_DAYS = 90
MAX_NEWS_RESULTS = 5

# (metric name, weight, question) -- weights sum to 100
RUBRIC = [
    ("BRAND_RELEVANCE", 20,
     "Does this connect naturally to the brand, category, consumer, problem, or mission?"),
    ("FOUNDER_RIGHT_TO_SPEAK", 20,
     "Does she have genuine experience, access, insight, or a strong POV that makes her worth listening to?"),
    ("CONVERSATION_HEAT", 15,
     "Is this currently being discussed, searched, debated, or shared?"),
    ("INSIGHT_POV_POTENTIAL", 15,
     'Can we say something beyond "X is amazing" or summarise the news?'),
    ("B2B_NETWORK_RELEVANCE", 15,
     "Could this attract customers, partners, distributors, retailers, investors, creators, or relevant industry people?"),
    ("CONTENT_DISTINCTIVENESS", 10,
     "Would this post look meaningfully different from 100 other LinkedIn posts on the same topic?"),
    ("TIMELINESS_SHELF_LIFE", 5,
     "Is there a reason to post now rather than next month?"),
]


def load_skill_text() -> str:
    return SKILL_FILE.read_text(encoding="utf-8")


def download_telegram_file(token: str, file_id: str) -> bytes:
    """Voice notes arrive as a file_id, not raw bytes -- this resolves it
    to an actual downloadable file via Telegram's two-step file API."""
    api = f"https://api.telegram.org/bot{token}"
    file_info = requests.get(f"{api}/getFile", params={"file_id": file_id}, timeout=20)
    file_info.raise_for_status()
    file_path = file_info.json()["result"]["file_path"]

    file_resp = requests.get(
        f"https://api.telegram.org/file/bot{token}/{file_path}", timeout=30
    )
    file_resp.raise_for_status()
    return file_resp.content


def transcribe_voice_note(client, audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
    """Telegram voice notes are OGG/Opus by default; Gemini accepts audio
    directly, so we skip any separate speech-to-text service."""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            "Transcribe this voice note exactly, word for word, in English. "
            "Output ONLY the transcription, nothing else -- no preamble, no notes.",
        ],
    )
    return (response.text or "").strip()


def send_telegram_message(token: str, chat_id: int, text: str) -> None:
    api = f"https://api.telegram.org/bot{token}"
    for i in range(0, len(text), 4000):
        requests.post(
            f"{api}/sendMessage",
            data={"chat_id": chat_id, "text": text[i:i + 4000]},
            timeout=20,
        )


def is_note_substantive(client, note: str) -> tuple[bool, str, float]:
    """Scores the note against the rubric (each metric 1-10, weighted to a
    single 1-10 score). Returns (decision, reason, weighted_score).
    Anything scoring strictly above SCORE_THRESHOLD becomes a draft."""
    if len(note.strip()) < MIN_NOTE_LENGTH:
        return False, "too short to contain a real claim or observation", 0.0

    rubric_lines = "\n".join(
        f"- {name} (weight {weight}/100): {question}" for name, weight, question in RUBRIC
    )
    score_lines = "\n".join(f"{name}: <1-10>" for name, _, _ in RUBRIC)

    prompt = f"""You are scoring a raw note from Meera, founder of a skincare
brand, on whether it's worth turning into a LinkedIn post. Score each
metric below from 1 (not at all) to 10 (extremely strong) based on what
this specific note gives you to work with.

{rubric_lines}

Be strict -- most raw notes (greetings, internal logistics, vendor/admin
details, vague one-liners) should score low across the board.

Note:
---
{note}
---

Reply with exactly these lines and nothing else:
{score_lines}
REASON: <one short sentence explaining the overall call>
"""
    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    text = response.text or ""

    scores = {}
    for name, _, _ in RUBRIC:
        match = re.search(rf"{name}\s*:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        scores[name] = max(1.0, min(10.0, float(match.group(1)))) if match else 1.0

    reason_match = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE)
    reason = reason_match.group(1).strip() if reason_match else "no reason returned"

    weighted_score = sum(scores[name] * weight for name, weight, _ in RUBRIC) / 100
    weighted_score = round(weighted_score, 1)

    decision = weighted_score > SCORE_THRESHOLD
    return decision, reason, weighted_score


def _news_search_query(client, note: str) -> str:
    prompt = f"""Give ONE short Google News search query (3-6 words, no
punctuation, no quotes) for the current news/industry angle relevant to
this note's topic. Reply with ONLY the query, nothing else.

Note:
---
{note}
---
"""
    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    query = (response.text or "").strip().splitlines()[0] if response.text else note[:60]
    return query[:100]


def fetch_recent_news(
    query: str, max_results: int = MAX_NEWS_RESULTS, max_age_days: int = MAX_NEWS_AGE_DAYS
) -> list[dict]:
    """Pulls real, dated results from Google News RSS and filters out
    anything older than max_age_days -- this is our only source of
    external "current events" grounding, so drafting can never hallucinate
    a news item that wasn't actually retrieved here."""
    try:
        resp = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
            timeout=15,
        )
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
    except (requests.RequestException, ElementTree.ParseError):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    results = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date_raw = item.findtext("pubDate")
        if not title or not link or not pub_date_raw:
            continue
        try:
            pub_date = parsedate_to_datetime(pub_date_raw)
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if pub_date < cutoff:
            continue
        results.append({
            "title": title,
            "link": link,
            "pub_date": pub_date.strftime("%Y-%m-%d"),
            "source": (item.findtext("source") or "").strip(),
        })
        if len(results) >= max_results:
            break
    return results


def draft_linkedin_post(client, note: str, skill_text: str) -> dict:
    """Returns {"post": str, "citations": [{"title","link","pub_date","source"}, ...]}."""
    query = _news_search_query(client, note)
    news_items = fetch_recent_news(query)

    if news_items:
        news_block = "\n".join(
            f"{i + 1}. \"{item['title']}\" ({item['source'] or 'unknown source'}, "
            f"{item['pub_date']}) - {item['link']}"
            for i, item in enumerate(news_items)
        )
    else:
        news_block = "(no news results found from the last 90 days for this topic)"

    prompt = f"""{skill_text}

---

Using ONLY the voice rules above, draft ONE LinkedIn post based on the raw
note below from Meera.

Here is a list of REAL, CURRENT news items, all published within the last
{MAX_NEWS_AGE_DAYS} days -- this is the ONLY external news/data source you
are allowed to reference:
{news_block}

Hard rules against hallucination:
- You may only reference a news item, statistic, or study if it is in the
  list above. Never invent a source, publication, statistic, or event
  that isn't in that list.
- If nothing in the list is genuinely relevant to this note's topic,
  write the post from the note alone with no external claim added. A
  post with no news angle is far better than one with a fabricated one.
- Do not invent specific numbers or details that aren't in the note or in
  the list above. If the note says "pH dropped by about 0.4 units"
  without exact before/after values, write it exactly that way.

Raw note from Meera:
---
{note}
---

Output in EXACTLY this format and nothing else:
POST:
<the finished LinkedIn post text -- no markdown, no headers -- signed "Meera" on its own line>
CITED: <comma-separated item numbers from the list above that you actually used, or NONE>
"""
    response = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
    text = response.text or ""

    post_match = re.search(r"POST:\s*(.*?)\s*CITED:", text, re.DOTALL | re.IGNORECASE)
    post = post_match.group(1).strip() if post_match else text.strip()

    cited_match = re.search(r"CITED:\s*(.+)", text, re.IGNORECASE)
    cited_raw = cited_match.group(1).strip() if cited_match else "NONE"

    citations = []
    if cited_raw.upper() != "NONE":
        for token in re.findall(r"\d+", cited_raw):
            idx = int(token) - 1
            if 0 <= idx < len(news_items):
                citations.append(news_items[idx])

    return {"post": post, "citations": citations}


def format_citations_message(citations: list[dict]) -> str:
    if not citations:
        return (
            "Sources: none -- no external news was cited in this draft "
            "(either nothing relevant was found in Google News from the "
            "last 90 days, or the post is based only on your note)."
        )
    lines = [f"Sources used for this draft (all within the last {MAX_NEWS_AGE_DAYS} days):"]
    for c in citations:
        lines.append(f"- {c['title']} ({c.get('source', '')}, {c['pub_date']}): {c['link']}")
    return "\n".join(lines)
