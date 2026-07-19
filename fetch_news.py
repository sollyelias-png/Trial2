#!/usr/bin/env python3
"""
Property Wire — Gemini-powered fetcher.

Runs outside Claude entirely (intended to be triggered by GitHub Actions on
a schedule). For each source URL, asks Gemini (with Google Search grounding
enabled) to find recent hotel/hospitality project news, extracts a
structured JSON record per project, merges it into the accumulated feed
file, de-duplicates, and writes the result back to disk.

The output file (feed.json) is committed back to the repo by the GitHub
Actions workflow, which makes it available at a public raw.githubusercontent.com
URL — that's what the HTML artifact reads.

Environment:
    GEMINI_API_KEY   required. Set as a GitHub Actions secret.

Usage:
    python fetch_news.py
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RECENCY_DAYS = 8
MODEL_NAME = "gemini-flash-latest"  # auto-updating alias; currently Gemini 3.5 Flash
MAX_PROJECTS_PER_SOURCE = 4
MAX_COMPANIES_PER_PROJECT = 8
REQUEST_PACING_SECONDS = 3          # gap between calls, keeps us well under free-tier rate limits
MAX_RETRIES = 4
FEED_PATH = os.path.join(os.path.dirname(__file__), "docs", "feed.json")

SOURCES = [
    "https://hotelsmag.com/",
    "https://hotelbusiness.com/",
    "https://hotelbusiness.com/category/design/renovations/",
    "https://hotelbusiness.com/category/development/ownership/",
    "https://hotelbusiness.com/category/development/acquisitions/",
    "https://hotelbusiness.com/category/development/brands/",
    "https://hotelbusiness.com/category/development/openings-pipeline/",
    "https://hotelbusiness.com/category/development/independent-boutique/",
    "https://www.hotel-online.com/",
    "https://www.hotel-online.com/categories/property-news",
    "https://www.hotel-online.com/categories/acquisitions",
    "https://www.hotel-online.com/categories/architecture-design",
    "https://www.hotel-online.com/categories/hotel-development",
    "https://www.hotel-online.com/categories/new-hotel-openings",
    "https://www.hotel-online.com/categories/renovations-rebranding",
    "https://www.hospitalitynet.org/",
    "https://www.hospitalitynet.org/development",
    "https://www.hospitalitynet.org/design-architecture",
    "https://www.staymagazine.ca/news",
    "https://www.hospitalityupgrade.com/news",
    "https://usa.boutiquehotelier.com/news/development/",
    "https://usa.boutiquehotelier.com/news/transactions/",
    "https://www.hotelinteractive.com/",
    "https://www.4hoteliers.com/",
    "https://www.hotelinvestmenttoday.com/development",
    "https://www.hotelinvestmenttoday.com/Latest-News",
    "https://www.hotelinvestmenttoday.com/deals/management",
]

VALID_ROLES = {"architect", "construction", "development", "operator", "owner", "other"}


def normalize_role(role):
    r = (role or "").lower()
    if "architect" in r or "design" in r:
        return "architect"
    if "construct" in r or "contractor" in r or "builder" in r:
        return "construction"
    if "develop" in r:
        return "development"
    if "operat" in r or "manage" in r:
        return "operator"
    if "own" in r:
        return "owner"
    return "other"


def hostname_of(url):
    try:
        return urlparse(url).hostname.replace("www.", "") if urlparse(url).hostname else ""
    except Exception:
        return ""


def is_within_recency_window(date_str):
    if not date_str:
        return True  # no date given -> assume new, per spec
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return True  # unparseable -> assume new
    return (datetime.now(timezone.utc) - d) <= timedelta(days=RECENCY_DAYS + 0.5)


def repair_and_parse_array(raw_text):
    """Same truncation-tolerant parsing used in the artifact version."""
    text = raw_text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    start = text.find("[")
    if start == -1:
        raise ValueError("no JSON array found in response")
    text = text[start:]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    depth = 0
    last_complete_end = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 1 and ch == "}":
                last_complete_end = i

    if last_complete_end == -1:
        raise ValueError("could not recover any complete entries from truncated response")

    return json.loads(text[: last_complete_end + 1] + "]")


def build_prompt(source_url):
    parsed = urlparse(source_url)
    host = (parsed.hostname or "").replace("www.", "")
    path = parsed.path.rstrip("/")
    suggested_query = f"site:{host}{path} hotel" if path else f"site:{host} hotel renovation development acquisition news"
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return f"""Use Google Search to find recent articles from this specific page/section: {source_url}
Suggested search query: {suggested_query}

Only include articles that are specifically about hotel renovations, new construction/development, new builds, acquisitions, ownership changes, or similar project news. Skip generic listicles, award roundups, opinion pieces, or anything not tied to a specific property/project.

Only include articles published within the last {RECENCY_DAYS} days (today is {today_str}). If you can't determine a specific publish date but the article appears current, include it anyway and leave date as an empty string — do not guess a date.

For each distinct project you find, extract:
- project_name: the property or project name
- summary: one concise sentence on what the news is
- source_url: the direct article URL
- date: publish date if known, in YYYY-MM-DD format, else ""
- companies: an array covering EVERY company AND EVERY named individual mentioned in the article about this project. Be exhaustive — completeness matters more than brevity here. Rules:
  - Format: {{"company": "...", "role": "architect"|"construction"|"development"|"operator"|"owner"|"other", "person": "...", "title": "..."}}
  - If a company has multiple named people mentioned, include a SEPARATE entry for each person (same company/role, different person/title) — do not merge multiple people into one entry.
  - If a person is named but their company/firm affiliation isn't clear from the article, still include them with company left as "" and role "other".
  - If a company is mentioned with no named individual, include it once with person and title left as "".
  - Do not skip any named individual anywhere in the article, even minor mentions.

Return ONLY a raw JSON array of project objects, nothing else — no markdown fences, no explanation, no text before or after. At most {MAX_PROJECTS_PER_SOURCE} projects, each with at most {MAX_COMPANIES_PER_PROJECT} company/person entries. If nothing relevant and recent is found, return exactly: []"""


def call_gemini(client, prompt):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            return response.text or ""
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if "429" in msg or "resource_exhausted" in msg or "rate" in msg:
                wait = 5 * attempt
                print(f"    rate limited, retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            raise
    raise last_error


def item_key(item):
    return (item.get("project_name", "").strip().lower(), item.get("date") or "nodate")


def fetch_source(client, source_url):
    prompt = build_prompt(source_url)
    raw_text = call_gemini(client, prompt)

    if not raw_text.strip():
        return []

    raw_items = repair_and_parse_array(raw_text)
    if not isinstance(raw_items, list):
        return []

    host = hostname_of(source_url)
    now_iso = datetime.now(timezone.utc).isoformat()
    results = []

    for raw in raw_items:
        if not isinstance(raw, dict) or not raw.get("project_name"):
            continue
        if not is_within_recency_window(raw.get("date")):
            continue

        companies = []
        for c in (raw.get("companies") or []):
            if not isinstance(c, dict):
                continue
            companies.append({
                "company": c.get("company", "") or "",
                "role": normalize_role(c.get("role")),
                "person": c.get("person", "") or "",
                "title": c.get("title", "") or "",
            })

        results.append({
            "project_name": raw["project_name"],
            "companies": companies,
            "summary": raw.get("summary", "") or "",
            "source_url": raw.get("source_url", "") or "",
            "source_domain": host,
            "date": raw.get("date", "") or "",
            "added_at": now_iso,
        })

    return results


def load_existing_feed():
    if not os.path.exists(FEED_PATH):
        return []
    try:
        with open(FEED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("items", []) if isinstance(data, dict) else data
    except Exception:
        return []


def save_feed(items):
    os.makedirs(os.path.dirname(FEED_PATH), exist_ok=True)
    items.sort(key=lambda it: it.get("date") or it.get("added_at") or "", reverse=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recency_window_days": RECENCY_DAYS,
        "source_count": len(SOURCES),
        "items": items,
    }
    with open(FEED_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    feed_items = load_existing_feed()
    existing_keys = {item_key(it) for it in feed_items}

    total_added = 0
    errors = []

    for i, source_url in enumerate(SOURCES, start=1):
        print(f"[{i}/{len(SOURCES)}] Checking {source_url} ...")
        try:
            new_items = fetch_source(client, source_url)
        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append((source_url, str(e)))
            continue

        added_here = 0
        for item in new_items:
            key = item_key(item)
            if key in existing_keys:
                continue
            existing_keys.add(key)
            feed_items.append(item)
            added_here += 1

        total_added += added_here
        print(f"    +{added_here} new")

        if i < len(SOURCES):
            time.sleep(REQUEST_PACING_SECONDS)

    save_feed(feed_items)

    print(f"\nDone. {total_added} new items added. Feed now has {len(feed_items)} total items.")
    if errors:
        print(f"\n{len(errors)} source(s) failed:")
        for url, err in errors:
            print(f"  - {url}: {err}")
        # Don't fail the whole workflow run over individual source errors —
        # partial results are still useful. Change to sys.exit(1) if you'd
        # rather the Action show as failed whenever any source errors out.


if __name__ == "__main__":
    main()
