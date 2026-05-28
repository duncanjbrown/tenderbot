#!/usr/bin/env python3
"""Tenderbot: find UK government tenders matching your interests."""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import anthropic
import requests
from pydantic import BaseModel

FIND_TENDER_API = "https://www.find-tender.service.gov.uk/api/1.0/ocdsReleasePackages"
BATCH_SIZE = 20
DEFAULT_CACHE_PATH = ".tenderbot_cache.json"
DEFAULT_RESULTS_CACHE_PATH = ".tenderbot_results_cache.json"


class TenderMatch(BaseModel):
    ocid: str
    notice_id: str = ""
    title: str
    relevant: bool
    reason: str
    value: str | None = None


class BatchResult(BaseModel):
    results: list[TenderMatch]


TENDER_TAGS = {"tender", "tenderAmendment", "tenderUpdate"}


def save_cache(releases: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(releases, f)


def load_cache(path: str) -> list[dict] | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_results_cache(results: list[TenderMatch], interests: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"interests": interests, "results": [r.model_dump() for r in results]},
            f,
        )


def load_results_cache(path: str) -> tuple[str, list[TenderMatch]] | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["interests"], [TenderMatch(**r) for r in data["results"]]
    except FileNotFoundError:
        return None


def fetch_tenders(hours_back: int = 24) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    releases = []
    url: str | None = (
        f"{FIND_TENDER_API}?updatedFrom={since}&limit=100"
    )

    while url:
        for attempt in range(5):
            resp = requests.get(url, timeout=30)
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429 and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise
            break
        data = resp.json()

        batch = data.get("releases", [])
        # Keep only tender-stage releases (filter client-side; stages param causes 502)
        releases.extend(
            r for r in batch if TENDER_TAGS.intersection(r.get("tag") or [])
        )

        # Follow the pre-built next URL directly to avoid cursor encoding issues
        url = (data.get("links") or {}).get("next") if batch else None

    return releases


def summarise(release: dict) -> dict:
    tender = release.get("tender", {}) or {}
    buyer = release.get("buyer", {}) or {}
    value = tender.get("value", {}) or {}
    return {
        "notice_id": release.get("id", ""),
        "ocid": release.get("ocid", ""),
        "title": tender.get("title") or "(no title)",
        "description": (tender.get("description") or "")[:400],
        "buyer": buyer.get("name") or "",
        "value": (
            f"{value.get('amount')} {value.get('currency')}"
            if value.get("amount")
            else None
        ),
    }


def enrich_matches(results: list[TenderMatch], releases: list[dict]) -> None:
    summaries = {r.get("ocid", ""): summarise(r) for r in releases}
    for match in results:
        s = summaries.get(match.ocid, {})
        match.notice_id = s.get("notice_id", "")
        match.value = s.get("value")


def evaluate(
    releases: list[dict], interests: str, client: anthropic.Anthropic
) -> list[TenderMatch]:
    system = (
        "You are a procurement analyst evaluating UK government tender notices.\n"
        f"The user is interested in: {interests}\n\n"
        "For each tender, decide whether it is relevant to those interests. "
        "A tender is relevant if its title, description, or buying organisation "
        "meaningfully relates to the stated interests. Be liberal — if there's a "
        "reasonable connection, mark it relevant."
    )

    all_results: list[TenderMatch] = []

    for i in range(0, len(releases), BATCH_SIZE):
        batch = [summarise(r) for r in releases[i : i + BATCH_SIZE]]
        end = i + len(batch)
        print(f"  Evaluating tenders {i + 1}–{end} of {len(releases)}…", file=sys.stderr)

        response = client.messages.parse(
            model="claude-haiku-4-5",
            max_tokens=4096,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Evaluate these {len(batch)} tenders for relevance to: {interests}\n\n"
                        + json.dumps(batch, indent=2)
                    ),
                }
            ],
            output_format=BatchResult,
        )

        if response.parsed_output:
            all_results.extend(response.parsed_output.results)

    return all_results


def render_html(hits: list[TenderMatch], total: int, interests: str, hours: int = 24) -> str:
    from html import escape
    from datetime import date

    def card(match: TenderMatch) -> str:
        url = escape(f"https://www.find-tender.service.gov.uk/Notice/{match.notice_id}")
        value_html = f'<p class="value">{escape(match.value)}</p>' if match.value else ""
        return f"""
        <article>
          <h2><a href="{url}">{escape(match.title)}</a></h2>
{value_html}
          <p class="reason">{escape(match.reason)}</p>
        </article>"""

    if hits:
        body = "\n".join(card(m) for m in hits)
    else:
        body = '<p class="empty">No matching tenders found.</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tenderbot — {escape(date.today().isoformat())}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; background: #f5f5f5; color: #1a1a1a; padding: 1rem; }}
    header {{ max-width: 640px; margin: 0 auto 1.5rem; }}
    h1 {{ font-size: 1.25rem; font-weight: 700; }}
    .meta {{ margin-top: .25rem; font-size: .875rem; color: #555; }}
    main {{ max-width: 640px; margin: 0 auto; display: flex; flex-direction: column; gap: 1rem; }}
    article {{ background: #fff; border-radius: 8px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    h2 {{ font-size: 1rem; font-weight: 600; margin-bottom: .5rem; }}
.value {{ font-size: .875rem; color: #333; font-weight: 600; margin-bottom: .25rem; }}
    .reason {{ font-size: .875rem; color: #444; line-height: 1.4; }}
    .empty {{ max-width: 640px; margin: 0 auto; color: #666; }}
  </style>
</head>
<body>
  <header>
    <h1>Tenderbot</h1>
    <p class="meta">{len(hits)} match(es) from {total} tenders &nbsp;·&nbsp; {escape(date.today().isoformat())}</p>
    <p class="meta">Last {hours} hours &nbsp;·&nbsp; Interests: {escape(interests)}</p>
  </header>
  <main>
    {body}
  </main>
</body>
</html>"""


def render_index_html(dates: list[str]) -> str:
    from html import escape

    sorted_dates = sorted(dates, reverse=True)
    if sorted_dates:
        items = "\n".join(
            f'    <li><a href="{escape(d)}/index.html">{escape(d)}</a></li>'
            for d in sorted_dates
        )
        body = f"<ul>\n{items}\n  </ul>"
    else:
        body = "<p>No results yet.</p>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tenderbot</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #f5f5f5; color: #1a1a1a; padding: 1rem; max-width: 640px; margin: 0 auto; }}
    h1 {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; }}
    ul {{ list-style: none; padding: 0; display: flex; flex-direction: column; gap: .5rem; }}
    li a {{ text-decoration: none; color: #1a6cbc; font-size: .9rem; }}
    li a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Tenderbot</h1>
  {body}
</body>
</html>"""


def _update_public_index(public_dir: str) -> None:
    import os
    import re

    date_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    try:
        entries = os.listdir(public_dir)
    except FileNotFoundError:
        entries = []
    dates = [e for e in entries if date_pat.match(e) and os.path.isdir(os.path.join(public_dir, e))]
    html = render_index_html(dates)
    with open(os.path.join(public_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find UK tenders matching your interests using Claude."
    )
    parser.add_argument(
        "--interests",
        default="health, NHS, prevention, interoperability",
        help="Your areas of interest (default: health, NHS, prevention, interoperability)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Hours back to search (default: 24)",
    )
    parser.add_argument(
        "--cache",
        default=DEFAULT_CACHE_PATH,
        help=f"Cache file path (default: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip the API fetch and re-run evaluation against the last cached results",
    )
    parser.add_argument(
        "--results-cache",
        default=DEFAULT_RESULTS_CACHE_PATH,
        help=f"Results cache file path (default: {DEFAULT_RESULTS_CACHE_PATH})",
    )
    parser.add_argument(
        "--use-results-cache",
        action="store_true",
        help="Skip the API fetch and Claude evaluation, re-render from last cached results",
    )
    args = parser.parse_args()

    if args.use_results_cache:
        cached = load_results_cache(args.results_cache)
        if cached is None:
            print(f"No results cache found at {args.results_cache}. Run without --use-results-cache first.", file=sys.stderr)
            raise SystemExit(1)
        cached_interests, results = cached
        if cached_interests != args.interests:
            print(f"Warning: cached results were evaluated against different interests:", file=sys.stderr)
            print(f"  cached:    {cached_interests}", file=sys.stderr)
            print(f"  requested: {args.interests}", file=sys.stderr)
        print(f"Loaded {len(results)} result(s) from results cache.", file=sys.stderr)
        hits = [r for r in results if r.relevant]
        _print_and_render(hits, results, args)
        return

    if args.use_cache:
        releases = load_cache(args.cache)
        if releases is None:
            print(f"No cache found at {args.cache}. Run without --use-cache first.", file=sys.stderr)
            raise SystemExit(1)
        print(f"Loaded {len(releases)} tender release(s) from cache.", file=sys.stderr)
    else:
        print(f"Fetching tenders from the last {args.hours} hours…", file=sys.stderr)
        releases = fetch_tenders(args.hours)
        print(f"Found {len(releases)} tender release(s).", file=sys.stderr)
        save_cache(releases, args.cache)
        print(f"Cached to {args.cache}.", file=sys.stderr)

    if not releases:
        print("No tenders found in the specified time window.")
        return

    print(f"Evaluating against interests: {args.interests}", file=sys.stderr)
    client = anthropic.Anthropic()
    results = evaluate(releases, args.interests, client)
    enrich_matches(results, releases)
    save_results_cache(results, args.interests, args.results_cache)
    print(f"Results cached to {args.results_cache}.", file=sys.stderr)

    hits = [r for r in results if r.relevant]
    _print_and_render(hits, results, args)


def _print_and_render(hits: list[TenderMatch], results: list[TenderMatch], args) -> None:
    import os
    from datetime import date

    today = date.today().isoformat()
    out_dir = os.path.join("public", today)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "index.html")

    html = render_html(hits, len(results), args.interests, hours=args.hours)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Results written to {out_path}", file=sys.stderr)

    _update_public_index("public")
    print("Index updated at public/index.html", file=sys.stderr)


if __name__ == "__main__":
    main()
