"""
core/dataforseo_runner.py

DataForSEO AI Optimization — ChatGPT LLM Scraper (Live Advanced endpoint).
Switched from LLM Responses to LLM Scraper for richer data.

Scraper endpoint gives us:
  - brand_entities: auto-detected brands with category
  - sources: with thumbnails, snippets, publication dates
  - markdown: full formatted response
  - check_url: direct link to verify on ChatGPT
  - fan_out_queries: what ChatGPT searched internally
  - item_types: text/table/images/products breakdown

To revert to LLM Responses endpoint, change _SCRAPER_URL back to:
  https://api.dataforseo.com/v3/ai_optimization/chat_gpt/llm_responses/live
and restore the old payload format.

Response structure (confirmed from scraper API docs):
tasks[0].result[0]
  .markdown          → full formatted response text
  .check_url         → direct ChatGPT verification URL
  .fan_out_queries   → what ChatGPT searched internally
  .item_types        → response format types
  .brand_entities[]  → auto-detected brands with category/urls
  .sources[]         → cited sources with thumbnail, snippet, publication_date
  .items[]           → structured response elements (text, table, images etc)
    type = "chat_gpt_text"
      .markdown      → response text
      .sources[]     → sources for this section
      .brand_entities[] → brands in this section
"""

import os
import base64
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# TOOL REGISTRY
# =============================================================================

ALL_TOOLS = {
    "ChatGPT (DataForSEO)": {
        "description": (
            "Real ChatGPT responses via DataForSEO LLM Scraper API. "
            "Scrapes the actual ChatGPT search interface for richer results "
            "including brand entities, source thumbnails, and fan-out queries."
        ),
        "requires_key": "DATAFORSEO_LOGIN",
        "free": False,
    }
}

TOOL_NAME = "ChatGPT (DataForSEO)"


# =============================================================================
# USAGE TRACKER
# =============================================================================

_usage: dict = {
    TOOL_NAME: {"calls": 0, "estimated_tokens": 0, "errors": 0, "free_limit": 0}
}


def get_usage_stats() -> dict:
    return _usage


def reset_usage_stats():
    for t in _usage:
        _usage[t]["calls"] = 0
        _usage[t]["estimated_tokens"] = 0
        _usage[t]["errors"] = 0


def _track(text: str, is_error: bool = False):
    _usage[TOOL_NAME]["calls"] += 1
    if is_error:
        _usage[TOOL_NAME]["errors"] += 1
    else:
        _usage[TOOL_NAME]["estimated_tokens"] += int(len(text.split()) * 1.3)


# =============================================================================
# CREDENTIAL RESOLUTION
# =============================================================================

def _get_auth() -> tuple:
    """
    Returns (login, password) for HTTPBasicAuth.
    Priority:
    1. DATAFORSEO_LOGIN + DATAFORSEO_PASSWORD env vars
    2. DATAFORSEO_API_KEY env var containing Base64(login:password)
    """
    login    = os.environ.get("DATAFORSEO_LOGIN", "").strip()
    password = os.environ.get("DATAFORSEO_PASSWORD", "").strip()
    if login and password:
        return login, password

    b64 = os.environ.get("DATAFORSEO_API_KEY", "").strip()
    if b64:
        try:
            decoded = base64.b64decode(b64).decode("utf-8")
            if ":" in decoded:
                l, p = decoded.split(":", 1)
                return l.strip(), p.strip()
        except Exception:
            pass

    return "", ""


def check_key_exists(requires_key: str) -> bool:
    login, password = _get_auth()
    return bool(login and password)


# =============================================================================
# LOCATION NAME MAP
# Scraper uses location_name (full name) not ISO codes
# =============================================================================

_COUNTRY_LOCATION = {
    "united states":  "United States",
    "united kingdom": "United Kingdom",
    "canada":         "Canada",
    "australia":      "Australia",
    "germany":        "Germany",
    "india":          "India",
    "pakistan":       "Pakistan",
    "france":         "France",
    "spain":          "Spain",
    "italy":          "Italy",
    "brazil":         "Brazil",
    "netherlands":    "Netherlands",
    "singapore":      "Singapore",
    "uae":            "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    "south africa":   "South Africa",
    "global":         "United States",  # default for global
}


# =============================================================================
# CORE API CALL — LLM Scraper
# =============================================================================

_SCRAPER_URL = "https://api.dataforseo.com/v3/ai_optimization/chat_gpt/llm_scraper/live/advanced"


def _call_scraper(keyword: str, country: str = "") -> dict:
    """
    Sends one keyword/prompt to DataForSEO ChatGPT LLM Scraper.

    Returns:
    {
        "text":            str,   # full response text (from markdown)
        "markdown":        str,   # full markdown response
        "sources":         list,  # [{url, title, domain, thumbnail,
                                  #   snippet, source_name, publication_date}]
        "brand_entities":  list,  # [{title, category, urls}]
        "fan_out_queries": list,  # [str, ...]
        "check_url":       str,   # direct ChatGPT verification URL
        "item_types":      list,  # response format types
        "web_searched":    bool,
    }
    """
    login, password = _get_auth()
    if not login or not password:
        return {
            "text": (
                "ERROR: DataForSEO credentials not set. "
                "Add DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD in the API Keys sidebar."
            ),
            "markdown": "",
            "sources": [],
            "brand_entities": [],
            "fan_out_queries": [],
            "check_url": "",
            "item_types": [],
            "web_searched": False,
        }

    # Resolve location name
    location = _COUNTRY_LOCATION.get(country.lower().strip(), "United States")

    payload = [{
        "keyword":       keyword.strip()[:2000],  # scraper allows 2000 chars
        "location_name": location,
        "language_name": "English",
        "force_web_search": True,
    }]

    try:
        resp = requests.post(
            _SCRAPER_URL,
            json=payload,
            auth=HTTPBasicAuth(login, password),
            timeout=120,  # API docs say up to 120 seconds
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return _error("DataForSEO request timed out (120s).")
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        return _error(f"DataForSEO HTTP {code}: {body}")
    except requests.exceptions.RequestException as e:
        return _error(f"DataForSEO connection failed: {e}")

    # ── Parse response ────────────────────────────────────────────────────────
    try:
        tasks = data.get("tasks") or []
        if not tasks:
            return _error("DataForSEO returned no tasks.")

        task        = tasks[0]
        status_code = task.get("status_code", 0)
        status_msg  = task.get("status_message", "")

        if status_code != 20000:
            return _error(f"DataForSEO task failed (code {status_code}): {status_msg}")

        results = task.get("result") or []
        if not results:
            return _error("DataForSEO returned empty result array.")

        result = results[0]

        # ── Top-level fields ─────────────────────────────────────────────────
        full_markdown = (result.get("markdown") or "").strip()
        check_url     = (result.get("check_url") or "").strip()
        fan_out       = result.get("fan_out_queries") or []
        item_types    = result.get("item_types") or []

        # ── Brand entities (top-level) ───────────────────────────────────────
        brand_entities = _parse_brand_entities(result.get("brand_entities") or [])

        # ── Sources (top-level — what ChatGPT actually cited) ────────────────
        sources = _parse_sources(result.get("sources") or [])

        # ── Extract plain text from items for brand detection ────────────────
        # Use markdown as primary text source, fall back to items
        text_parts = []
        if full_markdown:
            # Strip markdown image syntax for clean text
            import re as _re
            clean = _re.sub(r'!\[.*?\]\(.*?\)', '', full_markdown)
            clean = _re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)
            text_parts.append(clean.strip())
        else:
            # Fall back to extracting text from items array
            for item in result.get("items") or []:
                item_type = item.get("type", "")
                if item_type == "chat_gpt_text":
                    md = (item.get("markdown") or "").strip()
                    if md:
                        text_parts.append(md)
                    # Also get sources from items if not at top level
                    if not sources:
                        sources.extend(
                            _parse_sources(item.get("sources") or [])
                        )
                    # Also get brand entities from items
                    if not brand_entities:
                        brand_entities.extend(
                            _parse_brand_entities(item.get("brand_entities") or [])
                        )

        full_text = "\n\n".join(text_parts).strip()

        if not full_text and not full_markdown:
            return _error("DataForSEO returned no text content.")

        # Use markdown as display text if available, plain text for detection
        display_text = full_markdown or full_text

        return {
            "text":            full_text or full_markdown,
            "markdown":        full_markdown,
            "sources":         sources,
            "brand_entities":  brand_entities,
            "fan_out_queries": [str(q) for q in fan_out if q],
            "check_url":       check_url,
            "item_types":      item_types,
            "web_searched":    True,
        }

    except Exception as e:
        return _error(f"Failed to parse DataForSEO response: {e}")


def _error(msg: str) -> dict:
    """Returns a consistent error response dict."""
    return {
        "text":            f"ERROR: {msg}",
        "markdown":        "",
        "sources":         [],
        "brand_entities":  [],
        "fan_out_queries": [],
        "check_url":       "",
        "item_types":      [],
        "web_searched":    False,
    }


def _parse_sources(raw_sources: list) -> list:
    """
    Parses the sources array from the scraper response.
    Returns list of dicts with all available fields.
    """
    parsed = []
    seen_urls = set()
    for s in raw_sources:
        if not isinstance(s, dict):
            continue
        url = (s.get("url") or "").strip()
        # Clean utm tracking params
        if "?utm_source=" in url:
            url = url.split("?utm_source=")[0]
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        domain = (s.get("domain") or "").strip()
        # Clean markdown link format from domain if present
        if domain.startswith("["):
            import re as _re
            m = _re.search(r'\[([^\]]+)\]', domain)
            if m:
                domain = m.group(1)

        parsed.append({
            "url":              url,
            "title":            (s.get("title") or domain or "Source").strip(),
            "domain":           domain or url.replace("https://", "").replace("http://", "").split("/")[0],
            "thumbnail":        (s.get("thumbnail") or "").strip(),
            "snippet":          (s.get("snippet") or "").strip(),
            "source_name":      (s.get("source_name") or "").strip(),
            "publication_date": (s.get("publication_date") or "").strip(),
            "markdown":         (s.get("markdown") or "").strip(),
        })
    return parsed


def _parse_brand_entities(raw_entities: list) -> list:
    """
    Parses brand_entities from the scraper response.
    Returns list of dicts: {title, category, urls}
    """
    parsed = []
    seen = set()
    for e in raw_entities:
        if not isinstance(e, dict):
            continue
        title = (e.get("title") or "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        urls = []
        for u in e.get("urls") or []:
            if isinstance(u, dict) and u.get("url"):
                urls.append(u["url"])
        parsed.append({
            "title":    title,
            "category": (e.get("category") or "brand").strip(),
            "urls":     urls,
        })
    return parsed


# =============================================================================
# PUBLIC INTERFACE — matches shape app.py expects from old runner
# =============================================================================

def run_selected_tools(query: str, selected_tools: list, country: str = "") -> dict:
    """
    Runs query through DataForSEO ChatGPT LLM Scraper.
    Returns dict keyed by tool name — matches old ai_runner.run_selected_tools.
    """
    results = {}

    if TOOL_NAME not in selected_tools:
        return results

    response = _call_scraper(query, country=country)
    is_error = response["text"].startswith("ERROR")
    _track(response["text"], is_error=is_error)
    results[TOOL_NAME] = response

    return results