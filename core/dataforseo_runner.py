"""
core/dataforseo_runner.py

DataForSEO AI Optimization — ChatGPT LLM Responses (Live endpoint).
Replaces the old ai_runner.py for tracking runs.

To revert to old runner, change the import in app.py back to:
    from core.ai_runner import ALL_TOOLS, run_selected_tools, ...

Response structure (confirmed from live API docs):
tasks[0].result[0].items[]
  - type = "reasoning"  → skip (internal chain of thought)
  - type = "message"    → use this
      .sections[]
          .type = "text"
          .text = actual response text
          .annotations[] = [{title, url}, ...]
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
            "Real ChatGPT responses via DataForSEO LLM Responses API. "
            "Routes prompts through the actual ChatGPT interface, not the OpenAI API directly."
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
# ISO COUNTRY CODE MAP
# =============================================================================

_COUNTRY_ISO = {
    "united states": "US", "united kingdom": "GB", "canada": "CA",
    "australia": "AU", "germany": "DE", "india": "IN", "pakistan": "PK",
    "france": "FR", "spain": "ES", "italy": "IT", "brazil": "BR",
    "netherlands": "NL", "singapore": "SG", "uae": "AE",
    "united arab emirates": "AE", "south africa": "ZA",
}


# =============================================================================
# CORE API CALL
# =============================================================================

_LIVE_URL = "https://api.dataforseo.com/v3/ai_optimization/chat_gpt/llm_responses/live"
_MODEL    = "gpt-4.1-mini"


def _call_live(prompt: str, country: str = "") -> dict:
    """
    Sends one prompt to DataForSEO ChatGPT LLM Responses Live endpoint.

    Confirmed response path (from API docs + live example):
      tasks[0].result[0].items[]
        → skip type="reasoning"
        → use  type="message"
             .sections[].text       ← response text
             .sections[].annotations[].url/title  ← sources
    """
    login, password = _get_auth()
    if not login or not password:
        return {
            "text": (
                "ERROR: DataForSEO credentials not set. "
                "Add DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD in the API Keys sidebar."
            ),
            "sources": [],
            "web_searched": False,
        }

    payload = [{
        "model_name": _MODEL,
        "user_prompt": prompt.strip()[:500],   # API hard limit
        "web_search": True,
        "max_output_tokens": 1024,
    }]

    # Add country code for geo-aware web search
    country_iso = _COUNTRY_ISO.get(country.lower().strip(), "")
    if country_iso:
        payload[0]["web_search_country_iso_code"] = country_iso

    try:
        resp = requests.post(
            _LIVE_URL,
            json=payload,
            auth=HTTPBasicAuth(login, password),
            timeout=120,   # API docs say up to 120 seconds
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {"text": "ERROR: DataForSEO request timed out (120s).", "sources": [], "web_searched": False}
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        return {"text": f"ERROR: DataForSEO HTTP {code}: {body}", "sources": [], "web_searched": False}
    except requests.exceptions.RequestException as e:
        return {"text": f"ERROR: DataForSEO connection failed: {e}", "sources": [], "web_searched": False}

    # ── Parse response ────────────────────────────────────────────────────────
    try:
        tasks = data.get("tasks") or []
        if not tasks:
            return {"text": "ERROR: DataForSEO returned no tasks.", "sources": [], "web_searched": False}

        task        = tasks[0]
        status_code = task.get("status_code", 0)
        status_msg  = task.get("status_message", "")

        if status_code != 20000:
            return {
                "text": f"ERROR: DataForSEO task failed (code {status_code}): {status_msg}",
                "sources": [],
                "web_searched": False,
            }

        results = task.get("result") or []
        if not results:
            return {"text": "ERROR: DataForSEO returned empty result array.", "sources": [], "web_searched": False}

        result = results[0]
        items  = result.get("items") or []

        text_parts = []
        sources    = []

        for item in items:
            item_type = item.get("type", "")

            # Skip reasoning items — they are internal chain-of-thought, not the response
            if item_type == "reasoning":
                continue

            # Only process message items
            if item_type != "message":
                continue

            for section in item.get("sections") or []:
                # Extract response text
                t = (section.get("text") or "").strip()
                if t:
                    text_parts.append(t)

                # Extract source annotations (confirmed inside sections)
                for ann in section.get("annotations") or []:
                    url   = (ann.get("url") or "").strip()
                    title = (ann.get("title") or url or "Source").strip()
                    if url:
                        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
                        # Strip ?utm_source=openai tracking params for clean display
                        clean_url = url.split("?utm_source=")[0] if "?utm_source=" in url else url
                        sources.append({"url": clean_url, "title": title, "domain": domain})

        full_text = "\n\n".join(text_parts).strip()

        if not full_text:
            return {
                "text": "ERROR: DataForSEO returned no message content.",
                "sources": [],
                "web_searched": result.get("web_search", False),
            }

        return {
            "text": full_text,
            "sources": sources,
            "web_searched": result.get("web_search", True),
        }

    except Exception as e:
        return {
            "text": f"ERROR: Failed to parse DataForSEO response: {e}",
            "sources": [],
            "web_searched": False,
        }


# =============================================================================
# PUBLIC INTERFACE — matches shape app.py expects from old ai_runner
# =============================================================================

def run_selected_tools(query: str, selected_tools: list, country: str = "") -> dict:
    """
    Runs query through DataForSEO ChatGPT LLM Responses.
    Returns dict keyed by tool name — matches old ai_runner.run_selected_tools.
    """
    results = {}

    if TOOL_NAME not in selected_tools:
        return results

    response = _call_live(query, country=country)
    is_error = response["text"].startswith("ERROR")
    _track(response["text"], is_error=is_error)
    results[TOOL_NAME] = response

    return results