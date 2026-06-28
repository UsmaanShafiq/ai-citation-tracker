"""
core/dataforseo_runner.py

DataForSEO AI Optimization — ChatGPT LLM Responses (Live endpoint).
This replaces the old ai_runner.py for tracking runs.

The old ai_runner.py is preserved untouched and can be re-enabled by
changing the import line in app.py back to:
    from core.ai_runner import ALL_TOOLS, run_selected_tools, ...

Endpoint: POST https://api.dataforseo.com/v3/ai_optimization/chat_gpt/llm_responses/live

Auth: HTTPBasicAuth(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD)
  OR store Base64(login:password) in DATAFORSEO_API_KEY env var.

Why Live over Standard:
  Live returns immediately — no polling loop needed.
  Standard requires task_post then task_get polling.
  For a Streamlit tracking run, Live is simpler and more reliable.
"""

import os
import base64
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# TOOL REGISTRY — shape matches what app.py expects from ai_runner
# =============================================================================

ALL_TOOLS = {
    "ChatGPT (DataForSEO)": {
        "description": (
            "Real ChatGPT responses via DataForSEO LLM Responses API. "
            "Routes prompts through the actual ChatGPT interface rather than "
            "the OpenAI API directly, giving results closer to what users see."
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
    1. DATAFORSEO_LOGIN + DATAFORSEO_PASSWORD env vars (recommended)
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
_MODEL    = "gpt-4.1-mini"   # fast, web-search capable, cost-effective


def _call_live(prompt: str, country: str = "") -> dict:
    """
    Sends one prompt to DataForSEO ChatGPT LLM Responses Live endpoint.

    Returns:
        {
            "text": str,          # full response text
            "sources": list,      # [{url, title, domain}, ...]
            "web_searched": bool,
        }
    """
    login, password = _get_auth()
    if not login or not password:
        return {
            "text": (
                "ERROR: DataForSEO credentials not configured. "
                "Add DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD in the API Keys sidebar."
            ),
            "sources": [],
            "web_searched": False,
        }

    # Truncate to 500 char API limit
    user_prompt = prompt.strip()[:500]

    payload = [{
        "model_name": _MODEL,
        "user_prompt": user_prompt,
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
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {"text": "ERROR: DataForSEO request timed out (90s). Try again.", "sources": [], "web_searched": False}
    except requests.exceptions.HTTPError as e:
        return {"text": f"ERROR: DataForSEO HTTP error {e.response.status_code}: {e}", "sources": [], "web_searched": False}
    except requests.exceptions.RequestException as e:
        return {"text": f"ERROR: DataForSEO connection failed: {e}", "sources": [], "web_searched": False}

    # Parse response structure:
    # data.tasks[0].result[0].items[].sections[].text
    # data.tasks[0].result[0].items[].annotations[].url / .title
    try:
        tasks = data.get("tasks", [])
        if not tasks:
            return {"text": "ERROR: DataForSEO returned no tasks.", "sources": [], "web_searched": False}

        task = tasks[0]
        status_code = task.get("status_code", 0)
        status_msg  = task.get("status_message", "")

        if status_code != 20000:
            return {
                "text": f"ERROR: DataForSEO task failed (status {status_code}): {status_msg}",
                "sources": [],
                "web_searched": False,
            }

        results = task.get("result") or []
        if not results:
            return {"text": "ERROR: DataForSEO returned empty result.", "sources": [], "web_searched": False}

        result  = results[0]
        items   = result.get("items") or []

        text_parts = []
        sources    = []

        for item in items:
            # Extract text from sections
            for section in item.get("sections") or []:
                t = (section.get("text") or "").strip()
                if t:
                    text_parts.append(t)

            # Extract source citations from annotations
            for ann in item.get("annotations") or []:
                url   = (ann.get("url") or "").strip()
                title = (ann.get("title") or url or "Source").strip()
                if url:
                    domain = url.replace("https://", "").replace("http://", "").split("/")[0]
                    sources.append({"url": url, "title": title, "domain": domain})

        full_text = "\n\n".join(text_parts).strip()

        # Some models put text directly on result rather than in items
        if not full_text:
            full_text = (result.get("text") or "").strip()

        if not full_text:
            return {"text": "ERROR: DataForSEO returned no text content.", "sources": [], "web_searched": True}

        return {
            "text": full_text,
            "sources": sources,
            "web_searched": True,
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