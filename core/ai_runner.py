import os
import time
import importlib
import requests
# from groq import Groq  # GROQ DISABLED
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]

CHATGPT_DEFAULT_MODEL = "gpt-5.5-instant"
CHATGPT_FALLBACK_MODELS = ["gpt-5.4-mini", "gpt-5.4-nano"]
# High enough for full ChatGPT-style answers (tables, comparisons, multi-section replies).
CHATGPT_MAX_OUTPUT_TOKENS = 16384

CHATGPT_SYSTEM_INSTRUCTION = (
    "You are ChatGPT answering in the web interface. Match that depth, structure, and tone.\n\n"
    "MANDATORY FOR TOOL / VENDOR / SOFTWARE / SERVICE QUESTIONS:\n"
    "1. NEVER respond with generic advice like 'research online', 'check G2', or 'consult professionals'.\n"
    "2. ALWAYS name at least 10 specific real products, companies, or vendors by name.\n"
    "3. ALWAYS include a markdown comparison table with columns like: Tool | Best for | Pros | Cons.\n"
    "4. ALWAYS include a 'My recommendation' or 'Best pick by use case' section with 2-3 scenarios.\n"
    "5. Write at least 400 words. Use clear headers, bullet points, and a closing summary.\n"
    "6. If the user query is a short keyword phrase, treat it as a buyer asking for recommendations "
    "and answer as if they typed a full question in ChatGPT.\n"
    "7. Do not artificially shorten. Give the same rich answer a user would see on chat.openai.com."
)


def _wrap_chatgpt_user_query(query: str) -> str:
    """Expand bare keywords into a full buyer question so the model answers like ChatGPT web."""
    q = (query or "").strip()
    if not q:
        return q
    word_count = len(q.split())
    is_question = q.endswith("?") or any(
        q.lower().startswith(w) for w in ("what", "which", "who", "how", "where", "when", "as a", "i ", "we ")
    )
    if word_count <= 6 and not is_question:
        return (
            f"{q}\n\n"
            "I'm evaluating options and want a detailed answer like ChatGPT would give on the web. "
            "Please recommend specific named tools or vendors (at least 10), include a comparison table, "
            "explain pros and cons, and give clear recommendations by use case."
        )
    if not is_question:
        return (
            f"{q}\n\n"
            "Please answer in full ChatGPT style: name specific products/vendors, use a comparison table, "
            "and give practical recommendations — not generic steps to research."
        )
    return q


def _extract_responses_api_text(response) -> str:
    """Collect full assistant text from all output blocks (not just the first snippet)."""
    parts = []
    if hasattr(response, "output_text") and response.output_text:
        parts.append(response.output_text.strip())

    if hasattr(response, "output"):
        for item in response.output:
            if getattr(item, "type", "") != "message":
                continue
            for block in getattr(item, "content", []) or []:
                text = getattr(block, "text", None)
                if text and text.strip():
                    parts.append(text.strip())

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return "\n\n".join(unique).strip()


def _extract_responses_api_sources(response) -> list:
    sources = []
    if not hasattr(response, "output"):
        return sources
    for item in response.output:
        if getattr(item, "type", "") != "message":
            continue
        for block in getattr(item, "content", []) or []:
            for ann in getattr(block, "annotations", []) or []:
                url = getattr(ann, "url", "")
                if url:
                    parts = url.split("/")
                    domain = parts[2] if len(parts) > 2 else url
                    sources.append({
                        "title": getattr(ann, "title", domain),
                        "url": url,
                        "domain": domain,
                    })
    return sources


def _is_openai_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(x in msg for x in ["429", "rate_limit", "quota", "resource_exhausted", "insufficient_quota"])

# =============================================================================
# USAGE TRACKER
# =============================================================================

usage_stats = {
    # "Groq_Llama3":   {"calls": 0, "estimated_tokens": 0, "errors": 0, "free_limit": 14400},  # GROQ DISABLED
    # "Groq_Mixtral":  {"calls": 0, "estimated_tokens": 0, "errors": 0, "free_limit": 14400},  # GROQ DISABLED
    "Perplexity":    {"calls": 0, "estimated_tokens": 0, "errors": 0, "free_limit": 1000},
    "Gemini":        {"calls": 0, "estimated_tokens": 0, "errors": 0, "free_limit": 1500},
    "ChatGPT":       {"calls": 0, "estimated_tokens": 0, "errors": 0, "free_limit": 0},
    "Claude":        {"calls": 0, "estimated_tokens": 0, "errors": 0, "free_limit": 0},
}

def track_usage(tool_name: str, response_text: str, is_error: bool = False):
    if tool_name not in usage_stats:
        usage_stats[tool_name] = {"calls": 0, "estimated_tokens": 0, "errors": 0, "free_limit": 0}
    usage_stats[tool_name]["calls"] += 1
    if is_error:
        usage_stats[tool_name]["errors"] += 1
    else:
        estimated = len(response_text.split()) * 1.3
        usage_stats[tool_name]["estimated_tokens"] += int(estimated)

def get_usage_stats() -> dict:
    return usage_stats

def reset_usage_stats():
    for tool in usage_stats:
        usage_stats[tool]["calls"] = 0
        usage_stats[tool]["estimated_tokens"] = 0
        usage_stats[tool]["errors"] = 0


# =============================================================================
# FREE TOOLS - GROQ DISABLED
# To re-enable Groq: uncomment the functions below and move entries back to ALL_TOOLS
# =============================================================================

# def run_on_groq_llama(query: str) -> str:
#     for attempt in range(3):
#         try:
#             groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#             response = groq_client.chat.completions.create(
#                 model="llama-3.3-70b-versatile",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and completely based on what is being asked."},
#                     {"role": "user", "content": query}
#                 ],
#                 max_tokens=800,
#                 temperature=0.7
#             )
#             result = response.choices[0].message.content.strip()
#             track_usage("Groq_Llama3", result)
#             return result
#         except Exception as e:
#             if "rate" in str(e).lower() and attempt < 2:
#                 wait = (attempt + 1) * 5
#                 time.sleep(wait)
#             else:
#                 track_usage("Groq_Llama3", "", is_error=True)
#                 return f"ERROR: {str(e)}"


# def run_on_groq_mixtral(query: str) -> str:
#     try:
#         groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#         response = groq_client.chat.completions.create(
#             model="mixtral-8x7b-32768",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and completely based on what is being asked."},
#                 {"role": "user", "content": query}
#             ],
#             max_tokens=800,
#             temperature=0.7
#         )
#         result = response.choices[0].message.content.strip()
#         track_usage("Groq_Mixtral", result)
#         return result
#     except Exception as e:
#         track_usage("Groq_Mixtral", "", is_error=True)
#         return run_on_groq_llama(query)


# =============================================================================
# PAID TOOLS (uncomment function + add to ALL_TOOLS to activate)
# =============================================================================

def run_on_perplexity(query: str) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and completely based on what is being asked."},
                {"role": "user", "content": query}
            ],
            "max_tokens": 800
        }
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"].strip()
        track_usage("Perplexity", result)
        return result
    except Exception as e:
        track_usage("Perplexity", "", is_error=True)
        return f"ERROR: {str(e)}"


def run_on_gemini(query: str) -> str:
    try:
        key = os.getenv("GEMINI_API_KEY")

        # New Gemini SDK: google-genai
        try:
            try:
                from google import genai
            except ImportError:
                genai = importlib.import_module("google.genai")
            client = genai.Client(api_key=key)
            result = None
            for model_name in GEMINI_MODELS:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=query
                    )
                    result = (response.text or "").strip()
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if (
                        "404" in msg
                        or "not_found" in msg
                        or "not found" in msg
                        or "429" in msg
                        or "rate_limit" in msg
                        or "resource_exhausted" in msg
                        or "quota" in msg
                    ):
                        continue
                    raise
            if result is None:
                raise RuntimeError("No supported Gemini model found for this API key/project.")
        except (ImportError, ModuleNotFoundError):
            # Legacy Gemini SDK fallback: google-generativeai
            genai_legacy = importlib.import_module("google.generativeai")
            genai_legacy.configure(api_key=key)
            result = None
            for model_name in GEMINI_MODELS:
                try:
                    model = genai_legacy.GenerativeModel(model_name)
                    response = model.generate_content(query)
                    result = (response.text or "").strip()
                    break
                except Exception as e:
                    msg = str(e).lower()
                    if (
                        "404" in msg
                        or "not_found" in msg
                        or "not found" in msg
                        or "429" in msg
                        or "rate_limit" in msg
                        or "resource_exhausted" in msg
                        or "quota" in msg
                    ):
                        continue
                    raise
            if result is None:
                raise RuntimeError("No supported Gemini model found for this API key/project.")

        track_usage("Gemini", result)
        return result
    except Exception as e:
        track_usage("Gemini", "", is_error=True)
        return f"ERROR: {str(e)}"


def run_on_chatgpt(query: str, country: str = "") -> dict:
    """
    Runs query through GPT-5.5 Instant with live web search enabled.
    Falls back to GPT-5.4 mini / nano when rate limits are hit.
    Returns: {"text": str, "sources": list, "web_searched": bool}
    This matches the ChatGPT web interface behaviour - grounded in live web results.
    country: optional ISO country code or name for geographic relevance (e.g. "US", "India")
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        web_search_error = None
        models_to_try = [CHATGPT_DEFAULT_MODEL] + CHATGPT_FALLBACK_MODELS
        user_input = _wrap_chatgpt_user_query(query)

        for model_name in models_to_try:
            # ── Method 1: Responses API with web_search tool ────────────────
            try:
                response = client.responses.create(
                    model=model_name,
                    instructions=CHATGPT_SYSTEM_INSTRUCTION,
                    tools=[{"type": "web_search"}],
                    input=user_input,
                    max_output_tokens=CHATGPT_MAX_OUTPUT_TOKENS
                )

                result_text = _extract_responses_api_text(response)
                sources = _extract_responses_api_sources(response)

                status = getattr(response, "status", None)
                if status == "incomplete":
                    web_search_error = f"{model_name} response incomplete (hit output limit)"

                if result_text:
                    track_usage("ChatGPT", result_text)
                    return {
                        "text": result_text,
                        "sources": sources,
                        "web_searched": True,
                        "model_used": model_name,
                        "incomplete": status == "incomplete",
                    }
                web_search_error = f"{model_name} Responses API returned empty text"

            except Exception as e1:
                if _is_openai_rate_limit_error(e1) and model_name != models_to_try[-1]:
                    web_search_error = f"{model_name} rate limited, trying fallback"
                    continue
                web_search_error = f"{model_name} Responses API: {str(e1)}"

            # ── Method 2: Chat Completions (standard) ─────────────────────────
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": CHATGPT_SYSTEM_INSTRUCTION},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=CHATGPT_MAX_OUTPUT_TOKENS
                )
                result_text = completion.choices[0].message.content.strip()
                if result_text:
                    track_usage("ChatGPT", result_text)
                    return {
                        "text": result_text,
                        "sources": [],
                        "web_searched": False,
                        "model_used": model_name,
                        "fallback_reason": web_search_error or ""
                    }
                web_search_error = (web_search_error or "") + f" | {model_name}: empty response"

            except Exception as e2:
                if _is_openai_rate_limit_error(e2) and model_name != models_to_try[-1]:
                    web_search_error = f"{model_name} rate limited, trying fallback"
                    continue
                web_search_error = (web_search_error or "") + f" | {model_name}: {str(e2)}"

        if web_search_error:
            if "web_search_errors" not in usage_stats:
                usage_stats["web_search_errors"] = []
            usage_stats["web_search_errors"].append(str(web_search_error)[:200])

        track_usage("ChatGPT", "", is_error=True)
        return {"text": f"ERROR: {web_search_error or 'All ChatGPT models failed'}", "sources": [], "web_searched": False}

    except Exception as e:
        track_usage("ChatGPT", "", is_error=True)
        return {"text": f"ERROR: {str(e)}", "sources": [], "web_searched": False}


def run_on_claude(query: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": query}]
        )
        result = message.content[0].text.strip()
        track_usage("Claude", result)
        return result
    except Exception as e:
        track_usage("Claude", "", is_error=True)
        return f"ERROR: {str(e)}"


# =============================================================================
# ALL TOOLS REGISTRY
# Every tool that exists in this system, active or not.
# The frontend reads this to build the toggle UI.
# =============================================================================

# =============================================================================
# GROQ TOOLS - currently disabled. To re-enable:
# 1. Uncomment the functions above
# 2. Uncomment DISABLED_TOOLS below
# 3. Move the entries into ALL_TOOLS
# =============================================================================
# DISABLED_TOOLS = {
#     "Groq_Llama3": {
#         "fn": run_on_groq_llama,
#         "free": True,
#         "requires_key": "GROQ_API_KEY",
#         "cost_per_call": 0.0,
#         "free_limit": 14400,
#         "description": "Llama 3.3 70B via Groq. Free tier."
#     },
#     "Groq_Mixtral": {
#         "fn": run_on_groq_mixtral,
#         "free": True,
#         "requires_key": "GROQ_API_KEY",
#         "cost_per_call": 0.0,
#         "free_limit": 14400,
#         "description": "Mixtral 8x7B via Groq. Free tier."
#     },
# }

ALL_TOOLS = {
    "Perplexity": {
        "fn": run_on_perplexity,
        "free": False,
        "requires_key": "PERPLEXITY_API_KEY",
        "cost_per_call": 0.005,
        "free_limit": 0,
        "description": "Web-grounded answers. Most accurate. $5 free credit."
    },
    "Gemini": {
        "fn": run_on_gemini,
        "free": False,
        "requires_key": "GEMINI_API_KEY",
        "cost_per_call": 0.0,
        "free_limit": 0,
        "description": "Gemini API via key. Quota/billing depends on your Google project."
    },
    "ChatGPT": {
        "fn": run_on_chatgpt,
        "free": False,
        "requires_key": "OPENAI_API_KEY",
        "cost_per_call": 0.005,
        "free_limit": 0,
        "description": "GPT-5.5 Instant (fallback: GPT-5.4 mini/nano). Paid only."
    },
    "Claude": {
        "fn": run_on_claude,
        "free": False,
        "requires_key": "ANTHROPIC_API_KEY",
        "cost_per_call": 0.003,
        "free_limit": 0,
        "description": "Claude Sonnet. Strong reasoning. Paid only."
    },
}


def run_selected_tools(query: str, selected_tools: list, country: str = "") -> dict:
    """
    Runs query through only the selected tools.
    Returns dict of {tool_name: result} where result is either:
    - {"text": str, "sources": list, "web_searched": bool} for tools with web search (ChatGPT)
    - A plain string for tools without web search (Perplexity, Gemini, Claude)
    app.py handles both formats transparently.
    country: passed to tools that support geographic relevance (ChatGPT web search)
    """
    results = {}
    for tool_name in selected_tools:
        if tool_name in ALL_TOOLS:
            tool_fn = ALL_TOOLS[tool_name]["fn"]
            # Pass country to tools that support it
            import inspect
            fn_params = inspect.signature(tool_fn).parameters
            if "country" in fn_params:
                results[tool_name] = tool_fn(query, country=country)
            else:
                results[tool_name] = tool_fn(query)
            time.sleep(2)
    return results


def get_active_tool_names() -> list:
    return []  # Groq disabled


def check_key_exists(env_key: str) -> bool:
    val = os.getenv(env_key)
    return bool(val and val.strip() and "paste_your" not in val.lower())