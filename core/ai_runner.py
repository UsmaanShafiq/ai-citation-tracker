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


def run_on_chatgpt(query: str) -> dict:
    """
    Runs query through GPT-4o with live web search enabled.
    Returns: {"text": str, "sources": list, "web_searched": bool}
    This matches the ChatGPT web interface behaviour - grounded in live web results.
    Universal dict format - future tools (Perplexity, Gemini, Claude) return same structure.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Use OpenAI Responses API with web_search_preview tool
        # This is what makes results match what users see in ChatGPT web interface
        try:
            response = client.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=query,
                max_output_tokens=800
            )

            result_text = ""
            sources = []

            for item in response.output:
                if hasattr(item, "type"):
                    if item.type == "message":
                        for block in item.content:
                            if hasattr(block, "text"):
                                result_text = block.text.strip()
                            if hasattr(block, "annotations"):
                                for ann in block.annotations:
                                    if hasattr(ann, "url"):
                                        domain = ann.url.split("/")[2] if "/" in ann.url else ann.url
                                        sources.append({
                                            "title": getattr(ann, "title", domain),
                                            "url": ann.url,
                                            "domain": domain
                                        })

            # Fallback if text empty
            if not result_text and hasattr(response, "output_text"):
                result_text = (response.output_text or "").strip()

            track_usage("ChatGPT", result_text)
            return {"text": result_text, "sources": sources, "web_searched": True}

        except Exception:
            # Fallback to standard completion if web search API unavailable
            fallback = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and completely based on what is being asked."},
                    {"role": "user", "content": query}
                ],
                max_tokens=800
            )
            result = fallback.choices[0].message.content.strip()
            track_usage("ChatGPT", result)
            return {"text": result, "sources": [], "web_searched": False}

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
        "description": "GPT-4o. Most widely used AI. Paid only."
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


def run_selected_tools(query: str, selected_tools: list) -> dict:
    """
    Runs query through only the selected tools.
    Returns dict of {tool_name: result} where result is either:
    - {"text": str, "sources": list, "web_searched": bool} for tools with web search (ChatGPT)
    - A plain string for tools without web search (Perplexity, Gemini, Claude)
    app.py handles both formats transparently.
    """
    results = {}
    for tool_name in selected_tools:
        if tool_name in ALL_TOOLS:
            tool_fn = ALL_TOOLS[tool_name]["fn"]
            results[tool_name] = tool_fn(query)
            time.sleep(2)
    return results


def get_active_tool_names() -> list:
    return []  # Groq disabled


def check_key_exists(env_key: str) -> bool:
    val = os.getenv(env_key)
    return bool(val and val.strip() and "paste_your" not in val.lower())