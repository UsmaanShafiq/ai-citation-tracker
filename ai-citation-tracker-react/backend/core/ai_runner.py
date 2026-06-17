import os
import re
import requests
from typing import Optional

# ── Groq ──────────────────────────────────────────────────────────────────────
def run_groq(query: str, api_key: str, model: str = "llama3-8b-8192") -> str:
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── Perplexity ────────────────────────────────────────────────────────────────
def run_perplexity(query: str, api_key: str) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 1024,
        }
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── Gemini ────────────────────────────────────────────────────────────────────
def run_gemini(query: str, api_key: str) -> str:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=query,
        )
        return response.text or ""
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── OpenAI / ChatGPT ──────────────────────────────────────────────────────────
def run_openai(query: str, api_key: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── Anthropic / Claude ────────────────────────────────────────────────────────
def run_anthropic(query: str, api_key: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": query}],
        )
        return message.content[0].text or ""
    except Exception as e:
        return f"ERROR: {str(e)}"


# ── Dispatcher ────────────────────────────────────────────────────────────────
ALL_MODELS = {
    "ChatGPT": {"fn": run_openai, "key_env": "OPENAI_API_KEY"},
    "Perplexity": {"fn": run_perplexity, "key_env": "PERPLEXITY_API_KEY"},
    "Gemini": {"fn": run_gemini, "key_env": "GEMINI_API_KEY"},
    "Groq": {"fn": run_groq, "key_env": "GROQ_API_KEY"},
    "Claude": {"fn": run_anthropic, "key_env": "ANTHROPIC_API_KEY"},
}

def run_prompt_on_model(prompt: str, model_name: str, api_keys: dict) -> str:
    """
    Run a single prompt on a single model.
    api_keys: {"ChatGPT": "sk-...", "Gemini": "AI...", ...}
    """
    if model_name not in ALL_MODELS:
        return f"ERROR: Unknown model {model_name}"

    model_info = ALL_MODELS[model_name]
    api_key = api_keys.get(model_name, "")

    if not api_key:
        env_key = model_info["key_env"]
        api_key = os.getenv(env_key, "")

    if not api_key:
        return f"ERROR: No API key for {model_name}"

    return model_info["fn"](prompt, api_key)


def run_prompt_on_all_models(prompt: str, models: list, api_keys: dict) -> dict:
    """
    Run a prompt across multiple models.
    Returns: {"ChatGPT": "response...", "Gemini": "response...", ...}
    """
    results = {}
    for model_name in models:
        results[model_name] = run_prompt_on_model(prompt, model_name, api_keys)
    return results
