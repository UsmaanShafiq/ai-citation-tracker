import json
import importlib
import os
import random
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]
TARGET_QUERY_COUNT = 10

QUERY_CATEGORIES = [
    {"name": "awareness", "count": 5},
    {"name": "comparison", "count": 5},
    {"name": "pain_point", "count": 5},
    {"name": "buying_intent", "count": 5},
    {"name": "specific_filter", "count": 5},
]


def call_llm(
    prompt: str,
    preferred_backend: str = "groq",
    allowed_backends: list = None
) -> str:
    """
    Calls LLM using preferred backend with auto fallback.
    """
    if allowed_backends:
        backends = [b for b in allowed_backends if b in {"groq", "gemini"}]
    else:
        backends = ["gemini", "groq"] if preferred_backend == "gemini" else ["groq", "gemini"]

    if not backends:
        raise RuntimeError("No allowed LLM backends configured for query generation.")
    last_error = None

    for backend in backends:
        try:
            if backend == "gemini":
                key = os.getenv("GEMINI_API_KEY", "")
                if not key or "paste_your" in key.lower():
                    continue
                # New Gemini SDK: google-genai
                try:
                    try:
                        from google import genai
                    except ImportError:
                        genai = importlib.import_module("google.genai")
                    client = genai.Client(api_key=key)
                    for model_name in GEMINI_MODELS:
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=prompt
                            )
                            return (response.text or "").strip()
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
                    raise RuntimeError("No supported Gemini model found for this API key/project.")
                except (ImportError, ModuleNotFoundError):
                    pass

                # Legacy Gemini SDK fallback: google-generativeai
                try:
                    genai_legacy = importlib.import_module("google.generativeai")
                    genai_legacy.configure(api_key=key)
                    for model_name in GEMINI_MODELS:
                        try:
                            model = genai_legacy.GenerativeModel(model_name)
                            response = model.generate_content(prompt)
                            return (response.text or "").strip()
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
                    raise RuntimeError("No supported Gemini model found for this API key/project.")
                except (ImportError, ModuleNotFoundError):
                    raise RuntimeError(
                        "Gemini SDK not installed. Install one of: "
                        "'pip install google-genai' (recommended) or "
                        "'pip install google-generativeai'."
                    )

            else:
                key = os.getenv("GROQ_API_KEY", "")
                if not key or "paste_your" in key.lower():
                    continue
                from groq import Groq
                client = Groq(api_key=key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8000,
                    temperature=0.8
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "rate" in error_str or "429" in error_str or "quota" in error_str:
                continue
            raise

    raise RuntimeError(f"All backends failed or rate limited. Last error: {last_error}")


def generate_all_queries(
    icp_data: dict,
    website_url: str = None,
    competitors: list = None,
    preferred_backend: str = "groq",
    allowed_backends: list = None
) -> list:

    competitors_text = ", ".join(competitors) if competitors else "not specified"

    icp_formatted = f"""
Industries: {', '.join(icp_data.get('industries', []))}
Company sizes: {', '.join(icp_data.get('company_sizes', []))}
Geographies: {', '.join(icp_data.get('geographies', []))}
Buyer roles: {', '.join(icp_data.get('buyer_roles', []))}
Pain points: {', '.join(icp_data.get('pain_points', []))}
Budget ranges: {', '.join(icp_data.get('budget_ranges', []))}
Tech stacks: {', '.join(icp_data.get('tech_stacks', []))}
Company stages: {', '.join(icp_data.get('company_stages', []))}
Compliance needs: {', '.join(icp_data.get('compliance_needs', []))}
"""

    prompt = f"""You are an expert B2B SaaS market researcher and buyer behavior analyst. Your job is to generate hyper-specific AI search queries that real buyers type when evaluating software solutions.

## INPUTS
Ideal Client Profile (ICP):
{icp_formatted}

Competitor Names: {competitors_text}

## YOUR RESEARCH MINDSET
1. Inhabit the buyer's world. Real person, frustrated, time-pressured, skeptical of vendor marketing.
2. Channel social discussion forums. Think Reddit, G2, Capterra, LinkedIn, Slack. Use their vocabulary not vendor vocabulary.
3. Map the buying journey across: problem recognition, solution exploration, vendor evaluation, validation, decision.

## QUERY CONSTRUCTION RULES
Each query MUST include at least 5 of these filter dimensions. Vary filters so no two queries feel alike:
- Geography, Company Size/Stage, Team Size, Budget Signal, Industry Vertical
- Job Title/Persona, Pain Point, Trigger/Timing, Emotional State, Situational Context
- Specific Feature Need, Competitor Context, Buying Stage, Integration Need

## QUALITY CHECKS
- No two queries should have the same filter combination
- Must read like something a human types into ChatGPT, not a keyword string
- At least 3 queries must carry emotional language like "sick of", "fed up", "nothing seems to work"
- At least 4 queries must be late-stage high-intent
- Span at least 4 different industries or verticals
- Zero vendor/marketing language. No "robust", "seamless", "end-to-end"
- Do NOT mention the company being tracked in any query
- Every query must feel like it belongs in a Reddit thread or Slack community

## OUTPUT FORMAT
Return a JSON array of exactly 10 objects. Each object:
{{
  "query_id": 1,
  "query": "<natural buyer language>",
  "buying_stage": "<problem_recognition | solution_exploration | vendor_evaluation | validation | decision>",
  "filters_applied": {{
    "geography": "<value or null>",
    "company_size": "<value or null>",
    "team_size": "<value or null>",
    "budget_signal": "<value or null>",
    "industry_vertical": "<value or null>",
    "persona": "<value or null>",
    "pain_point": "<value or null>",
    "trigger": "<value or null>",
    "emotional_state": "<value or null>",
    "situational_context": "<value or null>",
    "feature_need": "<value or null>",
    "competitor_context": "<value or null>",
    "integration_need": "<value or null>"
  }},
  "filter_count": <integer minimum 5>,
  "rationale": "<1 sentence why a real buyer would search this>"
}}

Return ONLY the JSON array. No preamble. No explanation. No markdown fences."""

    raw = call_llm(
        prompt,
        preferred_backend=preferred_backend,
        allowed_backends=allowed_backends
    )

    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        queries = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            queries = json.loads(raw[start:end])
        else:
            raise RuntimeError(f"Could not parse query JSON. Raw:\n{raw[:500]}")

    normalized = []
    for q in queries[:TARGET_QUERY_COUNT]:
        normalized.append({
            "query": q["query"],
            "category": q["buying_stage"],
            "filters": q["filters_applied"],
            "filter_count": q.get("filter_count", 5),
            "rationale": q.get("rationale", ""),
            "query_id": q.get("query_id", 0)
        })

    return normalized