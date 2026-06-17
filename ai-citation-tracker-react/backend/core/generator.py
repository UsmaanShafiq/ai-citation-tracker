import json
import re
import os
from backend.config import settings


def _call_ai_for_json(prompt: str, api_keys: dict) -> str:
    """
    Call an available AI model to get JSON output.
    Tries Groq first (free), then Gemini, then OpenAI.
    """
    # Try Groq first (free)
    groq_key = api_keys.get("Groq") or os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception:
            pass

    # Try Gemini
    gemini_key = api_keys.get("Gemini") or os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            return response.text or ""
        except Exception:
            pass

    # Try OpenAI
    openai_key = api_keys.get("ChatGPT") or os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            return response.choices[0].message.content or ""
        except Exception:
            pass

    raise Exception("No available AI model to generate topics. Please add at least one API key.")


def _parse_json_from_response(text: str) -> list:
    """Safely extract JSON array from AI response."""
    # Strip markdown code blocks if present
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and any(isinstance(v, list) for v in result.values()):
            for v in result.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass

    # Try extracting JSON array from text
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return []


def generate_topics(brand_data: dict, api_keys: dict, count: int = 5) -> list:
    """
    Generate AI-powered search topics based on brand profile.
    Returns list of topic strings.
    """
    prompt = f"""You are an AI search visibility expert. Based on the brand information below, generate {count} realistic search topics that potential customers would use when searching for solutions like this brand.

Brand Information:
- Brand Name: {brand_data.get('name', '')}
- Domain: {brand_data.get('domain', '')}
- Products/Services: {', '.join(brand_data.get('products', []))}
- Target Customers: {', '.join(brand_data.get('customers', []))}
- Key Features: {', '.join(brand_data.get('key_features', []))}
- Business Type: {brand_data.get('business_type', '')}
- Country: {brand_data.get('country', 'United States')}

Generate {count} search topics that:
1. Represent real search intents potential customers would have
2. Are specific enough to be meaningful
3. Cover different aspects of the brand's offerings
4. Are phrased as short keyword-style topics (not full questions)

Respond ONLY with a JSON array of strings. No explanation, no markdown, just the array.
Example: ["best accounting software for small businesses", "invoice tracking software", "online bookkeeping software"]

Return exactly {count} topics."""

    raw = _call_ai_for_json(prompt, api_keys)
    topics = _parse_json_from_response(raw)

    # Fallback: extract quoted strings if JSON parsing fails
    if not topics:
        topics = re.findall(r'"([^"]{10,100})"', raw)

    return [t.strip() for t in topics if t.strip()][:count]


def generate_prompts_for_topic(topic: str, brand_data: dict, api_keys: dict, count: int = 5) -> list:
    """
    Generate realistic AI search prompts for a given topic.
    Returns list of prompt strings.
    """
    prompt = f"""You are an AI search visibility expert. For the topic "{topic}", generate {count} realistic conversational prompts that potential customers would type into AI tools like ChatGPT, Perplexity, or Gemini when looking for solutions.

Brand Context:
- Brand: {brand_data.get('name', '')}
- Products: {', '.join(brand_data.get('products', []))}
- Customers: {', '.join(brand_data.get('customers', []))}
- Business Type: {brand_data.get('business_type', '')}

Generate {count} prompts that:
1. Sound natural and conversational, like real user queries
2. Vary in style: some short keyword-style, some full questions, some comparison queries, some persona-based
3. Are specifically about the topic but written from the customer's perspective
4. Would realistically trigger a recommendation for this type of solution

Respond ONLY with a JSON array of strings. No explanation, no markdown, just the array.
Example: ["best accounting software for small businesses", "What's the best accounting software right now?", "Can you compare QuickBooks and Xero?"]

Return exactly {count} prompts."""

    raw = _call_ai_for_json(prompt, api_keys)
    prompts = _parse_json_from_response(raw)

    # Fallback
    if not prompts:
        prompts = re.findall(r'"([^"]{10,200})"', raw)

    # Always include the topic itself as first prompt
    result = [topic] + [p.strip() for p in prompts if p.strip() and p.strip().lower() != topic.lower()]

    return result[:count]


def calculate_visibility(results: list, brand_name: str, brand_domain: str) -> dict:
    """
    Calculate visibility metrics from a list of PromptResult objects.
    Returns overall and per-model/per-topic visibility percentages.
    """
    if not results:
        return {}

    models = list(set(r.model for r in results))
    topics = list(set(r.prompt.topic.name for r in results if r.prompt and r.prompt.topic))

    # Overall visibility per model
    by_model = {}
    for model in models:
        model_results = [r for r in results if r.model == model]
        mentions = sum(1 for r in model_results if r.brand_mentioned)
        total = len(model_results)
        by_model[model] = {
            "mentions": mentions,
            "total": total,
            "pct": round((mentions / total) * 100) if total > 0 else 0,
        }

    # Visibility per topic per model
    by_topic = {}
    for topic in topics:
        topic_results = [r for r in results if r.prompt and r.prompt.topic and r.prompt.topic.name == topic]
        by_topic[topic] = {}
        for model in models:
            model_topic_results = [r for r in topic_results if r.model == model]
            mentions = sum(1 for r in model_topic_results if r.brand_mentioned)
            total = len(model_topic_results)
            by_topic[topic][model] = {
                "mentions": mentions,
                "total": total,
                "pct": round((mentions / total) * 100) if total > 0 else 0,
            }

    # Overall across all models
    total_mentions = sum(1 for r in results if r.brand_mentioned)
    total_results = len(results)
    overall_pct = round((total_mentions / total_results) * 100) if total_results > 0 else 0

    return {
        "overall_pct": overall_pct,
        "total_mentions": total_mentions,
        "total_results": total_results,
        "by_model": by_model,
        "by_topic": by_topic,
    }
