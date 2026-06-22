import streamlit as st
import pandas as pd
import os
import sys
import json
import re
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from core.scorer import calculate_citation_share, calculate_citation_share_by_topic, calculate_citation_share_by_group
from core.ai_runner import ALL_TOOLS, run_selected_tools, check_key_exists, get_usage_stats, reset_usage_stats
from core.brand_detector import detect_brands


# =============================================================================
# HELPERS
# =============================================================================

def format_error_message(raw_error: str) -> str:
    msg = (raw_error or "").strip()
    low = msg.lower()
    if any(x in low for x in ["context_length_exceeded", "maximum context length", "context window", "token limit", "too many tokens", "prompt is too long"]):
        return "Request too large for the selected model. Reduce topics or query count, or switch models."
    if any(x in low for x in ["429", "rate_limit", "resource_exhausted", "quota"]):
        return "API quota or rate limit reached. Wait and retry, or enable another provider as fallback."
    if "no module named" in low or "sdk not installed" in low:
        return "Required SDK is not installed. Install dependencies from requirements.txt and retry."
    return msg


def _get_key(env_key: str) -> str:
    """
    Read API key - checks os.environ first, then re-reads .env file directly.
    This handles the case where a key was saved in the same Streamlit session
    (os.getenv caches the env at startup and won't reflect new saves).
    """
    # First try os.environ (works if key was set before app started)
    val = os.environ.get(env_key, "").strip()
    if val and "paste_your" not in val.lower():
        return val
    # Fall back to reading .env file directly (handles same-session saves)
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(env_key + "="):
                    v = line.split("=", 1)[1].strip()
                    if v and "paste_your" not in v.lower():
                        os.environ[env_key] = v  # update for future calls
                        return v
    return ""


def _call_ai_for_json(prompt: str) -> str:
    """Call best available AI model for JSON generation (topics/prompts)."""
    # GROQ DISABLED - uncomment below to re-enable
    # groq_key = _get_key("GROQ_API_KEY")
    # if groq_key:
    #     try:
    #         from groq import Groq
    #         client = Groq(api_key=groq_key)
    #         resp = client.chat.completions.create(
    #             model="llama-3.3-70b-versatile",
    #             messages=[{"role": "user", "content": prompt}],
    #             max_tokens=2000,
    #             temperature=0.7,
    #         )
    #         return resp.choices[0].message.content or ""
    #     except Exception:
    #         pass

    # Try Gemini
    gemini_key = _get_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return resp.text or ""
        except Exception:
            pass

    # Try OpenAI
    openai_key = _get_key("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            pass

    raise Exception("No available AI model. Please add at least one API key (Gemini, OpenAI, or Perplexity).")


def _parse_json_list(text: str) -> list:
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        for v in result.values():
            if isinstance(v, list):
                return v
    except Exception:
        pass
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # fallback: extract quoted strings
    return re.findall(r'"([^"]{5,200})"', text)



def fetch_brand_website(domain: str) -> str:
    """
    Visits the brand's website and extracts meaningful text content.
    Returns extracted text or empty string if fetch fails.
    Cleans up navigation, scripts, and boilerplate to keep only real content.
    """
    import requests
    from urllib.parse import urlparse

    # Normalize domain to full URL
    domain = domain.strip().rstrip("/")
    if not domain.startswith("http"):
        url = "https://" + domain
    else:
        url = domain

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # Try multiple URL variants to maximize success rate
    urls_to_try = [url]
    if not url.startswith("https://www."):
        urls_to_try.append(url.replace("https://", "https://www.", 1))
    if url.startswith("https://"):
        urls_to_try.append(url.replace("https://", "http://", 1))

    html = ""
    for try_url in urls_to_try:
        try:
            response = requests.get(try_url, headers=headers, timeout=12)
            response.raise_for_status()
            html = response.text
            break
        except Exception:
            continue

    try:
        if not html:
            return ""

        # Remove scripts, styles, nav, footer boilerplate
        import re as _re
        html = _re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", html, flags=_re.IGNORECASE)
        html = _re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", html, flags=_re.IGNORECASE)
        html = _re.sub(r"<nav[^>]*>[\s\S]*?</nav>", " ", html, flags=_re.IGNORECASE)
        html = _re.sub(r"<footer[^>]*>[\s\S]*?</footer>", " ", html, flags=_re.IGNORECASE)
        html = _re.sub(r"<header[^>]*>[\s\S]*?</header>", " ", html, flags=_re.IGNORECASE)

        # Strip all remaining HTML tags
        text = _re.sub(r"<[^>]+>", " ", html)

        # Clean up whitespace
        text = _re.sub(r"[ \t\n\r]+", " ", text).strip()

        # Keep only meaningful portion (first 3000 chars covers homepage content)
        text = text[:3000]

        # Basic quality check - if too short it probably failed
        if len(text.strip()) < 100:
            return ""

        return text.strip()

    except Exception:
        return ""


def ai_generate_topics(brand_data: dict) -> list:
    """
    Two-stage topic generation:
    Stage 1 - Brand Intelligence: understand the brand deeply using available AI
    Stage 2 - Topic Generation: generate realistic buyer search topics from that intelligence

    Fully model-agnostic - uses _call_ai_for_json which works with any AI model.
    When new models are added to the system, this function benefits automatically.
    """
    business_type = brand_data.get("business_type", "")
    products = ", ".join(brand_data.get("products", []))
    customers = ", ".join(brand_data.get("customers", []))
    key_features = ", ".join(brand_data.get("key_features", []))
    domain = brand_data.get("domain", "")
    brand_name = brand_data["name"]
    country = brand_data.get("country", "")

    # Website content - primary source of truth
    website_text = brand_data.get("_website_text", "")
    if not website_text and domain:
        website_text = fetch_brand_website(domain)

    # Build raw context from all available sources
    raw_context = ""
    if website_text:
        raw_context += "WEBSITE CONTENT:\n" + website_text[:2000] + "\n\n"
    raw_context += (
        "Brand name: " + brand_name + "\n"
        "Business type: " + business_type + "\n"
        "What they offer: " + products + "\n"
        "Who they serve: " + customers + "\n"
        "Key differentiators: " + key_features + "\n"
    )

    # ── STAGE 1: Brand Intelligence ───────────────────────────────────────────
    # Ask the AI to deeply understand the brand before generating topics.
    # This is model-agnostic - works with GPT, Gemini, Claude, Groq, or any future model.
    # The intelligence layer removes the need for keyword rules and mechanical filters.
    # Include buyer insights from Step 1.5 if available
    buyer_insights = brand_data.get("_buyer_insights", [])
    buyer_insights_text = ""
    if buyer_insights:
        buyer_insights_text = "\nDIRECT BUYER INSIGHTS (answers from the brand owner - highest priority):\n"
        for qa in buyer_insights:
            buyer_insights_text += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"

    intelligence_prompt = (
        "You are an AI visibility analyst. Read the brand information below carefully.\n\n"
        + raw_context
        + buyer_insights_text + "\n"
        "Based on this, answer these 4 questions in JSON format:\n"
        "{\n"
        "  \"what_they_offer\": \"One sentence describing exactly what this brand offers in plain language\",\n"
        "  \"exact_buyer\": \"One sentence describing the specific person who needs this, their role and situation\",\n"
        "  \"buyer_search_intent\": \"What would this buyer type into an AI tool when they are ready to find and choose a solution like this one?\",\n"
        "  \"unique_angle\": \"What makes this brand stand out from obvious alternatives in 1 sentence?\"\n"
        "}\n\n"
        "Be specific to THIS brand. Do not use generic category labels.\n"
        "For example, not \'patent software\' but \'patent management software for startup founders who want to file without expensive counsel\'\n"
        "Respond ONLY with valid JSON. No explanation, no markdown."
    )

    brand_intelligence = {}
    try:
        intel_raw = _call_ai_for_json(intelligence_prompt)
        parsed_intel = _parse_json_list(intel_raw)
        if isinstance(parsed_intel, list) and len(parsed_intel) > 0:
            brand_intelligence = parsed_intel[0] if isinstance(parsed_intel[0], dict) else {}
        elif isinstance(intel_raw, str):
            import json as _json
            try:
                brand_intelligence = _json.loads(intel_raw)
            except Exception:
                brand_intelligence = {}
    except Exception:
        brand_intelligence = {}

    # Build enriched context from brand intelligence
    if brand_intelligence:
        enriched_context = (
            "BRAND INTELLIGENCE (AI-generated understanding of this brand):\n"
            + f"What they offer: {brand_intelligence.get('what_they_offer', '')}\n"
            + f"Exact buyer: {brand_intelligence.get('exact_buyer', '')}\n"
            + f"Buyer search intent: {brand_intelligence.get('buyer_search_intent', '')}\n"
            + f"Unique angle: {brand_intelligence.get('unique_angle', '')}\n\n"
            + "RAW BRAND DATA (supplement only):\n"
            + raw_context
        )
    else:
        enriched_context = raw_context

    # Country context
    country_note = f"User location: {country}\n" if country and country.lower() != "global" else ""
    country_topic_note = (
        f"- For 1 topic, add '{country}' to make it location-specific\n"
        if country and country.lower() not in ["global", "united states", ""] else ""
    )

    # ── STAGE 2: Topic Generation from Brand Intelligence ─────────────────────
    # Topics are generated from genuine understanding, not keyword rules.
    # Works for any brand type: software, agency, service, marketplace, etc.
    topic_prompt = (
        "You track brand visibility across AI tools like ChatGPT, Perplexity, Claude, and Gemini.\n\n"
        "Based on the brand intelligence below, generate 7 search topics that the EXACT BUYER "
        "described would type into an AI tool when they are ready to find and choose a solution.\n\n"
        + enriched_context
        + country_note + "\n"
        "RULES:\n"
        "- Topics must be what the EXACT BUYER would search - not generic category searches\n"
        "- Every topic must cause an AI to respond by naming specific brands or providers\n"
        "- Do NOT generate informational, educational, or how-to topics\n"
        "- Do NOT include the brand name '" + brand_name + "' in any topic\n"
        "- Every topic must contain a category word: agency, agencies, firm, service, services, "
        "tool, tools, software, platform, company, companies, provider, providers\n"
        "- Use specific language from the brand intelligence above - not generic category labels\n"
        "- Mix: 3 topics based on unique features/differentiators, 4 based on general buyer intent\n"
        + country_topic_note +
        "\nFor each topic also provide one sentence of buyer intent (why they search this).\n"
        "Respond ONLY with a JSON array of 7 objects: "
        "[{\"topic\": \"...\" , \"intent\": \"...\"}, ...]\n"
        "No markdown, no explanation."
    )
    raw = _call_ai_for_json(topic_prompt)
    parsed_raw = _parse_json_list(raw)
    brand_lower = brand_name.lower()
    topics = []
    topic_intents = {}
    for item in parsed_raw:
        if isinstance(item, dict):
            t = item.get("topic", item.get("name", "")).strip()
            intent = item.get("intent", item.get("reason", "")).strip()
            if t:
                topics.append(t)
                if intent:
                    topic_intents[t] = intent
        elif isinstance(item, str) and item.strip():
            topics.append(item.strip())
    # Store intents in session state for display
    if "brand_data" in st.session_state:
        st.session_state.brand_data["_topic_intents"] = topic_intents

    # ── Niche anchor check ───────────────────────────────────────────────────
    # If the brand's key features contain specific niche terms (patent, legal, medical etc.)
    # then every topic must also contain at least one of those niche terms.
    # This prevents topics like "idea capture tools" when the brand is about "patent idea capture"
    # Fully dynamic - derived from user's own key features and products

    niche_keywords = set()
    all_brand_text = (products + " " + key_features + " " + " ".join(bd.get("products", [])) if "bd" in dir() else products + " " + key_features).lower()

    # Extract meaningful niche words (3+ chars, not common stopwords)
    stopwords = {"and", "the", "for", "with", "our", "not", "from", "that", "this",
                 "are", "was", "has", "have", "will", "been", "your", "their", "its"}
    for word in all_brand_text.split():
        word = word.strip(".,;:()-/")
        if len(word) >= 4 and word not in stopwords:
            niche_keywords.add(word)

    # Keep only words that are truly niche-specific (appear in features but not common English)
    common_english = {"best", "tool", "tools", "software", "platform", "service", "company",
                      "based", "using", "user", "users", "data", "team", "teams", "work",
                      "free", "easy", "simple", "fast", "high", "real", "time", "open",
                      "source", "access", "cost", "price", "need", "help", "make", "build",
                      "track", "manage", "capture", "create", "search", "find", "save"}
    niche_keywords = niche_keywords - common_english

    def _topic_has_niche_context(topic_text, niche_kws):
        """Check if topic contains at least one niche keyword from the brand's space."""
        if not niche_kws:
            return True  # No specific niche detected, skip check
        topic_lower = topic_text.lower()
        return any(kw in topic_lower for kw in niche_kws)

    # Filter out topics that are too vague or likely to produce wrong results
    # These are patterns that make ChatGPT give generic advice instead of brand names
    # Works for any industry - just checking for structural/intent issues
    bad_patterns = [
        "strategies", "tips", "how to", "examples", "guide", "tutorial",
        "best practices", "introduction", "overview", "explained", "meaning",
        "definition", "what is", "benefits of", "advantages of",
    ]
    # Standalone vague nouns that only work when combined with agency/firm/service
    vague_standalone = [
        "providers", "solutions", "platforms", "consultants", "consultancies",
        "writing services", "content writing", "writing service"
    ]

    filtered = []
    for t in topics:
        t = t.strip()
        if not t:
            continue
        if brand_lower in t.lower():
            continue
        t_lower = t.lower()
        # Skip if contains bad educational/informational patterns
        if any(bp in t_lower for bp in bad_patterns):
            continue
        # Skip vague standalone nouns unless paired with agency/firm/service/company
        has_vague = any(v in t_lower for v in vague_standalone)
        has_anchor = any(a in t_lower for a in ["agency", "agencies", "firm", "firms", "service", "services", "company", "companies"])
        if has_vague and not has_anchor:
            continue
        # Skip topics that lost niche context (e.g. "idea capture tools" when brand is patent-focused)
        if not _topic_has_niche_context(t, niche_keywords):
            continue
        filtered.append(t)

    return filtered[:5]


def ai_generate_prompts(topic: str, brand_data: dict) -> list:
    brand_name = brand_data["name"]
    products = ", ".join(brand_data.get("products", []))
    customers = ", ".join(brand_data.get("customers", []))
    business_type = brand_data.get("business_type", "")
    domain = brand_data.get("domain", "")

    # Reuse cached website text if already fetched, else fetch again
    website_text = brand_data.get("_website_text", "")
    if not website_text and domain:
        website_text = fetch_brand_website(domain)

    # Build context block
    if website_text:
        context_block = (
            "Website content (primary source of truth for what this brand does):\n"
            + website_text[:1500]
            + "\n\nForm data: " + products + " | Customers: " + customers
        )
    else:
        context_block = (
            "What they offer: " + products + "\n"
            "Who buys from them: " + customers
        )

    # Build competitor context - rotate which competitors are used
    # so different topics get different comparison pairs
    competitors = brand_data.get("competitors", [])
    competitor_context = ""
    if competitors:
        # Use topic name to deterministically pick different competitors per topic
        # This ensures variety across topics without randomness
        topic_hash = sum(ord(c) for c in topic) % max(len(competitors), 1)
        # Pick 2 competitors starting from the hash offset
        rotated = competitors[topic_hash:] + competitors[:topic_hash]
        comp_for_comparison = rotated[:2] if len(rotated) >= 2 else rotated
        comp_for_alternative = rotated[2:3] if len(rotated) >= 3 else rotated[:1]

        competitor_context = (
            "\nKnown competitors in this space: " + ", ".join(competitors) +
            "\nFor the COMPARISON prompt, compare these two specifically: " +
            " vs ".join(comp_for_comparison) +
            "\nFor the ALTERNATIVE prompt, ask for alternatives to: " +
            (comp_for_alternative[0] if comp_for_alternative else competitors[0])
        )

    # Get country for location-aware prompts
    country = brand_data.get("country", "")

    # Derive solution word dynamically from business type - works for any industry
    bt_lower = business_type.lower()
    if any(w in bt_lower for w in ["agency", "service", "studio", "consultancy"]):
        solution_word = "agency or service provider"
        avoid_line = "- NEVER use the word \'tool\' or \'software\' - this is a service not a tool\n"
    elif any(w in bt_lower for w in ["saas", "software"]):
        solution_word = "tool or software"
        avoid_line = ""
    elif any(w in bt_lower for w in ["ecommerce", "dtc"]):
        solution_word = "platform or store"
        avoid_line = ""
    elif any(w in bt_lower for w in ["marketplace", "aggregator"]):
        solution_word = "marketplace or platform"
        avoid_line = ""
    else:
        solution_word = "product or service"
        avoid_line = ""

    # Rotate persona role per topic
    customers_list = brand_data.get("customers", [])
    persona_role = customers_list[sum(ord(c) for c in topic) % len(customers_list)] if customers_list else "professional"

    # Country as system context not in prompt text
    country_system = ""
    country_suffix = ""
    if country and country.lower() not in ["global", "united states", ""]:
        country_system = f"\nUser location context: {country}. Tailor relevance to this market."
        country_suffix = f" {country}"

    # ── Decide whether comparison is relevant for this topic ────────────────
    if competitors and comp_for_comparison:
        if len(comp_for_comparison) >= 2:
            comparison_line = (f"3. COMPARISON: How does {comp_for_comparison[0]} compare to "
                               f"{comp_for_comparison[1]} for this use case? "
                               f"(Only include if naturally relevant to topic)")
        else:
            comparison_line = (f"3. ALTERNATIVE: What are the best alternatives to "
                               f"{comp_for_comparison[0]} for this topic?")
    else:
        comparison_line = ("3. SCENARIO: A specific real-world situation where someone needs "
                           f"a {solution_word} for this topic and asks for a recommendation")

    prompt = (
        "You generate search prompts for AI visibility tracking.\n\n"
        "Your job: write 5 prompts a REAL PERSON would type into ChatGPT "
        "when looking for a " + solution_word + ".\n\n"
        "Context:\n"
        + context_block
        + competitor_context
        + country_system + "\n\n"
        "HOW A REAL PERSON TYPES:\n"
        "- Short and direct: \'best saas content agency\' not \'What is the most optimal agency??\'\n"
        "- Conversational: \'who should I use for prior art search\'\n"
        "- Natural: no formal structure, no AI-sounding language\n"
        "- No country names in the prompt text\n\n"
        "GENERATE EXACTLY 5 PROMPTS for TOPIC: \"" + topic + "\"\n\n"
        "1. DIRECT SEARCH: 3-5 words exactly as typed in a search bar. No question mark.\n"
        "2. NATURAL QUESTION: Casual question about this topic. Like asking a friend.\n"
        + comparison_line + "\n"
        "4. PERSONA: I am a " + persona_role + " at a [company type]. "
        "I need [specific need from this topic]. Who do you recommend?\n"
        "5. COLLOQUIAL: Ultra short 2-5 words, informal like a quick text. Must end with ?\n\n"
        "RULES:\n"
        "- NEVER mention \"" + brand_name + "\"\n"
        "- Every prompt must make ChatGPT NAME SPECIFIC BRANDS\n"
        "- Every prompt must be about TOPIC: \"" + topic + "\"\n"
        "- Prompt 3 (comparison/scenario): only include if it naturally fits the topic\n"
        "- No year numbers, no country names\n"
        "- Persona MUST end with a clear recommendation ask\n"
        "- BAD: bare noun phrases, knowledge questions, vague statements\n"
        "\nReturn ONLY a JSON array of exactly 5 strings. No markdown."
    )

    import re as _re
    year_pattern = _re.compile(r"\b(20[0-9]{2})\b")
    brand_lower = brand_name.lower()
    topic_lower = topic.lower().strip()

    # Universal structural validator - works for any niche, any industry
    # A prompt is GOOD if it will force ChatGPT to name specific brands
    # A prompt is BAD if ChatGPT will respond with generic advice, tools, or nothing

    # Trigger words that force brand/agency citations in any niche
    CITATION_TRIGGERS = [
        "recommend", "suggest", "which", "what's the best", "who should",
        "compare", "vs", "versus", "alternative", "alternatives",
        "top", "best", "leading", "agency", "agencies", "firm", "firms",
        "company", "companies", "tool", "tools", "software", "platform",
        "service provider", "vendor", "who do you", "any suggestions",
        "what do you", "which one", "who offers"
    ]

    # Words that indicate a statement (not a question) with no brand ask
    STATEMENT_SIGNALS = [
        "how to", "what is", "what are the benefits", "what does",
        "explain", "guide", "tutorial", "tips for", "ways to",
        "understanding", "introduction to", "overview of", "learn",
        "difference between", "why is", "when should", "how does",
        "what makes", "how can i improve", "how do i"
    ]

    def is_citation_producing(p):
        """
        Returns True if this prompt will force ChatGPT to name specific brands.
        Universal logic - works for any industry, any niche.
        """
        pl = p.lower().strip()

        # Rule 1: Must have at least one citation trigger
        has_trigger = any(trigger in pl for trigger in CITATION_TRIGGERS)

        # Rule 2: Must NOT be a pure knowledge/information question
        is_informational = any(signal in pl for signal in STATEMENT_SIGNALS)

        # Rule 3: Persona prompts must end with a recommendation ask
        is_persona = pl.startswith("i ") or pl.startswith("i'm") or pl.startswith("i am")
        if is_persona:
            has_ask = any(ask in pl for ask in ["recommend", "suggest", "what do you", "any suggestions", "who should"])
            return has_ask

        # Rule 4: Short noun phrases without triggers are bad (e.g. "bottom-funnel content providers")
        words = pl.split()
        if len(words) <= 5 and not has_trigger and "?" not in pl:
            return False

        return has_trigger and not is_informational

    def is_bad_prompt(p):
        """Returns True if this prompt should be rejected and regenerated."""
        return not is_citation_producing(p)

    def process_raw(raw_prompts):
        """Clean, filter, and validate a list of raw prompts."""
        cleaned = []
        for p in raw_prompts:
            p = p.strip()
            if not p:
                continue
            # Strip year numbers
            p = year_pattern.sub("", p).strip()
            p = " ".join(p.split())
            # Skip if contains brand name
            if brand_lower in p.lower():
                continue
            # Skip exact duplicates of topic
            if p.lower() == topic_lower:
                continue
            # Skip duplicates already in list
            if p.lower() in [x.lower() for x in cleaned]:
                continue
            # Skip prompts that will produce bad results
            if is_bad_prompt(p):
                continue
            cleaned.append(p)
        return cleaned

    # First attempt
    raw = _call_ai_for_json(prompt)
    prompts = _parse_json_list(raw)
    result = [topic] + process_raw(prompts)

    # Add country variant if needed and not US
    if country_suffix and len(result) < 5:
        location_prompt = topic + country_suffix
        if location_prompt.lower() not in [r.lower() for r in result]:
            result.append(location_prompt)

    # If still under 5, regenerate and fill gaps - up to 2 extra attempts
    attempts = 0
    while len(result) < 5 and attempts < 2:
        attempts += 1
        retry_prompt = (
            prompt +
            f"\n\nIMPORTANT: The previous response was not enough. "
            f"Generate 4 MORE prompts that are DIFFERENT from these already generated:\n" +
            "\n".join(f"- {r}" for r in result)
        )
        try:
            raw2 = _call_ai_for_json(retry_prompt)
            extra = _parse_json_list(raw2)
            new_clean = process_raw(extra)
            # Add only what we still need
            for p in new_clean:
                if p.lower() not in [r.lower() for r in result]:
                    result.append(p)
                if len(result) >= 5:
                    break
        except Exception:
            break

    return result[:5]


# Common English words falsely detected as brand names - filter these out
BRAND_FALSE_POSITIVES = {
    "relevance", "influence", "impact", "reach", "clarity", "signal",
    "momentum", "velocity", "foundation", "element", "essence", "origin",
    "focus", "vision", "mission", "purpose", "strategy", "content",
    "agency", "studio", "digital", "creative", "media", "marketing",
    "growth", "scale", "pipeline", "revenue", "performance", "results",
    "insights", "analytics", "data", "intelligence", "solutions", "services",
    "platform", "engine", "framework", "system", "network", "hub",
    "monday", "notion", "canvas", "spectrum", "horizon", "zenith",
    "apex", "summit", "peak", "core", "pulse", "spark", "flow",
    "bridge", "link", "connect", "sync", "align", "engage",
    "here", "there", "this", "that", "these", "those",
    "first", "second", "third", "next", "last", "new", "old",
    "best", "top", "great", "good", "better", "well", "more",
    "also", "just", "only", "even", "still", "already", "yet",
    "however", "therefore", "because", "although", "unless", "whether",
}

def is_false_positive_brand(name: str) -> bool:
    """Returns True if this name is likely a false positive, not a real brand."""
    name_lower = name.lower().strip()
    # Single common words
    if name_lower in BRAND_FALSE_POSITIVES:
        return True
    # Very short (1-2 chars) or very long (40+ chars)
    if len(name_lower) <= 2 or len(name_lower) > 40:
        return True
    # All lowercase single word that is a common adjective/verb
    if name_lower == name and " " not in name and len(name) < 8:
        return True
    return False


def extract_linked_sites(text: str) -> list:
    url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2})|[/?#&=+@:!,;])+', re.IGNORECASE)
    urls = list(dict.fromkeys(url_pattern.findall(text)))
    sites = []
    for i, url in enumerate(urls[:8], 1):
        domain = re.sub(r'https?://(www\.)?', '', url).split('/')[0]
        sites.append({"rank": i, "domain": domain, "url": url})
    return sites



def tag_input(label: str, session_key: str, placeholder: str = "", help_text: str = "") -> list:
    """Renders a tag/pill input. User types values and clicks Add to create pills."""
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    # Trick to clear the input field after adding a pill:
    # We use a counter as part of the widget key. Incrementing it forces
    # Streamlit to create a fresh empty input widget on the next rerun.
    clear_key = f"{session_key}_clear_counter"
    if clear_key not in st.session_state:
        st.session_state[clear_key] = 0

    tags = st.session_state[session_key]

    st.markdown(f"**{label}**")
    if help_text:
        st.caption(help_text)

    if tags:
        # Show all pills as styled HTML in one flowing line
        pills_html = "".join([
            f'<span style="display:inline-block;background:#1e3a5f;color:#7dd3fc;'
            f'border:1px solid #2563eb;border-radius:999px;padding:4px 14px;'
            f'margin:3px 4px;font-size:13px;font-weight:500;">{t}</span>'
            for t in tags
        ])
        st.markdown(
            f'<div style="padding:6px 0 8px 0;line-height:2.4;">{pills_html}</div>',
            unsafe_allow_html=True
        )

        # Individual remove buttons in a compact row
        remove_idx = None
        num_cols = min(len(tags), 5)
        rm_cols = st.columns(num_cols)
        for idx, tag in enumerate(tags):
            with rm_cols[idx % num_cols]:
                short = (tag[:14] + "…") if len(tag) > 14 else tag
                if st.button(
                    f"✕ {short}",
                    key=f"{session_key}_rm_{idx}",
                    use_container_width=True,
                    help=f"Remove: {tag}"
                ):
                    remove_idx = idx
        if remove_idx is not None:
            st.session_state[session_key].pop(remove_idx)
            st.rerun()

    # Input + Add + Clear row
    col_in, col_add, col_clear = st.columns([5, 1, 1])
    with col_in:
        new_val = st.text_input(
            label,
            key=f"{session_key}_input_{st.session_state[clear_key]}",
            placeholder=placeholder,
            label_visibility="collapsed"
        )
    with col_add:
        st.write("")
        if st.button("+ Add", key=f"{session_key}_add_btn", use_container_width=True):
            items = [v.strip() for v in new_val.split(",") if v.strip()]
            added = False
            for item in items:
                if item and item not in st.session_state[session_key]:
                    st.session_state[session_key].append(item)
                    added = True
            if added:
                st.session_state[clear_key] += 1
                st.rerun()
    with col_clear:
        if tags:
            st.write("")
            if st.button("Clear all", key=f"{session_key}_clear_btn", use_container_width=True):
                st.session_state[session_key] = []
                st.session_state[clear_key] += 1
                st.rerun()

    return st.session_state[session_key]

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(page_title="AI Citation Tracker", page_icon="📡", layout="wide")

st.title("📡 AI Citation Tracker")
st.caption("Track how often your brand appears in AI-generated recommendations")
st.divider()

# =============================================================================
# SESSION STATE INIT
# =============================================================================

for key in ["step", "brand_data", "topics", "selected_topics", "prompts_by_topic",
            "selected_prompts", "all_results", "run_complete"]:
    if key not in st.session_state:
        if key == "step":
            st.session_state[key] = 1
        elif key in ["topics", "selected_topics", "all_results"]:
            st.session_state[key] = []
        elif key in ["prompts_by_topic", "selected_prompts"]:
            st.session_state[key] = {}
        elif key == "run_complete":
            st.session_state[key] = False
        else:
            st.session_state[key] = None

# =============================================================================
# SIDEBAR - API Keys & Tools (always visible)
# =============================================================================

with st.sidebar:
    st.header("API Keys & Tools")

    def read_env() -> dict:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        existing = {}
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        existing[k.strip()] = v.strip()
        return existing

    def write_env(data: dict):
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        with open(env_path, "w") as f:
            for k, v in data.items():
                f.write(f"{k}={v}\n")

    with st.expander("Manage API Keys"):
        key_fields = {
            # "GROQ_API_KEY": "Groq (Free)",  # GROQ DISABLED
            "PERPLEXITY_API_KEY": "Perplexity",
            "GEMINI_API_KEY": "Gemini (Free tier)",
            "OPENAI_API_KEY": "OpenAI / ChatGPT",
            "ANTHROPIC_API_KEY": "Claude / Anthropic",
        }
        updated_keys = {}
        keys_to_remove = []
        for env_key, label in key_fields.items():
            current = os.getenv(env_key, "")
            has_key = bool(current and current.strip() and "paste_your" not in current.lower())
            masked = current[:8] + "..." if has_key else ""
            col_input, col_remove = st.columns([4, 1])
            with col_input:
                new_val = st.text_input(label, value="", placeholder=masked if masked else "Paste key here",
                                        type="password", key=f"key_input_{env_key}")
                if new_val.strip():
                    updated_keys[env_key] = new_val.strip()
            with col_remove:
                st.write("")
                st.write("")
                if has_key:
                    if st.button("✕", key=f"remove_{env_key}"):
                        keys_to_remove.append(env_key)
        if keys_to_remove:
            existing = read_env()
            for k in keys_to_remove:
                existing.pop(k, None)
                os.environ.pop(k, None)
            write_env(existing)
            st.success(f"Removed {len(keys_to_remove)} key(s).")
            st.rerun()
        if st.button("Save Keys", use_container_width=True):
            if updated_keys:
                existing = read_env()
                existing.update(updated_keys)
                write_env(existing)
                for k, v in updated_keys.items():
                    os.environ[k] = v
                st.success(f"Saved {len(updated_keys)} key(s).")
                st.rerun()
            else:
                st.warning("No keys entered.")

    st.divider()
    st.subheader("AI Tools")
    st.caption("Toggle which tools to use for tracking.")
    selected_tools = []
    for tool_name, tool_info in ALL_TOOLS.items():
        key_exists = check_key_exists(tool_info["requires_key"])
        is_free = tool_info["free"]
        if is_free and key_exists:
            label = f"{tool_name} ✅ Free"
            default, disabled = True, False
        elif not is_free and key_exists:
            label = f"{tool_name} 🔑 Key found"
            default, disabled = True, False
        elif not key_exists and not is_free:
            label = f"{tool_name} 🔒 Needs key"
            default, disabled = False, True
        else:
            label = f"{tool_name} ✅ Free"
            default, disabled = True, False
        toggled = st.checkbox(label, value=default, disabled=disabled,
                              help=tool_info["description"], key=f"tool_{tool_name}")
        if toggled and not disabled:
            selected_tools.append(tool_name)

    if not selected_tools:
        st.warning("Select at least one tool")

    st.divider()
    stats = get_usage_stats()
    any_usage = any(stats[t]["calls"] > 0 for t in stats)
    if any_usage:
        st.subheader("Session Usage")
        for tool_name in selected_tools:
            if tool_name in stats and stats[tool_name]["calls"] > 0:
                s = stats[tool_name]
                st.markdown(f"**{tool_name}**: {s['calls']} calls, ~{s['estimated_tokens']} tokens")
        if st.button("Reset Stats"):
            reset_usage_stats()
            st.rerun()

    st.divider()
    if st.button("🔄 Start Over", use_container_width=True):
        for key in ["step", "brand_data", "topics", "selected_topics",
                    "prompts_by_topic", "selected_prompts", "all_results", "run_complete",
                    "step1_products", "step1_customers", "step1_key_features", "step1_competitors"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# =============================================================================
# STEP INDICATOR
# =============================================================================

step_names = ["Brand Details", "Topics", "Prompts", "Run & Results"]
cols = st.columns(4)
_cur_step = st.session_state.step
# Map step "1b" to position 1.5 for the indicator
_step_num = 1.5 if _cur_step == "1b" else (int(_cur_step) if isinstance(_cur_step, int) else 1)
for i, (col, name) in enumerate(zip(cols, step_names), 1):
    with col:
        if i < _step_num:
            st.success(f"✓ {name}")
        elif i == int(_step_num):
            st.info(f"▶ {name}")
        else:
            st.caption(f"{i}.  {name}")

st.divider()

# =============================================================================
# STEP 1: BRAND DETAILS
# =============================================================================

if st.session_state.step == 1:
    st.subheader("Step 1: Brand Details")
    st.caption("Enter your brand information. This is used to generate relevant topics and prompts.")

    # Pre-populate from saved brand_data if user navigated back
    _saved = st.session_state.get("brand_data") or {}

    col1, col2 = st.columns(2)
    with col1:
        brand_name = st.text_input("Brand Name *", value=_saved.get("name", ""), placeholder="e.g. Concurate")
    with col2:
        brand_domain = st.text_input("Brand Domain/URL *", value=_saved.get("domain", ""), placeholder="e.g. concurate.com")

    # Restore pills from saved brand_data when user navigates back
    if "step1_products" not in st.session_state and _saved.get("products"):
        st.session_state["step1_products"] = list(_saved["products"])
    if "step1_customers" not in st.session_state and _saved.get("customers"):
        st.session_state["step1_customers"] = list(_saved["customers"])
    if "step1_key_features" not in st.session_state and _saved.get("key_features"):
        st.session_state["step1_key_features"] = list(_saved["key_features"])

    products_list = tag_input(
        "Your Products and Services",
        session_key="step1_products",
        placeholder="Type a product/service and click Add",
        help_text="List all possible ways customers may describe your products/services"
    )
    customers_list = tag_input(
        "Your Target Customers",
        session_key="step1_customers",
        placeholder="Type a customer persona and click Add",
        help_text="Briefly list your different ideal customer personas"
    )
    key_features_list = tag_input(
        "Key Features / Differentiators",
        session_key="step1_key_features",
        placeholder="Type a feature or benefit and click Add",
        help_text="List important features, benefits and differentiators"
    )

    _btype_options = ["SaaS / Software", "Agency / Service Business", "Ecommerce / DTC Brand",
                     "Marketplace / Aggregator", "Training / Education",
                     "Healthcare / Medical", "Legal / Law Firm", "Financial Services",
                     "Media / Publishing", "Retail / Physical Store", "Other (type your own)"]
    _btype_saved = _saved.get("business_type", "SaaS / Software")
    # If saved value is a custom one not in the list, default to Other
    _btype_idx = _btype_options.index(_btype_saved) if _btype_saved in _btype_options else len(_btype_options) - 1
    _btype_selected = st.selectbox(
        "Business Type",
        options=_btype_options,
        index=_btype_idx,
        help="Select the closest match or choose Other to type your own"
    )
    # Show custom input when Other is selected
    if _btype_selected == "Other (type your own)":
        _custom_default = _btype_saved if _btype_saved not in _btype_options else ""
        business_type = st.text_input(
            "Describe your business type",
            value=_custom_default,
            placeholder="e.g. Training / Education, Non-profit, Government, Hardware, SaaS marketplace...",
            key="step1_custom_business_type"
        )
        if not business_type.strip():
            business_type = "Other"
    else:
        business_type = _btype_selected

    col3, col4 = st.columns(2)
    _country_options = ["United States", "United Kingdom", "Canada", "Australia",
                         "Germany", "India", "Pakistan", "Global"]
    _country_default = _saved.get("country", "United States")
    _country_idx = _country_options.index(_country_default) if _country_default in _country_options else 0
    country = st.selectbox("Country", options=_country_options, index=_country_idx)

    # Restore competitors pills when navigating back
    if "step1_competitors" not in st.session_state and _saved.get("competitors"):
        st.session_state["step1_competitors"] = list(_saved["competitors"])

    competitors_list_input = tag_input(
        "Your Direct Competitors (recommended)",
        session_key="step1_competitors",
        placeholder="Type a competitor name and click Add",
        help_text="Add brands at a similar scale to yours. These will be tracked separately from dominant platforms like Google."
    )

    st.divider()

    if st.button("Next: Generate Topics →", type="primary"):
        if not brand_name.strip() or not brand_domain.strip():
            st.error("Brand Name and Domain are required.")
        elif not products_list:
            st.error("Please add at least one product or service.")
        elif not competitors_list_input:
            st.error("⚠️ Competitors are required. Add at least one direct competitor for accurate comparison prompts and benchmarking.")
        else:
            # Check if brand data has changed from previous session
            # Ensure prev_data is always a dict even if session state has None
            prev_data = st.session_state.get("brand_data") or {}
            new_data = {
                "name": brand_name.strip(),
                "domain": brand_domain.strip(),
                "products": products_list,
                "customers": customers_list,
                "key_features": key_features_list,
                "business_type": business_type,
                "country": country,
                "competitors": competitors_list_input,
            }
            # If anything changed, clear all downstream cached data
            data_changed = (
                prev_data.get("name") != new_data["name"] or
                prev_data.get("domain") != new_data["domain"] or
                prev_data.get("products") != new_data["products"] or
                prev_data.get("customers") != new_data["customers"] or
                prev_data.get("key_features") != new_data["key_features"] or
                prev_data.get("business_type") != new_data["business_type"] or
                prev_data.get("country") != new_data["country"] or
                prev_data.get("competitors") != new_data["competitors"]
            )
            st.session_state.brand_data = new_data
            if data_changed:
                for key in ["topics", "selected_topics", "prompts_by_topic",
                            "selected_prompts", "all_results", "run_complete"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # Force website re-fetch since domain may have changed
                st.session_state.brand_data.pop("_website_text", None)
            st.session_state.step = "1b"
            st.rerun()

# =============================================================================
# STEP 1B: SMART QUESTIONS (AI-generated based on brand)
# =============================================================================

elif st.session_state.step == "1b":
    bd = st.session_state.brand_data

    # ── Generate questions AND auto-fill answers from website ─────────────────
    if "_smart_qa" not in st.session_state:

        # Fetch website if not already done
        website_text = bd.get("_website_text", "")
        if not website_text and bd.get("domain"):
            with st.spinner("🌐 Reading your website..."):
                website_text = fetch_brand_website(bd.get("domain", ""))
                st.session_state.brand_data["_website_text"] = website_text

        q_context = (
            f"Brand: {bd.get('name')}\n"
            f"Domain: {bd.get('domain')}\n"
            f"Business type: {bd.get('business_type')}\n"
            f"What they offer: {', '.join(bd.get('products', []))}\n"
            f"Who they serve: {', '.join(bd.get('customers', []))}\n"
            f"Key differentiators: {', '.join(bd.get('key_features', []))}\n"
            f"Competitors: {', '.join(bd.get('competitors', []))}\n"
        )
        if website_text:
            q_context = f"Website content:\n{website_text[:2000]}\n\n" + q_context

        with st.spinner("🧠 Analyzing your brand and generating insights..."):
            auto_prompt = (
                "You are an AI visibility consultant. Read this brand profile carefully:\n\n"
                + q_context + "\n"
                "Based on your deep understanding of this brand, generate 4 questions AND "
                "answer each one yourself using the brand information above.\n\n"
                "The questions should uncover:\n"
                "1. What buyers search for when ready to choose this brand\n"
                "2. What pain they have before finding this brand\n"
                "3. What makes buyers switch from competitors to this brand\n"
                "4. The most specific search phrase a real buyer would type\n\n"
                "Answer each question as if you are the brand's expert analyst "
                "who has read their website and understands their buyers deeply.\n"
                "Answers must be specific, realistic, and based only on the brand info above.\n\n"
                "Return ONLY a JSON array of 4 objects:\n"
                '[{"question": "...", "answer": "..."},...]\n'
                "No markdown, no explanation."
            )
            try:
                raw_qa = _call_ai_for_json(auto_prompt)
                parsed_qa = _parse_json_list(raw_qa)
                qa_pairs = []
                for item in parsed_qa:
                    if isinstance(item, dict) and item.get("question") and item.get("answer"):
                        qa_pairs.append({
                            "question": item["question"].strip(),
                            "answer": item["answer"].strip()
                        })
                if len(qa_pairs) < 4:
                    qa_pairs = [
                        {"question": f"What would a buyer type into ChatGPT when ready to choose {bd.get('name')}?",
                         "answer": f"They would likely search for {', '.join(bd.get('products', ['a solution like this'])[:2])}"},
                        {"question": "What pain does the buyer have before finding this brand?",
                         "answer": f"They struggle with managing {', '.join(bd.get('key_features', ['their needs'])[:1])} without the right tool."},
                        {"question": "Which competitor do buyers settle for if they can't find this brand?",
                         "answer": f"They often end up with {bd.get('competitors', ['a larger competitor'])[0] if bd.get('competitors') else 'a larger competitor'}."},
                        {"question": "Describe the ideal buyer in one sentence.",
                         "answer": f"A {', '.join(bd.get('customers', ['professional'])[:1])[0:50]} who needs {', '.join(bd.get('products', ['this solution'])[:1])}."}
                    ][:4 - len(qa_pairs)]
            except Exception:
                qa_pairs = [
                    {"question": f"What would a buyer type into ChatGPT when ready to choose {bd.get('name')}?", "answer": ""},
                    {"question": "What pain does the buyer have before finding this brand?", "answer": ""},
                    {"question": "Which competitor do buyers usually settle for instead?", "answer": ""},
                    {"question": "Describe the ideal buyer in one sentence.", "answer": ""}
                ]
            st.session_state["_smart_qa"] = qa_pairs

    qa_pairs = st.session_state.get("_smart_qa", [])

    # ── Header card ───────────────────────────────────────────────────────────
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3a5f 0%, #0f2340 100%);
                    border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
                    border: 1px solid #2563eb33;'>
            <div style='font-size: 13px; color: #7dd3fc; font-weight: 600;
                        letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;'>
                Step 1.5 of 4 · Brand Understanding
            </div>
            <div style='font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 10px;'>
                We read your website and filled these in 🤖
            </div>
            <div style='font-size: 15px; color: #94a3b8; line-height: 1.6;'>
                Based on your website and brand details, we answered these questions about your buyers.
                <br>
                <span style='color: #7dd3fc; font-weight: 500;'>Review and edit</span> 
                anything that looks wrong — or just click 
                <span style='color: #7dd3fc; font-weight: 500;'>Looks Good →</span> to continue.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Q&A cards with editable answers ──────────────────────────────────────
    icons = ["🔍", "💡", "🏁", "👤"]
    edited_answers = {}

    for idx, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        auto_answer = qa.get("answer", "")
        icon = icons[idx] if idx < len(icons) else "💬"

        st.markdown(f"""
            <div style='background: #0f1f35; border: 1px solid #1e3a5f;
                        border-left: 3px solid #2563eb; border-radius: 10px;
                        padding: 14px 20px; margin-bottom: 4px;'>
                <span style='color: #7dd3fc; font-size: 16px;'>{icon}</span>
                <span style='color: #e2e8f0; font-size: 14px; font-weight: 600;
                             margin-left: 10px;'>{question}</span>
            </div>
        """, unsafe_allow_html=True)

        edited_answers[f"q{idx}"] = st.text_area(
            label=f"q{idx}",
            value=auto_answer,
            key=f"smart_qa_{idx}",
            label_visibility="collapsed",
            height=80,
            help="Auto-filled from your website. Edit if needed."
        )

    # ── Action buttons ────────────────────────────────────────────────────────
    st.write("")
    col_back, col_skip, col_next = st.columns([1, 1, 2])

    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            for k in ["_smart_qa", "_smart_questions"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    with col_skip:
        if st.button("Skip this step", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

    with col_next:
        if st.button("✅ Looks Good → Generate Topics", type="primary", use_container_width=True):
            # Save edited Q&A into brand_data
            final_qa = []
            for idx, qa in enumerate(qa_pairs):
                answer = edited_answers.get(f"q{idx}", "").strip()
                if answer:
                    final_qa.append({"question": qa.get("question", ""), "answer": answer})
            st.session_state.brand_data["_buyer_insights"] = final_qa
            for key in ["topics", "selected_topics", "prompts_by_topic",
                        "selected_prompts", "all_results", "run_complete"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 2
            st.rerun()

# =============================================================================
# STEP 2: TOPICS
# =============================================================================

elif st.session_state.step == 2:
    bd = st.session_state.brand_data
    st.subheader("Step 2: Topics")
    st.caption("These are the topics for which we'll measure your brand visibility. Choose which to keep or add your own.")

    # Auto-generate if not done yet
    if not st.session_state.topics:

        # ── Step 1: Show iframe preview + fetch website in parallel ──────────
        domain = bd.get("domain", "")

        # Build the full URL for iframe display
        iframe_url = domain if domain.startswith("http") else f"https://{domain}"

        # Show the website in an iframe so the user can see we are visiting it
        st.markdown("#### 🌐 Visiting your website...")
        st.caption(f"We are reading: **{iframe_url}**")

        # Embed the actual website in an iframe
        st.components.v1.iframe(
            src=iframe_url,
            height=320,
            scrolling=True
        )

        # Now fetch the content in background
        with st.spinner("📖 Reading and extracting content from your website..."):
            if "_website_text" not in st.session_state.brand_data:
                website_text = fetch_brand_website(domain)
                st.session_state.brand_data["_website_text"] = website_text
            else:
                website_text = st.session_state.brand_data.get("_website_text", "")

        # ── Step 2: Show what was understood from the website ─────────────────
        if website_text:
            # Use AI to summarize what was understood about the brand from the website
            understanding_prompt = (
                "You just read the homepage of a brand. Based on the content below, "
                "write a SHORT 3-4 sentence summary explaining:\n"
                "1. What this brand does\n"
                "2. Who their customers are\n"
                "3. What makes them different\n\n"
                "Be specific and factual. Only use what is actually on the page.\n"
                "Do not invent anything. Write in plain English.\n\n"
                "Website content:\n" + website_text[:3000]
            )
            with st.spinner("🧠 Reading and understanding your website..."):
                try:
                    brand_understanding = _call_ai_for_json(
                        understanding_prompt + "\n\nRespond with plain text only, no JSON, no bullet points."
                    )
                except Exception:
                    brand_understanding = ""

            st.success(f"✅ Website fetched successfully from {bd.get('domain', '')}")
            with st.expander("📋 Brand Intelligence Summary", expanded=True):

                # ── Section 1: From Website ───────────────────────────────
                st.markdown("### 🌐 From Your Website")
                st.caption(f"Fetched from: {bd.get('domain', '')}")
                if brand_understanding:
                    st.info(brand_understanding)
                else:
                    st.info(website_text[:400] + "...")

                st.divider()

                # ── Section 2: From Form ──────────────────────────────────
                st.markdown("### 📝 From Your Form")
                st.caption("This is what you entered manually in Step 1.")

                form_cols = st.columns(2)
                with form_cols[0]:
                    if bd.get("products"):
                        st.markdown("**📦 Products / Services**")
                        for p in bd["products"]:
                            st.markdown(f"- {p}")
                    if bd.get("customers"):
                        st.markdown("**👥 Target Customers**")
                        for c in bd["customers"]:
                            st.markdown(f"- {c}")
                with form_cols[1]:
                    if bd.get("key_features"):
                        st.markdown("**⭐ Key Differentiators**")
                        for f in bd["key_features"]:
                            st.markdown(f"- {f}")
                    if bd.get("competitors"):
                        st.markdown("**🏁 Direct Competitors**")
                        for comp in bd["competitors"]:
                            st.markdown(f"- {comp}")
                    st.markdown(f"**🌍 Country:** {bd.get('country', 'United States')}")
                    st.markdown(f"**🏢 Business Type:** {bd.get('business_type', '')}")

                st.divider()
                st.success("✅ Topics will be generated using BOTH your website content and your form data.")
        else:
            st.warning(
                "⚠️ Could not read your website automatically. "
                "Topics will be generated from your form data only. "
                "Check that your domain is correct and publicly accessible."
            )
            with st.expander("📋 Form Data Being Used", expanded=True):
                st.markdown("### 📝 From Your Form")
                st.caption("Website could not be fetched — using form data only.")
                form_cols2 = st.columns(2)
                with form_cols2[0]:
                    if bd.get("products"):
                        st.markdown("**📦 Products / Services**")
                        for p in bd["products"]:
                            st.markdown(f"- {p}")
                    if bd.get("customers"):
                        st.markdown("**👥 Target Customers**")
                        for c in bd["customers"]:
                            st.markdown(f"- {c}")
                with form_cols2[1]:
                    if bd.get("key_features"):
                        st.markdown("**⭐ Key Differentiators**")
                        for f in bd["key_features"]:
                            st.markdown(f"- {f}")
                    if bd.get("competitors"):
                        st.markdown("**🏁 Direct Competitors**")
                        for comp in bd["competitors"]:
                            st.markdown(f"- {comp}")
                    st.markdown(f"**🌍 Country:** {bd.get('country', 'United States')}")

        # ── Step 3: Generate topics ───────────────────────────────────────────
        with st.spinner("🔍 Generating topics based on your brand profile..."):
            try:
                generated = ai_generate_topics(st.session_state.brand_data)
                st.session_state.topics = generated
                st.session_state.selected_topics = list(generated)
            except Exception as e:
                st.error(f"Topic generation failed: {format_error_message(str(e))}")
                st.info("You can add topics manually below.")
                st.session_state.topics = []
                st.session_state.selected_topics = []

    if st.session_state.topics:
        selected_count = len(st.session_state.selected_topics)
        st.success(f"✓ {selected_count} topic{'s' if selected_count != 1 else ''} selected")

        topic_intents = st.session_state.brand_data.get("_topic_intents", {})
        st.write("**AI Generated Topics** (uncheck to remove, click ✏️ to edit):")
        for t_idx, topic in enumerate(st.session_state.topics):
            checked = topic in st.session_state.selected_topics
            col_check, col_edit = st.columns([8, 1])
            with col_check:
                new_checked = st.checkbox(f"✦ {topic}", value=checked, key=f"topic_check_{t_idx}")
                if new_checked and topic not in st.session_state.selected_topics:
                    st.session_state.selected_topics.append(topic)
                elif not new_checked and topic in st.session_state.selected_topics:
                    st.session_state.selected_topics.remove(topic)
                if topic in topic_intents:
                    st.caption(f"💡 {topic_intents[topic]}")
            with col_edit:
                edit_key = f"topic_edit_mode_{t_idx}"
                if st.button("✏️", key=f"topic_edit_btn_{t_idx}", help="Edit this topic"):
                    st.session_state[edit_key] = True

            # Show inline editor if edit mode is active
            if st.session_state.get(f"topic_edit_mode_{t_idx}"):
                new_val = st.text_input(
                    "Edit topic",
                    value=topic,
                    key=f"topic_edit_input_{t_idx}",
                    label_visibility="collapsed"
                )
                save_col, cancel_col = st.columns([1, 1])
                with save_col:
                    if st.button("✅ Save", key=f"topic_save_{t_idx}", use_container_width=True):
                        new_val = new_val.strip()
                        if new_val and new_val != topic:
                            # Update in topics list
                            was_selected = topic in st.session_state.selected_topics
                            st.session_state.topics[t_idx] = new_val
                            if was_selected:
                                st.session_state.selected_topics.remove(topic)
                                st.session_state.selected_topics.append(new_val)
                        st.session_state[f"topic_edit_mode_{t_idx}"] = False
                        st.rerun()
                with cancel_col:
                    if st.button("✖ Cancel", key=f"topic_cancel_{t_idx}", use_container_width=True):
                        st.session_state[f"topic_edit_mode_{t_idx}"] = False
                        st.rerun()

    st.divider()
    col_add1, col_add2 = st.columns([4, 1])
    with col_add1:
        new_topic = st.text_input("Add a custom topic", placeholder="e.g. B2B SaaS content agency",
                                  key="custom_topic_input")
    with col_add2:
        st.write("")
        st.write("")
        if st.button("+ Add"):
            if new_topic.strip() and new_topic.strip() not in st.session_state.topics:
                st.session_state.topics.append(new_topic.strip())
                st.session_state.selected_topics.append(new_topic.strip())
                st.rerun()

    st.divider()
    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← Back"):
            # Clear ALL cached data so everything regenerates fresh from Step 1 edits
            st.session_state.step = 1
            for key in ["topics", "selected_topics", "prompts_by_topic",
                        "selected_prompts", "all_results", "run_complete"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear cached website text so it re-fetches with any domain changes
            if "brand_data" in st.session_state:
                st.session_state.brand_data.pop("_website_text", None)
            st.rerun()
    with col_next:
        if st.button("Next: Generate Prompts →", type="primary"):
            if not st.session_state.selected_topics:
                st.error("Please select at least one topic.")
            else:
                st.session_state.step = 3
                st.rerun()

# =============================================================================
# STEP 3: PROMPTS
# =============================================================================

elif st.session_state.step == 3:
    bd = st.session_state.brand_data
    st.subheader("Step 3: Review Prompts")
    st.caption("These prompts will be sent to each AI model to check if your brand is mentioned.")

    # Generate prompts for each selected topic FIRST before showing count
    for topic in st.session_state.selected_topics:
        if topic not in st.session_state.prompts_by_topic:
            with st.spinner(f"Generating prompts for: {topic}..."):
                try:
                    prompts = ai_generate_prompts(topic, st.session_state.brand_data)
                    st.session_state.prompts_by_topic[topic] = prompts
                    st.session_state.selected_prompts[topic] = list(prompts)
                except Exception as e:
                    st.warning(f"Could not generate prompts for '{topic}': {format_error_message(str(e))}")
                    st.session_state.prompts_by_topic[topic] = [topic]
                    st.session_state.selected_prompts[topic] = [topic]

    # Count AFTER generation so numbers are accurate
    total_topics = len(st.session_state.selected_topics)
    all_generated = all(
        t in st.session_state.prompts_by_topic
        for t in st.session_state.selected_topics
    )
    total_prompts = sum(
        len(st.session_state.selected_prompts.get(t, []))
        for t in st.session_state.selected_topics
    )
    if all_generated and total_topics > 0:
        avg_per_topic = round(total_prompts / total_topics, 1)
        st.info(f"**{total_topics} topics × {avg_per_topic} avg prompts = {total_prompts} total prompts** per AI model")
    else:
        st.info(f"**{total_topics} topics** — generating prompts...")

    # Display prompts per topic in accordion style
    for t_idx, topic in enumerate(st.session_state.selected_topics):
        prompts = st.session_state.prompts_by_topic.get(topic, [])
        selected = st.session_state.selected_prompts.get(topic, [])

        with st.expander(f"**{topic}** ({len(selected)} prompts selected)", expanded=True):
            for p_idx, prompt in enumerate(prompts):
                checked = prompt in selected
                col_check, col_edit = st.columns([8, 1])
                with col_check:
                    cb_key = f"prompt_t{t_idx}_p{p_idx}"
                    new_checked = st.checkbox(prompt, value=checked, key=cb_key)
                    if new_checked and prompt not in st.session_state.selected_prompts.get(topic, []):
                        st.session_state.selected_prompts.setdefault(topic, []).append(prompt)
                    elif not new_checked and prompt in st.session_state.selected_prompts.get(topic, []):
                        st.session_state.selected_prompts[topic].remove(prompt)
                with col_edit:
                    pedit_key = f"prompt_edit_mode_t{t_idx}_p{p_idx}"
                    if st.button("✏️", key=f"prompt_edit_btn_t{t_idx}_p{p_idx}", help="Edit this prompt"):
                        st.session_state[pedit_key] = True

                # Inline editor for this prompt
                if st.session_state.get(f"prompt_edit_mode_t{t_idx}_p{p_idx}"):
                    new_val = st.text_input(
                        "Edit prompt",
                        value=prompt,
                        key=f"prompt_edit_input_t{t_idx}_p{p_idx}",
                        label_visibility="collapsed"
                    )
                    psave_col, pcancel_col = st.columns([1, 1])
                    with psave_col:
                        if st.button("✅ Save", key=f"prompt_save_t{t_idx}_p{p_idx}", use_container_width=True):
                            new_val = new_val.strip()
                            if new_val and new_val != prompt:
                                was_selected = prompt in st.session_state.selected_prompts.get(topic, [])
                                st.session_state.prompts_by_topic[topic][p_idx] = new_val
                                if was_selected:
                                    if prompt in st.session_state.selected_prompts.get(topic, []):
                                        st.session_state.selected_prompts[topic].remove(prompt)
                                    st.session_state.selected_prompts.setdefault(topic, []).append(new_val)
                            st.session_state[f"prompt_edit_mode_t{t_idx}_p{p_idx}"] = False
                            st.rerun()
                    with pcancel_col:
                        if st.button("✖ Cancel", key=f"prompt_cancel_t{t_idx}_p{p_idx}", use_container_width=True):
                            st.session_state[f"prompt_edit_mode_t{t_idx}_p{p_idx}"] = False
                            st.rerun()

            # Add custom prompt
            custom_key = f"custom_prompt_t{t_idx}"
            custom_prompt = st.text_input("+ Add prompt", key=custom_key,
                                          placeholder="Type a custom prompt and press Enter")
            if custom_prompt.strip() and custom_prompt.strip() not in prompts:
                if st.button("Add", key=f"add_prompt_btn_t{t_idx}"):
                    st.session_state.prompts_by_topic[topic].append(custom_prompt.strip())
                    st.session_state.selected_prompts.setdefault(topic, []).append(custom_prompt.strip())
                    st.rerun()

    st.divider()
    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← Back"):
            # Clear prompts and results so they regenerate if topics changed
            st.session_state.step = 2
            for key in ["prompts_by_topic", "selected_prompts", "all_results", "run_complete"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col_next:
        total_selected = sum(len(v) for v in st.session_state.selected_prompts.values())
        if st.button(f"Start Tracking ({total_selected} prompts x {len(selected_tools)} models) →",
                     type="primary", disabled=len(selected_tools) == 0):
            if total_selected == 0:
                st.error("Please select at least one prompt.")
            else:
                st.session_state.step = 4
                st.rerun()

# =============================================================================
# STEP 4: RUN & RESULTS
# =============================================================================

elif st.session_state.step == 4:
    bd = st.session_state.brand_data
    brand_name = bd["name"]
    brand_domain = bd["domain"]
    competitors_list = bd.get("competitors", [])

    # Determine business type string
    btype = bd.get("business_type", "")
    if "Agency" in btype or "Service" in btype:
        detected_business_type = "service"
    elif "SaaS" in btype or "Software" in btype:
        detected_business_type = "software"
    else:
        detected_business_type = "other"

    # Run tracking if not done
    if not st.session_state.run_complete:
        st.subheader("Running Tracking...")

        # Build flat list of prompts with topic metadata
        all_prompts = []
        for topic, prompts in st.session_state.selected_prompts.items():
            for prompt in prompts:
                all_prompts.append({"query": prompt, "topic": topic, "category": "awareness"})

        progress_bar = st.progress(0)
        status_text = st.empty()
        live_feed = st.empty()
        call_count = 0
        all_results = []
        exhausted_tools = set()
        live_log = []
        total_calls = len(all_prompts) * len(selected_tools)

        for i, q in enumerate(all_prompts):
            status_text.text(f"Running {i+1}/{len(all_prompts)}: {q['query'][:70]}...")
            active_tools = [t for t in selected_tools if t not in exhausted_tools]
            if not active_tools:
                st.warning("All tools rate-limited. Stopping early.")
                break

            tool_responses = run_selected_tools(q["query"], active_tools)

            for tool_name, response_text in tool_responses.items():
                if response_text.startswith("ERROR"):
                    clean_error = response_text.split("ERROR:", 1)[-1].strip()
                    if any(x in clean_error.lower() for x in ["429", "rate_limit", "resource_exhausted", "quota"]):
                        exhausted_tools.add(tool_name)
                    call_count += 1
                    progress_bar.progress(min(call_count / total_calls, 1.0))
                    continue

                brand_data_detected = detect_brands(
                    response_text,
                    brand_name,
                    icp_text=f"{', '.join(bd.get('products', []))} {', '.join(bd.get('customers', []))}",
                    business_type=detected_business_type,
                    user_competitors=competitors_list
                )
                # Filter false positive brand names from results
                brand_data_detected["all_brands"] = [
                    b for b in brand_data_detected.get("all_brands", [])
                    if not is_false_positive_brand(b)
                ]

                linked_sites = extract_linked_sites(response_text)

                all_results.append({
                    "query": q["query"],
                    "topic": q["topic"],
                    "category": q["category"],
                    "query_group": "C",
                    "tool": tool_name,
                    "response": response_text,
                    "brands_detected": brand_data_detected,
                    "linked_sites": linked_sites,
                })

                # Update live feed
                mentioned = brand_data_detected.get("target_mentioned", False)
                live_log.append({
                    "prompt": q["query"][:70] + ("..." if len(q["query"]) > 70 else ""),
                    "model": tool_name,
                    "detected": mentioned
                })
                feed_lines = []
                for log in live_log[-8:]:  # Show last 8 entries
                    icon = "✅" if log["detected"] else "⚪"
                    feed_lines.append(f"{icon} **{log['model']}** — {log['prompt']}")
                live_feed.markdown("**Live Results:**\n" + "\n".join(feed_lines))

                call_count += 1
                progress_bar.progress(min(call_count / total_calls, 1.0))

        status_text.text("Done.")
        st.session_state.all_results = all_results
        st.session_state.run_complete = True

        # Play a notification sound when results are ready
        # Uses Web Audio API via JavaScript - no external files needed
        st.components.v1.html("""
        <script>
        (function() {
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                function beep(freq, start, duration, vol) {
                    var o = ctx.createOscillator();
                    var g = ctx.createGain();
                    o.connect(g);
                    g.connect(ctx.destination);
                    o.frequency.value = freq;
                    o.type = 'sine';
                    g.gain.setValueAtTime(0, ctx.currentTime + start);
                    g.gain.linearRampToValueAtTime(vol, ctx.currentTime + start + 0.01);
                    g.gain.linearRampToValueAtTime(0, ctx.currentTime + start + duration);
                    o.start(ctx.currentTime + start);
                    o.stop(ctx.currentTime + start + duration + 0.1);
                }
                // Two-tone pleasant notification
                beep(600, 0,    0.15, 0.3);
                beep(900, 0.18, 0.25, 0.3);
            } catch(e) {}
        })();
        </script>
        """, height=0)

        st.rerun()

    # ==========================================================================
    # DISPLAY RESULTS
    # ==========================================================================

    else:
        all_results = st.session_state.all_results
        st.subheader(f"Results: {brand_name}")
        st.caption(f"Domain: {brand_domain} | Country: {bd.get('country', 'US')} | Business Type: {bd.get('business_type', '')}")

        if not all_results:
            st.error("No results returned. Check your API keys and try again.")
        else:
            scores = calculate_citation_share(all_results, brand_name)

            # ── Overall metrics ──────────────────────────────────────────
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Citation Share", f"{scores['overall_citation_share']}%")
            c2.metric("Total Mentions", f"{scores['total_mentions']} / {scores['total_queries_run']}")
            c3.metric("Position Score", f"{scores['position_score_pct']}%")
            c4.metric("Topics Tracked", len(st.session_state.selected_topics))

            # ── Competitor benchmark ──────────────────────────────────────
            st.divider()
            st.subheader("📊 How You Compare Against Competitors")
            st.caption("Your brand vs competitors mentioned by AI across all prompts.")

            real_comps = scores.get("real_competitors", [])
            total_q_b = scores["total_queries_run"]
            your_pct = scores["overall_citation_share"]
            user_comps = bd.get("competitors", [])
            detected_map = {b.lower(): c for b, c in real_comps}

            bench_rows = [{"Brand": f"🎯 {brand_name} (You)", "AI Mentions": scores["total_mentions"],
                           "Visibility": f"{your_pct}%", "Note": "Your Brand"}]
            for comp in user_comps:
                mentions = detected_map.get(comp.lower(), 0)
                pct = round((mentions / total_q_b) * 100) if total_q_b > 0 else 0
                bench_rows.append({"Brand": comp, "AI Mentions": mentions,
                                   "Visibility": f"{pct}%", "Note": "Your Competitor"})
            for bc, cnt in real_comps[:8]:
                if bc.lower() not in [c.lower() for c in user_comps] and bc.lower() != brand_name.lower():
                    pct = round((cnt / total_q_b) * 100) if total_q_b > 0 else 0
                    bench_rows.append({"Brand": bc, "AI Mentions": cnt,
                                       "Visibility": f"{pct}%", "Note": "Also Detected"})
            st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)

            # ── Why not appearing insight ─────────────────────────────────
            if your_pct == 0 and real_comps:
                top_name, top_cnt = real_comps[0]
                top_pct = round((top_cnt / total_q_b) * 100)
                st.warning(
                    f"**{brand_name} has 0% AI visibility.** "
                    f"The most mentioned brand was **{top_name}** at {top_pct}% of responses. "
                    f"To improve visibility, {brand_name} needs more published content, case studies, "
                    f"and citations online that AI models can learn from."
                )
            elif your_pct > 0 and real_comps:
                top_name, top_cnt = real_comps[0]
                top_pct = round((top_cnt / total_q_b) * 100)
                if top_pct > your_pct:
                    st.info(
                        f"**{brand_name}** has {your_pct}% visibility. "
                        f"**{top_name}** leads at {top_pct}%. "
                        f"Gap to close: {top_pct - your_pct}%."
                    )

            # ── Charts ───────────────────────────────────────────────────
            st.subheader("Visibility Overview")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                tool_chart_data = {
                    tool: data["share_pct"]
                    for tool, data in scores["citation_share_by_tool"].items()
                }
                if tool_chart_data:
                    st.markdown("**Visibility % by AI Model**")
                    chart_df = pd.DataFrame({
                        "AI Model": list(tool_chart_data.keys()),
                        "Visibility %": list(tool_chart_data.values())
                    })
                    st.bar_chart(chart_df.set_index("AI Model"), use_container_width=True, color="#2563eb")

            with chart_col2:
                topic_scores_chart = calculate_citation_share_by_topic(all_results, brand_name)
                if topic_scores_chart:
                    st.markdown("**Visibility % by Topic**")
                    t_df = pd.DataFrame({
                        "Topic": [t[:30] + "..." if len(t) > 30 else t for t in topic_scores_chart.keys()],
                        "Visibility %": [v["share_pct"] for v in topic_scores_chart.values()]
                    })
                    st.bar_chart(t_df.set_index("Topic"), use_container_width=True, color="#16a34a")

            # ── Recommendation Share + Context breakdown ─────────────────
            ctx = scores["context_breakdown"]
            total_runs = scores["total_queries_run"] or 1
            recommended = ctx.get("recommended", 0)
            mentioned = ctx.get("mentioned", 0)
            warned = ctx.get("warned_against", 0)
            not_mentioned = ctx.get("not_mentioned", 0)
            rec_share = round((recommended / total_runs) * 100)
            mention_share = round((mentioned / total_runs) * 100)

            st.subheader("📊 Brand Visibility Breakdown")
            st.caption(
                "**Recommendation Share** = actively named as a top pick. "
                "**Mention Share** = casually listed. "
                "Recommendation Share is the most valuable metric."
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🏆 Recommendation Share", f"{rec_share}%",
                      f"{recommended}/{total_runs} prompts",
                      help="Brand actively recommended as top pick")
            m2.metric("🔵 Mention Share", f"{mention_share}%",
                      f"{mentioned}/{total_runs} prompts",
                      help="Brand mentioned but not as top pick")
            m3.metric("🔴 Warned Against", warned,
                      f"{round((warned/total_runs)*100)}%",
                      help="Brand mentioned with caution")
            m4.metric("⚪ Not Mentioned", not_mentioned,
                      f"{round((not_mentioned/total_runs)*100)}%",
                      help="Brand did not appear")
            if rec_share > 0:
                st.success(f"✅ Your brand is actively recommended in {rec_share}% of AI responses.")
            elif mention_share > 0:
                st.info(f"ℹ️ Your brand appears in {mention_share}% of responses but is not the top recommendation.")
            else:
                st.warning("⚠️ Your brand has 0% AI visibility across all prompts.")

            # ── Visibility per model table ────────────────────────────────
            st.subheader("Visibility % by AI Model")
            tool_rows = []
            for tool, data in scores["citation_share_by_tool"].items():
                tool_rows.append({
                    "AI Model": tool,
                    "Visibility %": f"{data['share_pct']}%",
                    "Brand Mentions": f"{data['mentions']} / {data['total_queries']}",
                })
            st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)

            # ── Per topic breakdown ───────────────────────────────────────
            st.subheader("Visibility % by Topic")
            topic_scores = calculate_citation_share_by_topic(all_results, brand_name)

            for topic, t_data in topic_scores.items():
                pct = t_data['share_pct']
                with st.expander(f"**{topic}** — {pct}% visibility", expanded=True):

                    # Per model breakdown for this topic
                    topic_results = [r for r in all_results if r.get("topic") == topic]
                    models_in_topic = list(set(r["tool"] for r in topic_results))

                    model_cols = st.columns(len(models_in_topic)) if models_in_topic else []
                    for col, model in zip(model_cols, models_in_topic):
                        model_results = [r for r in topic_results if r["tool"] == model]
                        mentions = sum(1 for r in model_results if r["brands_detected"]["target_mentioned"])
                        total = len(model_results)
                        col.metric(model, f"{round((mentions/total)*100) if total else 0}%",
                                   f"{mentions}/{total} prompts")

                    st.write("**Prompts:**")

                    # Per prompt per model table
                    prompt_texts = list(set(r["query"] for r in topic_results))
                    for prompt_text in prompt_texts:
                        prompt_results = [r for r in topic_results if r["query"] == prompt_text]

                        # Build badge row
                        badge_cols = st.columns([3] + [1] * len(models_in_topic))
                        with badge_cols[0]:
                            st.caption(f"+ {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
                        for col, model in zip(badge_cols[1:], models_in_topic):
                            model_result = next((r for r in prompt_results if r["tool"] == model), None)
                            if model_result and model_result["brands_detected"]["target_mentioned"]:
                                col.success("Brand")
                            elif model_result:
                                col.caption("—")
                            else:
                                col.caption("—")

                        # Drill-down expander for each prompt
                        with st.expander(f"View details: {prompt_text[:60]}..."):
                            for r in prompt_results:
                                st.markdown(f"**{r['tool']}**")
                                mentioned = r["brands_detected"]["target_mentioned"]
                                context = r["brands_detected"]["target_context"]
                                position = r["brands_detected"]["target_position"]

                                if mentioned:
                                    st.success(f"Brand mentioned | Context: {context} | Position: #{position}")
                                else:
                                    st.warning("Brand not mentioned")

                                # Linked sites
                                linked = r.get("linked_sites", [])
                                if linked:
                                    st.caption("**Linked Sites:**")
                                    link_rows = [{"#": s["rank"], "Domain": s["domain"], "URL": s["url"]} for s in linked]
                                    st.dataframe(pd.DataFrame(link_rows), use_container_width=True, hide_index=True)

                                # AI Response - highlighted brand mentions
                                response = r.get("response", "")
                                exp_label = "✅ View AI Response (brand found)" if mentioned else "View AI Response"
                                with st.expander(exp_label):
                                    if response:
                                        # Highlight all variants of brand name
                                        import re as _re
                                        highlighted = response
                                        # Try exact name and spaced variants
                                        variants_to_highlight = [brand_name]
                                        if " " not in brand_name:
                                            spaced = _re.sub(r"([a-z])([A-Z])", r"\1 \2", brand_name)
                                            if spaced != brand_name:
                                                variants_to_highlight.append(spaced)
                                        for v in variants_to_highlight:
                                            highlighted = _re.sub(
                                                r"\b" + _re.escape(v) + r"\b",
                                                f"**🟡 {v}**",
                                                highlighted,
                                                flags=_re.IGNORECASE
                                            )
                                        st.markdown(highlighted[:3000])
                                        if mentioned:
                                            st.success(f"✅ {brand_name} appears in this response")
                                    else:
                                        st.caption("No response recorded.")

            # ── Competitor ranking - 3 dynamic categories ───────────────
            st.subheader("Brands Detected in AI Responses")
            st.caption("Brands are automatically classified at runtime by AI — no hardcoding, works for any industry.")

            real_competitors = scores.get("real_competitors", [])
            dominant_platforms = scores.get("dominant_platforms", [])
            government_bodies = scores.get("government_bodies", [])
            total_q = scores["total_queries_run"]

            tab1, tab2, tab3 = st.tabs([
                f"Direct Competitors ({len(real_competitors)})",
                f"Dominant Platforms ({len(dominant_platforms)})",
                f"Government / Official Bodies ({len(government_bodies)})"
            ])

            with tab1:
                st.caption("Commercial tools and services at a similar scale to yours — your real competition.")

                # Always show user-entered competitors even if not detected
                # This gives them a 0% visibility baseline which is useful data
                user_competitors = bd.get("competitors", [])
                detected_names = [b.lower() for b, _ in real_competitors]

                # Add any user competitors that were not detected with 0 mentions
                zero_mention_competitors = [
                    (comp, 0) for comp in user_competitors
                    if comp.strip() and comp.strip().lower() not in detected_names
                ]

                all_competitors_display = list(real_competitors[:20]) + zero_mention_competitors

                if all_competitors_display:
                    comp_df = pd.DataFrame(all_competitors_display, columns=["Brand", "Mentions"])
                    comp_df["Appearance Rate"] = comp_df["Mentions"].apply(
                        lambda x: f"{round((x / total_q) * 100)}%" if total_q > 0 else "0%"
                    )
                    comp_df["Status"] = comp_df["Mentions"].apply(
                        lambda x: "✅ Detected" if x > 0 else "⚪ Not mentioned"
                    )
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                    if zero_mention_competitors:
                        st.caption(f"⚪ {len(zero_mention_competitors)} competitor(s) you entered were not mentioned by AI in this run.")
                else:
                    st.info("No direct competitors detected. Add known competitors in Step 1 for better tracking.")

            with tab2:
                st.caption("Large established commercial platforms. These dominate AI responses but are not your direct competition.")
                if dominant_platforms:
                    dom_df = pd.DataFrame(
                        [(b, c, f"{r}%") for b, c, r in dominant_platforms[:15]],
                        columns=["Brand", "Mentions", "Appearance Rate"]
                    )
                    st.dataframe(dom_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No dominant platforms detected.")

            with tab3:
                st.caption("Government agencies, regulatory bodies, and official databases. These appear as authoritative references, not competitors.")
                if government_bodies:
                    gov_df = pd.DataFrame(
                        [(b, c, f"{r}%") for b, c, r in government_bodies[:15]],
                        columns=["Organization", "Mentions", "Appearance Rate"]
                    )
                    st.dataframe(gov_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No government or official bodies detected.")

            # ── Brand mention context ────────────────────────────────────
            st.subheader("How Your Brand Was Mentioned")
            ctx = scores["context_breakdown"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recommended", ctx.get("recommended", 0))
            c2.metric("Mentioned", ctx.get("mentioned", 0))
            c3.metric("Warned Against", ctx.get("warned_against", 0))
            c4.metric("Not Mentioned", ctx.get("not_mentioned", 0))

            # ── Full results table ────────────────────────────────────────
            st.subheader("Full Results Table")
            rows = []
            for r in all_results:
                rows.append({
                    "Topic": r.get("topic", ""),
                    "AI Model": r["tool"],
                    "Prompt": r["query"],
                    "Brand Mentioned": "Yes" if r["brands_detected"]["target_mentioned"] else "No",
                    "Position": r["brands_detected"]["target_position"] or "-",
                    "Context": r["brands_detected"]["target_context"],
                    "Linked Sites": len(r.get("linked_sites", [])),
                    "Brands Found": ", ".join(r["brands_detected"]["all_brands"][:5]),
                    "Response Preview": r["response"][:200] + "..." if len(r.get("response", "")) > 200 else r.get("response", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True,
                         column_config={"Prompt": st.column_config.TextColumn(width="large"),
                                        "Response Preview": st.column_config.TextColumn(width="large")})

            # ── CSV download ──────────────────────────────────────────────
            csv_rows = []
            for r in all_results:
                csv_rows.append({
                    "Topic": r.get("topic", ""),
                    "AI Model": r["tool"],
                    "Prompt": r["query"],
                    "Brand Mentioned": "Yes" if r["brands_detected"]["target_mentioned"] else "No",
                    "Position": r["brands_detected"]["target_position"] or "-",
                    "Context": r["brands_detected"]["target_context"],
                    "Brands Found": ", ".join(r["brands_detected"]["all_brands"][:5]),
                    "Linked Sites": "; ".join([s["url"] for s in r.get("linked_sites", [])]),
                    "Full Response": r.get("response", ""),
                })
            csv = pd.DataFrame(csv_rows).to_csv(index=False)
            st.download_button(
                label="Download Full CSV",
                data=csv,
                file_name=f"citation_{brand_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

            # ── Re-run options ────────────────────────────────────────────
            st.divider()
            st.subheader("Run Again")

            rerun_col1, rerun_col2 = st.columns([1, 1])

            with rerun_col1:
                if st.button("🔄 Run Again with Same Settings", type="primary", use_container_width=True):
                    st.session_state.run_complete = False
                    st.session_state.all_results = []
                    st.rerun()

            with rerun_col2:
                # Country switcher - keeps all topics and prompts, just changes country
                st.markdown("**🌍 Re-run with a different country**")
                _country_options = ["United States", "United Kingdom", "Canada", "Australia",
                                    "Germany", "India", "Pakistan", "Global"]
                _current_country = bd.get("country", "United States")
                _current_idx = _country_options.index(_current_country) if _current_country in _country_options else 0
                new_country = st.selectbox(
                    "Select country",
                    options=_country_options,
                    index=_current_idx,
                    key="rerun_country_select",
                    label_visibility="collapsed"
                )
                if st.button("🚀 Re-run with this country", use_container_width=True):
                    # Update country in brand_data and clear cached website text
                    # so topics/prompts regenerate with new country context
                    st.session_state.brand_data["country"] = new_country
                    # Clear cached prompts so they regenerate with new country
                    st.session_state.prompts_by_topic = {}
                    st.session_state.selected_prompts = {}
                    st.session_state.topics = []
                    st.session_state.selected_topics = []
                    st.session_state.run_complete = False
                    st.session_state.all_results = []
                    st.session_state.step = 2
                    st.rerun()
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from core.scorer import calculate_citation_share, calculate_citation_share_by_topic, calculate_citation_share_by_group
from core.ai_runner import ALL_TOOLS, run_selected_tools, check_key_exists, get_usage_stats, reset_usage_stats
from core.brand_detector import detect_brands


# =============================================================================
# HELPERS
# =============================================================================

def format_error_message(raw_error: str) -> str:
    msg = (raw_error or "").strip()
    low = msg.lower()
    if any(x in low for x in ["context_length_exceeded", "maximum context length", "context window", "token limit", "too many tokens", "prompt is too long"]):
        return "Request too large for the selected model. Reduce topics or query count, or switch models."
    if any(x in low for x in ["429", "rate_limit", "resource_exhausted", "quota"]):
        return "API quota or rate limit reached. Wait and retry, or enable another provider as fallback."
    if "no module named" in low or "sdk not installed" in low:
        return "Required SDK is not installed. Install dependencies from requirements.txt and retry."
    return msg


def _get_key(env_key: str) -> str:
    """
    Read API key - checks os.environ first, then re-reads .env file directly.
    This handles the case where a key was saved in the same Streamlit session
    (os.getenv caches the env at startup and won't reflect new saves).
    """
    # First try os.environ (works if key was set before app started)
    val = os.environ.get(env_key, "").strip()
    if val and "paste_your" not in val.lower():
        return val
    # Fall back to reading .env file directly (handles same-session saves)
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(env_key + "="):
                    v = line.split("=", 1)[1].strip()
                    if v and "paste_your" not in v.lower():
                        os.environ[env_key] = v  # update for future calls
                        return v
    return ""


def _call_ai_for_json(prompt: str) -> str:
    """Call best available AI model for JSON generation (topics/prompts)."""
    # GROQ DISABLED - uncomment below to re-enable
    # groq_key = _get_key("GROQ_API_KEY")
    # if groq_key:
    #     try:
    #         from groq import Groq
    #         client = Groq(api_key=groq_key)
    #         resp = client.chat.completions.create(
    #             model="llama-3.3-70b-versatile",
    #             messages=[{"role": "user", "content": prompt}],
    #             max_tokens=2000,
    #             temperature=0.7,
    #         )
    #         return resp.choices[0].message.content or ""
    #     except Exception:
    #         pass

    # Try Gemini
    gemini_key = _get_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return resp.text or ""
        except Exception:
            pass

    # Try OpenAI
    openai_key = _get_key("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            pass

    raise Exception("No available AI model. Please add at least one API key (Gemini, OpenAI, or Perplexity).")


def _parse_json_list(text: str) -> list:
    text = re.sub(r'```(?:json)?', '', text).strip().rstrip('`').strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        for v in result.values():
            if isinstance(v, list):
                return v
    except Exception:
        pass
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # fallback: extract quoted strings
    return re.findall(r'"([^"]{5,200})"', text)



def fetch_brand_website(domain: str) -> str:
    """
    Visits the brand's website and extracts meaningful text content.
    Returns extracted text or empty string if fetch fails.
    Cleans up navigation, scripts, and boilerplate to keep only real content.
    """
    import requests
    from urllib.parse import urlparse

    # Normalize domain to full URL
    domain = domain.strip().rstrip("/")
    if not domain.startswith("http"):
        url = "https://" + domain
    else:
        url = domain

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # Try multiple URL variants to maximize success rate
    urls_to_try = [url]
    if not url.startswith("https://www."):
        urls_to_try.append(url.replace("https://", "https://www.", 1))
    if url.startswith("https://"):
        urls_to_try.append(url.replace("https://", "http://", 1))

    html = ""
    for try_url in urls_to_try:
        try:
            response = requests.get(try_url, headers=headers, timeout=12)
            response.raise_for_status()
            html = response.text
            break
        except Exception:
            continue

    try:
        if not html:
            return ""

        # Remove scripts, styles, nav, footer boilerplate
        import re as _re
        html = _re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", html, flags=_re.IGNORECASE)
        html = _re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", html, flags=_re.IGNORECASE)
        html = _re.sub(r"<nav[^>]*>[\s\S]*?</nav>", " ", html, flags=_re.IGNORECASE)
        html = _re.sub(r"<footer[^>]*>[\s\S]*?</footer>", " ", html, flags=_re.IGNORECASE)
        html = _re.sub(r"<header[^>]*>[\s\S]*?</header>", " ", html, flags=_re.IGNORECASE)

        # Strip all remaining HTML tags
        text = _re.sub(r"<[^>]+>", " ", html)

        # Clean up whitespace
        text = _re.sub(r"[ \t\n\r]+", " ", text).strip()

        # Keep only meaningful portion (first 3000 chars covers homepage content)
        text = text[:3000]

        # Basic quality check - if too short it probably failed
        if len(text.strip()) < 100:
            return ""

        return text.strip()

    except Exception:
        return ""


def ai_generate_topics(brand_data: dict) -> list:
    """
    Two-stage topic generation:
    Stage 1 - Brand Intelligence: understand the brand deeply using available AI
    Stage 2 - Topic Generation: generate realistic buyer search topics from that intelligence

    Fully model-agnostic - uses _call_ai_for_json which works with any AI model.
    When new models are added to the system, this function benefits automatically.
    """
    business_type = brand_data.get("business_type", "")
    products = ", ".join(brand_data.get("products", []))
    customers = ", ".join(brand_data.get("customers", []))
    key_features = ", ".join(brand_data.get("key_features", []))
    domain = brand_data.get("domain", "")
    brand_name = brand_data["name"]
    country = brand_data.get("country", "")

    # Website content - primary source of truth
    website_text = brand_data.get("_website_text", "")
    if not website_text and domain:
        website_text = fetch_brand_website(domain)

    # Build raw context from all available sources
    raw_context = ""
    if website_text:
        raw_context += "WEBSITE CONTENT:\n" + website_text[:2000] + "\n\n"
    raw_context += (
        "Brand name: " + brand_name + "\n"
        "Business type: " + business_type + "\n"
        "What they offer: " + products + "\n"
        "Who they serve: " + customers + "\n"
        "Key differentiators: " + key_features + "\n"
    )

    # ── STAGE 1: Brand Intelligence ───────────────────────────────────────────
    # Ask the AI to deeply understand the brand before generating topics.
    # This is model-agnostic - works with GPT, Gemini, Claude, Groq, or any future model.
    # The intelligence layer removes the need for keyword rules and mechanical filters.
    # Include buyer insights from Step 1.5 if available
    buyer_insights = brand_data.get("_buyer_insights", [])
    buyer_insights_text = ""
    if buyer_insights:
        buyer_insights_text = "\nDIRECT BUYER INSIGHTS (answers from the brand owner - highest priority):\n"
        for qa in buyer_insights:
            buyer_insights_text += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"

    intelligence_prompt = (
        "You are an AI visibility analyst. Read the brand information below carefully.\n\n"
        + raw_context
        + buyer_insights_text + "\n"
        "Based on this, answer these 4 questions in JSON format:\n"
        "{\n"
        "  \"what_they_offer\": \"One sentence describing exactly what this brand offers in plain language\",\n"
        "  \"exact_buyer\": \"One sentence describing the specific person who needs this, their role and situation\",\n"
        "  \"buyer_search_intent\": \"What would this buyer type into an AI tool when they are ready to find and choose a solution like this one?\",\n"
        "  \"unique_angle\": \"What makes this brand stand out from obvious alternatives in 1 sentence?\"\n"
        "}\n\n"
        "Be specific to THIS brand. Do not use generic category labels.\n"
        "For example, not \'patent software\' but \'patent management software for startup founders who want to file without expensive counsel\'\n"
        "Respond ONLY with valid JSON. No explanation, no markdown."
    )

    brand_intelligence = {}
    try:
        intel_raw = _call_ai_for_json(intelligence_prompt)
        parsed_intel = _parse_json_list(intel_raw)
        if isinstance(parsed_intel, list) and len(parsed_intel) > 0:
            brand_intelligence = parsed_intel[0] if isinstance(parsed_intel[0], dict) else {}
        elif isinstance(intel_raw, str):
            import json as _json
            try:
                brand_intelligence = _json.loads(intel_raw)
            except Exception:
                brand_intelligence = {}
    except Exception:
        brand_intelligence = {}

    # Build enriched context from brand intelligence
    if brand_intelligence:
        enriched_context = (
            "BRAND INTELLIGENCE (AI-generated understanding of this brand):\n"
            + f"What they offer: {brand_intelligence.get('what_they_offer', '')}\n"
            + f"Exact buyer: {brand_intelligence.get('exact_buyer', '')}\n"
            + f"Buyer search intent: {brand_intelligence.get('buyer_search_intent', '')}\n"
            + f"Unique angle: {brand_intelligence.get('unique_angle', '')}\n\n"
            + "RAW BRAND DATA (supplement only):\n"
            + raw_context
        )
    else:
        enriched_context = raw_context

    # Country context
    country_note = f"User location: {country}\n" if country and country.lower() != "global" else ""
    country_topic_note = (
        f"- For 1 topic, add '{country}' to make it location-specific\n"
        if country and country.lower() not in ["global", "united states", ""] else ""
    )

    # ── STAGE 2: Topic Generation from Brand Intelligence ─────────────────────
    # Topics are generated from genuine understanding, not keyword rules.
    # Works for any brand type: software, agency, service, marketplace, etc.
    topic_prompt = (
        "You track brand visibility across AI tools like ChatGPT, Perplexity, Claude, and Gemini.\n\n"
        "Based on the brand intelligence below, generate 7 search topics that the EXACT BUYER "
        "described would type into an AI tool when they are ready to find and choose a solution.\n\n"
        + enriched_context
        + country_note + "\n"
        "RULES:\n"
        "- Topics must be what the EXACT BUYER would search - not generic category searches\n"
        "- Every topic must cause an AI to respond by naming specific brands or providers\n"
        "- Do NOT generate informational, educational, or how-to topics\n"
        "- Do NOT include the brand name '" + brand_name + "' in any topic\n"
        "- Every topic must contain a category word: agency, agencies, firm, service, services, "
        "tool, tools, software, platform, company, companies, provider, providers\n"
        "- Use specific language from the brand intelligence above - not generic category labels\n"
        "- Mix: 3 topics based on unique features/differentiators, 4 based on general buyer intent\n"
        + country_topic_note +
        "\nFor each topic also provide one sentence of buyer intent (why they search this).\n"
        "Respond ONLY with a JSON array of 7 objects: "
        "[{\"topic\": \"...\" , \"intent\": \"...\"}, ...]\n"
        "No markdown, no explanation."
    )
    raw = _call_ai_for_json(topic_prompt)
    parsed_raw = _parse_json_list(raw)
    brand_lower = brand_name.lower()
    topics = []
    topic_intents = {}
    for item in parsed_raw:
        if isinstance(item, dict):
            t = item.get("topic", item.get("name", "")).strip()
            intent = item.get("intent", item.get("reason", "")).strip()
            if t:
                topics.append(t)
                if intent:
                    topic_intents[t] = intent
        elif isinstance(item, str) and item.strip():
            topics.append(item.strip())
    # Store intents in session state for display
    if "brand_data" in st.session_state:
        st.session_state.brand_data["_topic_intents"] = topic_intents

    # ── Niche anchor check ───────────────────────────────────────────────────
    # If the brand's key features contain specific niche terms (patent, legal, medical etc.)
    # then every topic must also contain at least one of those niche terms.
    # This prevents topics like "idea capture tools" when the brand is about "patent idea capture"
    # Fully dynamic - derived from user's own key features and products

    niche_keywords = set()
    all_brand_text = (products + " " + key_features + " " + " ".join(bd.get("products", [])) if "bd" in dir() else products + " " + key_features).lower()

    # Extract meaningful niche words (3+ chars, not common stopwords)
    stopwords = {"and", "the", "for", "with", "our", "not", "from", "that", "this",
                 "are", "was", "has", "have", "will", "been", "your", "their", "its"}
    for word in all_brand_text.split():
        word = word.strip(".,;:()-/")
        if len(word) >= 4 and word not in stopwords:
            niche_keywords.add(word)

    # Keep only words that are truly niche-specific (appear in features but not common English)
    common_english = {"best", "tool", "tools", "software", "platform", "service", "company",
                      "based", "using", "user", "users", "data", "team", "teams", "work",
                      "free", "easy", "simple", "fast", "high", "real", "time", "open",
                      "source", "access", "cost", "price", "need", "help", "make", "build",
                      "track", "manage", "capture", "create", "search", "find", "save"}
    niche_keywords = niche_keywords - common_english

    def _topic_has_niche_context(topic_text, niche_kws):
        """Check if topic contains at least one niche keyword from the brand's space."""
        if not niche_kws:
            return True  # No specific niche detected, skip check
        topic_lower = topic_text.lower()
        return any(kw in topic_lower for kw in niche_kws)

    # Filter out topics that are too vague or likely to produce wrong results
    # These are patterns that make ChatGPT give generic advice instead of brand names
    # Works for any industry - just checking for structural/intent issues
    bad_patterns = [
        "strategies", "tips", "how to", "examples", "guide", "tutorial",
        "best practices", "introduction", "overview", "explained", "meaning",
        "definition", "what is", "benefits of", "advantages of",
    ]
    # Standalone vague nouns that only work when combined with agency/firm/service
    vague_standalone = [
        "providers", "solutions", "platforms", "consultants", "consultancies",
        "writing services", "content writing", "writing service"
    ]

    filtered = []
    for t in topics:
        t = t.strip()
        if not t:
            continue
        if brand_lower in t.lower():
            continue
        t_lower = t.lower()
        # Skip if contains bad educational/informational patterns
        if any(bp in t_lower for bp in bad_patterns):
            continue
        # Skip vague standalone nouns unless paired with agency/firm/service/company
        has_vague = any(v in t_lower for v in vague_standalone)
        has_anchor = any(a in t_lower for a in ["agency", "agencies", "firm", "firms", "service", "services", "company", "companies"])
        if has_vague and not has_anchor:
            continue
        filtered.append(t)

    return filtered[:5]


def ai_generate_prompts(topic: str, brand_data: dict) -> list:
    brand_name = brand_data["name"]
    products = ", ".join(brand_data.get("products", []))
    customers = ", ".join(brand_data.get("customers", []))
    business_type = brand_data.get("business_type", "")
    domain = brand_data.get("domain", "")

    # Reuse cached website text if already fetched, else fetch again
    website_text = brand_data.get("_website_text", "")
    if not website_text and domain:
        website_text = fetch_brand_website(domain)

    # Build context block
    if website_text:
        context_block = (
            "Website content (primary source of truth for what this brand does):\n"
            + website_text[:1500]
            + "\n\nForm data: " + products + " | Customers: " + customers
        )
    else:
        context_block = (
            "What they offer: " + products + "\n"
            "Who buys from them: " + customers
        )

    # Build competitor context - rotate which competitors are used
    # so different topics get different comparison pairs
    competitors = brand_data.get("competitors", [])
    competitor_context = ""
    if competitors:
        # Use topic name to deterministically pick different competitors per topic
        # This ensures variety across topics without randomness
        topic_hash = sum(ord(c) for c in topic) % max(len(competitors), 1)
        # Pick 2 competitors starting from the hash offset
        rotated = competitors[topic_hash:] + competitors[:topic_hash]
        comp_for_comparison = rotated[:2] if len(rotated) >= 2 else rotated
        comp_for_alternative = rotated[2:3] if len(rotated) >= 3 else rotated[:1]

        competitor_context = (
            "\nKnown competitors in this space: " + ", ".join(competitors) +
            "\nFor the COMPARISON prompt, compare these two specifically: " +
            " vs ".join(comp_for_comparison) +
            "\nFor the ALTERNATIVE prompt, ask for alternatives to: " +
            (comp_for_alternative[0] if comp_for_alternative else competitors[0])
        )

    # Get country for location-aware prompts
    country = brand_data.get("country", "")

    # Derive solution word dynamically from business type - works for any industry
    bt_lower = business_type.lower()
    if any(w in bt_lower for w in ["agency", "service", "studio", "consultancy"]):
        solution_word = "agency or service provider"
        avoid_line = "- NEVER use the word \'tool\' or \'software\' - this is a service not a tool\n"
    elif any(w in bt_lower for w in ["saas", "software"]):
        solution_word = "tool or software"
        avoid_line = ""
    elif any(w in bt_lower for w in ["ecommerce", "dtc"]):
        solution_word = "platform or store"
        avoid_line = ""
    elif any(w in bt_lower for w in ["marketplace", "aggregator"]):
        solution_word = "marketplace or platform"
        avoid_line = ""
    else:
        solution_word = "product or service"
        avoid_line = ""

    # Rotate persona role per topic
    customers_list = brand_data.get("customers", [])
    persona_role = customers_list[sum(ord(c) for c in topic) % len(customers_list)] if customers_list else "professional"

    # Country as system context not in prompt text
    country_system = ""
    country_suffix = ""
    if country and country.lower() not in ["global", "united states", ""]:
        country_system = f"\nUser location context: {country}. Tailor relevance to this market."
        country_suffix = f" {country}"

    # ── Decide whether comparison is relevant for this topic ────────────────
    if competitors and comp_for_comparison:
        if len(comp_for_comparison) >= 2:
            comparison_line = (f"3. COMPARISON: How does {comp_for_comparison[0]} compare to "
                               f"{comp_for_comparison[1]} for this use case? "
                               f"(Only include if naturally relevant to topic)")
        else:
            comparison_line = (f"3. ALTERNATIVE: What are the best alternatives to "
                               f"{comp_for_comparison[0]} for this topic?")
    else:
        comparison_line = ("3. SCENARIO: A specific real-world situation where someone needs "
                           f"a {solution_word} for this topic and asks for a recommendation")

    prompt = (
        "You generate search prompts for AI visibility tracking.\n\n"
        "Your job: write 5 prompts a REAL PERSON would type into ChatGPT "
        "when looking for a " + solution_word + ".\n\n"
        "Context:\n"
        + context_block
        + competitor_context
        + country_system + "\n\n"
        "HOW A REAL PERSON TYPES:\n"
        "- Short and direct: \'best saas content agency\' not \'What is the most optimal agency??\'\n"
        "- Conversational: \'who should I use for prior art search\'\n"
        "- Natural: no formal structure, no AI-sounding language\n"
        "- No country names in the prompt text\n\n"
        "GENERATE EXACTLY 5 PROMPTS for TOPIC: \"" + topic + "\"\n\n"
        "1. DIRECT SEARCH: 3-5 words exactly as typed in a search bar. No question mark.\n"
        "2. NATURAL QUESTION: Casual question about this topic. Like asking a friend.\n"
        + comparison_line + "\n"
        "4. PERSONA: I am a " + persona_role + " at a [company type]. "
        "I need [specific need from this topic]. Who do you recommend?\n"
        "5. COLLOQUIAL: Ultra short 2-5 words, informal like a quick text. Must end with ?\n\n"
        "RULES:\n"
        "- NEVER mention \"" + brand_name + "\"\n"
        "- Every prompt must make ChatGPT NAME SPECIFIC BRANDS\n"
        "- Every prompt must be about TOPIC: \"" + topic + "\"\n"
        "- Prompt 3 (comparison/scenario): only include if it naturally fits the topic\n"
        "- No year numbers, no country names\n"
        "- Persona MUST end with a clear recommendation ask\n"
        "- BAD: bare noun phrases, knowledge questions, vague statements\n"
        "\nReturn ONLY a JSON array of exactly 5 strings. No markdown."
    )

    import re as _re
    year_pattern = _re.compile(r"\b(20[0-9]{2})\b")
    brand_lower = brand_name.lower()
    topic_lower = topic.lower().strip()

    # Universal structural validator - works for any niche, any industry
    # A prompt is GOOD if it will force ChatGPT to name specific brands
    # A prompt is BAD if ChatGPT will respond with generic advice, tools, or nothing

    # Trigger words that force brand/agency citations in any niche
    CITATION_TRIGGERS = [
        "recommend", "suggest", "which", "what's the best", "who should",
        "compare", "vs", "versus", "alternative", "alternatives",
        "top", "best", "leading", "agency", "agencies", "firm", "firms",
        "company", "companies", "tool", "tools", "software", "platform",
        "service provider", "vendor", "who do you", "any suggestions",
        "what do you", "which one", "who offers"
    ]

    # Words that indicate a statement (not a question) with no brand ask
    STATEMENT_SIGNALS = [
        "how to", "what is", "what are the benefits", "what does",
        "explain", "guide", "tutorial", "tips for", "ways to",
        "understanding", "introduction to", "overview of", "learn",
        "difference between", "why is", "when should", "how does",
        "what makes", "how can i improve", "how do i"
    ]

    def is_citation_producing(p):
        """
        Returns True if this prompt will force ChatGPT to name specific brands.
        Universal logic - works for any industry, any niche.
        """
        pl = p.lower().strip()

        # Rule 1: Must have at least one citation trigger
        has_trigger = any(trigger in pl for trigger in CITATION_TRIGGERS)

        # Rule 2: Must NOT be a pure knowledge/information question
        is_informational = any(signal in pl for signal in STATEMENT_SIGNALS)

        # Rule 3: Persona prompts must end with a recommendation ask
        is_persona = pl.startswith("i ") or pl.startswith("i'm") or pl.startswith("i am")
        if is_persona:
            has_ask = any(ask in pl for ask in ["recommend", "suggest", "what do you", "any suggestions", "who should"])
            return has_ask

        # Rule 4: Short noun phrases without triggers are bad (e.g. "bottom-funnel content providers")
        words = pl.split()
        if len(words) <= 5 and not has_trigger and "?" not in pl:
            return False

        return has_trigger and not is_informational

    def is_bad_prompt(p):
        """Returns True if this prompt should be rejected and regenerated."""
        return not is_citation_producing(p)

    def process_raw(raw_prompts):
        """Clean, filter, and validate a list of raw prompts."""
        cleaned = []
        for p in raw_prompts:
            p = p.strip()
            if not p:
                continue
            # Strip year numbers
            p = year_pattern.sub("", p).strip()
            p = " ".join(p.split())
            # Skip if contains brand name
            if brand_lower in p.lower():
                continue
            # Skip exact duplicates of topic
            if p.lower() == topic_lower:
                continue
            # Skip duplicates already in list
            if p.lower() in [x.lower() for x in cleaned]:
                continue
            # Skip prompts that will produce bad results
            if is_bad_prompt(p):
                continue
            cleaned.append(p)
        return cleaned

    # First attempt
    raw = _call_ai_for_json(prompt)
    prompts = _parse_json_list(raw)
    result = [topic] + process_raw(prompts)

    # Add country variant if needed and not US
    if country_suffix and len(result) < 5:
        location_prompt = topic + country_suffix
        if location_prompt.lower() not in [r.lower() for r in result]:
            result.append(location_prompt)

    # If still under 5, regenerate and fill gaps - up to 2 extra attempts
    attempts = 0
    while len(result) < 5 and attempts < 2:
        attempts += 1
        retry_prompt = (
            prompt +
            f"\n\nIMPORTANT: The previous response was not enough. "
            f"Generate 4 MORE prompts that are DIFFERENT from these already generated:\n" +
            "\n".join(f"- {r}" for r in result)
        )
        try:
            raw2 = _call_ai_for_json(retry_prompt)
            extra = _parse_json_list(raw2)
            new_clean = process_raw(extra)
            # Add only what we still need
            for p in new_clean:
                if p.lower() not in [r.lower() for r in result]:
                    result.append(p)
                if len(result) >= 5:
                    break
        except Exception:
            break

    return result[:5]


# Common English words falsely detected as brand names - filter these out
BRAND_FALSE_POSITIVES = {
    "relevance", "influence", "impact", "reach", "clarity", "signal",
    "momentum", "velocity", "foundation", "element", "essence", "origin",
    "focus", "vision", "mission", "purpose", "strategy", "content",
    "agency", "studio", "digital", "creative", "media", "marketing",
    "growth", "scale", "pipeline", "revenue", "performance", "results",
    "insights", "analytics", "data", "intelligence", "solutions", "services",
    "platform", "engine", "framework", "system", "network", "hub",
    "monday", "notion", "canvas", "spectrum", "horizon", "zenith",
    "apex", "summit", "peak", "core", "pulse", "spark", "flow",
    "bridge", "link", "connect", "sync", "align", "engage",
    "here", "there", "this", "that", "these", "those",
    "first", "second", "third", "next", "last", "new", "old",
    "best", "top", "great", "good", "better", "well", "more",
    "also", "just", "only", "even", "still", "already", "yet",
    "however", "therefore", "because", "although", "unless", "whether",
}

def is_false_positive_brand(name: str) -> bool:
    """Returns True if this name is likely a false positive, not a real brand."""
    name_lower = name.lower().strip()
    # Single common words
    if name_lower in BRAND_FALSE_POSITIVES:
        return True
    # Very short (1-2 chars) or very long (40+ chars)
    if len(name_lower) <= 2 or len(name_lower) > 40:
        return True
    # All lowercase single word that is a common adjective/verb
    if name_lower == name and " " not in name and len(name) < 8:
        return True
    return False


def extract_linked_sites(text: str) -> list:
    url_pattern = re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2})|[/?#&=+@:!,;])+', re.IGNORECASE)
    urls = list(dict.fromkeys(url_pattern.findall(text)))
    sites = []
    for i, url in enumerate(urls[:8], 1):
        domain = re.sub(r'https?://(www\.)?', '', url).split('/')[0]
        sites.append({"rank": i, "domain": domain, "url": url})
    return sites



def tag_input(label: str, session_key: str, placeholder: str = "", help_text: str = "") -> list:
    """Renders a tag/pill input. User types values and clicks Add to create pills."""
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    # Trick to clear the input field after adding a pill:
    # We use a counter as part of the widget key. Incrementing it forces
    # Streamlit to create a fresh empty input widget on the next rerun.
    clear_key = f"{session_key}_clear_counter"
    if clear_key not in st.session_state:
        st.session_state[clear_key] = 0

    tags = st.session_state[session_key]

    st.markdown(f"**{label}**")
    if help_text:
        st.caption(help_text)

    if tags:
        # Show all pills as styled HTML in one flowing line
        pills_html = "".join([
            f'<span style="display:inline-block;background:#1e3a5f;color:#7dd3fc;'
            f'border:1px solid #2563eb;border-radius:999px;padding:4px 14px;'
            f'margin:3px 4px;font-size:13px;font-weight:500;">{t}</span>'
            for t in tags
        ])
        st.markdown(
            f'<div style="padding:6px 0 8px 0;line-height:2.4;">{pills_html}</div>',
            unsafe_allow_html=True
        )

        # Individual remove buttons in a compact row
        remove_idx = None
        num_cols = min(len(tags), 5)
        rm_cols = st.columns(num_cols)
        for idx, tag in enumerate(tags):
            with rm_cols[idx % num_cols]:
                short = (tag[:14] + "…") if len(tag) > 14 else tag
                if st.button(
                    f"✕ {short}",
                    key=f"{session_key}_rm_{idx}",
                    use_container_width=True,
                    help=f"Remove: {tag}"
                ):
                    remove_idx = idx
        if remove_idx is not None:
            st.session_state[session_key].pop(remove_idx)
            st.rerun()

    # Input + Add + Clear row
    col_in, col_add, col_clear = st.columns([5, 1, 1])
    with col_in:
        new_val = st.text_input(
            label,
            key=f"{session_key}_input_{st.session_state[clear_key]}",
            placeholder=placeholder,
            label_visibility="collapsed"
        )
    with col_add:
        st.write("")
        if st.button("+ Add", key=f"{session_key}_add_btn", use_container_width=True):
            items = [v.strip() for v in new_val.split(",") if v.strip()]
            added = False
            for item in items:
                if item and item not in st.session_state[session_key]:
                    st.session_state[session_key].append(item)
                    added = True
            if added:
                st.session_state[clear_key] += 1
                st.rerun()
    with col_clear:
        if tags:
            st.write("")
            if st.button("Clear all", key=f"{session_key}_clear_btn", use_container_width=True):
                st.session_state[session_key] = []
                st.session_state[clear_key] += 1
                st.rerun()

    return st.session_state[session_key]

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(page_title="AI Citation Tracker", page_icon="📡", layout="wide")

st.title("📡 AI Citation Tracker")
st.caption("Track how often your brand appears in AI-generated recommendations")
st.divider()

# =============================================================================
# SESSION STATE INIT
# =============================================================================

for key in ["step", "brand_data", "topics", "selected_topics", "prompts_by_topic",
            "selected_prompts", "all_results", "run_complete"]:
    if key not in st.session_state:
        if key == "step":
            st.session_state[key] = 1
        elif key in ["topics", "selected_topics", "all_results"]:
            st.session_state[key] = []
        elif key in ["prompts_by_topic", "selected_prompts"]:
            st.session_state[key] = {}
        elif key == "run_complete":
            st.session_state[key] = False
        else:
            st.session_state[key] = None

# =============================================================================
# SIDEBAR - API Keys & Tools (always visible)
# =============================================================================

with st.sidebar:
    st.header("API Keys & Tools")

    def read_env() -> dict:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        existing = {}
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        existing[k.strip()] = v.strip()
        return existing

    def write_env(data: dict):
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        with open(env_path, "w") as f:
            for k, v in data.items():
                f.write(f"{k}={v}\n")

    with st.expander("Manage API Keys"):
        key_fields = {
            # "GROQ_API_KEY": "Groq (Free)",  # GROQ DISABLED
            "PERPLEXITY_API_KEY": "Perplexity",
            "GEMINI_API_KEY": "Gemini (Free tier)",
            "OPENAI_API_KEY": "OpenAI / ChatGPT",
            "ANTHROPIC_API_KEY": "Claude / Anthropic",
        }
        updated_keys = {}
        keys_to_remove = []
        for env_key, label in key_fields.items():
            current = os.getenv(env_key, "")
            has_key = bool(current and current.strip() and "paste_your" not in current.lower())
            masked = current[:8] + "..." if has_key else ""
            col_input, col_remove = st.columns([4, 1])
            with col_input:
                new_val = st.text_input(label, value="", placeholder=masked if masked else "Paste key here",
                                        type="password", key=f"key_input_{env_key}")
                if new_val.strip():
                    updated_keys[env_key] = new_val.strip()
            with col_remove:
                st.write("")
                st.write("")
                if has_key:
                    if st.button("✕", key=f"remove_{env_key}"):
                        keys_to_remove.append(env_key)
        if keys_to_remove:
            existing = read_env()
            for k in keys_to_remove:
                existing.pop(k, None)
                os.environ.pop(k, None)
            write_env(existing)
            st.success(f"Removed {len(keys_to_remove)} key(s).")
            st.rerun()
        if st.button("Save Keys", use_container_width=True):
            if updated_keys:
                existing = read_env()
                existing.update(updated_keys)
                write_env(existing)
                for k, v in updated_keys.items():
                    os.environ[k] = v
                st.success(f"Saved {len(updated_keys)} key(s).")
                st.rerun()
            else:
                st.warning("No keys entered.")

    st.divider()
    st.subheader("AI Tools")
    st.caption("Toggle which tools to use for tracking.")
    selected_tools = []
    for tool_name, tool_info in ALL_TOOLS.items():
        key_exists = check_key_exists(tool_info["requires_key"])
        is_free = tool_info["free"]
        if is_free and key_exists:
            label = f"{tool_name} ✅ Free"
            default, disabled = True, False
        elif not is_free and key_exists:
            label = f"{tool_name} 🔑 Key found"
            default, disabled = True, False
        elif not key_exists and not is_free:
            label = f"{tool_name} 🔒 Needs key"
            default, disabled = False, True
        else:
            label = f"{tool_name} ✅ Free"
            default, disabled = True, False
        toggled = st.checkbox(label, value=default, disabled=disabled,
                              help=tool_info["description"], key=f"tool_{tool_name}")
        if toggled and not disabled:
            selected_tools.append(tool_name)

    if not selected_tools:
        st.warning("Select at least one tool")

    st.divider()
    stats = get_usage_stats()
    any_usage = any(stats[t]["calls"] > 0 for t in stats)
    if any_usage:
        st.subheader("Session Usage")
        for tool_name in selected_tools:
            if tool_name in stats and stats[tool_name]["calls"] > 0:
                s = stats[tool_name]
                st.markdown(f"**{tool_name}**: {s['calls']} calls, ~{s['estimated_tokens']} tokens")
        if st.button("Reset Stats"):
            reset_usage_stats()
            st.rerun()

    st.divider()
    if st.button("🔄 Start Over", use_container_width=True):
        for key in ["step", "brand_data", "topics", "selected_topics",
                    "prompts_by_topic", "selected_prompts", "all_results", "run_complete",
                    "step1_products", "step1_customers", "step1_key_features", "step1_competitors"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# =============================================================================
# STEP INDICATOR
# =============================================================================

step_names = ["Brand Details", "Topics", "Prompts", "Run & Results"]
cols = st.columns(4)
_cur_step = st.session_state.step
# Map step "1b" to position 1.5 for the indicator
_step_num = 1.5 if _cur_step == "1b" else (int(_cur_step) if isinstance(_cur_step, int) else 1)
for i, (col, name) in enumerate(zip(cols, step_names), 1):
    with col:
        if i < _step_num:
            st.success(f"✓ {name}")
        elif i == int(_step_num):
            st.info(f"▶ {name}")
        else:
            st.caption(f"{i}.  {name}")

st.divider()

# =============================================================================
# STEP 1: BRAND DETAILS
# =============================================================================

if st.session_state.step == 1:
    st.subheader("Step 1: Brand Details")
    st.caption("Enter your brand information. This is used to generate relevant topics and prompts.")

    # Pre-populate from saved brand_data if user navigated back
    _saved = st.session_state.get("brand_data") or {}

    col1, col2 = st.columns(2)
    with col1:
        brand_name = st.text_input("Brand Name *", value=_saved.get("name", ""), placeholder="e.g. Concurate")
    with col2:
        brand_domain = st.text_input("Brand Domain/URL *", value=_saved.get("domain", ""), placeholder="e.g. concurate.com")

    # Restore pills from saved brand_data when user navigates back
    if "step1_products" not in st.session_state and _saved.get("products"):
        st.session_state["step1_products"] = list(_saved["products"])
    if "step1_customers" not in st.session_state and _saved.get("customers"):
        st.session_state["step1_customers"] = list(_saved["customers"])
    if "step1_key_features" not in st.session_state and _saved.get("key_features"):
        st.session_state["step1_key_features"] = list(_saved["key_features"])

    products_list = tag_input(
        "Your Products and Services",
        session_key="step1_products",
        placeholder="Type a product/service and click Add",
        help_text="List all possible ways customers may describe your products/services"
    )
    customers_list = tag_input(
        "Your Target Customers",
        session_key="step1_customers",
        placeholder="Type a customer persona and click Add",
        help_text="Briefly list your different ideal customer personas"
    )
    key_features_list = tag_input(
        "Key Features / Differentiators",
        session_key="step1_key_features",
        placeholder="Type a feature or benefit and click Add",
        help_text="List important features, benefits and differentiators"
    )

    _btype_options = ["SaaS / Software", "Agency / Service Business", "Ecommerce / DTC Brand",
                     "Marketplace / Aggregator", "Other / Not Sure"]
    _btype_default = _saved.get("business_type", "SaaS / Software")
    _btype_idx = _btype_options.index(_btype_default) if _btype_default in _btype_options else 0
    business_type = st.selectbox(
        "Business Type",
        options=_btype_options,
        index=_btype_idx,
        help="Controls how competitors are detected"
    )

    col3, col4 = st.columns(2)
    _country_options = ["United States", "United Kingdom", "Canada", "Australia",
                         "Germany", "India", "Pakistan", "Global"]
    _country_default = _saved.get("country", "United States")
    _country_idx = _country_options.index(_country_default) if _country_default in _country_options else 0
    country = st.selectbox("Country", options=_country_options, index=_country_idx)

    # Restore competitors pills when navigating back
    if "step1_competitors" not in st.session_state and _saved.get("competitors"):
        st.session_state["step1_competitors"] = list(_saved["competitors"])

    competitors_list_input = tag_input(
        "Your Direct Competitors (recommended)",
        session_key="step1_competitors",
        placeholder="Type a competitor name and click Add",
        help_text="Add brands at a similar scale to yours. These will be tracked separately from dominant platforms like Google."
    )

    st.divider()

    if st.button("Next: Generate Topics →", type="primary"):
        if not brand_name.strip() or not brand_domain.strip():
            st.error("Brand Name and Domain are required.")
        elif not products_list:
            st.error("Please add at least one product or service.")
        elif not competitors_list_input:
            st.error("⚠️ Competitors are required. Add at least one direct competitor for accurate comparison prompts and benchmarking.")
        else:
            # Check if brand data has changed from previous session
            # Ensure prev_data is always a dict even if session state has None
            prev_data = st.session_state.get("brand_data") or {}
            new_data = {
                "name": brand_name.strip(),
                "domain": brand_domain.strip(),
                "products": products_list,
                "customers": customers_list,
                "key_features": key_features_list,
                "business_type": business_type,
                "country": country,
                "competitors": competitors_list_input,
            }
            # If anything changed, clear all downstream cached data
            data_changed = (
                prev_data.get("name") != new_data["name"] or
                prev_data.get("domain") != new_data["domain"] or
                prev_data.get("products") != new_data["products"] or
                prev_data.get("customers") != new_data["customers"] or
                prev_data.get("key_features") != new_data["key_features"] or
                prev_data.get("business_type") != new_data["business_type"] or
                prev_data.get("country") != new_data["country"] or
                prev_data.get("competitors") != new_data["competitors"]
            )
            st.session_state.brand_data = new_data
            if data_changed:
                for key in ["topics", "selected_topics", "prompts_by_topic",
                            "selected_prompts", "all_results", "run_complete"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # Force website re-fetch since domain may have changed
                st.session_state.brand_data.pop("_website_text", None)
            st.session_state.step = "1b"
            st.rerun()

# =============================================================================
# STEP 1B: SMART QUESTIONS (AI-generated based on brand)
# =============================================================================

elif st.session_state.step == "1b":
    bd = st.session_state.brand_data

    # ── Generate questions AND auto-fill answers from website ─────────────────
    if "_smart_qa" not in st.session_state:

        # Fetch website if not already done
        website_text = bd.get("_website_text", "")
        if not website_text and bd.get("domain"):
            with st.spinner("🌐 Reading your website..."):
                website_text = fetch_brand_website(bd.get("domain", ""))
                st.session_state.brand_data["_website_text"] = website_text

        q_context = (
            f"Brand: {bd.get('name')}\n"
            f"Domain: {bd.get('domain')}\n"
            f"Business type: {bd.get('business_type')}\n"
            f"What they offer: {', '.join(bd.get('products', []))}\n"
            f"Who they serve: {', '.join(bd.get('customers', []))}\n"
            f"Key differentiators: {', '.join(bd.get('key_features', []))}\n"
            f"Competitors: {', '.join(bd.get('competitors', []))}\n"
        )
        if website_text:
            q_context = f"Website content:\n{website_text[:2000]}\n\n" + q_context

        with st.spinner("🧠 Analyzing your brand and generating insights..."):
            auto_prompt = (
                "You are an AI visibility consultant. Read this brand profile carefully:\n\n"
                + q_context + "\n"
                "Based on your deep understanding of this brand, generate 4 questions AND "
                "answer each one yourself using the brand information above.\n\n"
                "The questions should uncover:\n"
                "1. What buyers search for when ready to choose this brand\n"
                "2. What pain they have before finding this brand\n"
                "3. What makes buyers switch from competitors to this brand\n"
                "4. The most specific search phrase a real buyer would type\n\n"
                "Answer each question as if you are the brand's expert analyst "
                "who has read their website and understands their buyers deeply.\n"
                "Answers must be specific, realistic, and based only on the brand info above.\n\n"
                "Return ONLY a JSON array of 4 objects:\n"
                '[{"question": "...", "answer": "..."},...]\n'
                "No markdown, no explanation."
            )
            try:
                raw_qa = _call_ai_for_json(auto_prompt)
                parsed_qa = _parse_json_list(raw_qa)
                qa_pairs = []
                for item in parsed_qa:
                    if isinstance(item, dict) and item.get("question") and item.get("answer"):
                        qa_pairs.append({
                            "question": item["question"].strip(),
                            "answer": item["answer"].strip()
                        })
                if len(qa_pairs) < 4:
                    qa_pairs = [
                        {"question": f"What would a buyer type into ChatGPT when ready to choose {bd.get('name')}?",
                         "answer": f"They would likely search for {', '.join(bd.get('products', ['a solution like this'])[:2])}"},
                        {"question": "What pain does the buyer have before finding this brand?",
                         "answer": f"They struggle with managing {', '.join(bd.get('key_features', ['their needs'])[:1])} without the right tool."},
                        {"question": "Which competitor do buyers settle for if they can't find this brand?",
                         "answer": f"They often end up with {bd.get('competitors', ['a larger competitor'])[0] if bd.get('competitors') else 'a larger competitor'}."},
                        {"question": "Describe the ideal buyer in one sentence.",
                         "answer": f"A {', '.join(bd.get('customers', ['professional'])[:1])[0:50]} who needs {', '.join(bd.get('products', ['this solution'])[:1])}."}
                    ][:4 - len(qa_pairs)]
            except Exception:
                qa_pairs = [
                    {"question": f"What would a buyer type into ChatGPT when ready to choose {bd.get('name')}?", "answer": ""},
                    {"question": "What pain does the buyer have before finding this brand?", "answer": ""},
                    {"question": "Which competitor do buyers usually settle for instead?", "answer": ""},
                    {"question": "Describe the ideal buyer in one sentence.", "answer": ""}
                ]
            st.session_state["_smart_qa"] = qa_pairs

    qa_pairs = st.session_state.get("_smart_qa", [])

    # ── Header card ───────────────────────────────────────────────────────────
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1e3a5f 0%, #0f2340 100%);
                    border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
                    border: 1px solid #2563eb33;'>
            <div style='font-size: 13px; color: #7dd3fc; font-weight: 600;
                        letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;'>
                Step 1.5 of 4 · Brand Understanding
            </div>
            <div style='font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 10px;'>
                We read your website and filled these in 🤖
            </div>
            <div style='font-size: 15px; color: #94a3b8; line-height: 1.6;'>
                Based on your website and brand details, we answered these questions about your buyers.
                <br>
                <span style='color: #7dd3fc; font-weight: 500;'>Review and edit</span> 
                anything that looks wrong — or just click 
                <span style='color: #7dd3fc; font-weight: 500;'>Looks Good →</span> to continue.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Q&A cards with editable answers ──────────────────────────────────────
    icons = ["🔍", "💡", "🏁", "👤"]
    edited_answers = {}

    for idx, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        auto_answer = qa.get("answer", "")
        icon = icons[idx] if idx < len(icons) else "💬"

        st.markdown(f"""
            <div style='background: #0f1f35; border: 1px solid #1e3a5f;
                        border-left: 3px solid #2563eb; border-radius: 10px;
                        padding: 14px 20px; margin-bottom: 4px;'>
                <span style='color: #7dd3fc; font-size: 16px;'>{icon}</span>
                <span style='color: #e2e8f0; font-size: 14px; font-weight: 600;
                             margin-left: 10px;'>{question}</span>
            </div>
        """, unsafe_allow_html=True)

        edited_answers[f"q{idx}"] = st.text_area(
            label=f"q{idx}",
            value=auto_answer,
            key=f"smart_qa_{idx}",
            label_visibility="collapsed",
            height=80,
            help="Auto-filled from your website. Edit if needed."
        )

    # ── Action buttons ────────────────────────────────────────────────────────
    st.write("")
    col_back, col_skip, col_next = st.columns([1, 1, 2])

    with col_back:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            for k in ["_smart_qa", "_smart_questions"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    with col_skip:
        if st.button("Skip this step", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

    with col_next:
        if st.button("✅ Looks Good → Generate Topics", type="primary", use_container_width=True):
            # Save edited Q&A into brand_data
            final_qa = []
            for idx, qa in enumerate(qa_pairs):
                answer = edited_answers.get(f"q{idx}", "").strip()
                if answer:
                    final_qa.append({"question": qa.get("question", ""), "answer": answer})
            st.session_state.brand_data["_buyer_insights"] = final_qa
            for key in ["topics", "selected_topics", "prompts_by_topic",
                        "selected_prompts", "all_results", "run_complete"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 2
            st.rerun()

# =============================================================================
# STEP 2: TOPICS
# =============================================================================

elif st.session_state.step == 2:
    bd = st.session_state.brand_data
    st.subheader("Step 2: Topics")
    st.caption("These are the topics for which we'll measure your brand visibility. Choose which to keep or add your own.")

    # Auto-generate if not done yet
    if not st.session_state.topics:

        # ── Step 1: Show iframe preview + fetch website in parallel ──────────
        domain = bd.get("domain", "")

        # Build the full URL for iframe display
        iframe_url = domain if domain.startswith("http") else f"https://{domain}"

        # Show the website in an iframe so the user can see we are visiting it
        st.markdown("#### 🌐 Visiting your website...")
        st.caption(f"We are reading: **{iframe_url}**")

        # Embed the actual website in an iframe
        st.components.v1.iframe(
            src=iframe_url,
            height=320,
            scrolling=True
        )

        # Now fetch the content in background
        with st.spinner("📖 Reading and extracting content from your website..."):
            if "_website_text" not in st.session_state.brand_data:
                website_text = fetch_brand_website(domain)
                st.session_state.brand_data["_website_text"] = website_text
            else:
                website_text = st.session_state.brand_data.get("_website_text", "")

        # ── Step 2: Show what was understood from the website ─────────────────
        if website_text:
            # Use AI to summarize what was understood about the brand from the website
            understanding_prompt = (
                "You just read the homepage of a brand. Based on the content below, "
                "write a SHORT 3-4 sentence summary explaining:\n"
                "1. What this brand does\n"
                "2. Who their customers are\n"
                "3. What makes them different\n\n"
                "Be specific and factual. Only use what is actually on the page.\n"
                "Do not invent anything. Write in plain English.\n\n"
                "Website content:\n" + website_text[:3000]
            )
            with st.spinner("🧠 Reading and understanding your website..."):
                try:
                    brand_understanding = _call_ai_for_json(
                        understanding_prompt + "\n\nRespond with plain text only, no JSON, no bullet points."
                    )
                except Exception:
                    brand_understanding = ""

            st.success(f"✅ Website fetched successfully from {bd.get('domain', '')}")
            with st.expander("📋 Brand Intelligence Summary", expanded=True):

                # ── Section 1: From Website ───────────────────────────────
                st.markdown("### 🌐 From Your Website")
                st.caption(f"Fetched from: {bd.get('domain', '')}")
                if brand_understanding:
                    st.info(brand_understanding)
                else:
                    st.info(website_text[:400] + "...")

                st.divider()

                # ── Section 2: From Form ──────────────────────────────────
                st.markdown("### 📝 From Your Form")
                st.caption("This is what you entered manually in Step 1.")

                form_cols = st.columns(2)
                with form_cols[0]:
                    if bd.get("products"):
                        st.markdown("**📦 Products / Services**")
                        for p in bd["products"]:
                            st.markdown(f"- {p}")
                    if bd.get("customers"):
                        st.markdown("**👥 Target Customers**")
                        for c in bd["customers"]:
                            st.markdown(f"- {c}")
                with form_cols[1]:
                    if bd.get("key_features"):
                        st.markdown("**⭐ Key Differentiators**")
                        for f in bd["key_features"]:
                            st.markdown(f"- {f}")
                    if bd.get("competitors"):
                        st.markdown("**🏁 Direct Competitors**")
                        for comp in bd["competitors"]:
                            st.markdown(f"- {comp}")
                    st.markdown(f"**🌍 Country:** {bd.get('country', 'United States')}")
                    st.markdown(f"**🏢 Business Type:** {bd.get('business_type', '')}")

                st.divider()
                st.success("✅ Topics will be generated using BOTH your website content and your form data.")
        else:
            st.warning(
                "⚠️ Could not read your website automatically. "
                "Topics will be generated from your form data only. "
                "Check that your domain is correct and publicly accessible."
            )
            with st.expander("📋 Form Data Being Used", expanded=True):
                st.markdown("### 📝 From Your Form")
                st.caption("Website could not be fetched — using form data only.")
                form_cols2 = st.columns(2)
                with form_cols2[0]:
                    if bd.get("products"):
                        st.markdown("**📦 Products / Services**")
                        for p in bd["products"]:
                            st.markdown(f"- {p}")
                    if bd.get("customers"):
                        st.markdown("**👥 Target Customers**")
                        for c in bd["customers"]:
                            st.markdown(f"- {c}")
                with form_cols2[1]:
                    if bd.get("key_features"):
                        st.markdown("**⭐ Key Differentiators**")
                        for f in bd["key_features"]:
                            st.markdown(f"- {f}")
                    if bd.get("competitors"):
                        st.markdown("**🏁 Direct Competitors**")
                        for comp in bd["competitors"]:
                            st.markdown(f"- {comp}")
                    st.markdown(f"**🌍 Country:** {bd.get('country', 'United States')}")

        # ── Step 3: Generate topics ───────────────────────────────────────────
        with st.spinner("🔍 Generating topics based on your brand profile..."):
            try:
                generated = ai_generate_topics(st.session_state.brand_data)
                st.session_state.topics = generated
                st.session_state.selected_topics = list(generated)
            except Exception as e:
                st.error(f"Topic generation failed: {format_error_message(str(e))}")
                st.info("You can add topics manually below.")
                st.session_state.topics = []
                st.session_state.selected_topics = []

    if st.session_state.topics:
        selected_count = len(st.session_state.selected_topics)
        st.success(f"✓ {selected_count} topic{'s' if selected_count != 1 else ''} selected")

        topic_intents = st.session_state.brand_data.get("_topic_intents", {})
        st.write("**AI Generated Topics** (uncheck to remove, click ✏️ to edit):")
        for t_idx, topic in enumerate(st.session_state.topics):
            checked = topic in st.session_state.selected_topics
            col_check, col_edit = st.columns([8, 1])
            with col_check:
                new_checked = st.checkbox(f"✦ {topic}", value=checked, key=f"topic_check_{t_idx}")
                if new_checked and topic not in st.session_state.selected_topics:
                    st.session_state.selected_topics.append(topic)
                elif not new_checked and topic in st.session_state.selected_topics:
                    st.session_state.selected_topics.remove(topic)
                if topic in topic_intents:
                    st.caption(f"💡 {topic_intents[topic]}")
            with col_edit:
                edit_key = f"topic_edit_mode_{t_idx}"
                if st.button("✏️", key=f"topic_edit_btn_{t_idx}", help="Edit this topic"):
                    st.session_state[edit_key] = True

            # Show inline editor if edit mode is active
            if st.session_state.get(f"topic_edit_mode_{t_idx}"):
                new_val = st.text_input(
                    "Edit topic",
                    value=topic,
                    key=f"topic_edit_input_{t_idx}",
                    label_visibility="collapsed"
                )
                save_col, cancel_col = st.columns([1, 1])
                with save_col:
                    if st.button("✅ Save", key=f"topic_save_{t_idx}", use_container_width=True):
                        new_val = new_val.strip()
                        if new_val and new_val != topic:
                            # Update in topics list
                            was_selected = topic in st.session_state.selected_topics
                            st.session_state.topics[t_idx] = new_val
                            if was_selected:
                                st.session_state.selected_topics.remove(topic)
                                st.session_state.selected_topics.append(new_val)
                        st.session_state[f"topic_edit_mode_{t_idx}"] = False
                        st.rerun()
                with cancel_col:
                    if st.button("✖ Cancel", key=f"topic_cancel_{t_idx}", use_container_width=True):
                        st.session_state[f"topic_edit_mode_{t_idx}"] = False
                        st.rerun()

    st.divider()
    col_add1, col_add2 = st.columns([4, 1])
    with col_add1:
        new_topic = st.text_input("Add a custom topic", placeholder="e.g. B2B SaaS content agency",
                                  key="custom_topic_input")
    with col_add2:
        st.write("")
        st.write("")
        if st.button("+ Add"):
            if new_topic.strip() and new_topic.strip() not in st.session_state.topics:
                st.session_state.topics.append(new_topic.strip())
                st.session_state.selected_topics.append(new_topic.strip())
                st.rerun()

    st.divider()
    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← Back"):
            # Clear ALL cached data so everything regenerates fresh from Step 1 edits
            st.session_state.step = 1
            for key in ["topics", "selected_topics", "prompts_by_topic",
                        "selected_prompts", "all_results", "run_complete"]:
                if key in st.session_state:
                    del st.session_state[key]
            # Clear cached website text so it re-fetches with any domain changes
            if "brand_data" in st.session_state:
                st.session_state.brand_data.pop("_website_text", None)
            st.rerun()
    with col_next:
        if st.button("Next: Generate Prompts →", type="primary"):
            if not st.session_state.selected_topics:
                st.error("Please select at least one topic.")
            else:
                st.session_state.step = 3
                st.rerun()

# =============================================================================
# STEP 3: PROMPTS
# =============================================================================

elif st.session_state.step == 3:
    bd = st.session_state.brand_data
    st.subheader("Step 3: Review Prompts")
    st.caption("These prompts will be sent to each AI model to check if your brand is mentioned.")

    # Generate prompts for each selected topic FIRST before showing count
    for topic in st.session_state.selected_topics:
        if topic not in st.session_state.prompts_by_topic:
            with st.spinner(f"Generating prompts for: {topic}..."):
                try:
                    prompts = ai_generate_prompts(topic, st.session_state.brand_data)
                    st.session_state.prompts_by_topic[topic] = prompts
                    st.session_state.selected_prompts[topic] = list(prompts)
                except Exception as e:
                    st.warning(f"Could not generate prompts for '{topic}': {format_error_message(str(e))}")
                    st.session_state.prompts_by_topic[topic] = [topic]
                    st.session_state.selected_prompts[topic] = [topic]

    # Count AFTER generation so numbers are accurate
    total_topics = len(st.session_state.selected_topics)
    all_generated = all(
        t in st.session_state.prompts_by_topic
        for t in st.session_state.selected_topics
    )
    total_prompts = sum(
        len(st.session_state.selected_prompts.get(t, []))
        for t in st.session_state.selected_topics
    )
    if all_generated and total_topics > 0:
        avg_per_topic = round(total_prompts / total_topics, 1)
        st.info(f"**{total_topics} topics × {avg_per_topic} avg prompts = {total_prompts} total prompts** per AI model")
    else:
        st.info(f"**{total_topics} topics** — generating prompts...")

    # Display prompts per topic in accordion style
    for t_idx, topic in enumerate(st.session_state.selected_topics):
        prompts = st.session_state.prompts_by_topic.get(topic, [])
        selected = st.session_state.selected_prompts.get(topic, [])

        with st.expander(f"**{topic}** ({len(selected)} prompts selected)", expanded=True):
            for p_idx, prompt in enumerate(prompts):
                checked = prompt in selected
                col_check, col_edit = st.columns([8, 1])
                with col_check:
                    cb_key = f"prompt_t{t_idx}_p{p_idx}"
                    new_checked = st.checkbox(prompt, value=checked, key=cb_key)
                    if new_checked and prompt not in st.session_state.selected_prompts.get(topic, []):
                        st.session_state.selected_prompts.setdefault(topic, []).append(prompt)
                    elif not new_checked and prompt in st.session_state.selected_prompts.get(topic, []):
                        st.session_state.selected_prompts[topic].remove(prompt)
                with col_edit:
                    pedit_key = f"prompt_edit_mode_t{t_idx}_p{p_idx}"
                    if st.button("✏️", key=f"prompt_edit_btn_t{t_idx}_p{p_idx}", help="Edit this prompt"):
                        st.session_state[pedit_key] = True

                # Inline editor for this prompt
                if st.session_state.get(f"prompt_edit_mode_t{t_idx}_p{p_idx}"):
                    new_val = st.text_input(
                        "Edit prompt",
                        value=prompt,
                        key=f"prompt_edit_input_t{t_idx}_p{p_idx}",
                        label_visibility="collapsed"
                    )
                    psave_col, pcancel_col = st.columns([1, 1])
                    with psave_col:
                        if st.button("✅ Save", key=f"prompt_save_t{t_idx}_p{p_idx}", use_container_width=True):
                            new_val = new_val.strip()
                            if new_val and new_val != prompt:
                                was_selected = prompt in st.session_state.selected_prompts.get(topic, [])
                                st.session_state.prompts_by_topic[topic][p_idx] = new_val
                                if was_selected:
                                    if prompt in st.session_state.selected_prompts.get(topic, []):
                                        st.session_state.selected_prompts[topic].remove(prompt)
                                    st.session_state.selected_prompts.setdefault(topic, []).append(new_val)
                            st.session_state[f"prompt_edit_mode_t{t_idx}_p{p_idx}"] = False
                            st.rerun()
                    with pcancel_col:
                        if st.button("✖ Cancel", key=f"prompt_cancel_t{t_idx}_p{p_idx}", use_container_width=True):
                            st.session_state[f"prompt_edit_mode_t{t_idx}_p{p_idx}"] = False
                            st.rerun()

            # Add custom prompt
            custom_key = f"custom_prompt_t{t_idx}"
            custom_prompt = st.text_input("+ Add prompt", key=custom_key,
                                          placeholder="Type a custom prompt and press Enter")
            if custom_prompt.strip() and custom_prompt.strip() not in prompts:
                if st.button("Add", key=f"add_prompt_btn_t{t_idx}"):
                    st.session_state.prompts_by_topic[topic].append(custom_prompt.strip())
                    st.session_state.selected_prompts.setdefault(topic, []).append(custom_prompt.strip())
                    st.rerun()

    st.divider()
    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← Back"):
            # Clear prompts and results so they regenerate if topics changed
            st.session_state.step = 2
            for key in ["prompts_by_topic", "selected_prompts", "all_results", "run_complete"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    with col_next:
        total_selected = sum(len(v) for v in st.session_state.selected_prompts.values())
        if st.button(f"Start Tracking ({total_selected} prompts x {len(selected_tools)} models) →",
                     type="primary", disabled=len(selected_tools) == 0):
            if total_selected == 0:
                st.error("Please select at least one prompt.")
            else:
                st.session_state.step = 4
                st.rerun()

# =============================================================================
# STEP 4: RUN & RESULTS
# =============================================================================

elif st.session_state.step == 4:
    bd = st.session_state.brand_data
    brand_name = bd["name"]
    brand_domain = bd["domain"]
    competitors_list = bd.get("competitors", [])

    # Determine business type string
    btype = bd.get("business_type", "")
    if "Agency" in btype or "Service" in btype:
        detected_business_type = "service"
    elif "SaaS" in btype or "Software" in btype:
        detected_business_type = "software"
    else:
        detected_business_type = "other"

    # Run tracking if not done
    if not st.session_state.run_complete:
        st.subheader("Running Tracking...")

        # Build flat list of prompts with topic metadata
        all_prompts = []
        for topic, prompts in st.session_state.selected_prompts.items():
            for prompt in prompts:
                all_prompts.append({"query": prompt, "topic": topic, "category": "awareness"})

        progress_bar = st.progress(0)
        status_text = st.empty()
        live_feed = st.empty()
        call_count = 0
        all_results = []
        exhausted_tools = set()
        live_log = []
        total_calls = len(all_prompts) * len(selected_tools)

        for i, q in enumerate(all_prompts):
            status_text.text(f"Running {i+1}/{len(all_prompts)}: {q['query'][:70]}...")
            active_tools = [t for t in selected_tools if t not in exhausted_tools]
            if not active_tools:
                st.warning("All tools rate-limited. Stopping early.")
                break

            tool_responses = run_selected_tools(q["query"], active_tools)

            for tool_name, response_text in tool_responses.items():
                if response_text.startswith("ERROR"):
                    clean_error = response_text.split("ERROR:", 1)[-1].strip()
                    if any(x in clean_error.lower() for x in ["429", "rate_limit", "resource_exhausted", "quota"]):
                        exhausted_tools.add(tool_name)
                    call_count += 1
                    progress_bar.progress(min(call_count / total_calls, 1.0))
                    continue

                brand_data_detected = detect_brands(
                    response_text,
                    brand_name,
                    icp_text=f"{', '.join(bd.get('products', []))} {', '.join(bd.get('customers', []))}",
                    business_type=detected_business_type,
                    user_competitors=competitors_list
                )
                # Filter false positive brand names from results
                brand_data_detected["all_brands"] = [
                    b for b in brand_data_detected.get("all_brands", [])
                    if not is_false_positive_brand(b)
                ]

                linked_sites = extract_linked_sites(response_text)

                all_results.append({
                    "query": q["query"],
                    "topic": q["topic"],
                    "category": q["category"],
                    "query_group": "C",
                    "tool": tool_name,
                    "response": response_text,
                    "brands_detected": brand_data_detected,
                    "linked_sites": linked_sites,
                })

                # Update live feed
                mentioned = brand_data_detected.get("target_mentioned", False)
                live_log.append({
                    "prompt": q["query"][:70] + ("..." if len(q["query"]) > 70 else ""),
                    "model": tool_name,
                    "detected": mentioned
                })
                feed_lines = []
                for log in live_log[-8:]:  # Show last 8 entries
                    icon = "✅" if log["detected"] else "⚪"
                    feed_lines.append(f"{icon} **{log['model']}** — {log['prompt']}")
                live_feed.markdown("**Live Results:**\n" + "\n".join(feed_lines))

                call_count += 1
                progress_bar.progress(min(call_count / total_calls, 1.0))

        status_text.text("Done.")
        st.session_state.all_results = all_results
        st.session_state.run_complete = True

        # Play a notification sound when results are ready
        # Uses Web Audio API via JavaScript - no external files needed
        st.components.v1.html("""
        <script>
        (function() {
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                function beep(freq, start, duration, vol) {
                    var o = ctx.createOscillator();
                    var g = ctx.createGain();
                    o.connect(g);
                    g.connect(ctx.destination);
                    o.frequency.value = freq;
                    o.type = 'sine';
                    g.gain.setValueAtTime(0, ctx.currentTime + start);
                    g.gain.linearRampToValueAtTime(vol, ctx.currentTime + start + 0.01);
                    g.gain.linearRampToValueAtTime(0, ctx.currentTime + start + duration);
                    o.start(ctx.currentTime + start);
                    o.stop(ctx.currentTime + start + duration + 0.1);
                }
                // Two-tone pleasant notification
                beep(600, 0,    0.15, 0.3);
                beep(900, 0.18, 0.25, 0.3);
            } catch(e) {}
        })();
        </script>
        """, height=0)

        st.rerun()

    # ==========================================================================
    # DISPLAY RESULTS
    # ==========================================================================

    else:
        all_results = st.session_state.all_results
        st.subheader(f"Results: {brand_name}")
        st.caption(f"Domain: {brand_domain} | Country: {bd.get('country', 'US')} | Business Type: {bd.get('business_type', '')}")

        if not all_results:
            st.error("No results returned. Check your API keys and try again.")
        else:
            scores = calculate_citation_share(all_results, brand_name)

            # ── Overall metrics ──────────────────────────────────────────
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Citation Share", f"{scores['overall_citation_share']}%")
            c2.metric("Total Mentions", f"{scores['total_mentions']} / {scores['total_queries_run']}")
            c3.metric("Position Score", f"{scores['position_score_pct']}%")
            c4.metric("Topics Tracked", len(st.session_state.selected_topics))

            # ── Competitor benchmark ──────────────────────────────────────
            st.divider()
            st.subheader("📊 How You Compare Against Competitors")
            st.caption("Your brand vs competitors mentioned by AI across all prompts.")

            real_comps = scores.get("real_competitors", [])
            total_q_b = scores["total_queries_run"]
            your_pct = scores["overall_citation_share"]
            user_comps = bd.get("competitors", [])
            detected_map = {b.lower(): c for b, c in real_comps}

            bench_rows = [{"Brand": f"🎯 {brand_name} (You)", "AI Mentions": scores["total_mentions"],
                           "Visibility": f"{your_pct}%", "Note": "Your Brand"}]
            for comp in user_comps:
                mentions = detected_map.get(comp.lower(), 0)
                pct = round((mentions / total_q_b) * 100) if total_q_b > 0 else 0
                bench_rows.append({"Brand": comp, "AI Mentions": mentions,
                                   "Visibility": f"{pct}%", "Note": "Your Competitor"})
            for bc, cnt in real_comps[:8]:
                if bc.lower() not in [c.lower() for c in user_comps] and bc.lower() != brand_name.lower():
                    pct = round((cnt / total_q_b) * 100) if total_q_b > 0 else 0
                    bench_rows.append({"Brand": bc, "AI Mentions": cnt,
                                       "Visibility": f"{pct}%", "Note": "Also Detected"})
            st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)

            # ── Why not appearing insight ─────────────────────────────────
            if your_pct == 0 and real_comps:
                top_name, top_cnt = real_comps[0]
                top_pct = round((top_cnt / total_q_b) * 100)
                st.warning(
                    f"**{brand_name} has 0% AI visibility.** "
                    f"The most mentioned brand was **{top_name}** at {top_pct}% of responses. "
                    f"To improve visibility, {brand_name} needs more published content, case studies, "
                    f"and citations online that AI models can learn from."
                )
            elif your_pct > 0 and real_comps:
                top_name, top_cnt = real_comps[0]
                top_pct = round((top_cnt / total_q_b) * 100)
                if top_pct > your_pct:
                    st.info(
                        f"**{brand_name}** has {your_pct}% visibility. "
                        f"**{top_name}** leads at {top_pct}%. "
                        f"Gap to close: {top_pct - your_pct}%."
                    )

            # ── Charts ───────────────────────────────────────────────────
            st.subheader("Visibility Overview")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                tool_chart_data = {
                    tool: data["share_pct"]
                    for tool, data in scores["citation_share_by_tool"].items()
                }
                if tool_chart_data:
                    st.markdown("**Visibility % by AI Model**")
                    chart_df = pd.DataFrame({
                        "AI Model": list(tool_chart_data.keys()),
                        "Visibility %": list(tool_chart_data.values())
                    })
                    st.bar_chart(chart_df.set_index("AI Model"), use_container_width=True, color="#2563eb")

            with chart_col2:
                topic_scores_chart = calculate_citation_share_by_topic(all_results, brand_name)
                if topic_scores_chart:
                    st.markdown("**Visibility % by Topic**")
                    t_df = pd.DataFrame({
                        "Topic": [t[:30] + "..." if len(t) > 30 else t for t in topic_scores_chart.keys()],
                        "Visibility %": [v["share_pct"] for v in topic_scores_chart.values()]
                    })
                    st.bar_chart(t_df.set_index("Topic"), use_container_width=True, color="#16a34a")

            # ── Recommendation Share + Context breakdown ─────────────────
            ctx = scores["context_breakdown"]
            total_runs = scores["total_queries_run"] or 1
            recommended = ctx.get("recommended", 0)
            mentioned = ctx.get("mentioned", 0)
            warned = ctx.get("warned_against", 0)
            not_mentioned = ctx.get("not_mentioned", 0)
            rec_share = round((recommended / total_runs) * 100)
            mention_share = round((mentioned / total_runs) * 100)

            st.subheader("📊 Brand Visibility Breakdown")
            st.caption(
                "**Recommendation Share** = actively named as a top pick. "
                "**Mention Share** = casually listed. "
                "Recommendation Share is the most valuable metric."
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🏆 Recommendation Share", f"{rec_share}%",
                      f"{recommended}/{total_runs} prompts",
                      help="Brand actively recommended as top pick")
            m2.metric("🔵 Mention Share", f"{mention_share}%",
                      f"{mentioned}/{total_runs} prompts",
                      help="Brand mentioned but not as top pick")
            m3.metric("🔴 Warned Against", warned,
                      f"{round((warned/total_runs)*100)}%",
                      help="Brand mentioned with caution")
            m4.metric("⚪ Not Mentioned", not_mentioned,
                      f"{round((not_mentioned/total_runs)*100)}%",
                      help="Brand did not appear")
            if rec_share > 0:
                st.success(f"✅ Your brand is actively recommended in {rec_share}% of AI responses.")
            elif mention_share > 0:
                st.info(f"ℹ️ Your brand appears in {mention_share}% of responses but is not the top recommendation.")
            else:
                st.warning("⚠️ Your brand has 0% AI visibility across all prompts.")

            # ── Visibility per model table ────────────────────────────────
            st.subheader("Visibility % by AI Model")
            tool_rows = []
            for tool, data in scores["citation_share_by_tool"].items():
                tool_rows.append({
                    "AI Model": tool,
                    "Visibility %": f"{data['share_pct']}%",
                    "Brand Mentions": f"{data['mentions']} / {data['total_queries']}",
                })
            st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)

            # ── Per topic breakdown ───────────────────────────────────────
            st.subheader("Visibility % by Topic")
            topic_scores = calculate_citation_share_by_topic(all_results, brand_name)

            for topic, t_data in topic_scores.items():
                pct = t_data['share_pct']
                with st.expander(f"**{topic}** — {pct}% visibility", expanded=True):

                    # Per model breakdown for this topic
                    topic_results = [r for r in all_results if r.get("topic") == topic]
                    models_in_topic = list(set(r["tool"] for r in topic_results))

                    model_cols = st.columns(len(models_in_topic)) if models_in_topic else []
                    for col, model in zip(model_cols, models_in_topic):
                        model_results = [r for r in topic_results if r["tool"] == model]
                        mentions = sum(1 for r in model_results if r["brands_detected"]["target_mentioned"])
                        total = len(model_results)
                        col.metric(model, f"{round((mentions/total)*100) if total else 0}%",
                                   f"{mentions}/{total} prompts")

                    st.write("**Prompts:**")

                    # Per prompt per model table
                    prompt_texts = list(set(r["query"] for r in topic_results))
                    for prompt_text in prompt_texts:
                        prompt_results = [r for r in topic_results if r["query"] == prompt_text]

                        # Build badge row
                        badge_cols = st.columns([3] + [1] * len(models_in_topic))
                        with badge_cols[0]:
                            st.caption(f"+ {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
                        for col, model in zip(badge_cols[1:], models_in_topic):
                            model_result = next((r for r in prompt_results if r["tool"] == model), None)
                            if model_result and model_result["brands_detected"]["target_mentioned"]:
                                col.success("Brand")
                            elif model_result:
                                col.caption("—")
                            else:
                                col.caption("—")

                        # Drill-down expander for each prompt
                        with st.expander(f"View details: {prompt_text[:60]}..."):
                            for r in prompt_results:
                                st.markdown(f"**{r['tool']}**")
                                mentioned = r["brands_detected"]["target_mentioned"]
                                context = r["brands_detected"]["target_context"]
                                position = r["brands_detected"]["target_position"]

                                if mentioned:
                                    st.success(f"Brand mentioned | Context: {context} | Position: #{position}")
                                else:
                                    st.warning("Brand not mentioned")

                                # Linked sites
                                linked = r.get("linked_sites", [])
                                if linked:
                                    st.caption("**Linked Sites:**")
                                    link_rows = [{"#": s["rank"], "Domain": s["domain"], "URL": s["url"]} for s in linked]
                                    st.dataframe(pd.DataFrame(link_rows), use_container_width=True, hide_index=True)

                                # AI Response - highlighted brand mentions
                                response = r.get("response", "")
                                exp_label = "✅ View AI Response (brand found)" if mentioned else "View AI Response"
                                with st.expander(exp_label):
                                    if response:
                                        # Highlight all variants of brand name
                                        import re as _re
                                        highlighted = response
                                        # Try exact name and spaced variants
                                        variants_to_highlight = [brand_name]
                                        if " " not in brand_name:
                                            spaced = _re.sub(r"([a-z])([A-Z])", r"\1 \2", brand_name)
                                            if spaced != brand_name:
                                                variants_to_highlight.append(spaced)
                                        for v in variants_to_highlight:
                                            highlighted = _re.sub(
                                                r"\b" + _re.escape(v) + r"\b",
                                                f"**🟡 {v}**",
                                                highlighted,
                                                flags=_re.IGNORECASE
                                            )
                                        st.markdown(highlighted[:3000])
                                        if mentioned:
                                            st.success(f"✅ {brand_name} appears in this response")
                                    else:
                                        st.caption("No response recorded.")

            # ── Competitor ranking - 3 dynamic categories ───────────────
            st.subheader("Brands Detected in AI Responses")
            st.caption("Brands are automatically classified at runtime by AI — no hardcoding, works for any industry.")

            real_competitors = scores.get("real_competitors", [])
            dominant_platforms = scores.get("dominant_platforms", [])
            government_bodies = scores.get("government_bodies", [])
            total_q = scores["total_queries_run"]

            tab1, tab2, tab3 = st.tabs([
                f"Direct Competitors ({len(real_competitors)})",
                f"Dominant Platforms ({len(dominant_platforms)})",
                f"Government / Official Bodies ({len(government_bodies)})"
            ])

            with tab1:
                st.caption("Commercial tools and services at a similar scale to yours — your real competition.")

                # Always show user-entered competitors even if not detected
                # This gives them a 0% visibility baseline which is useful data
                user_competitors = bd.get("competitors", [])
                detected_names = [b.lower() for b, _ in real_competitors]

                # Add any user competitors that were not detected with 0 mentions
                zero_mention_competitors = [
                    (comp, 0) for comp in user_competitors
                    if comp.strip() and comp.strip().lower() not in detected_names
                ]

                all_competitors_display = list(real_competitors[:20]) + zero_mention_competitors

                if all_competitors_display:
                    comp_df = pd.DataFrame(all_competitors_display, columns=["Brand", "Mentions"])
                    comp_df["Appearance Rate"] = comp_df["Mentions"].apply(
                        lambda x: f"{round((x / total_q) * 100)}%" if total_q > 0 else "0%"
                    )
                    comp_df["Status"] = comp_df["Mentions"].apply(
                        lambda x: "✅ Detected" if x > 0 else "⚪ Not mentioned"
                    )
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                    if zero_mention_competitors:
                        st.caption(f"⚪ {len(zero_mention_competitors)} competitor(s) you entered were not mentioned by AI in this run.")
                else:
                    st.info("No direct competitors detected. Add known competitors in Step 1 for better tracking.")

            with tab2:
                st.caption("Large established commercial platforms. These dominate AI responses but are not your direct competition.")
                if dominant_platforms:
                    dom_df = pd.DataFrame(
                        [(b, c, f"{r}%") for b, c, r in dominant_platforms[:15]],
                        columns=["Brand", "Mentions", "Appearance Rate"]
                    )
                    st.dataframe(dom_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No dominant platforms detected.")

            with tab3:
                st.caption("Government agencies, regulatory bodies, and official databases. These appear as authoritative references, not competitors.")
                if government_bodies:
                    gov_df = pd.DataFrame(
                        [(b, c, f"{r}%") for b, c, r in government_bodies[:15]],
                        columns=["Organization", "Mentions", "Appearance Rate"]
                    )
                    st.dataframe(gov_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No government or official bodies detected.")

            # ── Brand mention context ────────────────────────────────────
            st.subheader("How Your Brand Was Mentioned")
            ctx = scores["context_breakdown"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recommended", ctx.get("recommended", 0))
            c2.metric("Mentioned", ctx.get("mentioned", 0))
            c3.metric("Warned Against", ctx.get("warned_against", 0))
            c4.metric("Not Mentioned", ctx.get("not_mentioned", 0))

            # ── Full results table ────────────────────────────────────────
            st.subheader("Full Results Table")
            rows = []
            for r in all_results:
                rows.append({
                    "Topic": r.get("topic", ""),
                    "AI Model": r["tool"],
                    "Prompt": r["query"],
                    "Brand Mentioned": "Yes" if r["brands_detected"]["target_mentioned"] else "No",
                    "Position": r["brands_detected"]["target_position"] or "-",
                    "Context": r["brands_detected"]["target_context"],
                    "Linked Sites": len(r.get("linked_sites", [])),
                    "Brands Found": ", ".join(r["brands_detected"]["all_brands"][:5]),
                    "Response Preview": r["response"][:200] + "..." if len(r.get("response", "")) > 200 else r.get("response", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True,
                         column_config={"Prompt": st.column_config.TextColumn(width="large"),
                                        "Response Preview": st.column_config.TextColumn(width="large")})

            # ── CSV download ──────────────────────────────────────────────
            csv_rows = []
            for r in all_results:
                csv_rows.append({
                    "Topic": r.get("topic", ""),
                    "AI Model": r["tool"],
                    "Prompt": r["query"],
                    "Brand Mentioned": "Yes" if r["brands_detected"]["target_mentioned"] else "No",
                    "Position": r["brands_detected"]["target_position"] or "-",
                    "Context": r["brands_detected"]["target_context"],
                    "Brands Found": ", ".join(r["brands_detected"]["all_brands"][:5]),
                    "Linked Sites": "; ".join([s["url"] for s in r.get("linked_sites", [])]),
                    "Full Response": r.get("response", ""),
                })
            csv = pd.DataFrame(csv_rows).to_csv(index=False)
            st.download_button(
                label="Download Full CSV",
                data=csv,
                file_name=f"citation_{brand_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

            # ── Re-run options ────────────────────────────────────────────
            st.divider()
            st.subheader("Run Again")

            rerun_col1, rerun_col2 = st.columns([1, 1])

            with rerun_col1:
                if st.button("🔄 Run Again with Same Settings", type="primary", use_container_width=True):
                    st.session_state.run_complete = False
                    st.session_state.all_results = []
                    st.rerun()

            with rerun_col2:
                # Country switcher - keeps all topics and prompts, just changes country
                st.markdown("**🌍 Re-run with a different country**")
                _country_options = ["United States", "United Kingdom", "Canada", "Australia",
                                    "Germany", "India", "Pakistan", "Global"]
                _current_country = bd.get("country", "United States")
                _current_idx = _country_options.index(_current_country) if _current_country in _country_options else 0
                new_country = st.selectbox(
                    "Select country",
                    options=_country_options,
                    index=_current_idx,
                    key="rerun_country_select",
                    label_visibility="collapsed"
                )
                if st.button("🚀 Re-run with this country", use_container_width=True):
                    # Update country in brand_data and clear cached website text
                    # so topics/prompts regenerate with new country context
                    st.session_state.brand_data["country"] = new_country
                    # Clear cached prompts so they regenerate with new country
                    st.session_state.prompts_by_topic = {}
                    st.session_state.selected_prompts = {}
                    st.session_state.topics = []
                    st.session_state.selected_topics = []
                    st.session_state.run_complete = False
                    st.session_state.all_results = []
                    st.session_state.step = 2
                    st.rerun()