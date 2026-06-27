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


def resolve_brand_terms(brand_data: dict) -> dict:
    """
    Context-aware term resolver. Reads website + form data, identifies ambiguous
    abbreviations, and resolves them to their correct meaning for THIS brand.
    Runs once per session, result cached in brand_data["_resolved_terms"].
    """
    import json as _json

    cached = brand_data.get("_resolved_terms", {})
    if cached:
        return cached

    website_text = brand_data.get("_website_text", "")
    products = ", ".join(brand_data.get("products", []))
    customers = ", ".join(brand_data.get("customers", []))
    key_features = ", ".join(brand_data.get("key_features", []))
    business_type = brand_data.get("business_type", "")
    brand_name = brand_data.get("name", "")

    AMBIGUOUS_TERMS = {
        "GEO": ["Generative Engine Optimization (marketing)", "Geographic / Geospatial"],
        "LLM": ["Large Language Model (AI)", "Master of Laws (legal degree)"],
        "IP": ["Intellectual Property", "Internet Protocol (networking)"],
        "PR": ["Public Relations / Digital PR", "Pull Request (software development)"],
        "ML": ["Machine Learning", "Markup Language"],
        "AR": ["Augmented Reality", "Accounts Receivable"],
        "NLP": ["Natural Language Processing", "No List Price"],
        "AEO": ["Answer Engine Optimization", "Account Executive Operations"],
        "LLMO": ["Large Language Model Optimization", "other"],
        "SEO": ["Search Engine Optimization", "Securities and Exchange Organization"],
        "CRO": ["Conversion Rate Optimization", "Chief Revenue Officer"],
        "GTM": ["Go-To-Market strategy", "Google Tag Manager"],
        "SEM": ["Search Engine Marketing", "Scanning Electron Microscope"],
        "PPC": ["Pay Per Click advertising", "Power PC"],
        "ATP": ["Authorized Training Partner", "Adenosine Triphosphate (biology)"],
        "NSE": ["Network Security Expert (Fortinet certification)", "National Stock Exchange"],
        "JNCIA": ["Juniper Networks Certified Internet Associate", "other"],
        "CDP": ["Customer Data Platform", "Continuing Development Programme"],
        "CMS": ["Content Management System", "Centers for Medicare Services"],
        "ROI": ["Return on Investment", "other"],
        "SLA": ["Service Level Agreement", "other"],
        "DTC": ["Direct to Consumer", "other"],
        "B2B": ["Business to Business", "other"],
        "B2C": ["Business to Consumer", "other"],
    }

    brand_context = ""
    if website_text:
        brand_context += "WEBSITE CONTENT:\n" + website_text[:2000] + "\n\n"
    brand_context += (
        "Brand: " + brand_name + "\n"
        "Business type: " + business_type + "\n"
        "Products/Services: " + products + "\n"
        "Customers: " + customers + "\n"
        "Key features: " + key_features + "\n"
    )

    import re as _re_terms
    all_brand_text = brand_context + products + " " + key_features
    # Only check terms that appear as standalone uppercase words (not substrings)
    terms_to_check = {}
    for term, meanings in AMBIGUOUS_TERMS.items():
        # Match whole word only — "PR" must not match inside "prior" or "search"
        pattern = _re_terms.compile(r'\b' + _re_terms.escape(term) + r'\b')
        if pattern.search(all_brand_text):
            terms_to_check[term] = meanings

    if not terms_to_check:
        brand_data["_resolved_terms"] = {}
        return {}

    terms_list = "\n".join(
        "- " + term + ": could mean " + " OR ".join(meanings)
        for term, meanings in terms_to_check.items()
    )

    resolver_prompt = (
        "You are resolving abbreviations for a brand based on their website and business context.\n\n"
        "BRAND CONTEXT:\n"
        + brand_context
        + "\nAMBIGUOUS TERMS FOUND IN THIS BRAND DATA:\n"
        + terms_list
        + "\n\nFor each term, decide which meaning applies to THIS brand based on their context.\n"
        + 'Return ONLY a JSON object: {"TERM": "full expanded meaning", ...}\n'
        + "Example for a marketing agency: "
        + '{"GEO": "Generative Engine Optimization", "PR": "Digital Public Relations"}\n'
        + "Example for a network IT company: "
        + '{"IP": "Internet Protocol", "ATP": "Authorized Training Partner"}\n'
        + "Only include terms clearly relevant to this brand.\n"
        + "Use the brand own language from their website.\n"
        + "Respond ONLY with valid JSON. No markdown."
    )

    try:
        raw = _call_ai_for_json(resolver_prompt)
        try:
            resolved = _json.loads(raw)
        except Exception:
            parsed = _parse_json_list(raw)
            resolved = parsed[0] if parsed and isinstance(parsed[0], dict) else {}
        resolved = {
            k: v for k, v in resolved.items()
            if isinstance(k, str) and isinstance(v, str) and v.strip()
        }
        brand_data["_resolved_terms"] = resolved
        return resolved
    except Exception:
        brand_data["_resolved_terms"] = {}
        return {}


def build_term_glossary(resolved_terms: dict) -> str:
    """Builds glossary string to inject into every AI prompt."""
    if not resolved_terms:
        return ""
    lines = ["TERM GLOSSARY (use these exact meanings, never use the abbreviation alone):"]
    for term, meaning in resolved_terms.items():
        lines.append("- " + term + " = " + meaning + " (always write the full phrase, never just '" + term + "')")
    return "\n".join(lines) + "\n\n"


def ai_generate_topics(brand_data: dict) -> list:
    """
    Topic generation following strict priority order:
    1. Products and Services — use exact phrases the user typed
    2. Key Features — use as topic modifiers
    3. Business Type — determines category word (tool/agency/training)
    4. Website scan — gap-filler only, not primary source
    """
    import json as _json

    business_type = brand_data.get("business_type", "")
    products_list  = brand_data.get("products", [])
    customers_list = brand_data.get("customers", [])
    features_list  = brand_data.get("key_features", [])
    competitors_list = brand_data.get("competitors", [])
    domain     = brand_data.get("domain", "")
    brand_name = brand_data["name"]
    country    = brand_data.get("country", "")

    website_text = brand_data.get("_website_text", "")
    if not website_text and domain:
        website_text = fetch_brand_website(domain)

    # ── TERM RESOLUTION ───────────────────────────────────────────────────────
    resolved_terms = resolve_brand_terms(brand_data)
    term_glossary  = build_term_glossary(resolved_terms)

    # ── PRIORITY 3: Derive category word from business type ───────────────────
    bt_lower = business_type.lower()
    if any(w in bt_lower for w in ["training", "education", "bootcamp", "courses"]):
        category_word    = "training"
        category_options = "training, courses, programs, providers, companies"
    elif any(w in bt_lower for w in ["agency", "service", "studio", "consultancy"]):
        category_word    = "agency"
        category_options = "agency, agencies, services, firm, company, provider"
    elif any(w in bt_lower for w in ["saas", "software", "platform", "tool"]):
        category_word    = "software"
        category_options = "software, tool, platform, solution, product"
    else:
        category_word    = "provider"
        category_options = "provider, company, solution, platform, service"

    # ── PRIORITY 1: Extract exact product phrases from user input ─────────────
    products_clean = [p.strip() for p in products_list if len(p.strip()) >= 3][:10]

    # ── PRIORITY 2: Extract top differentiators ───────────────────────────────
    features_clean = [f.strip() for f in features_list if len(f.strip()) >= 3][:6]

    # ── PRIORITY 4: Website scan as gap-filler only ───────────────────────────
    website_context = ""
    if website_text:
        website_context = (
            "\nWEBSITE CONTENT (use only to fill gaps not already in the form above):\n"
            + website_text[:1500] + "\n"
        )

    # ── Internal summary — prevents category drift ────────────────────────────
    internal_summary = (
        "INTERNAL SUMMARY (use this as generation context):\n"
        "This brand is a " + business_type + ".\n"
        "Their specific offerings are: " + ", ".join(products_clean[:3]) + ".\n"
        "Their strongest differentiators are: " + ", ".join(features_clean[:3]) + ".\n"
        "Their primary buyers are: " + ", ".join(customers_list[:3]) + ".\n\n"
    )

    country_topic = ""
    if country and country.lower() not in ["global", "united states", ""]:
        country_topic = "- Include 1 topic mentioning '" + country + "' for local relevance\n"

    products_str  = "\n".join("- " + p for p in products_clean)
    features_str  = "\n".join("- " + f for f in features_clean)

    topic_prompt = (
        "You track brand visibility in AI tools like ChatGPT and Perplexity.\n\n"
        + term_glossary
        + internal_summary
        + "PRIORITY 1 — PRODUCTS AND SERVICES (user typed these exactly — use their words):\n"
        + products_str + "\n\n"
        + "PRIORITY 2 — KEY DIFFERENTIATORS (use as topic modifiers):\n"
        + features_str + "\n\n"
        + "PRIORITY 3 — BUSINESS TYPE: " + business_type + "\n"
        + "Category words to use: " + category_options + "\n"
        + "NEVER use category words that contradict the business type.\n"
        + "e.g. never write 'patent search firms' for a SaaS company — write 'patent search software'\n\n"
        + website_context
        + "\nGenerate 5 short specific topics a buyer types when searching for this brand's category.\n\n"
        + "STRICT RULES:\n"
        + "1. Use the EXACT phrases from PRIORITY 1 above — do not summarise into category names\n"
        + "   If user typed 'patent search API' write a topic around 'patent search API'\n"
        + "   If user typed 'LinkedIn ghostwriting' write a topic around 'LinkedIn ghostwriting'\n"
        + "   If user typed 'Fortinet authorized partner' write a topic around 'Fortinet authorized partner'\n"
        + "2. Combine product phrases with differentiator modifiers where natural\n"
        + "   e.g. 'free AI patent search tool' (product: patent search + differentiator: free + AI)\n"
        + "   e.g. 'open source patent search platform' (product: patent search + differentiator: open source)\n"
        + "3. Every topic must use a category word matching PRIORITY 3 above\n"
        + "4. Topics are SHORT — 3-7 words. No sentences.\n"
        + "5. VARY endings — do not end all topics with the same word\n"
        + "6. No qualifiers: no 'best', 'top', 'leading'\n"
        + "7. No '" + brand_name + "' in any topic\n"
        + "8. EXPAND abbreviated terms using TERM GLOSSARY — never abbreviate\n"
        + country_topic
        + "\nGOOD EXAMPLES for a free open-source patent search SaaS:\n"
        + "- 'free AI patent search tool'\n"
        + "- 'open source patent search platform'\n"
        + "- 'patent search API for startups'\n"
        + "- 'semantic prior art search software'\n"
        + "- 'AI patent classification tool'\n\n"
        + "BAD EXAMPLES for the same brand:\n"
        + "- 'patent search services' (wrong category word — it is software not services)\n"
        + "- 'prior art search firms' (wrong category word — it is software not firms)\n"
        + "- 'patent research solutions' (vague — does not use specific product phrases)\n\n"
        + "GOOD EXAMPLES for a B2B SaaS content agency:\n"
        + "- 'B2B content marketing agency'\n"
        + "- 'SaaS content marketing services'\n"
        + "- 'LinkedIn ghostwriting for B2B'\n"
        + "- 'Generative Engine Optimization agency for SaaS'\n"
        + "- 'thought leadership content creation'\n\n"
        + "For each topic include one sentence of buyer intent.\n"
        + 'Respond ONLY with JSON: [{"topic": "...", "intent": "..."}, ...]\n'
        + "No markdown."
    )

    raw = _call_ai_for_json(topic_prompt)
    parsed_raw = _parse_json_list(raw)
    brand_lower = brand_name.lower()
    topics = []
    topic_intents = {}

    for item in parsed_raw:
        if isinstance(item, dict):
            t      = item.get("topic", "").strip()
            intent = item.get("intent", "").strip()
            if t:
                topics.append(t)
                if intent:
                    topic_intents[t] = intent
        elif isinstance(item, str) and item.strip():
            topics.append(item.strip())

    if "brand_data" in st.session_state:
        st.session_state.brand_data["_topic_intents"] = topic_intents

    # Post-generation: expand abbreviations (whole-word only)
    import re as _re_expand
    for abbr, expansion in (resolved_terms or {}).items():
        pat = _re_expand.compile(r'\b' + _re_expand.escape(abbr) + r'\b', _re_expand.IGNORECASE)
        topics = [pat.sub(expansion, t) for t in topics]

    # Fix plural misspellings
    import re as _re_tp
    PLURAL_FIXES_T = {
        r'\bagencys\b': 'agencies', r'\bcompanys\b': 'companies',
        r'\bprovidors\b': 'providers', r'\bsoftwares\b': 'software',
    }
    def fix_tp(text):
        for pat, rep in PLURAL_FIXES_T.items():
            text = _re_tp.sub(pat, rep, text, flags=_re_tp.IGNORECASE)
        return text

    bad_patterns = [
        "strategies", "tips", "how to", "examples", "guide", "tutorial",
        "best practices", "introduction", "overview", "explained",
        "what is", "benefits of", "advantages of",
    ]

    filtered = []
    for t in topics:
        t = fix_tp(t.strip())
        if not t or brand_lower in t.lower():
            continue
        if any(bp in t.lower() for bp in bad_patterns):
            continue
        filtered.append(t)

    # Fallback: build directly from user's product phrases
    if len(filtered) < 3:
        for p in products_clean[:6]:
            fallback = fix_tp(p + " " + category_word)
            if brand_lower not in fallback.lower() and fallback not in filtered:
                filtered.append(fallback)
            if len(filtered) >= 5:
                break

    return filtered[:5]



def ai_generate_prompts(topic: str, brand_data: dict) -> list:
    """
    Prompt generation with:
    - Fix 2a: Persona rotation — each prompt uses a different buyer from Target Customers
    - Fix 2b: Differentiator inclusion — key features appear in prompts naturally

    5-prompt structure per topic:
    1. Bare topic keyword
    2-3. Persona-based first-person prompts (different buyer each)
    4. Persona-based with differentiator reference
    5. Comparison or recommendation prompt
    """
    import json as _json

    brand_name     = brand_data.get("name", "")
    products_list  = brand_data.get("products", [])
    customers_list = brand_data.get("customers", [])
    features_list  = brand_data.get("key_features", [])
    business_type  = brand_data.get("business_type", "")
    competitors    = brand_data.get("competitors", [])
    domain         = brand_data.get("domain", "")

    website_text = brand_data.get("_website_text", "")
    if not website_text and domain:
        website_text = fetch_brand_website(domain)

    resolved_terms = brand_data.get("_resolved_terms", {})
    term_glossary  = build_term_glossary(resolved_terms)

    # ── Category word from business type ──────────────────────────────────────
    bt_lower = business_type.lower()
    if any(w in bt_lower for w in ["training", "education", "bootcamp"]):
        category_word = "training provider"
        avoid_word    = "NEVER use 'course' or 'tutorial' — ask for training companies\n"
    elif any(w in bt_lower for w in ["agency", "service", "studio"]):
        category_word = "agency"
        avoid_word    = "NEVER use 'tool' or 'software' — this is a service not a product\n"
    elif any(w in bt_lower for w in ["saas", "software", "platform"]):
        category_word = "software"
        avoid_word    = "NEVER use 'agency' or 'consultant' — this is software not a service\n"
    else:
        category_word = "provider"
        avoid_word    = ""

    brand_lower = brand_name.lower()

    # ── Fix 2a: Rotate through ALL customer personas ──────────────────────────
    # Each prompt in the 5 gets a DIFFERENT persona — no repeats within a topic
    customers_clean = [c.strip() for c in customers_list if c.strip()]
    if not customers_clean:
        customers_clean = ["professional", "business owner", "manager", "team lead", "researcher"]

    # Use topic hash to start at a different offset per topic
    topic_hash = sum(ord(c) for c in topic) % max(len(customers_clean), 1)
    rotated_customers = customers_clean[topic_hash:] + customers_clean[:topic_hash]
    # Ensure we have at least 4 unique personas for prompts 2-5
    while len(rotated_customers) < 4:
        rotated_customers = rotated_customers + rotated_customers
    personas = rotated_customers[:4]  # 4 unique personas for prompts 2-5

    # ── Fix 2b: Top differentiators to include in prompts ────────────────────
    features_clean = [f.strip() for f in features_list if f.strip()][:6]
    top_features   = features_clean[:3] if features_clean else []

    # ── Comparison setup ──────────────────────────────────────────────────────
    comp_for_comparison = []
    if competitors:
        offset = topic_hash % max(len(competitors), 1)
        rotated_comp = competitors[offset:] + competitors[:offset]
        comp_for_comparison = rotated_comp[:1]

    if comp_for_comparison:
        comparison_line = (
            "PROMPT 5 — COMPARISON:\n"
            "Ask how " + brand_name + " compares to " + comp_for_comparison[0] + " for this use case.\n"
            "This is the ONE prompt where you MAY use the brand name '" + brand_name + "'.\n"
            "CORRECT: 'How does " + brand_name + " compare to " + comp_for_comparison[0] + " for [topic use case]?'\n"
            "CORRECT: '" + brand_name + " vs " + comp_for_comparison[0] + " — which is better for [specific need]?'\n"
        )
    else:
        comparison_line = (
            "PROMPT 5 — RECOMMENDATION:\n"
            "Ask which " + category_word + " is best for the specific use case in the topic.\n"
            "CORRECT: 'Which " + category_word + " would you recommend for [specific need from topic]?'\n"
        )

    # ── Recommendation ask rotation ───────────────────────────────────────────
    REC_ASKS = [
        "What do you recommend?",
        "What would you suggest?",
        "Which one should I go with?",
        "Any recommendations?",
        "Who should I choose?",
        "What's your pick?",
        "Which would you choose?",
        "Can you point me in the right direction?",
    ]
    rec_ask_1 = REC_ASKS[topic_hash % len(REC_ASKS)]
    rec_ask_2 = REC_ASKS[(topic_hash + 2) % len(REC_ASKS)]
    rec_ask_3 = REC_ASKS[(topic_hash + 4) % len(REC_ASKS)]

    # ── Context for the prompt ────────────────────────────────────────────────
    context = (
        "Products: " + ", ".join(products_list[:5]) + "\n"
        + "Key differentiators: " + ", ".join(features_clean[:4]) + "\n"
        + "Buyers: " + ", ".join(customers_clean[:5]) + "\n"
    )

    features_instruction = ""
    if top_features:
        features_instruction = (
            "\nDIFFERENTIATOR INCLUSION RULE:\n"
            "The brand's key differentiators are: " + ", ".join(top_features) + "\n"
            "At least 1 of prompts 2-4 must naturally reference one of these differentiators.\n"
            "Example: if differentiator is 'free', one persona should say 'I need a free solution'.\n"
            "Example: if differentiator is 'authorized partner', one persona should say 'I need an authorized provider'.\n"
            "Example: if differentiator is 'open source', one persona should mention 'open source'.\n"
        )

    prompt = (
        "You generate search prompts for AI visibility tracking.\n\n"
        + term_glossary
        + "BRAND CONTEXT:\n"
        + context
        + features_instruction
        + "\nTOPIC: \"" + topic + "\"\n"
        + "SOLUTION TYPE: " + category_word + "\n"
        + avoid_word + "\n"
        + "PERSONAS TO USE (one per prompt, no repeats):\n"
        + "Prompt 2 persona: " + personas[0] + "\n"
        + "Prompt 3 persona: " + personas[1] + "\n"
        + "Prompt 4 persona: " + personas[2] + "\n"
        + "Prompt 5 persona: " + (personas[3] if len(personas) > 3 else personas[0]) + "\n\n"

        + "GENERATE EXACTLY 5 PROMPTS IN THIS EXACT STRUCTURE:\n\n"

        + "PROMPT 1 — PLAIN KEYWORD:\n"
        + "The topic name exactly as written. Nothing added. No question mark.\n"
        + "CORRECT: '" + topic + "'\n\n"

        + "PROMPT 2 — PERSONA (buyer: " + personas[0] + "):\n"
        + "First person singular. MUST say 'I am a " + personas[0] + "' — singular not plural.\n"
        + "References one of these differentiators naturally: " + ", ".join(top_features[:2] if top_features else ["the solution"]) + ".\n"
        + "Ends with exactly: 'What would you suggest?' or 'Who should I choose?'\n"
        + "CORRECT: 'I am a " + personas[0] + " and I need [differentiator] for [topic use case]. What would you suggest?'\n"
        + "WRONG: 'I am a " + personas[0] + "s' (never pluralise the persona)\n"
        + "WRONG: ends with anything other than 'What would you suggest?' or 'Who should I choose?'\n\n"

        + "PROMPT 3 — PERSONA (buyer: " + personas[1] + ", DIFFERENT from prompt 2):\n"
        + "First person singular. MUST say 'I am a " + personas[1] + "' — singular not plural.\n"
        + "Different differentiator angle from prompt 2. Use: " + ", ".join(top_features[1:3] if len(top_features) > 1 else ["the solution"]) + ".\n"
        + "Ends with exactly: 'Any recommendations?' or 'Which would you go with?'\n"
        + "CORRECT: 'I am a " + personas[1] + " looking for [different angle on topic]. Any recommendations?'\n"
        + "WRONG: same persona as prompt 2\n"
        + "WRONG: ends with anything other than 'Any recommendations?' or 'Which would you go with?'\n\n"

        + "PROMPT 4 — HOW-TO OR INFORMATIONAL:\n"
        + "Third person or neutral. NO persona. NO 'I am a'.\n"
        + "Starts with 'How do I' OR 'What is the best way to' OR 'Which " + category_word + "s offer'.\n"
        + "Focuses on the use case from the topic, not the buyer role.\n"
        + "CORRECT: 'How do I find a " + category_word + " that offers [topic use case]?'\n"
        + "CORRECT: 'Which " + category_word + "s offer [key feature from topic]?'\n"
        + "CORRECT: 'What is the best way to [use case from topic]?'\n"
        + "WRONG: starts with 'I am a' or contains a buyer persona\n\n"

        + "PROMPT 5 — COMPARISON:\n"
        + "Directly compares " + brand_name + " against " + (comp_for_comparison[0] if comp_for_comparison else "a competitor") + ".\n"
        + "This is the ONLY prompt where you may use the brand name '" + brand_name + "'.\n"
        + "CORRECT: 'How does " + brand_name + " compare to " + (comp_for_comparison[0] if comp_for_comparison else "alternatives") + " for [topic use case]?'\n"
        + "CORRECT: '" + brand_name + " vs " + (comp_for_comparison[0] if comp_for_comparison else "competitors") + " — which is better for [topic]?'\n\n"

        + "ABSOLUTE RULES:\n"
        + "- ALWAYS use singular persona: 'I am a researcher' NEVER 'I am a researchers'\n"
        + "- NEVER repeat the same persona in prompts 2 and 3\n"
        + "- NEVER generate advice-seeking prompts ('what should I look for')\n"
        + "- NEVER use year numbers\n"
        + "- NEVER abbreviate terms from TERM GLOSSARY above\n"
        + "- Every prompt must cause ChatGPT to name specific " + category_word + "s\n\n"

        + "GOOD EXAMPLE SET for topic 'free AI patent search tool':\n"
        + "1. 'free AI patent search tool'\n"
        + "2. 'I am a startup founder and I need a free AI tool to search patents before filing. What would you suggest?'\n"
        + "3. 'I am an inventor looking for an open source patent search platform with no login required. Any recommendations?'\n"
        + "4. 'Which patent search tools offer natural language queries and free access?'\n"
        + "5. 'How does PQAI compare to Google Patents for prior art search?'\n\n"

        + "BAD EXAMPLE SET (never do this):\n"
        + "1. 'What are the best AI tools for patent searching?' (not the bare topic)\n"
        + "2. 'I am a startup founders' (pluralised persona — WRONG)\n"
        + "3. 'I am a startup founder...' again (same persona repeated from prompt 2 — WRONG)\n"
        + "4. 'I am a researcher looking for...' (has persona — prompt 4 must be neutral)\n"
        + "5. 'Which tools are best?' (does not name brand vs competitor)\n\n"

        + "Return ONLY a JSON array of exactly 5 strings.\n"
        + "No markdown. No explanation.\n"
        + '["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"]'
    )

    import re as _re_year
    year_pattern = _re_year.compile(r'\b(20[0-9]{2})\b')

    CITATION_TRIGGERS = [
        "recommend", "suggest", "which", "best", "top", "leading",
        "compare", "vs", "versus", "alternative", "alternatives",
        "agency", "agencies", "firm", "firms", "company", "companies",
        "tool", "tools", "software", "platform", "provider", "providers",
        "who do you", "any recommendations", "what do you", "which one",
        "who offers", "who makes", "what would you", "what's your pick",
        "i am a", "i'm a", "i need",
    ]
    STATEMENT_SIGNALS = [
        "how to", "what is", "what are the benefits", "what does",
        "explain", "guide", "tutorial", "tips for", "ways to",
        "understanding", "introduction to", "overview of",
        "why is", "when should", "how does", "what makes",
        "how can i improve", "how do i", "what should i look for",
        "what criteria", "key features of", "what factors",
    ]

    PLURAL_FIXES_P = {
        r'\bagencys\b': 'agencies', r'\bcompanys\b': 'companies',
        r'\bprovidors\b': 'providers', r'\bsoftwares\b': 'software',
        r'\bsolution s\b': 'solutions',
    }
    import re as _re_plural

    def fix_plurals(text):
        for pat, rep in PLURAL_FIXES_P.items():
            text = _re_plural.sub(pat, rep, text, flags=_re_plural.IGNORECASE)
        return text

    def is_citation_producing(p):
        pl = p.lower().strip()
        has_trigger = any(t in pl for t in CITATION_TRIGGERS)
        is_informational = any(s in pl for s in STATEMENT_SIGNALS)
        is_persona = pl.startswith("i ") or pl.startswith("i'm") or pl.startswith("i am")
        if is_persona:
            return any(ask in pl for ask in [
                "recommend", "suggest", "what do you", "any recommendation",
                "who should", "which one", "what would", "what's your"
            ])
        return has_trigger and not is_informational

    def _too_similar(a, b):
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return False
        overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
        return overlap > 0.85

    def _is_acceptable(p, existing):
        if not p or len(p.strip()) < 5:
            return False
        if brand_lower in p.lower() and "vs" not in p.lower() and "compare" not in p.lower():
            return False
        if any(_too_similar(p, e) for e in existing):
            return False
        return True

    def _clean(p):
        p = year_pattern.sub("", p).strip()
        p = fix_plurals(p)

        def _singularise(w):
            if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
            if w.endswith("ers") and not w.endswith("eers"): return w[:-1]
            if w.endswith("ants") or w.endswith("ents"): return w[:-1]
            if w.endswith("ors") and len(w) > 4: return w[:-1]
            if w.endswith("sts"): return w[:-1]
            return w

        def _fix_persona(text):
            words = text.split()
            result = []
            for i, w in enumerate(words):
                if (i >= 3
                        and words[i-3].lower() == "i"
                        and words[i-2].lower() == "am"
                        and words[i-1].lower() in ("a", "an")):
                    w = _singularise(w)
                elif (i >= 4
                      and words[i-4].lower() == "i"
                      and words[i-3].lower() == "am"
                      and words[i-2].lower() in ("a", "an")):
                    w = _singularise(w)
                result.append(w)
            return " ".join(result)

        p = _fix_persona(p)
        return p
    # Generate prompts
    raw_prompts = _call_ai_for_json(prompt)
    parsed = _parse_json_list(raw_prompts)
    generated = [_clean(p) for p in parsed if isinstance(p, str) and p.strip()]

    # Retry once if too few
    if len(generated) < 4:
        raw2   = _call_ai_for_json(prompt)
        parsed2 = _parse_json_list(raw2)
        for p in parsed2:
            if isinstance(p, str) and p.strip():
                generated.append(_clean(p))

    # Bare topic always first
    topic_as_prompt = topic.strip()
    result = [topic_as_prompt]

    for p in generated:
        if _is_acceptable(p, result):
            result.append(p)
        if len(result) >= 5:
            break

    # Guaranteed fallbacks — natural, never weird
    _cw = category_word
    _buyer0 = personas[0] if personas else "professional"
    _buyer1 = personas[1] if len(personas) > 1 else "manager"
    guaranteed = [
        "I am a " + _buyer0 + " and I need " + topic + ". " + rec_ask_1,
        "I am a " + _buyer1 + " looking for " + topic + ". " + rec_ask_2,
        "which " + _cw + " is best for " + topic + "?",
        "best " + topic + " " + _cw,
        "top rated " + topic + " " + _cw + "s",
        "recommended " + topic + " for " + (_buyer0.split()[0] if _buyer0 else "teams"),
    ]
    for fb in guaranteed:
        if len(result) >= 5:
            break
        fb = _clean(fb)
        if _is_acceptable(fb, result):
            result.append(fb)

    # Hard pad — should never reach here
    while len(result) < 5:
        result.append("best " + topic)

    return result[:5]



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
_cur_step = st.session_state.step if isinstance(st.session_state.step, int) else 1
for i, (col, name) in enumerate(zip(cols, step_names), 1):
    with col:
        if i < _cur_step:
            st.success(f"✓ {name}")
        elif i == _cur_step:
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

        # ── Step 1: Fetch website content ────────────────────────────────────
        domain = bd.get("domain", "")
        iframe_url = domain if domain.startswith("http") else f"https://{domain}"

        st.markdown("#### 🌐 Reading your website...")
        st.caption(f"Fetching content from: **{iframe_url}**")

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

            tool_responses = run_selected_tools(q["query"], active_tools, country=bd.get("country", ""))

            for tool_name, raw_response in tool_responses.items():
                # Handle dict (web search) and plain string responses
                if isinstance(raw_response, dict):
                    response_text = raw_response.get("text", "") or ""
                    web_sources = raw_response.get("sources", [])
                    web_searched = raw_response.get("web_searched", False)
                else:
                    response_text = str(raw_response) if raw_response else ""
                    web_sources = []
                    web_searched = False

                # Auto-retry once on empty response
                if not response_text.strip() or len(response_text.strip()) < 30:
                    import time as _time
                    _time.sleep(3)
                    retry_resp = run_selected_tools(q["query"], [tool_name], country=bd.get("country", ""))
                    retry_raw = retry_resp.get(tool_name, "")
                    if isinstance(retry_raw, dict):
                        response_text = retry_raw.get("text", "") or ""
                        web_sources = retry_raw.get("sources", [])
                        web_searched = retry_raw.get("web_searched", False)
                    elif retry_raw:
                        response_text = str(retry_raw)

                if not response_text.strip():
                    call_count += 1
                    progress_bar.progress(min(call_count / total_calls, 1.0))
                    continue

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
                brand_data_detected["all_brands"] = [
                    b for b in brand_data_detected.get("all_brands", [])
                    if not is_false_positive_brand(b)
                ]
                linked_sites = web_sources if web_sources else extract_linked_sites(response_text)

                all_results.append({
                    "query": q["query"],
                    "topic": q["topic"],
                    "category": q["category"],
                    "query_group": "C",
                    "tool": tool_name,
                    "response": response_text,
                    "brands_detected": brand_data_detected,
                    "linked_sites": linked_sites,
                    "web_searched": web_searched,
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
            c3.metric("Avg Position", "N/A", help="Position tracking removed — LLM position detection was unreliable")
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
                                # Final safety check: verify brand actually in response text
                                import re as _re_check
                                _resp_text = r.get("response", "")
                                _brand_in_text = bool(_re_check.search(
                                    r"\b" + _re_check.escape(brand_name) + r"\b",
                                    _resp_text, _re_check.IGNORECASE
                                ))
                                if mentioned and not _brand_in_text:
                                    mentioned = False

                                if mentioned:
                                    st.success(f"✅ {brand_name} appears in this response | Context: {context}")
                                else:
                                    st.warning("Brand not mentioned")

                                # Web sources — shown like ChatGPT citations
                                linked = r.get("linked_sites", [])
                                web_searched_r = r.get("web_searched", False)
                                src_count = len(linked)

                                if web_searched_r and src_count > 0:
                                    st.markdown(f"🌐 **Web searched** · 📎 **{src_count} source{'s' if src_count != 1 else ''} used**")
                                elif web_searched_r:
                                    st.markdown("🌐 **Web searched**")

                                # AI Response expander
                                response = r.get("response", "")
                                if web_searched_r and src_count > 0:
                                    exp_label = (f"✅ View AI Response + {src_count} sources (brand found)"
                                                 if mentioned else f"View AI Response + {src_count} sources")
                                else:
                                    exp_label = "✅ View AI Response (brand found)" if mentioned else "View AI Response"
                                with st.expander(exp_label):
                                    # Show sources inside expander
                                    if web_searched_r and linked:
                                        st.markdown("**🌐 Web Sources Used**")
                                        st.caption("Response grounded in live web search — same as ChatGPT web interface.")
                                        for s in linked[:8]:
                                            title = s.get("title") or s.get("domain", "Source")
                                            url = s.get("url", "")
                                            domain = s.get("domain", "")
                                            if url:
                                                st.markdown(f"&nbsp;&nbsp;📎 [{title}]({url}) `{domain}`")
                                        st.divider()
                                    if response:
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

            # ── Competitor visibility (no tabs, clean table) ──────────────
            user_competitors = bd.get("competitors", [])
            detected_names_map = {b.lower(): c for b, c in real_competitors}
            comp_rows = [{"Brand": f"🎯 {brand_name} (You)", "AI Mentions": scores["total_mentions"],
                          "Visibility": f"{scores['overall_citation_share']}%", "Status": "Your Brand"}]
            for comp in user_competitors:
                mentions = detected_names_map.get(comp.lower(), 0)
                pct = round((mentions / total_q) * 100) if total_q > 0 else 0
                comp_rows.append({"Brand": comp, "AI Mentions": mentions,
                                  "Visibility": f"{pct}%",
                                  "Status": "✅ Detected" if mentions > 0 else "⚪ Not mentioned"})
            for brand_c, count in real_competitors[:8]:
                if brand_c.lower() not in [c.lower() for c in user_competitors] and brand_c.lower() != brand_name.lower():
                    pct = round((count / total_q) * 100) if total_q > 0 else 0
                    comp_rows.append({"Brand": brand_c, "AI Mentions": count,
                                      "Visibility": f"{pct}%", "Status": "Also detected"})
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

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