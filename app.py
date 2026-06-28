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


def _call_ai_for_prompts(system_prompt: str, user_message: str) -> str:
    """
    Dedicated call for prompt generation.
    System/user role split + temperature=0.8 for variety.
    Prevents repetitive duplicate outputs.
    """
    last_error = None
    openai_key = _get_key("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=2000,
                temperature=0.8,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_error = f"OpenAI error: {e}"
    gemini_key = _get_key("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai as _genai
            _client = _genai.Client(api_key=gemini_key)
            combined = system_prompt + "\n\n" + user_message
            _resp = _client.models.generate_content(
                model="gemini-2.0-flash",
                contents=combined,
                config={"temperature": 0.8}
            )
            return _resp.text or ""
        except Exception as e:
            last_error = f"Gemini error: {e}"
    detail = f" Last error: {last_error}" if last_error else ""
    raise Exception(f"No AI model available for prompt generation.{detail}")


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




# =============================================================================
# VALIDATION LIBRARY — Code-level fixes (priorities 1-10)
# =============================================================================
import re as _vre

# Priority 5: Abbreviation normalisation — long forms nobody types into ChatGPT
_EXPAND_TO_SHORT = {
    "Search Engine Optimization": "SEO",
    "Business to Business": "B2B",
    "Business-to-Business": "B2B",
    "Business to Consumer": "B2C",
    "Business-to-Consumer": "B2C",
    "Application Programming Interface": "API",
    "Large Language Model": "LLM",
    "Natural Language Processing": "NLP",
    "Customer Relationship Management": "CRM",
    "Software as a Service": "SaaS",
}

def _normalise_abbr(text: str) -> str:
    for long_form, short in _EXPAND_TO_SHORT.items():
        text = _vre.sub(_vre.escape(long_form), short, text, flags=_vre.IGNORECASE)
    return text

# Priority 8: Topic filler word removal
_FILLER = [
    r'\bavailable\b', r'\boptions\b', r'\bofferings\b',
    r'\bvarious\b', r'\bcomprehensive\b', r'\binnovative\b',
    r'\bcutting-edge\b', r'\bstate-of-the-art\b', r'\bworld-class\b',
]

def _clean_topic(topic: str) -> str:
    topic = _normalise_abbr(topic)
    for pat in _FILLER:
        topic = _vre.sub(pat, '', topic, flags=_vre.IGNORECASE)
    # "softwares" is not a word — software is uncountable
    topic = _vre.sub(r'\bsoftwares\b', 'software', topic, flags=_vre.IGNORECASE)
    # Strip banned first words — these belong in prompts not topic names
    _BANNED = r'^(best|top|compare|comparing|which|who|how|what|are|is|find|get|discover|explore)\s+'
    topic = _vre.sub(_BANNED, '', topic, flags=_vre.IGNORECASE).strip()
    # Enforce 6-word maximum
    words = topic.split()
    if len(words) > 6:
        topic = ' '.join(words[:6])
    return _vre.sub(r'\s+', ' ', topic).strip().strip(',').strip()

# Priority 3: Grammar — "I am a/an" + singular noun
_VOWELS = set('aeiouAEIOU')

def _fix_article_grammar(text: str) -> str:
    _UNCOUNTABLE = {"software", "hardware", "equipment", "information", "research", "advice"}
    def _fix(m):
        prefix  = m.group(1)
        article = m.group(2)
        noun    = m.group(3)
        rest    = m.group(4)
        # Uncountable nouns — drop article entirely
        if noun.lower() in _UNCOUNTABLE:
            return prefix.rstrip() + " " + noun + rest
        # Singularise
        if noun.endswith("ies") and len(noun) > 4:
            noun = noun[:-3] + "y"
        elif noun.endswith("ers") and not noun.endswith("eers"):
            noun = noun[:-1]
        elif noun.endswith("ants") or noun.endswith("ents"):
            noun = noun[:-1]
        elif noun.endswith("ors") and len(noun) > 4:
            noun = noun[:-1]
        elif noun.endswith("sts"):
            noun = noun[:-1]
        # Fix article for vowel sounds
        article = "an" if noun and noun[0] in _VOWELS else "a"
        return prefix + article + " " + noun + rest
    # Fix "I am a/an <noun>" patterns
    text = _vre.sub(
        r"(I(?:'m| am) )(a|an) ([A-Za-z]+)(.*)",
        _fix, text, flags=_vre.IGNORECASE
    )
    # Fix standalone "a software" / "an software" anywhere in text
    text = _vre.sub(r'\b(a|an) software\b', 'software', text, flags=_vre.IGNORECASE)
    # Fix softwares anywhere in prompt text
    text = _vre.sub(r'\bsoftwares\b', 'software', text, flags=_vre.IGNORECASE)
    return text

# Priority 7: Brand name capitalisation enforcement
def _fix_brand_cap(text: str, brand_name: str) -> str:
    if not brand_name:
        return text
    pat = _vre.compile(r'\b' + _vre.escape(brand_name) + r'\b', _vre.IGNORECASE)
    return pat.sub(brand_name, text)

# Priority 4: Business type vocabulary enforcement — wrong category word detection
_AGENCY_WORDS   = {'agency','agencies','firm','firms','consultant','consultants'}
_SOFTWARE_WORDS = {'software','tool','tools','platform','platforms','app','apps'}
_TRAINING_WORDS = {'training','course','courses','bootcamp','certification'}

def _has_category_violation(prompt: str, category_word: str) -> bool:
    pl  = prompt.lower()
    cw  = category_word.lower()
    if any(w in cw for w in ['agency','service']):
        return any(w in pl.split() for w in _SOFTWARE_WORDS)
    if any(w in cw for w in ['software','tool','platform']):
        return any(w in pl.split() for w in _AGENCY_WORDS)
    return False

# Priority 1: Structural duplication — first 6 words + semantic overlap
def _has_structural_dupes(prompts: list) -> bool:
    STOP = {'the','a','an','for','of','in','to','and','or','is',
            'are','what','which','how','who','best','top','any','i','we'}
    seen_start = set()
    seen_full  = set()
    for p in prompts:
        pl    = p.lower().strip()
        first6 = " ".join(pl.split()[:6])
        if pl in seen_full or first6 in seen_start:
            return True
        seen_full.add(pl)
        seen_start.add(first6)
    # Semantic overlap check between all pairs
    for i in range(len(prompts)):
        for j in range(i+1, len(prompts)):
            a = set(prompts[i].lower().split()) - STOP
            b = set(prompts[j].lower().split()) - STOP
            if a and b and len(a & b) / min(len(a), len(b)) > 0.80:
                return True
    return False

# Priority 2: How-to prompt presence
_HOWTO_STARTS = ['how do i','how can i','how does','how do companies',
                 "what's the best way to",'what is the best way to']

def _has_howto(prompts: list) -> bool:
    return any(p.lower().strip().startswith(s) for p in prompts for s in _HOWTO_STARTS)

# Priority 9: Topic coverage check
def _uncovered_products(topics: list, products_list: list) -> list:
    combined = " ".join(topics).lower()
    return [
        p for p in products_list
        if not any(w in combined for w in p.lower().split() if len(w) >= 4)
    ]

# Master cleaner — apply all code-level fixes to a prompt
def _clean_generated_prompt(p: str, brand_name: str, category_word: str) -> str:
    p = _normalise_abbr(p)
    p = _fix_article_grammar(p)
    p = _fix_brand_cap(p, brand_name)
    return p.strip()


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

    all_brand_text = (brand_context + products + " " + key_features).upper()
    terms_to_check = {
        term: meanings for term, meanings in AMBIGUOUS_TERMS.items()
        if term in all_brand_text
    }

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


def generate_gap_analysis(brand_data: dict, all_results: list, scores: dict) -> str:
    """
    Generates authentic gap analysis based on actual AI responses from the tracking run.
    Compares what competitors are being cited for vs what the client offers.
    Returns a structured analysis string.
    """
    import json as _json

    brand_name = brand_data.get("name", "")
    website_text = brand_data.get("_website_text", "")
    products = ", ".join(brand_data.get("products", []))
    key_features = ", ".join(brand_data.get("key_features", []))

    # Extract what ChatGPT actually said about competitors
    competitor_mentions = {}
    source_domains = set()
    response_snippets = []

    for r in all_results:
        resp_text = r.get("response", "")
        if resp_text and len(resp_text) > 50:
            response_snippets.append(resp_text[:300])

        brands = r.get("brands_detected", {}).get("all_brands", [])
        for b in brands:
            b_lower = b.lower()
            if b_lower != brand_name.lower():
                competitor_mentions[b] = competitor_mentions.get(b, 0) + 1

        for s in r.get("linked_sites", []):
            domain = s.get("domain", "")
            if domain and brand_name.lower() not in domain.lower():
                source_domains.add(domain)

    # Top competitors by frequency
    top_competitors = sorted(competitor_mentions.items(), key=lambda x: x[1], reverse=True)[:8]
    overall_visibility = scores.get("overall_citation_share", 0)

    # Build context for gap analysis
    gap_context = (
        "BRAND BEING ANALYZED:\n"
        "Name: " + brand_name + "\n"
        "What they offer: " + products + "\n"
        "Key differentiators: " + key_features + "\n"
    )
    if website_text:
        gap_context += "Website content:\n" + website_text[:1500] + "\n\n"

    gap_context += (
        "\nTRACKING RUN RESULTS:\n"
        "AI Visibility Score: " + str(overall_visibility) + "%\n"
        "Total prompts run: " + str(len(all_results)) + "\n\n"
        "BRANDS APPEARING INSTEAD OF " + brand_name.upper() + ":\n"
    )
    for brand, count in top_competitors:
        pct = round((count / len(all_results)) * 100)
        gap_context += "- " + brand + ": appeared in " + str(pct) + "% of responses\n"

    if source_domains:
        gap_context += "\nSOURCE WEBSITES CHATGPT CITED (that do NOT include " + brand_name + "):\n"
        for domain in list(source_domains)[:8]:
            gap_context += "- " + domain + "\n"

    if response_snippets:
        gap_context += "\nSAMPLE AI RESPONSES (showing why competitors are recommended):\n"
        for snippet in response_snippets[:3]:
            gap_context += "---\n" + snippet + "\n"

    gap_prompt = (
        "You are an AI visibility consultant analyzing why a brand is not appearing in AI search results.\n\n"
        + gap_context
        + "\nBased on this data, identify exactly why " + brand_name + " is not being recommended "
        "and what they need to do to appear.\n\n"
        "Generate a gap analysis with these 5 sections:\n\n"
        "1. VISIBILITY DIAGNOSIS\n"
        "One paragraph explaining exactly why " + brand_name + " has " + str(overall_visibility) + "% visibility. "
        "Be specific — reference the actual competitors appearing and what signals they have that " + brand_name + " lacks.\n\n"
        "2. CONTENT GAPS (3 specific gaps)\n"
        "What specific content does " + brand_name + " need to create to appear in AI responses? "
        "Reference the actual queries where competitors appeared. Be specific — not 'publish more content' "
        "but 'create a comparison page for [specific competitor] vs " + brand_name + " because ChatGPT cited [source] for this query'.\n\n"
        "3. CITATION FOOTPRINT GAPS (3 specific gaps)\n"
        "Which specific third-party sites cited competitors but not " + brand_name + "? "
        "What does the brand need to do to get listed on those same sites?\n\n"
        "4. POSITIONING GAPS (2 specific gaps)\n"
        "What specific language or proof points do competing brands use that " + brand_name + " does not? "
        "Reference actual response snippets from the tracking run above.\n\n"
        "5. TOP 3 PRIORITY ACTIONS\n"
        "The 3 highest-impact actions " + brand_name + " should take right now, in priority order. "
        "Each must be specific, actionable, and directly tied to the data above.\n\n"
        "Keep each section concise and evidence-based. No generic SEO advice. "
        "Every recommendation must reference actual data from this tracking run."
    )

    try:
        result = _call_ai_for_json(gap_prompt)
        # This returns plain text not JSON
        if result and len(result) > 100:
            return result
        return "Gap analysis could not be generated. Please try again."
    except Exception as e:
        return "Gap analysis error: " + str(e)


def ai_generate_topics(brand_data: dict) -> list:
    """
    Three-phase topic generation - universal, works for any business.
    Phase 1: Extract brand anchors deterministically from user data
    Phase 2: AI understands the business deeply
    Phase 3: Generate topics grounded in real brand terminology
    """
    import json as _json

    business_type = brand_data.get("business_type", "")
    products_list = brand_data.get("products", [])
    customers_list = brand_data.get("customers", [])
    features_list = brand_data.get("key_features", [])
    competitors_list = brand_data.get("competitors", [])
    domain = brand_data.get("domain", "")
    brand_name = brand_data["name"]
    country = brand_data.get("country", "")
    products = ", ".join(products_list)
    customers = ", ".join(customers_list)
    key_features = ", ".join(features_list)

    website_text = brand_data.get("_website_text", "")
    if not website_text and domain:
        website_text = fetch_brand_website(domain)

    # ── TERM RESOLUTION (silent, runs once per session) ──────────────────
    resolved_terms = resolve_brand_terms(brand_data)
    term_glossary = build_term_glossary(resolved_terms)

    # ── PHASE 1: Extract brand anchors from user data (no AI needed) ─────────
    # Finds specific terms the brand actually uses — product names, vendor names,
    # certification names, proprietary tool names. These become mandatory in topics.
    COMMON_WORDS = {
        "best", "tool", "tools", "software", "platform", "service", "services",
        "company", "agency", "firm", "provider", "solution", "product", "team",
        "based", "using", "user", "users", "data", "work", "free", "easy",
        "simple", "fast", "high", "real", "time", "open", "source", "access",
        "cost", "price", "need", "help", "make", "build", "track", "manage",
        "create", "search", "find", "save", "with", "that", "this", "from",
        "your", "their", "have", "will", "been", "more", "most", "very",
        "over", "also", "only", "like", "just", "into", "than", "some",
        "each", "such", "when", "where", "which", "what", "across", "through",
        "without", "within", "between", "focused", "driven", "leading", "expert",
        "award", "winning", "years", "since", "first", "content", "marketing",
        "digital", "online", "growth", "brand", "brands", "business", "businesses",
        "client", "clients", "results", "report", "strategy", "consulting",
        "management", "support", "global", "local", "enterprise", "scalable",
    }

    brand_anchors = []
    for item in products_list + features_list:
        item_clean = item.strip()
        words = item_clean.split()
        specific = [w for w in words if w.lower() not in COMMON_WORDS and len(w) >= 3]
        if len(words) >= 2 and len(specific) >= 1 and any(len(w) >= 5 for w in specific):
            brand_anchors.append(item_clean)
        elif len(words) == 1 and len(item_clean) >= 4 and item_clean.lower() not in COMMON_WORDS:
            brand_anchors.append(item_clean)

    seen_lower = set()
    unique_anchors = []
    for a in sorted(brand_anchors, key=len, reverse=True):
        if a.lower() not in seen_lower:
            seen_lower.add(a.lower())
            unique_anchors.append(a)
    brand_anchors = unique_anchors[:10]

    # ── PHASE 2: AI understands the business deeply ───────────────────────────
    buyer_insights = brand_data.get("_buyer_insights", [])
    buyer_insights_text = ""
    if buyer_insights:
        buyer_insights_text = "\nDIRECT BUYER INSIGHTS FROM BRAND OWNER:\n"
        for qa in buyer_insights:
            buyer_insights_text += "Q: " + qa["question"] + "\nA: " + qa["answer"] + "\n\n"

    raw_context = ""
    if website_text:
        raw_context += "WEBSITE CONTENT:\n" + website_text[:2500] + "\n\n"
    raw_context += (
        "Brand: " + brand_name + "\n"
        + "Business type: " + business_type + "\n"
        + "Products/Services: " + products + "\n"
        + "Target customers: " + customers + "\n"
        + "Key differentiators: " + key_features + "\n"
        + "Competitors: " + ", ".join(competitors_list) + "\n"
    )

    understanding_prompt = (
        "Read this brand information and answer in JSON format.\n\n"
        + term_glossary
        + raw_context
        + buyer_insights_text
        + "\nAnswer about THIS specific brand:\n"
        + '{\n'
        + '  "what_business_does": "One specific sentence. Use the brand own terminology from website.",\n'
        + '  "exact_buyer": "Job title, company type, and problem they need solved.",\n'
        + '  "buyer_searches": ["5 exact phrases buyer types into ChatGPT. Use brand product names. Start with best/top/which/who/compare."],\n'
        + '  "business_category": "Single word: agency OR software OR platform OR firm OR provider OR training company"\n'
        + '}\n\n'
        + "CRITICAL: Use actual product and vendor names from the website.\n"
        + "WRONG: best GEO agency. RIGHT: best agency for Generative Engine Optimization using BlueprintIQ\n"
        + "WRONG: cybersecurity training. RIGHT: best authorized Palo Alto Networks training company\n"
        + "Respond ONLY with valid JSON. No markdown."
    )

    brand_understanding = {}
    try:
        raw_intel = _call_ai_for_json(understanding_prompt)
        try:
            brand_understanding = _json.loads(raw_intel)
        except Exception:
            parsed = _parse_json_list(raw_intel)
            if parsed and isinstance(parsed[0], dict):
                brand_understanding = parsed[0]
    except Exception:
        brand_understanding = {}

    buyer_searches = brand_understanding.get("buyer_searches", [])
    business_category = brand_understanding.get("business_category", "agency")
    what_business_does = brand_understanding.get("what_business_does", "")
    exact_buyer = brand_understanding.get("exact_buyer", "")

    # ── PHASE 3: Generate topics grounded in brand anchors ────────────────────
    country_note = ("User country: " + country + "\n") if country and country.lower() not in ["global", ""] else ""
    country_topic_note = (
        "- Include 1 topic with '" + country + "' for local search\n"
        if country and country.lower() not in ["global", "united states", ""] else ""
    )
    anchor_list = "\n".join(("- " + a) for a in brand_anchors[:8]) if brand_anchors else "- (none extracted)"
    buyer_search_list = "\n".join(("- " + s) for s in buyer_searches[:5]) if buyer_searches else ""

    topic_prompt = (
        "You generate topic names for AI visibility tracking.\n\n"
        + term_glossary
        + "BRAND INPUT DATA (use exact phrases from here as the core of each topic):\n"
        + "Brand: " + brand_name + "\n"
        + "Business type: " + business_category + "\n"
        + "Anchor terms (use at least 3 of these in topics):\n"
        + anchor_list
        + "\n\nRAW BRAND DATA:\n"
        + raw_context
        + country_note
        + "\nGenerate 5 topic names. Each topic is a short noun phrase, not a question.\n\n"
        + "TOPIC NAME RULES — apply all strictly:\n"
        + "1. LENGTH: 3 to 6 words maximum. No exceptions.\n"
        + "2. FORMAT: Noun phrases only. Never a question. Never a command.\n"
        + "   Topics describe a category of thing, not an action or question.\n"
        + "3. BANNED FIRST WORDS: Never start a topic with best, top, compare, which,\n"
        + "   who, how, what, are, is, find, get, or any question word or superlative.\n"
        + "   These words belong in prompts, not topic names.\n"
        + "4. ONE CONCEPT: One clear idea per topic. Do not combine multiple ideas.\n"
        + "5. VOCABULARY: Use exact product names and differentiator words from the\n"
        + "   input data above. Add the business type word at the end if needed.\n"
        + "6. SOFTWARE RULE: Write 'software' never 'softwares'. Software is uncountable.\n"
        + "7. No '" + brand_name + "' in any topic.\n\n"
        + "GOOD TOPIC EXAMPLES (noun phrases, 3-6 words):\n"
        + "  For a patent search SaaS: 'free patent search tool', 'open source patent search',\n"
        + "  'AI patent search software', 'patent search API', 'semantic prior art search'\n"
        + "  For a content agency: 'B2B content marketing agency', 'SaaS content marketing services',\n"
        + "  'LinkedIn ghostwriting for B2B', 'thought leadership content creation'\n"
        + "  For a training company: 'Juniper Networks training providers',\n"
        + "  'authorized Palo Alto Networks training', 'Fortinet NSE certification training'\n\n"
        + "BAD TOPIC EXAMPLES (never generate these):\n"
        + "  'best free patent search tool software for legal firms' (starts with best, too long)\n"
        + "  'compare semantic patent search tools for researchers' (starts with compare)\n"
        + "  'which AI patent search solution offers the fastest results' (question, too long)\n"
        + "  'patent search softwares' (softwares is not a word)\n\n"
        + country_topic_note
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
            t = item.get("topic", "").strip()
            intent = item.get("intent", "").strip()
            if t:
                topics.append(t)
                if intent:
                    topic_intents[t] = intent
        elif isinstance(item, str) and item.strip():
            topics.append(item.strip())

    if "brand_data" in st.session_state:
        st.session_state.brand_data["_topic_intents"] = topic_intents

    bad_patterns = [
        "strategies", "tips", "how to", "examples", "guide", "tutorial",
        "best practices", "introduction", "overview", "explained",
        "what is", "benefits of", "advantages of",
    ]

    filtered = []
    for t in topics:
        # Priority 8: remove filler words, normalise abbreviations
        t = _clean_topic(t.strip())
        if not t or brand_lower in t.lower():
            continue
        if any(bp in t.lower() for bp in bad_patterns):
            continue
        filtered.append(t)

    # Priority 9: coverage check — ensure major products have a topic
    products_for_coverage = [p.strip() for p in products_list if len(p.strip()) >= 3][:8]
    uncovered = _uncovered_products(filtered, products_for_coverage)
    for missing in uncovered[:2]:
        fb = _clean_topic(missing + " " + category_word)
        if brand_lower not in fb.lower() and fb not in filtered and len(filtered) < 5:
            filtered.append(fb)

    # Fallback: build from products if still too few
    if len(filtered) < 3:
        for p in products_for_coverage[:6]:
            fallback = _clean_topic(p + " " + category_word)
            if brand_lower not in fallback.lower() and fallback not in filtered:
                filtered.append(fallback)
            if len(filtered) >= 5:
                break

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

    # Reuse resolved terms from topic generation
    resolved_terms = brand_data.get("_resolved_terms", {})
    term_glossary = build_term_glossary(resolved_terms)

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

    # Priority 10: Competitors LOCKED to user input only
    # Never pull competitor names from website scan or model knowledge
    competitors = brand_data.get("competitors", [])
    competitor_context = ""
    comp_for_comparison = []
    if competitors:
        _topic_hash_c = sum(ord(c) for c in topic) % max(len(competitors), 1)
        _rotated_comp = competitors[_topic_hash_c:] + competitors[:_topic_hash_c]
        comp_for_comparison = _rotated_comp[:1]  # ONE competitor per topic, rotated

        competitor_context = (
            "\nDIRECT COMPETITORS (from user input ONLY — never use other names): "
            + ", ".join(competitors)
            + "\nFor prompt 5, compare " + brand_name + " specifically against: "
            + (comp_for_comparison[0] if comp_for_comparison else competitors[0])
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

    # Priority 6: Persona rotation through full ICP list
    # Each topic gets a unique persona from Target Customers — no repeats until all used
    # Only uses personas from the user's actual input list — never invents roles
    customers_list = brand_data.get("customers", [])
    SINGULAR_MAP_P = {
        "startups": "startup founder", "inventors": "inventor",
        "researchers": "researcher", "developers": "developer",
        "attorneys": "patent attorney", "patent attorneys": "patent attorney",
        "ip professionals": "IP professional", "engineers": "engineer",
        "managers": "corporate training manager", "founders": "founder",
        "scientists": "scientist", "lawyers": "lawyer",
        "companies": "company owner", "businesses": "business owner",
        "cmos": "CMO", "marketing managers": "marketing manager",
    }
    def _make_singular(label):
        label = label.strip()
        ll = label.lower()
        if ll in SINGULAR_MAP_P:
            return SINGULAR_MAP_P[ll]
        for plural, singular in SINGULAR_MAP_P.items():
            if ll.endswith(" " + plural):
                return label.rsplit(" ", 1)[0] + " " + singular.split()[-1]
        if label.endswith("ies") and len(label) > 4: return label[:-3] + "y"
        if label.endswith("ers") and not label.endswith("eers"): return label[:-1]
        if label.endswith("ants") or label.endswith("ents"): return label[:-1]
        if label.endswith("ors") and len(label) > 4: return label[:-1]
        return label

    _customers_singular = [_make_singular(c) for c in customers_list if c.strip()]
    # Deduplicate
    _seen_c = set()
    _customers_unique = []
    for c in _customers_singular:
        if c.lower() not in _seen_c:
            _seen_c.add(c.lower())
            _customers_unique.append(c)
    if not _customers_unique:
        _customers_unique = ["startup founder", "researcher", "developer", "manager", "professional"]

    # Rotate starting point per topic hash
    _topic_hash = sum(ord(c) for c in topic) % max(len(_customers_unique), 1)
    _rotated_customers = _customers_unique[_topic_hash:] + _customers_unique[:_topic_hash]
    persona_role = _rotated_customers[0]  # Primary persona for this topic

    # Country as system context not in prompt text
    country_system = ""
    country_suffix = ""
    if country and country.lower() not in ["global", "united states", ""]:
        country_system = f"\nUser location context: {country}. Tailor relevance to this market."
        country_suffix = f" {country}"


    # ── Priority 10: Comparison uses brand vs ONE locked competitor ──────────
    # Rotates through the user's Direct Competitors list — never uses website data
    if competitors and comp_for_comparison:
        comparison_line = (
            "5. COMPARISON: Compare " + brand_name + " against " + comp_for_comparison[0] + ".\\n"
            "   MUST start with 'How does " + brand_name + "' or '" + brand_name + " vs'.\\n"
            "   This is the ONLY prompt that may name " + brand_name + " directly.\\n"
            "   CORRECT: 'How does " + brand_name + " compare to " + comp_for_comparison[0] + " for [topic]?'"
        )
    else:
        comparison_line = (
            "5. RECOMMENDATION: Which " + solution_word + " is best for [specific need from topic]?\\n"
            "   CORRECT: 'Which " + solution_word + " would you recommend for [specific need from topic]?'"
        )

    # ── Priority 4 & 6: Category word + countable prompt word ────────────────
    # cat_word = the actual solution type (used in context and rules)
    # prompt_word = always a COUNTABLE noun used in prompt 2/3 templates
    # "Which tools" / "Which agencies" / "Which training providers" — no grammar errors
    # "Which software" is BANNED — software is uncountable, causes "Which software offer"
    bt_lower_cw = business_type.lower()
    if any(w in bt_lower_cw for w in ["training", "education", "bootcamp"]):
        cat_word    = "training provider"
        prompt_word = "training provider"  # "Which training providers"
        p2_opener   = "Which training providers"
        p3_opener   = "Are there training providers"
        cat_ban     = "NEVER use software or agency — this is a training company\n"
    elif any(w in bt_lower_cw for w in ["agency", "service", "studio"]):
        cat_word    = "agency"
        prompt_word = "agency"             # "Which agencies"
        p2_opener   = "Which agencies"
        p3_opener   = "Are there agencies"
        cat_ban     = "NEVER use software or tool — this is a service not software\n"
    elif any(w in bt_lower_cw for w in ["saas", "software", "platform"]):
        cat_word    = "software"
        prompt_word = "tool"               # "Which tools" — never "Which software"
        p2_opener   = "Which tools"
        p3_opener   = "Are there tools"
        cat_ban     = "NEVER use agency or firm — this is software not a service\n"
    else:
        cat_word    = "provider"
        prompt_word = "provider"
        p2_opener   = "Which providers"
        p3_opener   = "Are there providers"
        cat_ban     = ""

    prompt = (
        "You are an expert GEO (Generative Engine Optimization) strategist who understands "
        "how real buyers search across every industry and niche.\n\n"
        + term_glossary
        + "BRAND CONTEXT:\n"
        "Business type: " + business_type + "\n"
        "Solution type: " + cat_word + "\n"
        + cat_ban
        + "Context: " + context_block + "\n"
        + competitor_context
        + country_system + "\n\n"
        "GENERATE EXACTLY 5 PROMPTS for TOPIC: \"" + topic + "\"\n\n"
        "PROMPT 1 — DIRECT SEARCH:\n"
        "The exact topic phrase as typed. Nothing added. No question mark.\n"
        "Output: '" + topic + "'\n\n"
        "PROMPT 2 — DISCOVERY (third person, no 'best', no 'I'):\n"
        "MUST start with: '" + p2_opener + "'\n"
        + cat_ban +
        "CORRECT: '" + p2_opener + " offer [topic feature]?'\n"
        "CORRECT: '" + p2_opener + " specialise in [topic]?'\n"
        "WRONG: 'Which software offer...' (software is uncountable — use tools/platforms)\n"
        "WRONG: 'best...' or any first-person opener\n\n"
        "PROMPT 3 — CASUAL DISCOVERY (third person, different opener from prompt 2):\n"
        "MUST start with: '" + p3_opener + "' OR 'Where can I find' OR 'Is there a' OR 'Who offers'\n"
        "Different intent from prompt 2. Never the same opening word.\n"
        + cat_ban +
        "CORRECT: '" + p3_opener + " that handle [topic]?'\n"
        "WRONG: same opening as prompt 2. WRONG: 'I' or 'we'.\n\n"
        "PROMPT 4 — PERSONA (first person, starts with I or we):\n"
        "MUST start with 'I' or 'we'. This is the ONLY first-person prompt.\n"
        "Persona: " + persona_role + " — singular. Never pluralise.\n"
        "Short and casual. References a key differentiator from the brand context.\n"
        "GRAMMAR: 'I am a " + persona_role + "' NEVER 'I am a " + persona_role + "s'\n"
        "Ends with: 'What do you recommend?' OR 'Any suggestions?' OR 'Which would you go with?'\n"
        "CORRECT: 'I am a " + persona_role + " and I need [differentiator] for [topic]. What do you recommend?'\n\n"
        "PROMPT 5 — COMPARISON (mandatory, always present):\n"
        "MUST be: 'How does " + brand_name + " compare to " + (comp_for_comparison[0] if comp_for_comparison else "[competitor]") + " for [topic]?'\n"
        "Brand name '" + brand_name + "' is REQUIRED in this prompt — exact capitalisation.\n"
        "Competitor must be from the Direct Competitors field only: " + (comp_for_comparison[0] if comp_for_comparison else "a named competitor") + "\n"
        "NEVER omit this prompt. NEVER use a competitor not from the input field.\n\n"
        "ABSOLUTE RULES:\n"
        "1. Prompts 1-4: NEVER mention '" + brand_name + "'\n"
        "2. Prompt 5 ALWAYS names '" + brand_name + "' and a competitor\n"
        "3. 'best' is BANNED from prompts 2, 3, and 4\n"
        "4. No two prompts start with the same word\n"
        "5. Prompts 2 and 3 are THIRD PERSON — never 'I' or 'we'\n"
        "6. Prompt 4 is the ONLY first-person prompt\n"
        "7. Every prompt contains at least one word from: '" + topic + "'\n"
        "8. No year numbers. No country names in prompt text\n"
        "9. Singular grammar: 'I am a researcher' NEVER 'I am a researchers'\n"
        "10. If two prompts look similar — rewrite one from a completely different angle\n"
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
        """Clean, filter, and validate a list of raw prompts using validation library."""
        cleaned = []
        for p in raw_prompts:
            p = p.strip()
            if not p:
                continue
            # Priority 5: normalise abbreviations (B2B, SEO etc)
            p = _normalise_abbr(p)
            # Strip year numbers
            p = year_pattern.sub("", p).strip()
            p = " ".join(p.split())
            # Priority 3: fix article grammar (I am a/an + singular)
            p = _fix_article_grammar(p)
            # Priority 7: enforce brand name capitalisation
            p = _fix_brand_cap(p, brand_name)
            # Skip if contains brand name in prompts 1-4
            if brand_lower in p.lower() and "vs" not in p.lower() and "compare" not in p.lower():
                continue
            # Skip exact duplicates of topic
            if p.lower() == topic_lower:
                continue
            # Skip duplicates already in list
            if p.lower() in [x.lower() for x in cleaned]:
                continue
            # Priority 4: reject wrong category word
            if hasattr(locals(), "cat_word") and _has_category_violation(p, cat_word):
                continue
            # Skip prompts that will produce bad results
            if is_bad_prompt(p):
                continue
            cleaned.append(p)
        return cleaned

    # First attempt — uses system/user split + temperature 0.8
    raw = _call_ai_for_prompts(
        "You are an expert GEO strategist generating search prompts. Follow all rules exactly.",
        prompt
    )
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
            raw2 = _call_ai_for_prompts(prompt, "Regenerate. Previous had issues. Make all 5 prompts structurally different.")
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

    # P3: Ensure exactly 5 prompts — regenerate missing ones
    attempts = 0
    while len(result) < 5 and attempts < 2:
        attempts += 1
        try:
            raw_pad = _call_ai_for_prompts(
                "You are an expert GEO strategist. Generate missing prompts.",
                prompt + "\n\nNEED MORE PROMPTS. Already have:\n" +
                "\n".join(f"- {r}" for r in result) +
                "\nGenerate only the missing prompts to reach a total of 5."
            )
            pad_clean = process_raw(_parse_json_list(raw_pad))
            for p in pad_clean:
                if len(result) >= 5:
                    break
                if p.lower() not in [r.lower() for r in result]:
                    result.append(p)
        except Exception:
            break

    # P2: Guarantee comparison prompt as prompt 5
    # Always "How does [Brand] compare to [Competitor] for [topic]?"
    _comp_name = comp_for_comparison[0] if comp_for_comparison else (
        competitors[0] if competitors else "alternatives"
    )
    _comparison = "How does " + brand_name + " compare to " + _comp_name + " for " + topic + "?"
    _comparison = _fix_brand_cap(_normalise_abbr(_comparison), brand_name)

    # Check if any prompt already contains both brand name and a competitor
    _has_comparison = any(
        brand_name.lower() in p.lower() and _comp_name.lower() in p.lower()
        for p in result
    )
    if not _has_comparison:
        if len(result) < 5:
            result.append(_comparison)
        else:
            result[4] = _comparison  # Always slot 5

    # P1: Structural duplicate check — auto-regenerate if dupes found
    if _has_structural_dupes(result):
        try:
            raw_regen = _call_ai_for_prompts(
                "You are an expert GEO strategist. Generate 5 completely different prompts.",
                prompt + "\n\nPREVIOUS GENERATION HAD DUPLICATE PROMPTS. "
                "Make every prompt different in opening word AND intent. "
                "No two prompts may start with the same word."
            )
            regen = _parse_json_list(raw_regen)
            regen_clean = process_raw(regen)
            if regen_clean and not _has_structural_dupes([topic] + regen_clean):
                result = ([topic] + regen_clean)[:5]
                # Re-inject comparison as prompt 5
                result[4] = _comparison
        except Exception:
            pass

    # P5: Ensure how-to prompt exists somewhere in prompts 2-4
    if not _has_howto(result):
        howto = "How do I find " + solution_word + " for " + topic + "?"
        howto = _fix_brand_cap(_normalise_abbr(howto), brand_name)
        if len(result) < 5:
            result.append(howto)
        else:
            result[2] = howto  # Slot 3 — leaves slots 4 and 5 intact

    # Final pad to guarantee exactly 5
    _pad = [
        "Which " + prompt_word + "s specialise in " + topic + "?",
        "Are there " + prompt_word + "s that offer " + topic + "?",
        "I need " + topic + " — any suggestions?",
        "Where can I find " + topic + "?",
        _comparison,
    ]
    _pad_i = 0
    while len(result) < 5:
        candidate = _fix_brand_cap(_normalise_abbr(_pad[_pad_i % len(_pad)]), brand_name)
        if candidate not in result:
            result.append(candidate)
        _pad_i += 1

    result = result[:5]
    # Always ensure prompt 5 is the comparison
    result[4] = _comparison
    return result


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

                                # Final safety check: verify brand actually in response text
                                # Catches any LLM hallucination from brand_detector
                                import re as _re_check
                                _resp_text = r.get("response", "")
                                _brand_in_text = bool(_re_check.search(
                                    r"\b" + _re_check.escape(brand_name) + r"\b",
                                    _resp_text, _re_check.IGNORECASE
                                ))
                                if mentioned and not _brand_in_text:
                                    mentioned = False  # Override false positive

                                if mentioned:
                                    st.success(f"Brand mentioned | Context: {context} | Position: #{position}")
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

            # ── Gap Finder ────────────────────────────────────────────────
            st.divider()
            st.subheader("🔍 Gap Finder")
            st.caption(
                "Analyzes your tracking results to find exactly why competitors appear and you don't. "
                "Evidence-based gaps — not generic SEO advice."
            )

            if st.button("🔍 Generate Gap Analysis", type="primary", use_container_width=False):
                with st.spinner("Analyzing AI responses to find your visibility gaps..."):
                    gap_analysis = generate_gap_analysis(bd, all_results, scores)
                    st.session_state["_gap_analysis"] = gap_analysis

            if st.session_state.get("_gap_analysis"):
                gap_text = st.session_state["_gap_analysis"]
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #1e3a5f 0%, #0f2340 100%);
                                border-radius: 12px; padding: 4px 24px 20px 24px;
                                border: 1px solid #2563eb44; margin: 16px 0;'>
                """, unsafe_allow_html=True)

                # Parse and render sections with styling
                sections = {
                    "1. VISIBILITY DIAGNOSIS": "🩺",
                    "2. CONTENT GAPS": "📝",
                    "3. CITATION FOOTPRINT GAPS": "🔗",
                    "4. POSITIONING GAPS": "🎯",
                    "5. TOP 3 PRIORITY ACTIONS": "⚡",
                }
                remaining = gap_text
                for section_title, icon in sections.items():
                    if section_title in remaining:
                        parts = remaining.split(section_title, 1)
                        before = parts[0].strip()
                        remaining = parts[1]
                        next_section_start = len(remaining)
                        for other in sections:
                            if other != section_title and other in remaining:
                                idx = remaining.find(other)
                                if idx < next_section_start:
                                    next_section_start = idx
                        section_body = remaining[:next_section_start].strip()
                        remaining = remaining[next_section_start:]

                        st.markdown(f"### {icon} {section_title.split('. ', 1)[1]}")
                        st.markdown(section_body)
                        st.markdown("---")

                if remaining.strip():
                    st.markdown(remaining.strip())

                st.markdown("</div>", unsafe_allow_html=True)

                # Download gap analysis
                st.download_button(
                    label="📥 Download Gap Analysis",
                    data=gap_text,
                    file_name=f"gap_analysis_{brand_name.lower().replace(' ', '_')}.txt",
                    mime="text/plain"
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