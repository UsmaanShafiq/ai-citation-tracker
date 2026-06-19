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
    # Try Groq first (free)
    groq_key = _get_key("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            pass

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

    raise Exception("No available AI model. Please add at least one API key (Groq is free).")


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

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text

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
    business_type = brand_data.get("business_type", "")
    products = ", ".join(brand_data.get("products", []))
    customers = ", ".join(brand_data.get("customers", []))
    key_features = ", ".join(brand_data.get("key_features", []))
    domain = brand_data.get("domain", "")

    # Visit the brand website for accurate context
    website_text = ""
    if domain:
        website_text = fetch_brand_website(domain)

    # Build context block - website content takes priority, form data as supplement
    if website_text:
        context_block = (
            "WEBSITE CONTENT (scraped directly from their homepage - use this as primary source of truth):\n"
            + website_text[:2000]
            + "\n\n"
            "FORM DATA (entered by user - use to supplement website content):\n"
            "What they do: " + products + "\n"
            "Who they serve: " + customers + "\n"
            "Key differentiators: " + key_features + "\n"
        )
    else:
        # Fallback: no website content, use form data only
        context_block = (
            "BRAND PROFILE (entered by user):\n"
            "What they do: " + products + "\n"
            "Who they serve: " + customers + "\n"
            "Key differentiators: " + key_features + "\n"
        )

    prompt = (
        "You are an AI search visibility expert helping track how often brands appear in AI-generated answers.\n"
        "Your job is to generate realistic search topics that a potential BUYER would type into an AI tool "
        "like ChatGPT, Perplexity, or Gemini when looking for a solution like this brand.\n\n"
        "IMPORTANT RULES:\n"
        "- Read ALL the brand information below carefully before generating anything\n"
        "- Infer the correct meaning of ALL industry terms and abbreviations from the brand context\n"
        "- Do NOT assume any fixed meaning for abbreviations - let the brand content guide you\n"
        "- Topics must reflect what a real buyer searches for, NOT what the brand wants to be known for\n"
        "- Do NOT include the brand name in any topic\n"
        "- Do NOT generate topics about software tools if this is a service/agency business\n"
        "- Business type: " + business_type + "\n"
        "- Think like a buyer who does not know this brand yet but has a problem this brand solves\n\n"
        "Brand: " + brand_data["name"] + "\n"
        "Domain: " + domain + "\n"
        "Country: " + brand_data.get("country", "United States") + "\n\n"
        + context_block +
        "\nBased on ALL the information above, generate exactly 5 short keyword-style search topics (3-8 words each).\n"
        "Each topic must represent a genuine search intent a potential customer would have.\n"
        "NEVER include the brand name in any topic.\n\n"
        "Respond ONLY with a valid JSON array of 5 strings. No explanation, no markdown, no preamble."
    )

    raw = _call_ai_for_json(prompt)
    topics = _parse_json_list(raw)
    # Filter out any topic that accidentally contains the brand name
    brand_lower = brand_data["name"].lower()
    topics = [t.strip() for t in topics if t.strip() and brand_lower not in t.lower()]
    return topics[:5]


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

    prompt = (
        "You are an AI search visibility expert helping track how often brands appear in AI-generated answers.\n"
        "For the topic below, generate 5 realistic prompts that a real potential buyer would type into "
        "ChatGPT, Perplexity, or Gemini when searching for a solution.\n\n"
        "Topic: " + topic + "\n\n"
        "Brand context (use this to understand the industry and buyer correctly):\n"
        + context_block + "\n"
        "Business type: " + business_type + "\n\n"
        "STRICT RULES - violating these makes the data useless:\n"
        "- NEVER mention the brand name \"" + brand_name + "\" in any prompt\n"
        "- NEVER mention any specific brand or company name in any prompt\n"
        "- Prompts must sound like a real buyer who does NOT know which brand they will choose yet\n"
        "- Vary the style: 1 short keyword, 2 full questions, 1 comparison, 1 persona-based\n"
        "- Prompts must be specific enough to realistically surface agency/service recommendations\n"
        "- Do NOT generate prompts about software tools or platforms if this is a service business\n\n"
        "Respond ONLY with a valid JSON array of 5 strings. No explanation, no markdown, no preamble."
    )

    raw = _call_ai_for_json(prompt)
    prompts = _parse_json_list(raw)

    # Strict filter: remove any prompt that contains the brand name
    brand_lower = brand_name.lower()
    clean = [p.strip() for p in prompts if p.strip() and brand_lower not in p.lower()]

    # Always include the bare topic as first prompt (most natural search)
    result = [topic] + [p for p in clean if p.lower() != topic.lower()]
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

    tags = st.session_state[session_key]

    st.markdown(f"**{label}**")
    if help_text:
        st.caption(help_text)

    if tags:
        pills_html = "".join([
            f'<span style="display:inline-block;background:#1e3a5f;color:#7dd3fc;'
            f'border:1px solid #2563eb;border-radius:999px;padding:3px 12px;'
            f'margin:2px 4px 2px 0;font-size:13px;">{t}</span>'
            for t in tags
        ])
        st.markdown(f'<div style="margin-bottom:6px">{pills_html}</div>', unsafe_allow_html=True)

    col_in, col_add, col_clear = st.columns([5, 1, 1])
    with col_in:
        new_val = st.text_input(
            label, key=f"{session_key}_input",
            placeholder=placeholder, label_visibility="collapsed"
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
                st.rerun()
    with col_clear:
        if tags:
            st.write("")
            if st.button("Clear", key=f"{session_key}_clear_btn", use_container_width=True):
                st.session_state[session_key] = []
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
            "GROQ_API_KEY": "Groq (Free)",
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
                    "step1_products", "step1_customers", "step1_key_features"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# =============================================================================
# STEP INDICATOR
# =============================================================================

step_names = ["Brand Details", "Topics", "Prompts", "Run & Results"]
cols = st.columns(4)
for i, (col, name) in enumerate(zip(cols, step_names), 1):
    with col:
        if i < st.session_state.step:
            st.success(f"✓ {name}")
        elif i == st.session_state.step:
            st.info(f"▶ {name}")
        else:
            st.caption(f"{i}. {name}")

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
    with col3:
        _country_options = ["United States", "United Kingdom", "Canada", "Australia",
                             "Germany", "India", "Pakistan", "Global"]
        _country_default = _saved.get("country", "United States")
        _country_idx = _country_options.index(_country_default) if _country_default in _country_options else 0
        country = st.selectbox("Country", options=_country_options, index=_country_idx)
    with col4:
        _saved_comps = ", ".join(_saved.get("competitors", []))
        competitors_input = st.text_input("Competitors (optional)",
                                          value=_saved_comps,
                                          placeholder="Animalz, Siege Media, Grow and Convert")

    st.divider()

    if st.button("Next: Generate Topics →", type="primary"):
        if not brand_name.strip() or not brand_domain.strip():
            st.error("Brand Name and Domain are required.")
        else:
            st.session_state.brand_data = {
                "name": brand_name.strip(),
                "domain": brand_domain.strip(),
                "products": products_list,
                "customers": customers_list,
                "key_features": key_features_list,
                "business_type": business_type,
                "country": country,
                "competitors": [c.strip() for c in competitors_input.split(",") if c.strip()],
            }
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
        with st.spinner("Visiting brand website for context..."):
            # Fetch website once and cache in brand_data to reuse in prompt generation
            if "_website_text" not in st.session_state.brand_data:
                website_text = fetch_brand_website(bd.get("domain", ""))
                st.session_state.brand_data["_website_text"] = website_text
                if website_text:
                    st.success(f"Website read successfully ({len(website_text)} chars). Generating topics...")
                else:
                    st.warning("Could not read website. Using form data only.")

        with st.spinner("Generating topics from your brand profile..."):
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

        st.write("**AI Generated Topics** (uncheck to remove):")
        for topic in st.session_state.topics:
            checked = topic in st.session_state.selected_topics
            new_checked = st.checkbox(f"✦ {topic}", value=checked, key=f"topic_check_{topic}")
            if new_checked and topic not in st.session_state.selected_topics:
                st.session_state.selected_topics.append(topic)
            elif not new_checked and topic in st.session_state.selected_topics:
                st.session_state.selected_topics.remove(topic)

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
            st.session_state.step = 1
            st.session_state.topics = []
            st.session_state.selected_topics = []
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
    st.caption("These prompts will be sent to each AI model to check if your brand is mentioned. 5 prompts per topic.")

    total_topics = len(st.session_state.selected_topics)
    total_prompts = total_topics * 5
    st.info(f"**{total_topics} topics x 5 prompts = {total_prompts} total prompts** per AI model")

    # Generate prompts for each selected topic if not done
    for topic in st.session_state.selected_topics:
        if topic not in st.session_state.prompts_by_topic:
            with st.spinner(f"Generating prompts for: {topic}..."):
                try:
                    # Pass full brand_data including cached website text
                    prompts = ai_generate_prompts(topic, st.session_state.brand_data)
                    st.session_state.prompts_by_topic[topic] = prompts
                    st.session_state.selected_prompts[topic] = list(prompts)
                except Exception as e:
                    st.warning(f"Could not generate prompts for '{topic}': {format_error_message(str(e))}")
                    st.session_state.prompts_by_topic[topic] = [topic]
                    st.session_state.selected_prompts[topic] = [topic]

    # Display prompts per topic in accordion style
    for topic in st.session_state.selected_topics:
        prompts = st.session_state.prompts_by_topic.get(topic, [])
        selected = st.session_state.selected_prompts.get(topic, [])

        with st.expander(f"**{topic}** ({len(selected)} prompts selected)", expanded=True):
            for prompt in prompts:
                checked = prompt in selected
                new_checked = st.checkbox(prompt, value=checked, key=f"prompt_{topic}_{prompt[:40]}")
                if new_checked and prompt not in st.session_state.selected_prompts.get(topic, []):
                    st.session_state.selected_prompts.setdefault(topic, []).append(prompt)
                elif not new_checked and prompt in st.session_state.selected_prompts.get(topic, []):
                    st.session_state.selected_prompts[topic].remove(prompt)

            # Add custom prompt
            custom_key = f"custom_prompt_{topic[:20]}"
            custom_prompt = st.text_input("+ Add prompt", key=custom_key,
                                          placeholder="Type a custom prompt and press Enter")
            if custom_prompt.strip() and custom_prompt.strip() not in prompts:
                if st.button("Add", key=f"add_prompt_btn_{topic[:20]}"):
                    st.session_state.prompts_by_topic[topic].append(custom_prompt.strip())
                    st.session_state.selected_prompts.setdefault(topic, []).append(custom_prompt.strip())
                    st.rerun()

    st.divider()
    col_back, col_next = st.columns([1, 3])
    with col_back:
        if st.button("← Back"):
            st.session_state.step = 2
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

        total_calls = len(all_prompts) * len(selected_tools)
        progress_bar = st.progress(0)
        status_text = st.empty()
        call_count = 0
        all_results = []
        exhausted_tools = set()

        for i, q in enumerate(all_prompts):
            status_text.text(f"Prompt {i+1}/{len(all_prompts)}: {q['query'][:70]}...")
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

                call_count += 1
                progress_bar.progress(min(call_count / total_calls, 1.0))

        status_text.text("Done.")
        st.session_state.all_results = all_results
        st.session_state.run_complete = True
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

            # ── Visibility per model ──────────────────────────────────────
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

                                # AI Response (collapsed)
                                with st.expander("Full AI Response"):
                                    response = r.get("response", "")
                                    # Highlight brand mentions
                                    highlighted = response.replace(
                                        brand_name,
                                        f"**:green[{brand_name}]**"
                                    )
                                    st.markdown(highlighted[:3000])

            # ── Competitor ranking ────────────────────────────────────────
            st.subheader("Top Competitor Brands Detected")
            if scores["competitor_ranking"]:
                comp_df = pd.DataFrame(scores["competitor_ranking"][:15], columns=["Brand", "Mentions"])
                comp_df["Share %"] = comp_df["Mentions"].apply(
                    lambda x: f"{round((x / scores['total_queries_run']) * 100)}%"
                )
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            else:
                st.info("No competitor brands detected.")

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

            # ── Re-run button ─────────────────────────────────────────────
            st.divider()
            if st.button("🔄 Run Again with Same Settings", type="primary"):
                st.session_state.run_complete = False
                st.session_state.all_results = []
                st.rerun()