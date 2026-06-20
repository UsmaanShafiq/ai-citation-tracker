import json
import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# BUSINESS TYPE DETECTION
# =============================================================================

SOFTWARE_SIGNALS = [
    "software", "platform", "tool", "saas", "app", "dashboard",
    "integration", "api", "pricing per seat", "free trial", "subscription"
]

SERVICE_SIGNALS = [
    "agency", "consultancy", "consultant", "studio", "service",
    "content partner", "writing team", "managed", "done for you",
    "retainer", "we write", "our team writes"
]



def detect_business_type(icp_text: str = "", brand_name: str = "") -> str:
    """
    Detects whether the brand is primarily a service/agency or a software/product.
    Returns "service" or "software" as a general signal for the LLM.
    This is used only as context hint - not for filtering results.
    """
    combined = (icp_text + " " + brand_name).lower()
    service_score = sum(1 for s in SERVICE_SIGNALS if s in combined)
    software_score = sum(1 for s in SOFTWARE_SIGNALS if s in combined)
    if service_score > software_score:
        return "service"
    return "software"


def pass_one_string_match(
    response_text: str,
    business_type: str = "software",
    target_brand: str = "",
    user_competitors: list = None
) -> list:
    """
    Pass 1: String matching to catch brands the LLM might miss.
    Only checks:
    1. The target brand itself (whole-word match)
    2. User-supplied competitors from the form (whole-word match)
    Does NOT use a hardcoded brand list - that approach breaks for global users
    across different industries. The LLM pass handles broader brand extraction.
    """
    import re as _re
    found = []

    # Build the list of brands to check from user-supplied data only
    brands_to_check = []
    if target_brand:
        brands_to_check.append(target_brand)
    if user_competitors:
        brands_to_check.extend([c.strip() for c in user_competitors if c.strip()])

    for brand in brands_to_check:
        pattern = _re.compile(r"\b" + _re.escape(brand) + r"\b", _re.IGNORECASE)
        if pattern.search(response_text):
            if brand not in found:
                found.append(brand)

    return found


def _call_llm_for_detection(prompt: str) -> str:
    """
    Calls the best available LLM for brand detection.
    Tries OpenAI first, then Groq as fallback.
    Returns cleaned JSON string.
    """
    import re as _re

    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and "paste_your" not in openai_key.lower():
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1
            )
            raw = result.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = _re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            return raw
        except Exception:
            pass

    # Fallback: Try Groq
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key and "paste_your" not in groq_key.lower():
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            result = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1
            )
            raw = result.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = _re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            return raw
        except Exception:
            pass

    # If both fail, return a safe default JSON
    return '{"all_brands": [], "target_mentioned": false, "target_position": 0, "target_context": "not_mentioned"}'



def pass_two_llm_detection(
    response_text: str,
    target_brand: str,
    business_type: str = "software"
) -> dict:
    """
    Universal LLM-based brand detection with dynamic classification.
    At runtime, classifies every detected brand into:
    - government: official bodies, regulatory agencies, government databases
    - dominant: large established commercial platforms (Google, Microsoft, etc.)
    - competitor: same-scale commercial tools and services
    Works for any industry without any hardcoding.
    """
    prompt = (
        f"Read the AI response below and extract all brands, tools, organizations, or services mentioned.\n\n"
        f"For each brand found, classify it into one of three categories:\n"
        f"- \"government\": government agencies, official regulatory bodies, national/international databases, "
        f"standards organizations (e.g. USPTO, FDA, WHO, SEC, WIPO, EPO, IEEE, ISO, any .gov entity)\n"
        f"- \"dominant\": large established commercial platforms or companies that dominate their industry "
        f"(e.g. Google, Microsoft, Amazon, Apple, Salesforce, Adobe) - companies worth billions\n"
        f"- \"competitor\": commercial tools, startups, SaaS products, agencies, or services at a similar "
        f"scale to the target brand that could be a real alternative\n\n"
        f"Return ONLY a valid JSON object. No explanation. No markdown fences.\n\n"
        f"Format:\n"
        f"{{\n"
        f"  \"all_brands\": [\n"
        f"    {{\"name\": \"Brand1\", \"category\": \"competitor\"}},\n"
        f"    {{\"name\": \"USPTO\", \"category\": \"government\"}},\n"
        f"    {{\"name\": \"Google\", \"category\": \"dominant\"}}\n"
        f"  ],\n"
        f"  \"target_mentioned\": true,\n"
        f"  \"target_position\": 1,\n"
        f"  \"target_context\": \"recommended\"\n"
        f"}}\n\n"
        f"Rules:\n"
        f"- all_brands: every brand/org mentioned in the response, with category assigned\n"
        f"- Only include brands that appear as recommendations, suggestions, or comparisons\n"
        f"- target_mentioned: true ONLY if \"{target_brand}\" appears explicitly by name\n"
        f"- target_position: position of \"{target_brand}\" (1=first, 0=not mentioned)\n"
        f"- target_context: one of recommended / mentioned / warned_against / not_mentioned\n\n"
        f"Target brand to track: {target_brand}\n\n"
        f"AI Response:\n"
        f"\"\"\"{response_text[:2000]}\"\"\""
    )

    try:
        raw = _call_llm_for_detection(prompt)
        parsed = json.loads(raw)

        # Normalize all_brands to always be a flat list of strings
        # while preserving category info separately
        raw_brands = parsed.get("all_brands", [])
        normalized_brands = []
        government_brands = []
        dominant_brands = []
        competitor_brands = []

        for item in raw_brands:
            if isinstance(item, dict):
                name = item.get("name", "").strip()
                category = item.get("category", "competitor")
            else:
                # Fallback if LLM returns plain strings
                name = str(item).strip()
                category = "competitor"

            if not name:
                continue

            normalized_brands.append(name)
            if category == "government":
                government_brands.append(name)
            elif category == "dominant":
                dominant_brands.append(name)
            else:
                competitor_brands.append(name)

        parsed["all_brands"] = normalized_brands
        parsed["government_brands"] = government_brands
        parsed["dominant_brands"] = dominant_brands
        parsed["competitor_brands"] = competitor_brands
        return parsed

    except Exception:
        return {
            "all_brands": [],
            "government_brands": [],
            "dominant_brands": [],
            "competitor_brands": [],
            "target_mentioned": False,
            "target_position": 0,
            "target_context": "not_mentioned"
        }



def detect_brands(
    response_text: str,
    target_brand: str,
    icp_text: str = "",
    business_type: str = None,
    user_competitors: list = None,
    custom_exclusions: list = None
) -> dict:
    """
    Main function. Auto-detects business type and runs both passes.
    user_competitors: brands entered by user in frontend - always included
    custom_exclusions: brands entered by user to exclude
    """
    if business_type is None:
        business_type = detect_business_type(icp_text, target_brand)

    # Pass 1: string match on target brand + user competitors only
    # No hardcoded lists - works correctly for any industry, any user
    string_matches = pass_one_string_match(
        response_text,
        business_type=business_type,
        target_brand=target_brand,
        user_competitors=user_competitors or []
    )

    # Pass 2: LLM extracts all brand mentions intelligently from the response
    llm_result = pass_two_llm_detection(response_text, target_brand, business_type)

    all_brands = llm_result.get("all_brands", [])
    for brand in string_matches:
        if brand not in all_brands:
            all_brands.append(brand)

    # Apply custom exclusions from frontend (user-defined, not hardcoded)
    if custom_exclusions:
        custom_lower = [t.lower() for t in custom_exclusions]
        all_brands = [b for b in all_brands if b.lower() not in custom_lower]

    # GROUND TRUTH check: the brand name must actually appear in the response text.
    # The regex is the final authority - LLM cannot override this.
    # We also generate smart variations to handle cases where the user typed
    # the brand name differently from how it appears in AI responses.
    # e.g. "siegemedia" typed without space matches "Siege Media" in responses.
    import re as _re

    def build_brand_variants(name: str) -> list:
        """Generate smart variations of a brand name for matching."""
        variants = [name]

        # If no spaces: try inserting a space before each capital or word boundary
        # e.g. "siegemedia" -> "siege media", "SiegeMedia" -> "Siege Media"
        if " " not in name:
            # lowercase with space inserted at camelCase boundaries
            spaced = _re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
            if spaced != name:
                variants.append(spaced)
            # try all lowercase with space at common split points
            # e.g. "siegemedia" - try splitting at each position
            lower = name.lower()
            for i in range(2, len(lower) - 1):
                variants.append(lower[:i] + " " + lower[i:])

        # Also try removing spaces (in case user typed "Siege Media" but AI says "SiegeMedia")
        if " " in name:
            variants.append(name.replace(" ", ""))
            variants.append(name.replace(" ", "").lower())

        return list(dict.fromkeys(variants))  # deduplicate, preserve order

    target_in_string = False
    for variant in build_brand_variants(target_brand):
        pattern = _re.compile(r"\b" + _re.escape(variant) + r"\b", _re.IGNORECASE)
        if pattern.search(response_text):
            target_in_string = True
            break

    # Only mark as mentioned if the brand ACTUALLY appears in the text.
    # The LLM result is used for context/position only when string match confirms presence.
    target_mentioned = target_in_string

    # Get context and position from LLM only when we confirmed the brand is present
    target_context = "not_mentioned"
    target_position = 0
    if target_in_string:
        target_context = llm_result.get("target_context", "mentioned")
        target_position = llm_result.get("target_position", 0)
        # If LLM says not_mentioned but string match found it, override context
        if target_context == "not_mentioned":
            target_context = "mentioned"

    return {
        "all_brands": all_brands,
        "competitor_brands": llm_result.get("competitor_brands", []),
        "government_brands": llm_result.get("government_brands", []),
        "dominant_brands": llm_result.get("dominant_brands", []),
        "target_mentioned": target_mentioned,
        "target_position": target_position,
        "target_context": target_context,
        "string_match_brands": string_matches,
        "llm_detected_brands": llm_result.get("all_brands", []),
        "business_type_detected": business_type
    }


# Run this file directly to test: python core/brand_detector.py
if __name__ == "__main__":
    # Quick sanity test
    test_response = "We recommend Acme Corp and BrandX for your needs. BrandY is also popular."
    result = detect_brands(test_response, target_brand="Acme Corp")
    print(f"Target mentioned: {result['target_mentioned']}")
    print(f"All brands: {result['all_brands']}")
    print(f"Context: {result['target_context']}")