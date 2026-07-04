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
    combined = (icp_text + " " + brand_name).lower()
    service_score = sum(1 for s in SERVICE_SIGNALS if s in combined)
    software_score = sum(1 for s in SOFTWARE_SIGNALS if s in combined)
    if service_score > software_score:
        return "service"
    return "software"


def confirm_with_brand_entities(brand_name: str, brand_entities: list) -> bool:
    """
    NEW: Checks if our target brand appears in the DataForSEO-detected brand_entities.
    This is a second confirmation layer — regex is still the final authority.
    Returns True if brand found in DataForSEO's auto-detected entity list.
    brand_entities: list of {title, category, urls} from the scraper response.
    """
    if not brand_entities or not brand_name:
        return False
    brand_lower = brand_name.lower().strip()
    for entity in brand_entities:
        entity_name = (entity.get("title") or "").lower().strip()
        if not entity_name:
            continue
        # Exact match or one contains the other
        if entity_name == brand_lower:
            return True
        if brand_lower in entity_name or entity_name in brand_lower:
            return True
    return False


def pass_one_string_match(
    response_text: str,
    business_type: str = "software",
    target_brand: str = "",
    user_competitors: list = None
) -> list:
    import re as _re
    found = []
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
    import re as _re

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
            if raw.startswith("```"):
                raw = _re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            return raw
        except Exception:
            pass

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

    return '{"all_brands": [], "target_mentioned": false, "target_context": "not_mentioned"}'


def pass_two_llm_detection(
    response_text: str,
    target_brand: str,
    business_type: str = "software"
) -> dict:
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
        f"  \"target_context\": \"recommended\"\n"
        f"}}\n\n"
        f"Rules:\n"
        f"- all_brands: every brand/org mentioned in the response, with category assigned\n"
        f"- Only include brands that appear as recommendations, suggestions, or comparisons\n"
        f"- target_mentioned: true ONLY if \"{target_brand}\" appears explicitly by name\n"
        f"- target_context: one of recommended / mentioned / warned_against / not_mentioned\n\n"
        f"Target brand to track: {target_brand}\n\n"
        f"AI Response:\n"
        f"\"\"\"{response_text[:2000]}\"\"\""
    )

    try:
        raw = _call_llm_for_detection(prompt)
        parsed = json.loads(raw)

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
            "target_context": "not_mentioned"
        }


def detect_brands(
    response_text: str,
    target_brand: str,
    icp_text: str = "",
    business_type: str = None,
    user_competitors: list = None,
    custom_exclusions: list = None,
    brand_entities: list = None,       # NEW: from DataForSEO scraper
) -> dict:
    """
    Main function. Auto-detects business type and runs both passes.
    brand_entities: optional list from DataForSEO scraper for second confirmation layer.
    """
    if business_type is None:
        business_type = detect_business_type(icp_text, target_brand)

    # Pass 1: string match on target brand + user competitors only
    string_matches = pass_one_string_match(
        response_text,
        business_type=business_type,
        target_brand=target_brand,
        user_competitors=user_competitors or []
    )

    # Pass 2: LLM extracts all brand mentions intelligently
    llm_result = pass_two_llm_detection(response_text, target_brand, business_type)

    all_brands = llm_result.get("all_brands", [])
    for brand in string_matches:
        if brand not in all_brands:
            all_brands.append(brand)

    if custom_exclusions:
        custom_lower = [t.lower() for t in custom_exclusions]
        all_brands = [b for b in all_brands if b.lower() not in custom_lower]

    # GROUND TRUTH: regex is the final authority
    import re as _re

    def build_brand_variants(name: str) -> list:
        variants = [name]
        if " " not in name:
            spaced = _re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
            if spaced != name:
                variants.append(spaced)
            lower = name.lower()
            for i in range(2, len(lower) - 1):
                variants.append(lower[:i] + " " + lower[i:])
        if " " in name:
            variants.append(name.replace(" ", ""))
            variants.append(name.replace(" ", "").lower())
        return list(dict.fromkeys(variants))

    target_in_string = False
    for variant in build_brand_variants(target_brand):
        pattern = _re.compile(r"\b" + _re.escape(variant) + r"\b", _re.IGNORECASE)
        if pattern.search(response_text):
            target_in_string = True
            break

    target_mentioned = target_in_string  # regex is final authority

    target_context = "not_mentioned"
    if target_in_string:
        target_context = llm_result.get("target_context", "mentioned")
        if target_context == "not_mentioned":
            target_context = "mentioned"

    # NEW: DataForSEO brand entity confirmation (second layer, non-authoritative)
    dataforseo_confirmed = confirm_with_brand_entities(
        target_brand, brand_entities or []
    )

    # NEW: Also enrich all_brands with DataForSEO brand entities
    # Add any brands DataForSEO detected that our LLM missed
    if brand_entities:
        for entity in brand_entities:
            entity_name = (entity.get("title") or "").strip()
            if entity_name and entity_name.lower() != target_brand.lower():
                if entity_name not in all_brands:
                    all_brands.append(entity_name)

    return {
        "all_brands":             all_brands,
        "competitor_brands":      llm_result.get("competitor_brands", []),
        "government_brands":      llm_result.get("government_brands", []),
        "dominant_brands":        llm_result.get("dominant_brands", []),
        "target_mentioned":       target_mentioned,
        "target_context":         target_context,
        "dataforseo_confirmed":   dataforseo_confirmed,   # NEW
        "string_match_brands":    string_matches,
        "llm_detected_brands":    llm_result.get("all_brands", []),
        "business_type_detected": business_type,
    }


if __name__ == "__main__":
    test_response = "We recommend Acme Corp and BrandX for your needs. BrandY is also popular."
    result = detect_brands(test_response, target_brand="Acme Corp")
    print(f"Target mentioned: {result['target_mentioned']}")
    print(f"All brands: {result['all_brands']}")
    print(f"Context: {result['target_context']}")
    print(f"DataForSEO confirmed: {result['dataforseo_confirmed']}")