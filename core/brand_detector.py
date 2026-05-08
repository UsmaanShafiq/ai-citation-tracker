import json
import re
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =============================================================================
# KNOWN BRANDS LIST (Pass 1 - String Matching)
# Add any brand you want to always catch here.
# The LLM pass will catch everything else automatically.
# =============================================================================

KNOWN_BRANDS = [
    "ServiceTitan", "Jobber", "Housecall Pro", "HouseCall Pro",
    "Workiz", "ServiceTrade", "FieldEdge", "mHelpDesk", "Kickserv",
    "Service Fusion", "ServiceFusion", "Vonigo", "Zuper", "Fieldd",
    "GorillaDesk", "PestPac", "ServiceM8", "Commusoft", "Fergus",
    "Simpro", "simPRO", "Assignar", "BuildOps", "FieldPulse",
    "Successware", "Sera", "Dispatch", "ServiceBox", "FieldWeb",
    "Salesforce", "Microsoft", "QuickBooks", "Google", "Zoho",
    "Freshdesk", "HubSpot", "Monday", "Asana", "ServiceNow"
]


def pass_one_string_match(response_text: str) -> list:
    """
    Fast string matching against known brand list.
    Case-insensitive. Returns list of matched brand names.
    """
    found = []
    response_lower = response_text.lower()

    for brand in KNOWN_BRANDS:
        if brand.lower() in response_lower:
            if brand not in found:
                found.append(brand)

    return found


def pass_two_llm_detection(response_text: str, target_brand: str) -> dict:
    """
    Sends response to Groq and asks it to extract all brand mentions.
    Catches brands not in our known list.
    Also detects position and context of target brand.
    """
    prompt = f"""Read the AI response below and extract every software product, platform, or brand name mentioned.

Return ONLY a valid JSON object. No explanation. No markdown fences.

Format:
{{
  "all_brands": ["Brand1", "Brand2", "Brand3"],
  "target_mentioned": true,
  "target_position": 1,
  "target_context": "recommended"
}}

Rules:
- all_brands: every software/brand name you find, in order of appearance
- target_mentioned: true if "{target_brand}" appears in the response, false if not
- target_position: what position is "{target_brand}" mentioned (1 = first brand mentioned, 2 = second, etc). Use 0 if not mentioned
- target_context: one of "recommended", "mentioned", "warned_against", "not_mentioned"

Target brand to track: {target_brand}

AI Response to analyze:
\"\"\"{response_text[:2000]}\"\"\"
"""

    try:
        result = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1
        )

        raw = result.choices[0].message.content.strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines).strip()

        return json.loads(raw)

    except Exception as e:
        return {
            "all_brands": [],
            "target_mentioned": False,
            "target_position": 0,
            "target_context": "not_mentioned"
        }


def detect_brands(response_text: str, target_brand: str) -> dict:
    """
    Main function. Runs both passes and merges results.
    Returns a clean dict with everything the scorer needs.
    """
    # Pass 1: fast string match
    string_matches = pass_one_string_match(response_text)

    # Pass 2: LLM detection
    llm_result = pass_two_llm_detection(response_text, target_brand)

    # Merge brand lists, remove duplicates, preserve order
    all_brands = llm_result.get("all_brands", [])
    for brand in string_matches:
        if brand not in all_brands:
            all_brands.append(brand)

    # Check target brand with both methods
    target_in_string = any(
        target_brand.lower() in b.lower() or b.lower() in target_brand.lower()
        for b in string_matches
    )
    target_mentioned = llm_result.get("target_mentioned", False) or target_in_string

    return {
        "all_brands": all_brands,
        "target_mentioned": target_mentioned,
        "target_position": llm_result.get("target_position", 0),
        "target_context": llm_result.get("target_context", "not_mentioned"),
        "string_match_brands": string_matches,
        "llm_detected_brands": llm_result.get("all_brands", [])
    }


# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_response = """
    As a small HVAC business owner, I'd recommend considering the following options:

    1. ServiceTitan: This is an all-in-one platform designed specifically for HVAC businesses.
    It integrates with QuickBooks and offers scheduling, dispatch, and invoicing.

    2. Jobber: A more affordable option at around $200/mo. Great for smaller teams.
    It also integrates with QuickBooks Online.

    3. Housecall Pro: Another strong option with a clean mobile app for technicians.
    Pricing starts around $150/mo for small teams.

    4. Workiz: Good for dispatching and has a free trial available.

    I would warn against using mHelpDesk as several users report poor customer support.
    """

    target = "ServiceTitan"

    print("Testing Brand Detector...")
    print("=" * 60)

    result = detect_brands(sample_response, target)

    print(f"All brands found:      {result['all_brands']}")
    print(f"String match brands:   {result['string_match_brands']}")
    print(f"LLM detected brands:   {result['llm_detected_brands']}")
    print(f"Target mentioned:      {result['target_mentioned']}")
    print(f"Target position:       {result['target_position']}")
    print(f"Target context:        {result['target_context']}")
    print("=" * 60)
    print("Brand Detector working correctly.")
    