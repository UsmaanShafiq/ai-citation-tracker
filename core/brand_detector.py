import json
import os
from groq import Groq
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

# =============================================================================
# KNOWN SERVICE BRANDS (agencies, studios, consultancies)
# =============================================================================

KNOWN_SERVICE_BRANDS = [
    "Animalz", "Grow and Convert", "Foundation", "Siege Media",
    "Codeless", "Optimist", "Beam Content", "Omniscient Digital",
    "Fenwick", "Pepper Content", "Verblio", "Contently",
    "Skyword", "NewsCred", "ClearVoice", "Brafton",
    "Column Five", "TopRank Marketing", "Knotch",
    "Scripted", "WriterAccess", "Crowd Content",
    "Content Harmony", "Eucalypt", "Concurate",
    "Influence and Co", "Rock Content", "Compose.ly",
    "Content Cucumber", "Fractl", "Relevance",
    "Walker Sands", "Velocity Partners",
]

# =============================================================================
# SOFTWARE TOOLS TO EXCLUDE for service businesses
# =============================================================================

SOFTWARE_TOOLS_TO_EXCLUDE = [
    "HubSpot", "Ahrefs", "SEMrush", "Google", "Google Analytics",
    "Salesforce", "Marketo", "WordPress", "Trello", "Asana",
    "Notion", "Hootsuite", "Buffer", "Sprout Social", "Moz",
    "Clearscope", "MarketMuse", "Surfer SEO", "BuzzSumo",
    "CoSchedule", "Loomly", "Later", "Mailchimp", "ActiveCampaign",
    "Pardot", "Eloqua", "Monday", "ClickUp", "Basecamp",
    "Slack", "Microsoft", "Zoom", "Webflow", "Squarespace",
    "QuickBooks", "Stripe", "Intercom", "Zendesk", "Freshdesk",
    "ServiceNow", "Zoho", "Pipedrive", "LinkedIn", "Twitter",
    "Facebook", "Instagram", "YouTube", "TikTok", "Reddit",
]


def detect_business_type(icp_text: str = "", brand_name: str = "") -> str:
    combined = (icp_text + " " + brand_name).lower()
    service_score = sum(1 for s in SERVICE_SIGNALS if s in combined)
    software_score = sum(1 for s in SOFTWARE_SIGNALS if s in combined)
    if service_score > software_score:
        return "service"
    return "software"


def pass_one_string_match(response_text: str, business_type: str = "software") -> list:
    found = []
    response_lower = response_text.lower()

    if business_type == "service":
        brand_list = KNOWN_SERVICE_BRANDS
    else:
        brand_list = KNOWN_SERVICE_BRANDS + [
            "ServiceTitan", "Jobber", "Housecall Pro", "Workiz",
            "ServiceTrade", "FieldEdge", "mHelpDesk", "Kickserv",
            "GorillaDesk", "PestPac", "ServiceM8", "Commusoft",
            "Simpro", "BuildOps", "FieldPulse", "Successware",
        ]

    for brand in brand_list:
        if brand.lower() in response_lower:
            if brand not in found:
                found.append(brand)

    return found


def pass_two_llm_detection(
    response_text: str,
    target_brand: str,
    business_type: str = "software"
) -> dict:

    if business_type == "service":
        brand_instruction = """IMPORTANT: This is for a SERVICE BUSINESS, specifically a content marketing agency.

A competitor is ONLY a brand recommended as a service provider, agency, consultancy, or content studio.

Do NOT include any of these as competitors. They are software tools not service providers:
HubSpot, Ahrefs, SEMrush, Google, Google Analytics, Salesforce, Marketo, WordPress,
Trello, Asana, Notion, Hootsuite, Buffer, Moz, Clearscope, MarketMuse, CoSchedule,
Monday, ClickUp, Slack, Microsoft, Zoom, LinkedIn, Mailchimp, ActiveCampaign.

Only include brands that are content agencies, content studios, or B2B content consultancies."""
    else:
        brand_instruction = """Extract every software product, platform, or SaaS tool mentioned.
Include all software brands and tools that appear as recommendations."""

    prompt = f"""Read the AI response below and extract brand mentions.

{brand_instruction}

Return ONLY a valid JSON object. No explanation. No markdown fences.

Format:
{{
  "all_brands": ["Brand1", "Brand2"],
  "target_mentioned": true,
  "target_position": 1,
  "target_context": "recommended"
}}

Rules:
- all_brands: brands matching the criteria above, in order of appearance
- target_mentioned: true if "{target_brand}" appears in the response
- target_position: position of "{target_brand}" (1 = first, 0 = not mentioned)
- target_context: one of "recommended", "mentioned", "warned_against", "not_mentioned"

Target brand to track: {target_brand}

AI Response to analyze:
\"\"\"{response_text[:2000]}\"\"\"
"""

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
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

        parsed = json.loads(raw)

        # Post-process: always remove software tools for service businesses
        if business_type == "service":
            exclude_lower = [t.lower() for t in SOFTWARE_TOOLS_TO_EXCLUDE]
            parsed["all_brands"] = [
                b for b in parsed.get("all_brands", [])
                if b.lower() not in exclude_lower
            ]

        return parsed

    except Exception as e:
        return {
            "all_brands": [],
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

    # Add user competitors to known brands list temporarily
    if user_competitors:
        for comp in user_competitors:
            comp_clean = comp.strip()
            if comp_clean and comp_clean not in KNOWN_SERVICE_BRANDS:
                KNOWN_SERVICE_BRANDS.append(comp_clean)

    string_matches = pass_one_string_match(response_text, business_type)
    llm_result = pass_two_llm_detection(response_text, target_brand, business_type)

    all_brands = llm_result.get("all_brands", [])
    for brand in string_matches:
        if brand not in all_brands:
            all_brands.append(brand)

    # Final filter: always remove software tools for service businesses
    if business_type == "service":
        exclude_lower = [t.lower() for t in SOFTWARE_TOOLS_TO_EXCLUDE]
        all_brands = [b for b in all_brands if b.lower() not in exclude_lower]

    # Apply custom exclusions from frontend
    if custom_exclusions:
        custom_lower = [t.lower() for t in custom_exclusions]
        all_brands = [b for b in all_brands if b.lower() not in custom_lower]

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
        "llm_detected_brands": llm_result.get("all_brands", []),
        "business_type_detected": business_type
    }


# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Test 1: Agency response (HubSpot and Ahrefs should be excluded)
    agency_response = """
    If you are looking for a B2B content agency that focuses on pipeline,
    I would look at Animalz, Grow and Convert, or Siege Media.
    Animalz is known for deep SaaS content. Grow and Convert focuses
    specifically on bottom-funnel content that drives signups.
    Some teams also use HubSpot for CMS and Ahrefs for keyword research
    but those are tools not agencies. Foundation is also worth evaluating
    for thought leadership content.
    """

    print("Test 1: Agency response")
    print("=" * 60)
    result = detect_brands(
        agency_response,
        target_brand="Concurate",
        icp_text="content marketing agency retainer B2B SaaS"
    )
    print(f"Business type:    {result['business_type_detected']}")
    print(f"Brands found:     {result['all_brands']}")
    print(f"HubSpot excluded: {'HubSpot' not in result['all_brands']}")
    print(f"Ahrefs excluded:  {'Ahrefs' not in result['all_brands']}")
    print()

    # Test 2: Software response
    software_response = """
    For HVAC scheduling, ServiceTitan is the most popular.
    Jobber is more affordable. Housecall Pro has a great mobile app.
    All three integrate with QuickBooks.
    """

    print("Test 2: Software response")
    print("=" * 60)
    result2 = detect_brands(
        software_response,
        target_brand="ServiceTitan",
        icp_text="HVAC software scheduling tool"
    )
    print(f"Business type: {result2['business_type_detected']}")
    print(f"Brands found:  {result2['all_brands']}")
    print("=" * 60)
    print("Tests complete.")