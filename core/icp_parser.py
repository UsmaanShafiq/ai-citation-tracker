import json
import os
import importlib
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]


def parse_icp_with_gemini(icp_text: str) -> dict:
    key = os.getenv("GEMINI_API_KEY")
    prompt = build_prompt(icp_text)

    # Try modern SDK first: google-genai
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
                return extract_json((response.text or ""))
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

    # Fallback to legacy SDK: google-generativeai
    try:
        genai_legacy = importlib.import_module("google.generativeai")
        genai_legacy.configure(api_key=key)
        for model_name in GEMINI_MODELS:
            try:
                model = genai_legacy.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return extract_json((response.text or ""))
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


def parse_icp_with_groq(icp_text: str) -> dict:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = build_prompt(icp_text)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.1
    )
    return extract_json(response.choices[0].message.content)


def build_prompt(icp_text: str) -> str:
    return f"""You are a B2B marketing analyst. Extract structured buyer profile data from the ICP document below.

Return ONLY valid JSON. No explanation. No markdown fences. No extra text.

Format exactly like this:
{{
  "industries": ["HVAC", "plumbing", "electrical"],
  "company_sizes": ["10-50 employees", "50-200 employees"],
  "geographies": ["United States", "Canada"],
  "buyer_roles": ["Operations Manager", "Owner"],
  "pain_points": ["scheduling chaos", "no job visibility"],
  "budget_ranges": ["under $300/mo", "$300 to $1000/mo"],
  "tech_stacks": ["QuickBooks", "Google Sheets"],
  "company_stages": ["growing SMB", "bootstrapped startup"],
  "compliance_needs": ["none", "SOC 2"]
}}

ICP Document:
{icp_text}"""


def extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()
    return json.loads(raw)


def validate_icp_data(icp_data: dict) -> dict:
    defaults = {
        "industries": ["general business"],
        "company_sizes": ["small to medium"],
        "geographies": ["United States"],
        "buyer_roles": ["decision maker"],
        "pain_points": ["operational inefficiency"],
        "budget_ranges": ["mid-range budget"],
        "tech_stacks": ["standard tools"],
        "company_stages": ["established business"],
        "compliance_needs": ["none"]
    }
    for key, default_val in defaults.items():
        if key not in icp_data or not icp_data[key]:
            icp_data[key] = default_val
    return icp_data


def parse_and_validate(icp_text: str, selected_tools: list = None) -> dict:
    """
    Main function. Uses only selected tool families for ICP parsing.
    """
    selected_tools = selected_tools or []
    backends = []
    if "Gemini" in selected_tools:
        backends.append("gemini")
    if "Groq_Llama3" in selected_tools or "Groq_Mixtral" in selected_tools:
        backends.append("groq")

    if not backends:
        raise RuntimeError("No selected tools available for ICP parsing.")

    last_error = None
    for backend in backends:
        try:
            if backend == "gemini":
                raw = parse_icp_with_gemini(icp_text)
            else:
                raw = parse_icp_with_groq(icp_text)
            clean = validate_icp_data(raw)
            return clean
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str or "quota" in error_str:
                continue
            raise

    raise RuntimeError(f"rate_limit_exceeded: {last_error}")