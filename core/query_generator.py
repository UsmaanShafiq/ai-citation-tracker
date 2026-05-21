import json
import importlib
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]

TARGET_QUERY_COUNT = 25

QUERY_CATEGORIES = [
    {"name": "provider_evaluating", "count": 9},
    {"name": "proof_seeking", "count": 8},
    {"name": "decision_ready", "count": 8},
]


def call_llm(
    prompt: str,
    preferred_backend: str = "groq",
    allowed_backends: list = None
) -> str:
    if allowed_backends:
        backends = [b for b in allowed_backends if b in {"groq", "gemini"}]
    else:
        backends = ["gemini", "groq"] if preferred_backend == "gemini" else ["groq", "gemini"]

    if not backends:
        raise RuntimeError("No allowed LLM backends configured.")

    last_error = None

    for backend in backends:
        try:
            if backend == "gemini":
                key = os.getenv("GEMINI_API_KEY", "")
                if not key or "paste_your" in key.lower():
                    continue
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
                            return (response.text or "").strip()
                        except Exception as e:
                            msg = str(e).lower()
                            if any(x in msg for x in ["404", "not_found", "not found", "429", "rate_limit", "resource_exhausted", "quota"]):
                                continue
                            raise
                    raise RuntimeError("No supported Gemini model available.")
                except (ImportError, ModuleNotFoundError):
                    pass

                try:
                    genai_legacy = importlib.import_module("google.generativeai")
                    genai_legacy.configure(api_key=key)
                    for model_name in GEMINI_MODELS:
                        try:
                            model = genai_legacy.GenerativeModel(model_name)
                            response = model.generate_content(prompt)
                            return (response.text or "").strip()
                        except Exception as e:
                            msg = str(e).lower()
                            if any(x in msg for x in ["404", "not_found", "not found", "429", "rate_limit", "resource_exhausted", "quota"]):
                                continue
                            raise
                    raise RuntimeError("No supported Gemini model available.")
                except (ImportError, ModuleNotFoundError):
                    raise RuntimeError("Gemini SDK not installed. Run: pip install google-genai")

            else:
                key = os.getenv("GROQ_API_KEY", "")
                if not key or "paste_your" in key.lower():
                    continue
                from groq import Groq
                client = Groq(api_key=key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8000,
                    temperature=0.8
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if any(x in error_str for x in ["rate", "429", "quota", "resource_exhausted"]):
                continue
            raise

    raise RuntimeError(f"All backends failed or rate limited. Last error: {last_error}")


def generate_all_queries(
    icp_data: dict,
    company_name: str = "",
    website_url: str = None,
    competitors: list = None,
    business_type: str = "",
    target_keywords: list = None,
    geography: str = "United States",
    preferred_backend: str = "groq",
    allowed_backends: list = None
) -> list:

    competitors_text = ", ".join(competitors) if competitors else "not specified"
    keywords_text = ", ".join(target_keywords) if target_keywords else "not specified"

    # Use geography from parameter, fall back to ICP data
    geo = geography or ", ".join(icp_data.get("geographies", ["United States"]))

    icp_formatted = f"""
Industries: {', '.join(icp_data.get('industries', []))}
Company sizes: {', '.join(icp_data.get('company_sizes', []))}
Geographies: {geo}
Buyer roles: {', '.join(icp_data.get('buyer_roles', []))}
Pain points: {', '.join(icp_data.get('pain_points', []))}
Budget ranges: {', '.join(icp_data.get('budget_ranges', []))}
Tech stacks: {', '.join(icp_data.get('tech_stacks', []))}
Company stages: {', '.join(icp_data.get('company_stages', []))}
Compliance needs: {', '.join(icp_data.get('compliance_needs', []))}
"""

    business_type_instruction = ""
    if business_type and business_type.strip() and business_type.lower() not in ["auto detect", "not specified", ""]:
        business_type_instruction = f"""**Business Type:** {business_type}
(This field is provided. Accept it as ground truth. Do not override or reinterpret it. Build your entire understanding of the buying motion, vocabulary, and filter dimensions around this business type.)"""
    else:
        business_type_instruction = f"""**Business Type:** not specified
(Infer the business type from the ICP and company name before generating any queries.)"""

    keyword_instruction = ""
    if target_keywords:
        keyword_instruction = f"""
## TARGET KEYWORDS
These are the specific phrases the user wants {company_name} to be discovered for in AI-generated answers:
{keywords_text}

Weave these keywords naturally into queries where they fit the buyer language and context.
Do not force them into every query. Use them where a real buyer would actually use that phrase.
Aim for meaningful coverage: each keyword should appear across multiple queries in natural varied forms.
Never make a query feel constructed around a keyword. The buyer situation and specificity must always come first.
"""

    prompt = f"""You are an expert B2B market researcher and buyer behavior analyst. Your job is to generate hyper-specific, bottom-of-the-funnel AI search queries that real buyers type when they are close to making a purchase decision in {company_name}'s category.

## INPUTS
**Company Name:** {company_name}

**Ideal Client Profile (ICP):**
{icp_formatted}

{business_type_instruction}

**Competitor Names (optional):** {competitors_text}
{keyword_instruction}

## STEP 1 — ESTABLISH BUSINESS CONTEXT

{"Accept the provided business type as ground truth. Do not second-guess or reinterpret it. Build your entire understanding of the buying motion, vocabulary, and filter dimensions around this business type." if business_type and business_type.strip() and business_type.lower() not in ["auto detect", "not specified", ""] else f"""Infer the business type by answering these questions internally before proceeding:
1. What is {company_name} actually selling — a software product, a professional service, an agency offering, a managed service, a marketplace, or something else? Read the ICP carefully. Do not assume.
2. How does buying actually happen in this category?
   - Software: buyer compares tools, does trials, checks G2, evaluates pricing tiers
   - Agency or consulting: buyer evaluates expertise, reads case studies, asks for referrals, judges thought leadership
   - Managed service: buyer focuses on outcomes and reliability, not features
3. What language does the buyer actually use when they are this close to a decision?
4. Which filter dimensions are irrelevant for this business type and should be dropped entirely?
Complete this reasoning before touching query generation."""}

## STEP 2 — BOTTOM-OF-FUNNEL ONLY

Every single query in this output must reflect a buyer who is close to a purchase decision. This is not awareness. This is not education. This is a buyer who has already done their research and is now in one of these three modes:

**provider_evaluating** — actively comparing specific companies, approaches, or options side by side
**proof_seeking** — looking for validation that a specific solution will work for their exact situation before committing
**decision_ready** — essentially ready to buy, doing final checks, looking for confirmation or a nudge

Do not generate queries for pain_aware or solution_aware stages. If a query sounds like the buyer is still figuring out what they need, it does not belong in this output.

Signs a query is genuinely bottom-of-funnel:
- It names or implies specific options being compared
- It asks for proof, references, case studies, or outcomes for a specific context
- It asks what others in a very specific situation actually chose
- It carries the weight of an imminent decision ("we need to decide by end of quarter," "about to sign," "final shortlist")
- It is specific enough that a generic answer would not satisfy it

## STEP 3 — INHABIT THE BUYER AT DECISION TIME

This buyer has already gone through awareness and consideration. They have probably talked to vendors, sat through demos, and read reviews. They are now trying to answer one of these underlying questions:
- "Is this the right choice for my exact situation?"
- "Has anyone like me done this and did it work?"
- "What am I missing before I commit?"
- "Is option A actually better than option B for a company like mine?"
- "What do people say about this when they are not trying to sell me something?"

Channel how this buyer would phrase these on Reddit, in a Slack community, on G2, or directly to an AI assistant. Use their words, their doubts, their specific context. Not seller language.

## FILTER DIMENSIONS

Each query MUST include at least 5 filter dimensions. Only use filters meaningful for the confirmed or inferred business type. Vary combinations across queries so no two feel alike.

### Universal Filters
- Geography
- Company Size / Stage
- Industry Vertical
- Job Title / Persona
- Pain Point (the specific frustration that has brought them to the edge of a decision)
- Trigger / Timing (end of quarter, after a failed hire, post-funding, contract renewal coming up)
- Emotional State (tired of waiting, under pressure to show results, anxious about making the wrong call)
- Situational Context

### For Software / SaaS Products
- Team Size
- Budget Signal (approved budget of X, cost-per-seat concern, switching cost consideration)
- Feature Need
- Integration Need
- Competitor Context (head-to-head comparison, migration from a specific tool, shortlist of two)

### For Agencies / Consulting / Professional Services
- Engagement Model (retainer vs project-based, embedded vs advisory, trial project before committing)
- Outcome / Proof Focus (pipeline results, content that ranked, clients they have worked with in this niche)
- Past Experience with Category (burned by a previous agency, first time outsourcing, had bad content before)
- Internal Capability Gap (no writer internally, founder has been doing it and cannot scale, junior team needs senior oversight)
- Cultural / Execution Fit (needs someone who understands technical buyers, wants strategic input not just execution)
- Proof / Trust Signal (wants to see work in their specific vertical, checking references, evaluating thought leadership quality)
- Competitor Context (comparing two agencies, weighing agency vs in-house hire, evaluating a shortlist)

## QUERY CONSTRUCTION RULES

- Every query must be bottom-of-funnel, no exceptions
- Queries must read like something a real buyer would type into an AI assistant or post in a Slack community when they are days or weeks away from a decision
- Use the buyer language not the seller language. No "robust," "seamless," "end-to-end," "best-in-class," "innovative," "strategic partner"
- At least 6 queries must reference a competitor naturally (only if competitors were provided)
- At least 4 queries must carry decision-pressure language ("need to decide," "about to sign," "final two options," "last thing I need to figure out")
- At least 4 queries must be proof-seeking, asking for real outcomes, references, or case studies in a specific context
- Queries must span at least 4 different industry verticals plausible for this ICP
- No two queries should have the same combination of filters
- Zero marketing or vendor language anywhere in the queries
- For provider_evaluating and decision_ready queries: you MAY mention {company_name} by name when it makes the query more realistic and specific, as a real buyer close to signing would
- For proof_seeking queries: do NOT mention {company_name} by name, keep them unbiased validation queries

## GOOD VS BAD EXAMPLES

For a B2B content marketing agency:

Good: "we're down to two content agencies for our Series B DevOps SaaS, one has more clients in our space but the other's writing quality is noticeably better, how do other B2B SaaS founders think about this tradeoff when they're about to sign"

Good: "looking for a B2B SaaS content marketing agency that has actually moved pipeline not just traffic, does anyone have direct experience with {company_name} or similar shops, we're a 60-person security software company in the US and need to decide in the next two weeks"

Bad: "what is the best content marketing agency for SaaS companies"
Why bad: too early stage, no specificity, no decision pressure, no filters

For a B2B SaaS tool:

Good: "we've been trialing both Tool A and Tool B for three weeks with our 8-person RevOps team, Tool A has better reporting but Tool B connects natively with our Salesforce setup, has anyone dealt with this exact tradeoff and what did you end up choosing"

Bad: "best RevOps tools for SaaS companies"
Why bad: awareness stage, no filters, not bottom-of-funnel

## OUTPUT FORMAT

Return a JSON array of exactly 25 objects. Each object must follow this schema:
{{
  "query_id": 1,
  "query": "<the actual search query in natural buyer language>",
  "buying_stage": "<provider_evaluating | proof_seeking | decision_ready>",
  "filters_applied": {{
    "geography": "<value or null>",
    "company_size": "<value or null>",
    "industry_vertical": "<value or null>",
    "persona": "<value or null>",
    "pain_point": "<value or null>",
    "trigger": "<value or null>",
    "emotional_state": "<value or null>",
    "situational_context": "<value or null>",
    "team_size": "<value or null>",
    "budget_signal": "<value or null>",
    "feature_need": "<value or null>",
    "integration_need": "<value or null>",
    "engagement_model": "<value or null>",
    "outcome_focus": "<value or null>",
    "capability_gap": "<value or null>",
    "past_experience": "<value or null>",
    "proof_signal": "<value or null>",
    "competitor_context": "<value or null>",
    "target_keyword_used": "<keyword from target keywords woven into this query, or null>"
  }},
  "filter_count": 5,
  "rationale": "<1 sentence explaining why a buyer this close to a decision would search exactly this>"
}}

Only include filter keys relevant to this business type. Null out all filters that do not apply to a given query.

## FINAL QUALITY CHECKLIST — verify before returning output
- Business type was either accepted from input or explicitly inferred, not assumed
- Every single query is bottom-of-funnel: provider_evaluating, proof_seeking, or decision_ready only
- No query could be mistaken for an awareness or consideration stage search
- Every query has a minimum filter_count of 5
- No two queries share the same combination of filters
- At least 6 queries reference a competitor naturally (if competitors were provided)
- At least 4 queries contain decision-pressure language
- At least 4 queries are proof-seeking with specific context
- Target keywords appear naturally across multiple queries (if provided)
- Queries span at least 4 different industry verticals plausible for this ICP
- Zero marketing or vendor language in any query
- Every query reads like something a real buyer would type when they are days or weeks away from committing

Return ONLY the JSON array. No preamble. No explanation. No markdown fences."""

    raw = call_llm(
        prompt,
        preferred_backend=preferred_backend,
        allowed_backends=allowed_backends
    )

    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        queries = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            queries = json.loads(raw[start:end])
        else:
            raise RuntimeError(f"Could not parse query JSON. Raw output:\n{raw[:500]}")

    normalized = []
    for q in queries[:TARGET_QUERY_COUNT]:
        normalized.append({
            "query": q["query"],
            "category": q["buying_stage"],
            "topic": "Auto",
            "filters": q.get("filters_applied", {}),
            "filter_count": q.get("filter_count", 5),
            "rationale": q.get("rationale", ""),
            "query_id": q.get("query_id", 0),
            "target_keyword_used": q.get("filters_applied", {}).get("target_keyword_used", None)
        })

    return normalized


def generate_queries_for_topic(
    topic: str,
    icp_data: dict,
    company_name: str = "",
    competitors: list = None,
    queries_per_topic: int = 5,
    geography: str = "United States",
    business_type: str = "",
    target_keywords: list = None,
    preferred_backend: str = "groq",
    allowed_backends: list = None
) -> list:
    """
    Generates bottom-of-funnel queries for a single topic using ICP as context.
    Returns list of query objects tagged with the topic.
    """
    competitors_text = ", ".join(competitors) if competitors else "not specified"
    keywords_text = ", ".join(target_keywords) if target_keywords else "not specified"

    icp_formatted = f"""
Industries: {', '.join(icp_data.get('industries', []))}
Company sizes: {', '.join(icp_data.get('company_sizes', []))}
Geographies: {geography}
Buyer roles: {', '.join(icp_data.get('buyer_roles', []))}
Pain points: {', '.join(icp_data.get('pain_points', []))}
Budget ranges: {', '.join(icp_data.get('budget_ranges', []))}
Company stages: {', '.join(icp_data.get('company_stages', []))}
"""

    bt = business_type if business_type and business_type.lower() not in ["auto detect", "not specified", ""] else "infer from ICP"

    prompt = f"""You are an expert B2B market researcher. Generate exactly {queries_per_topic} bottom-of-funnel buyer queries for the topic below.

Company: {company_name}
Topic: {topic}
Business Type: {bt}
Geography: {geography}
Competitors: {competitors_text}
Target Keywords: {keywords_text}

ICP Context:
{icp_formatted}

CRITICAL RULES:
- Every query must be BOTTOM-OF-FUNNEL only: provider_evaluating, proof_seeking, or decision_ready
- Every query must be specifically about the topic: "{topic}"
- Queries must sound like a real buyer typed them when days or weeks away from a decision
- Do NOT generate awareness or consideration queries
- At least 2 queries must carry decision-pressure language: "need to decide", "about to sign", "final shortlist"
- At least 1 query must be proof-seeking: asking for real outcomes, references, or case studies
- For provider_evaluating and decision_ready queries: you MAY mention {company_name} by name when realistic
- For proof_seeking queries: do NOT mention {company_name} by name
- Include geography "{geography}" naturally in at least 1 query
- If competitors provided, mention at least 1 naturally
- If target keywords provided, weave them naturally into queries where appropriate
- Each query must cover a different angle of the topic
- Zero marketing language

Return a JSON array of exactly {queries_per_topic} objects:
[
  {{
    "query_id": 1,
    "query": "<natural buyer language bottom-of-funnel query about {topic}>",
    "buying_stage": "<provider_evaluating | proof_seeking | decision_ready>",
    "filters_applied": {{
      "geography": "<value or null>",
      "company_size": "<value or null>",
      "industry_vertical": "<value or null>",
      "persona": "<value or null>",
      "pain_point": "<value or null>",
      "trigger": "<value or null>",
      "emotional_state": "<value or null>",
      "target_keyword_used": "<keyword used or null>"
    }},
    "filter_count": 5,
    "rationale": "<why a buyer this close to a decision would search this>"
  }}
]

Return ONLY the JSON array. No explanation. No markdown fences."""

    raw = call_llm(prompt, preferred_backend=preferred_backend, allowed_backends=allowed_backends)

    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        queries = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            queries = json.loads(raw[start:end])
        else:
            raise RuntimeError(f"Could not parse topic query JSON. Raw:\n{raw[:300]}")

    normalized = []
    for q in queries[:queries_per_topic]:
        normalized.append({
            "query": q["query"],
            "category": q.get("buying_stage", "provider_evaluating"),
            "topic": topic,
            "filters": q.get("filters_applied", {}),
            "filter_count": q.get("filter_count", 5),
            "rationale": q.get("rationale", ""),
            "query_id": q.get("query_id", 0),
            "target_keyword_used": q.get("filters_applied", {}).get("target_keyword_used", None)
        })

    return normalized