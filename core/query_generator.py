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

TARGET_QUERY_COUNT = 10

QUERY_CATEGORIES = [
    {"name": "pain_aware", "count": 5},
    {"name": "solution_aware", "count": 5},
    {"name": "provider_evaluating", "count": 5},
    {"name": "proof_seeking", "count": 5},
    {"name": "decision_ready", "count": 5},
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
    preferred_backend: str = "groq",
    allowed_backends: list = None
) -> list:

    competitors_text = ", ".join(competitors) if competitors else "not specified"

    icp_formatted = f"""
Industries: {', '.join(icp_data.get('industries', []))}
Company sizes: {', '.join(icp_data.get('company_sizes', []))}
Geographies: {', '.join(icp_data.get('geographies', []))}
Buyer roles: {', '.join(icp_data.get('buyer_roles', []))}
Pain points: {', '.join(icp_data.get('pain_points', []))}
Budget ranges: {', '.join(icp_data.get('budget_ranges', []))}
Tech stacks: {', '.join(icp_data.get('tech_stacks', []))}
Company stages: {', '.join(icp_data.get('company_stages', []))}
Compliance needs: {', '.join(icp_data.get('compliance_needs', []))}
"""

    prompt = f"""You are an expert B2B market researcher and buyer behavior analyst. Your job is to generate hyper-specific AI search queries that real buyers type when searching for solutions in {company_name}'s category which may be software, a service, an agency, a marketplace, or something else entirely. Do not assume the category. Infer it from the inputs.

CRITICAL INSTRUCTION: Read the ICP carefully before generating 
any query. If the company is an agency, consultancy, or service 
business, every single query must reflect someone looking to HIRE 
a service provider, not buy software. Generating software-buying 
queries for a service business is a complete failure of this task.

## INPUTS
Company Name: {company_name}
Ideal Client Profile (ICP): {icp_formatted}
Competitor Names (optional): {competitors_text}

## STEP 1 — UNDERSTAND THE BUSINESS CONTEXT BEFORE ANYTHING ELSE

Before generating a single query, answer these four questions internally. Your entire query generation must be grounded in these answers.

1. What is {company_name} actually selling?
Is it a software product, a professional service, a managed service, a marketplace, a community, a content product, or something else? Do not assume. Read the ICP carefully.

2. How does buying actually happen in this category?
Software: buyer searches for tools, compares on G2, does trials.
Agency or consulting: buyer searches for expertise, checks case studies, asks for referrals, evaluates thought leadership.
Managed service: buyer searches for outcomes not features.
Think through the actual decision journey for THIS category. Who gets involved? How long does it take? What triggers the search?

3. What does the buyer actually type into an AI or search engine?
A buyer looking for a content marketing agency does not search for software with editorial calendar features. They search for content agency that understands B2B SaaS or who writes good long-form content for developer tools companies. Ground all queries in the vocabulary of THIS buying motion.

4. Which filter dimensions are irrelevant for this business type?
If {company_name} is a service business, filters like needs SSO, requires API access, or SOC 2 compliance are meaningless. Drop them. Replace with dimensions that actually apply: engagement model preference, output quality bar, cultural fit, past agency experience.

Only proceed to query generation after completing this reasoning.

## STEP 2 — INHABIT THE BUYER'S WORLD

1. This is a real person, frustrated, time-pressured, skeptical of vendor marketing.
2. Think Reddit, G2, LinkedIn comments, Slack communities. Use their vocabulary not seller vocabulary. Buyers say our content is getting zero traction not we need a content strategy partner.
3. Queries must span all five buying stages. Do not cluster them in one stage.

## BUYING STAGE TAXONOMY
- pain_aware: they know something is wrong, not sure what the solution looks like
- solution_aware: they know the category of solution, searching for options
- provider_evaluating: actively comparing specific companies or approaches
- proof_seeking: looking for evidence it works for their specific situation
- decision_ready: close to committing, doing final validation

## FILTER DIMENSIONS
Each query MUST include at least 5 filter dimensions. Choose filters relevant to how buying happens for THIS business. Vary filters across queries so no two feel alike.

Universal filters for any business:
- geography
- company_size
- industry_vertical
- persona
- pain_point
- trigger
- emotional_state
- situational_context

For agencies and professional services use these:
- engagement_model: retainer vs project, embedded vs advisory, long-term vs one-off
- outcome_focus: pipeline generation, thought leadership, category creation, brand credibility
- past_experience: never hired agency before, burned by bad agency, in-house team failed
- capability_gap: no writer on team, founder doing it all, content team too junior, no strategic layer
- proof_signal: wants case studies in specific niche, needs references, judges by published work
- competitor_context: comparing agency X vs agency Y, debating agency vs hiring in-house

For software products use these instead:
- budget_signal
- feature_need
- integration_need
- competitor_context

## QUERY CONSTRUCTION RULES
- Must read like a human typing to an AI, not a keyword string
- Use buyer language not seller language
- No robust, seamless, end-to-end, best-in-class, strategic partner anywhere
- At least 5 queries reference a competitor naturally within the query if competitors were provided
- At least 3 queries carry emotional language: sick of, fed up, nothing seems to work, keeps falling apart
- At least 4 queries are late-stage high-intent: provider_evaluating, proof_seeking, or decision_ready
- Span at least 4 different industry verticals plausible for this ICP
- No two queries have the same filter combination
- Do NOT mention {company_name} in any query
- Every query feels like a Reddit thread, Slack message, or real AI conversation

## GOOD VS BAD EXAMPLES

For a content marketing agency:

Good: we are a 40-person DevOps SaaS company in the US, our founder used to write all the content but we just closed Series A and he does not have time anymore, how do we find a content agency that actually understands technical buyers and will not just churn out generic blog posts

Bad: best content marketing software for B2B SaaS companies with editorial calendar and SEO features

Why bad: a buyer looking for an agency is not looking for software. The query must reflect the actual buying motion.

For a B2B SaaS tool:

Good: our CS team is 4 people managing 300 accounts, we are losing track of renewals, tried Asana but it was not built for this, what are teams our size actually using

Bad: best project management software for customer success with renewal tracking

## FINAL QUALITY CHECKLIST apply before returning output
- Every query reflects the correct business type, no software filters in a services context
- Every query has minimum 5 filters
- No two queries share the same filter combination
- Buying stages distributed across all five, no clustering
- At least 3 queries contain emotional buyer language
- At least 4 queries are high-intent late-stage
- At least 5 queries reference a competitor naturally if competitors provided
- Queries span at least 4 different industry verticals
- Zero marketing or vendor language anywhere
- Every query reads like Reddit, Slack, or real AI chat

## OUTPUT FORMAT
Return a JSON array of exactly 25 objects. Each object must follow this schema exactly:
{{
  "query_id": 1,
  "query": "<natural buyer language>",
  "buying_stage": "<pain_aware | solution_aware | provider_evaluating | proof_seeking | decision_ready>",
  "filters_applied": {{
    "geography": "<value or null>",
    "company_size": "<value or null>",
    "industry_vertical": "<value or null>",
    "persona": "<value or null>",
    "pain_point": "<value or null>",
    "trigger": "<value or null>",
    "emotional_state": "<value or null>",
    "situational_context": "<value or null>",
    "engagement_model": "<value or null>",
    "outcome_focus": "<value or null>",
    "capability_gap": "<value or null>",
    "past_experience": "<value or null>",
    "proof_signal": "<value or null>",
    "competitor_context": "<value or null>"
  }},
  "filter_count": 5,
  "rationale": "<1 sentence why a real buyer would search this>"
}}

Only include filter keys relevant to this business type. Null out all filters that do not apply to a given query.

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
            "filters": q["filters_applied"],
            "filter_count": q.get("filter_count", 5),
            "rationale": q.get("rationale", ""),
            "query_id": q.get("query_id", 0)
        })

    return normalized