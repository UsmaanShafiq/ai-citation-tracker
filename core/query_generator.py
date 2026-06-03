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


def _build_main_prompt(
    company_name: str,
    icp_formatted: str,
    business_type: str,
    competitors_text: str,
    keywords_text: str,
    geography: str
) -> str:
    """Builds the full bottom-of-funnel prompt with 3-group distribution."""

    if business_type and business_type.strip() and business_type.lower() not in ["auto detect", "not specified", ""]:
        bt_section = f"""**Business Type:** {business_type}
(This field is provided. Accept it as ground truth. Do not override or reinterpret it. Build your entire understanding of the buying motion, vocabulary, and filter dimensions around this business type.)"""
        bt_step1 = f"Accept the provided business type ({business_type}) as ground truth. Do not second-guess it. Build your entire understanding of buying motion, vocabulary, and filter dimensions around this."
    else:
        bt_section = f"""**Business Type:** not specified
(Infer the business type from the ICP and company name before generating any queries.)"""
        bt_step1 = f"""Infer the business type by answering these questions internally:
1. What is {company_name} actually selling — software, professional service, agency offering, managed service, marketplace, or something else? Read the ICP carefully.
2. How does buying happen in this category? Software: trials, G2 comparisons. Agency: case studies, referrals, thought leadership. Managed service: outcomes not features.
3. What language does the buyer use at this stage? Ground all queries in that vocabulary.
4. Which filter dimensions are irrelevant for this type and should be dropped entirely?
Complete this reasoning before generating any queries."""

    keyword_section = ""
    if keywords_text and keywords_text != "not specified":
        keyword_section = f"""
**Target Keywords (optional):** {keywords_text}
(Weave these naturally into queries where they fit buyer language. Do not force into every query. Target keywords are especially important in Group C shortlisting queries where buyers use category language. Aim for meaningful coverage across all 25 queries.)"""

    return f"""You are an expert B2B market researcher and buyer behavior analyst. Your job is to generate hyper-specific, bottom-of-the-funnel AI search queries that real buyers type when they are close to creating a shortlist or making a purchase decision in {company_name}'s category.

## INPUTS

**Company Name:** {company_name}

**Ideal Client Profile (ICP):**
{icp_formatted}

{bt_section}
{keyword_section}

**Competitor Names (optional):** {competitors_text}

**Target Geography:** {geography}

## STEP 1 — ESTABLISH BUSINESS CONTEXT

{bt_step1}

## STEP 2 — BOTTOM-OF-FUNNEL ONLY

Every single query must reflect a buyer who is close to a purchase decision. Not awareness. Not education. A buyer in one of these three modes:

**provider_evaluating** — actively comparing specific companies, approaches, or options side by side
**proof_seeking** — looking for validation that a specific solution will work for their exact situation before committing
**decision_ready** — essentially ready to buy, doing final checks, looking for confirmation or a nudge

Do NOT generate queries for pain_aware or solution_aware stages. If a query sounds like the buyer is still figuring out what they need, it does not belong.

Signs a query is genuinely bottom-of-funnel:
- It names or implies specific options being compared
- It asks for proof, references, case studies, or outcomes for a specific context
- It asks what others in a very specific situation actually chose
- It carries the weight of an imminent decision ("we need to decide by end of quarter," "about to sign," "final shortlist")
- It is specific enough that a generic answer would not satisfy it

## STEP 3 — INHABIT THE BUYER AT DECISION TIME

This buyer has talked to vendors, sat through demos, read reviews. They are now asking:
- "Is this the right choice for my exact situation?"
- "Has anyone like me done this and did it work?"
- "What am I missing before I commit?"
- "Is option A actually better than option B for a company like mine?"
- "What do people say about this when they are not trying to sell me something?"

Channel how this buyer phrases questions on Reddit, Slack, G2, or directly to an AI assistant. Use their words, their doubts, their specific context. Not seller language.

## MANDATORY QUERY DISTRIBUTION — HARD RULE

The 25 queries MUST be split into exactly three groups. This is not a guideline. Counts must be exact.

### Group A — Company + Competitor (EXACTLY 5 queries)
These queries name {company_name} AND at least one competitor from the competitors list in the same query. The buyer already knows both names and is making a direct comparison or asking others who have evaluated both. Competitors must be woven in naturally.

RULE: Query must contain {company_name} name + at least one competitor name.

### Group B — Company Only (EXACTLY 10 queries)
These queries name {company_name} but do NOT mention any competitor. The buyer has {company_name} on their shortlist and is doing final validation — seeking proof, checking references, understanding scope, or looking for reassurance from people who have worked with them.

RULE: Query must contain {company_name} name only. Zero competitor names anywhere in the query.

### Group C — Shortlisting (EXACTLY 10 queries)
These queries name NEITHER {company_name} NOR any competitor. The buyer knows exactly what they need and what kind of provider they are looking for, but they have not yet landed on specific names. They are building their shortlist. These queries are highly specific about situation, vertical, outcome, and constraints — but they search by category not by company name. This is where target keywords matter most.

RULE: Zero company names. Neither {company_name} nor any competitor name appears anywhere.

ENFORCEMENT: Before finalizing output, count queries in each group. If count is not exactly 5 / 10 / 10, revise until it is. Do not output until distribution is correct.

## FILTER DIMENSIONS

Each query MUST include at least 5 filter dimensions. Only use filters meaningful for the confirmed business type. Vary combinations across queries so no two feel alike.

### Universal Filters (any business type)
- Geography
- Company Size / Stage
- Industry Vertical
- Job Title / Persona
- Pain Point (specific frustration that brought them to edge of decision)
- Trigger / Timing (end of quarter, after failed hire, post-funding, contract renewal)
- Emotional State (tired of waiting, under pressure, anxious about wrong call)
- Situational Context (remote-first, heavily regulated, multi-timezone, post-acquisition)

### For Agencies / Consulting / Professional Services
- Engagement Model (retainer vs project, embedded vs advisory, trial before committing)
- Outcome / Proof Focus (pipeline results, content that ranked, clients in this niche)
- Past Experience with Category (burned by previous agency, first time outsourcing)
- Internal Capability Gap (no writer, founder doing it all, junior team needs senior oversight)
- Cultural / Execution Fit (understands technical buyers, strategic input not just execution)
- Proof / Trust Signal (work in specific vertical, references, thought leadership quality)
- Competitor Context (comparing two agencies, agency vs in-house, evaluating shortlist)

### For Software / SaaS Products
- Team Size
- Budget Signal (approved budget, cost-per-seat, switching cost)
- Feature Need
- Integration Need
- Competitor Context (head-to-head comparison, migration, shortlist of two)

## HANDLING TARGET KEYWORDS

If target keywords are provided:
- Treat as specific phrases the user wants {company_name} to be discovered for in AI answers
- Weave naturally into queries where they fit buyer language
- Do NOT force into every query
- Target keywords are ESPECIALLY important in Group C queries — buyers use category language so keywords fit most naturally here
- Aim for meaningful coverage: each keyword should appear across multiple queries
- Never make a query feel constructed around a keyword

## QUERY CONSTRUCTION RULES

- Every query must be bottom-of-funnel, no exceptions
- Queries must read like something a real buyer types into an AI assistant or posts in Slack when days or weeks from a decision
- Buyer language not seller language. No "robust," "seamless," "end-to-end," "best-in-class," "innovative," "strategic partner"
- At least 4 queries must carry decision-pressure language: "need to decide," "about to sign," "final two options," "last thing I need to figure out" — distribute across groups
- At least 4 queries must be proof-seeking, asking for real outcomes, references, case studies in specific context — distribute across groups
- Queries must span at least 4 different industry verticals plausible for this ICP
- No two queries should have the same combination of filters
- Zero marketing or vendor language anywhere

## GOOD VS BAD EXAMPLES

### Group A (company + competitor):
GOOD: "we're down to {company_name} and Animalz for our Series B fintech SaaS — Animalz has more brand recognition but {company_name} seems more focused on pipeline and AI visibility — which one actually helped CMOs get qualified demos faster after SEO traffic plateaued"

### Group B (company only):
GOOD: "does {company_name} actually move pipeline or just produce content — we're a 90-person healthtech SaaS in the US and I've seen their site but want to hear from someone who has worked with them before we commit"

### Group C (shortlisting — no names):
GOOD: "which B2B SaaS content marketing agencies have actual case studies showing qualified inbound leads from content not just traffic — we're a 100-person legaltech company in the UK with $4-5k/month approved and need to build a shortlist this week"

BAD: "what is the best content marketing agency for SaaS companies" — awareness stage, no specificity, no decision pressure, no filters

## OUTPUT FORMAT

Return a JSON array of exactly 25 objects. Each object must follow this schema exactly:
{{
  "query_id": 1,
  "query_group": "<A | B | C>",
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
    "budget_signal": "<value or null>",
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

Only include filter keys relevant to this business type. Null out all filters that do not apply.

## FINAL QUALITY CHECKLIST — verify before returning output

- Business type accepted from input or explicitly inferred, not assumed
- Every single query is bottom-of-funnel: provider_evaluating, proof_seeking, or decision_ready only
- No query could be mistaken for awareness or consideration stage
- Every query has minimum filter_count of 5
- No two queries share the same filter combination
- EXACTLY 5 queries in Group A ({company_name} + at least one competitor name)
- EXACTLY 10 queries in Group B ({company_name} only, zero competitor names)
- EXACTLY 10 queries in Group C (zero company names, neither {company_name} nor any competitor)
- Group A queries each contain at least one competitor name
- Group C queries contain zero company or competitor names, category language only
- At least 4 queries contain decision-pressure language distributed across groups
- At least 4 queries are proof-seeking with specific context distributed across groups
- Target keywords appear naturally across multiple queries especially in Group C
- Queries span at least 4 different industry verticals plausible for this ICP
- Zero marketing or vendor language in any query
- Every query reads like something a real buyer would type when days or weeks from committing

Return ONLY the JSON array. No preamble. No explanation. No markdown fences."""


def _build_topic_prompt(
    topic: str,
    company_name: str,
    icp_formatted: str,
    business_type: str,
    competitors_text: str,
    keywords_text: str,
    geography: str,
    queries_per_topic: int
) -> str:
    """Builds prompt for topic-based query generation with group distribution."""

    bt = business_type if business_type and business_type.lower() not in ["auto detect", "not specified", ""] else "infer from ICP"

    # Calculate group distribution for topic queries
    # Scale 5/10/10 proportionally based on queries_per_topic
    if queries_per_topic >= 5:
        group_a = max(1, queries_per_topic // 5)
        group_c = (queries_per_topic - group_a) // 2
        group_b = queries_per_topic - group_a - group_c
    else:
        group_a = 1
        group_b = queries_per_topic - 2 if queries_per_topic > 2 else 1
        group_c = queries_per_topic - group_a - group_b

    return f"""You are an expert B2B market researcher. Generate exactly {queries_per_topic} bottom-of-funnel buyer queries for the topic below.

Company: {company_name}
Topic: {topic}
Business Type: {bt}
Geography: {geography}
Competitors: {competitors_text}
Target Keywords: {keywords_text}

ICP Context:
{icp_formatted}

MANDATORY GROUP DISTRIBUTION — HARD RULE (must be exact):
- Group A (Company + Competitor): exactly {group_a} queries — must name {company_name} AND at least one competitor
- Group B (Company Only): exactly {group_b} queries — must name {company_name} only, zero competitor names
- Group C (Shortlisting): exactly {group_c} queries — zero company names, neither {company_name} nor any competitor

CRITICAL RULES:
- Every query must be BOTTOM-OF-FUNNEL only: provider_evaluating, proof_seeking, or decision_ready
- Every query must be specifically about the topic: "{topic}"
- Queries must sound like a real buyer typed them when days or weeks away from a decision
- Do NOT generate awareness or consideration queries
- At least 1 query must carry decision-pressure language: "need to decide", "about to sign", "final shortlist"
- At least 1 query must be proof-seeking: asking for real outcomes, references, or case studies
- Group C queries: use target keywords naturally — this is where category language fits best
- Include geography "{geography}" naturally in at least 1 query
- At least 5 filter dimensions per query
- Zero marketing language

Return a JSON array of exactly {queries_per_topic} objects:
[
  {{
    "query_id": 1,
    "query_group": "<A | B | C>",
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
      "situational_context": "<value or null>",
      "engagement_model": "<value or null>",
      "outcome_focus": "<value or null>",
      "capability_gap": "<value or null>",
      "past_experience": "<value or null>",
      "proof_signal": "<value or null>",
      "competitor_context": "<value or null>",
      "target_keyword_used": "<keyword used or null>"
    }},
    "filter_count": 5,
    "rationale": "<why a buyer this close to a decision would search this>"
  }}
]

Return ONLY the JSON array. No explanation. No markdown fences."""


def _parse_and_normalize(raw: str, count: int, topic: str = "Auto") -> list:
    """Parse LLM JSON output and normalize to pipeline format."""
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
    for q in queries[:count]:
        normalized.append({
            "query": q["query"],
            "query_group": q.get("query_group", "C"),
            "category": q["buying_stage"],
            "topic": topic,
            "filters": q.get("filters_applied", {}),
            "filter_count": q.get("filter_count", 5),
            "rationale": q.get("rationale", ""),
            "query_id": q.get("query_id", 0),
            "target_keyword_used": q.get("filters_applied", {}).get("target_keyword_used", None)
        })

    return normalized


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

    prompt = _build_main_prompt(
        company_name=company_name,
        icp_formatted=icp_formatted,
        business_type=business_type,
        competitors_text=competitors_text,
        keywords_text=keywords_text,
        geography=geo
    )

    raw = call_llm(prompt, preferred_backend=preferred_backend, allowed_backends=allowed_backends)
    return _parse_and_normalize(raw, TARGET_QUERY_COUNT, topic="Auto")


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

    prompt = _build_topic_prompt(
        topic=topic,
        company_name=company_name,
        icp_formatted=icp_formatted,
        business_type=business_type,
        competitors_text=competitors_text,
        keywords_text=keywords_text,
        geography=geography,
        queries_per_topic=queries_per_topic
    )

    raw = call_llm(prompt, preferred_backend=preferred_backend, allowed_backends=allowed_backends)
    return _parse_and_normalize(raw, queries_per_topic, topic=topic)
