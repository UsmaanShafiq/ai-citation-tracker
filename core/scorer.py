from collections import defaultdict


def calculate_citation_share(results: list, target_brand: str) -> dict:
    """
    Takes the full list of query results and calculates citation share.

    results: list of dicts, each containing:
        - query: str
        - category: str
        - tool: str
        - brands_detected: dict (output from brand_detector)

    Returns a complete scoring report.
    """

    tools = list(set(r["tool"] for r in results))
    categories = ["provider_evaluating", "proof_seeking", "decision_ready"]

    # ─── Overall citation share per tool ─────────────────────────────────────
    tool_totals = defaultdict(int)
    tool_mentions = defaultdict(int)

    for r in results:
        tool = r["tool"]
        tool_totals[tool] += 1
        if r["brands_detected"]["target_mentioned"]:
            tool_mentions[tool] += 1

    citation_share_by_tool = {}
    for tool in tools:
        total = tool_totals[tool]
        mentions = tool_mentions[tool]
        citation_share_by_tool[tool] = {
            "mentions": mentions,
            "total_queries": total,
            "share_pct": round((mentions / total) * 100) if total > 0 else 0
        }

    # ─── Citation share per category ─────────────────────────────────────────
    citation_share_by_category = {}
    for category in categories:
        cat_results = [r for r in results if r["category"] == category]
        if not cat_results:
            continue

        cat_mentions = sum(
            1 for r in cat_results if r["brands_detected"]["target_mentioned"]
        )
        citation_share_by_category[category] = {
            "mentions": cat_mentions,
            "total_queries": len(cat_results),
            "share_pct": round((cat_mentions / len(cat_results)) * 100)
        }

    # ─── Competitor brand frequency - dynamic 3-category classification ─────────
    # Categories are assigned at runtime by the LLM based on what each brand is.
    # No hardcoding - works for any industry automatically.
    # government: USPTO, FDA, WHO, SEC etc (any official body)
    # dominant: Google, Microsoft, Amazon etc (billion-dollar platforms)
    # competitor: same-scale commercial tools and services

    total_queries = len(results)

    government_counts = defaultdict(int)
    dominant_counts = defaultdict(int)
    competitor_counts = defaultdict(int)

    for r in results:
        bd = r["brands_detected"]
        target_lower = target_brand.lower()

        for brand in bd.get("government_brands", []):
            if brand.strip().lower() != target_lower:
                government_counts[brand.strip()] += 1

        for brand in bd.get("dominant_brands", []):
            if brand.strip().lower() != target_lower:
                dominant_counts[brand.strip()] += 1

        for brand in bd.get("competitor_brands", []):
            if brand.strip().lower() != target_lower:
                competitor_counts[brand.strip()] += 1

        # Fallback: if categories not present (old data), use all_brands
        if not any([bd.get("government_brands"), bd.get("dominant_brands"), bd.get("competitor_brands")]):
            for brand in bd.get("all_brands", []):
                brand_clean = brand.strip()
                if brand_clean.lower() != target_lower:
                    competitor_counts[brand_clean] += 1

    # Sort each category by frequency
    real_competitors = sorted(competitor_counts.items(), key=lambda x: x[1], reverse=True)
    dominant_platforms = sorted(
        [(b, c, round((c/total_queries)*100)) for b, c in dominant_counts.items()],
        key=lambda x: x[1], reverse=True
    )
    government_bodies = sorted(
        [(b, c, round((c/total_queries)*100)) for b, c in government_counts.items()],
        key=lambda x: x[1], reverse=True
    )

    # Combined list for backward compatibility
    competitor_ranking = (
        [(b, c) for b, c in real_competitors] +
        [(b, c) for b, c, _ in dominant_platforms] +
        [(b, c) for b, c, _ in government_bodies]
    )

    # ─── Position tracking removed ───────────────────────────────────────────
    # LLM position detection was unreliable (often returned 0 even when brand
    # appeared at position 4 or 7). Removed to avoid misleading data.
    avg_position_score = 0
    position_score_pct = 0

    # ─── Context breakdown ────────────────────────────────────────────────────
    context_counts = defaultdict(int)
    for r in results:
        ctx = r["brands_detected"]["target_context"]
        context_counts[ctx] += 1

    # ─── Overall summary ──────────────────────────────────────────────────────
    total_queries = len(results)
    total_mentions = sum(
        1 for r in results if r["brands_detected"]["target_mentioned"]
    )
    overall_share = round((total_mentions / total_queries) * 100) if total_queries > 0 else 0

    return {
        "target_brand": target_brand,
        "total_queries_run": total_queries,
        "overall_citation_share": overall_share,
        "total_mentions": total_mentions,
        "avg_position_score": avg_position_score,
        "position_score_pct": position_score_pct,
        "citation_share_by_tool": citation_share_by_tool,
        "citation_share_by_category": citation_share_by_category,
        "competitor_ranking": competitor_ranking,
        "real_competitors": real_competitors,
        "dominant_platforms": dominant_platforms,
        "government_bodies": government_bodies,
        "context_breakdown": dict(context_counts)
    }


def format_report(score_data: dict) -> str:
    """
    Formats the score data into a readable text report.
    Used for terminal output and debugging.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"AI CITATION SHARE REPORT")
    lines.append(f"Brand: {score_data['target_brand']}")
    lines.append(f"Total queries run: {score_data['total_queries_run']}")
    lines.append("=" * 60)

    lines.append(f"\nOVERALL CITATION SHARE: {score_data['overall_citation_share']}%")
    lines.append(f"Total mentions: {score_data['total_mentions']} / {score_data['total_queries_run']} queries")


    lines.append("\nCITATION SHARE BY TOOL:")
    for tool, data in score_data["citation_share_by_tool"].items():
        lines.append(f"  {tool}: {data['share_pct']}%  ({data['mentions']}/{data['total_queries']} queries)")

    lines.append("\nCITATION SHARE BY CATEGORY:")
    for cat, data in score_data["citation_share_by_category"].items():
        lines.append(f"  {cat}: {data['share_pct']}%  ({data['mentions']}/{data['total_queries']} queries)")

    lines.append("\nTOP COMPETITORS MENTIONED:")
    for brand, count in score_data["competitor_ranking"][:10]:
        lines.append(f"  {brand}: {count} mentions")

    lines.append("\nCONTEXT BREAKDOWN (how target brand was mentioned):")
    for ctx, count in score_data["context_breakdown"].items():
        lines.append(f"  {ctx}: {count} times")

    lines.append("=" * 60)
    return "\n".join(lines)


# ─── TEST ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Simulated results (what the full pipeline will produce)
    sample_results = [
        {
            "query": "Best HVAC software for small US business?",
            "category": "awareness",
            "tool": "Groq_Llama3",
            "brands_detected": {
                "all_brands": ["ServiceTitan", "Jobber", "Housecall Pro"],
                "target_mentioned": True,
                "target_context": "recommended"
            }
        },
        {
            "query": "ServiceTitan vs Jobber for plumbing company?",
            "category": "comparison",
            "tool": "Groq_Llama3",
            "brands_detected": {
                "all_brands": ["Jobber", "Housecall Pro", "Workiz"],
                "target_mentioned": False,
                "target_context": "not_mentioned"
            }
        },
        {
            "query": "Help with scheduling chaos in my HVAC business",
            "category": "pain_point",
            "tool": "Groq_Mixtral",
            "brands_detected": {
                "all_brands": ["ServiceTitan", "Workiz"],
                "target_mentioned": True,
                "target_context": "recommended"
            }
        },
        {
            "query": "Ready to buy HVAC software under $300/mo",
            "category": "buying_intent",
            "tool": "Groq_Mixtral",
            "brands_detected": {
                "all_brands": ["Jobber", "ServiceTitan", "Housecall Pro"],
                "target_mentioned": True,
                "target_context": "recommended"
            }
        },
        {
            "query": "HVAC software Canada QuickBooks integration SOC2",
            "category": "specific_filter",
            "tool": "Groq_Llama3",
            "brands_detected": {
                "all_brands": ["ServiceTitan", "FieldEdge"],
                "target_mentioned": True,
                "target_context": "recommended"
            }
        },
    ]

    print("Testing Scorer...")

    score = calculate_citation_share(sample_results, "ServiceTitan")
    report = format_report(score)
    print(report)


def calculate_citation_share_by_group(results: list, target_brand: str) -> dict:
    """
    Groups results by query_group (A/B/C) and calculates citation share per group.
    Group A: Company + Competitor queries
    Group B: Company Only queries
    Group C: Shortlisting queries (no company names)
    """
    from collections import defaultdict

    group_labels = {
        "A": "Group A: Company + Competitor",
        "B": "Group B: Company Only",
        "C": "Group C: Shortlisting"
    }

    group_data = defaultdict(list)
    for r in results:
        g = r.get("query_group", "C")
        group_data[g].append(r)

    group_scores = {}
    for group_key in ["A", "B", "C"]:
        group_results = group_data.get(group_key, [])
        if not group_results:
            continue
        total = len(group_results)
        mentions = sum(1 for r in group_results if r["brands_detected"]["target_mentioned"])
        share = round((mentions / total) * 100) if total > 0 else 0
        group_scores[group_labels.get(group_key, group_key)] = {
            "total_queries": total,
            "mentions": mentions,
            "share_pct": share
        }

    return group_scores


def calculate_citation_share_by_topic(results: list, target_brand: str) -> dict:
    """
    Groups results by topic and calculates citation share per topic.
    """
    from collections import defaultdict
    topic_groups = defaultdict(list)
    for r in results:
        topic = r.get("topic", "General")
        topic_groups[topic].append(r)

    topic_scores = {}
    for topic, topic_results in topic_groups.items():
        total = len(topic_results)
        mentions = sum(1 for r in topic_results if r["brands_detected"]["target_mentioned"])
        share = round((mentions / total) * 100) if total > 0 else 0
        topic_scores[topic] = {
            "total_queries": total,
            "mentions": mentions,
            "share_pct": share
        }

    return topic_scores