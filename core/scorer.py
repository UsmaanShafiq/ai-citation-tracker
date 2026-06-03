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

    # ─── Competitor brand frequency ───────────────────────────────────────────
    SOFTWARE_EXCLUDE = {
        "hubspot", "ahrefs", "semrush", "google", "google analytics",
        "salesforce", "marketo", "wordpress", "trello", "asana",
        "notion", "hootsuite", "buffer", "moz", "clearscope",
        "marketmuse", "coschedule", "mailchimp", "activecampaign",
        "pardot", "monday", "clickup", "slack", "microsoft",
        "zoom", "webflow", "squarespace", "quickbooks", "stripe",
        "intercom", "zendesk", "linkedin", "twitter", "facebook",
        "instagram", "youtube", "canva", "sprout social", "buzzstream",
        "contentful", "upwork", "clutch", "goodfirms", "paypal",
        "square", "shopify", "woocommerce", "magento", "freshdesk",
        "servicenow", "zoho", "pipedrive", "reddit", "tiktok",
    }

    competitor_counts = defaultdict(int)
    for r in results:
        for brand in r["brands_detected"]["all_brands"]:
            brand_clean = brand.strip()
            if brand_clean.lower() == target_brand.lower():
                continue
            if brand_clean.lower() in SOFTWARE_EXCLUDE:
                continue
            competitor_counts[brand_clean] += 1

    # Sort by frequency
    competitor_ranking = sorted(
        competitor_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # ─── Position score ───────────────────────────────────────────────────────
    position_scores = []
    position_points = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}

    for r in results:
        pos = r["brands_detected"]["target_position"]
        if pos > 0:
            points = position_points.get(pos, 1)
        else:
            points = 0
        position_scores.append(points)

    avg_position_score = round(
        sum(position_scores) / len(position_scores), 2
    ) if position_scores else 0

    max_possible = 5 * len(results)
    position_score_pct = round(
        (sum(position_scores) / max_possible) * 100
    ) if max_possible > 0 else 0

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
    lines.append(f"Position score: {score_data['position_score_pct']}%")

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
                "target_position": 1,
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
                "target_position": 0,
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
                "target_position": 1,
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
                "target_position": 2,
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
                "target_position": 1,
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