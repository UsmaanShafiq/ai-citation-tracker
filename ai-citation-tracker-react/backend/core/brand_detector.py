import re
from typing import Optional

def extract_urls(text: str) -> list:
    """Extract all URLs from response text."""
    url_pattern = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2})|[/?#&=+@:!,;])+',
        re.IGNORECASE
    )
    return list(dict.fromkeys(url_pattern.findall(text)))  # deduplicated


def extract_linked_sites(text: str) -> list:
    """
    Extract cited sources/links from AI response.
    Returns list of {rank, title, url}
    """
    urls = extract_urls(text)
    sites = []
    for i, url in enumerate(urls[:10], 1):  # max 10
        # Try to extract a title from surrounding text
        domain = re.sub(r'https?://(www\.)?', '', url).split('/')[0]
        sites.append({
            "rank": i,
            "title": domain,
            "url": url,
        })
    return sites


def detect_brand(
    response_text: str,
    brand_name: str,
    brand_domain: str,
) -> dict:
    """
    Detect if a brand is mentioned in an AI response.
    Returns structured detection result.
    """
    text_lower = response_text.lower()
    brand_lower = brand_name.lower()
    domain_lower = brand_domain.lower().replace("https://", "").replace("http://", "").replace("www.", "")

    # Check if brand is mentioned
    brand_mentioned = brand_lower in text_lower or domain_lower in text_lower

    # Find position (which paragraph/sentence mentions it first)
    brand_position = None
    brand_context = "not_mentioned"

    if brand_mentioned:
        # Find approximate position
        sentences = re.split(r'[.!?]\s+', response_text)
        for i, sentence in enumerate(sentences):
            if brand_lower in sentence.lower() or domain_lower in sentence.lower():
                brand_position = i + 1
                break

        # Determine context
        context_window = ""
        idx = text_lower.find(brand_lower)
        if idx == -1:
            idx = text_lower.find(domain_lower)
        if idx != -1:
            start = max(0, idx - 100)
            end = min(len(response_text), idx + 200)
            context_window = response_text[start:end].lower()

        if any(w in context_window for w in ["recommend", "best", "top", "ideal", "perfect", "great for", "suggest"]):
            brand_context = "recommended"
        elif any(w in context_window for w in ["avoid", "warning", "not recommend", "poor", "issue", "problem"]):
            brand_context = "warned_against"
        else:
            brand_context = "mentioned"

    # Detect all brand names in response (simple extraction of capitalized proper nouns)
    all_brands = extract_brand_names(response_text)

    # Extract linked sites
    linked_sites = extract_linked_sites(response_text)

    return {
        "brand_mentioned": brand_mentioned,
        "brand_position": brand_position,
        "brand_context": brand_context,
        "linked_sites": linked_sites,
        "all_brands_detected": all_brands,
    }


def extract_brand_names(text: str) -> list:
    """
    Simple heuristic to extract likely brand/company names.
    Looks for capitalized multi-word phrases.
    """
    # Match capitalized words (potential brand names)
    pattern = re.compile(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b')
    candidates = pattern.findall(text)

    # Filter out common words
    stop_words = {
        "The", "This", "That", "These", "Those", "There", "Their", "They",
        "With", "From", "Into", "About", "After", "Before", "Between",
        "When", "Where", "What", "Which", "While", "How", "Why",
        "Also", "More", "Most", "Many", "Some", "Such", "Other",
        "Here", "Just", "Like", "Well", "Even", "Only", "Both",
        "I", "You", "He", "She", "We", "It", "Its", "My", "Your",
        "For", "And", "But", "Or", "So", "Yet", "Nor",
    }

    seen = set()
    brands = []
    for name in candidates:
        first_word = name.split()[0]
        if first_word not in stop_words and name not in seen and len(name) > 2:
            seen.add(name)
            brands.append(name)

    return brands[:20]  # cap at 20
