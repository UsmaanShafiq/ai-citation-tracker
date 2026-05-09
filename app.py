import streamlit as st
import pandas as pd
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(__file__))

from core.icp_parser import parse_and_validate
from core.query_generator import generate_all_queries
from core.ai_runner import ALL_TOOLS, run_selected_tools, check_key_exists, get_usage_stats, reset_usage_stats
from core.brand_detector import detect_brands
from core.scorer import calculate_citation_share


def format_error_message(raw_error: str) -> str:
    """
    Convert provider errors into clean, user-friendly messages.
    """
    msg = (raw_error or "").strip()
    low = msg.lower()

    # Token/context window style errors ("memory reached" from model side)
    if (
        "context_length_exceeded" in low
        or "maximum context length" in low
        or "context window" in low
        or "token limit" in low
        or "too many tokens" in low
        or "prompt is too long" in low
        or "memory" in low and "exceed" in low
    ):
        return "Request too large for the selected model (context/memory limit reached). Reduce ICP length or query count, or switch models."

    # Quota / rate limit errors
    if (
        "429" in low
        or "rate_limit" in low
        or "resource_exhausted" in low
        or "quota" in low
    ):
        return "API quota or rate limit reached. Wait and retry, add billing, or enable another checked provider as fallback."

    # Missing module / SDK errors
    if "no module named" in low or "sdk not installed" in low:
        return "Required SDK is not installed in this environment. Install dependencies from requirements.txt and retry."

    # Invalid model / provider mismatch errors
    if "not found" in low and "model" in low:
        return "Selected model is unavailable for this API key/project. Try another provider or update the model configuration."

    return msg

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AI Citation Tracker",
    page_icon="📡",
    layout="wide"
)

st.title("📡 AI Citation Tracker")
st.caption("Track how often your brand appears in AI-generated buying recommendations")
st.divider()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Setup")

    website_url = st.text_input(
        "Website URL",
        placeholder="https://concurate.com"
    )

    brand_name = st.text_input(
        "Brand Name to Track",
        placeholder="Concurate"
    )

    icp_text = st.text_area(
        "ICP Document",
        height=180,
        placeholder="Paste your Ideal Customer Profile here..."
    )

    competitors_input = st.text_input(
        "Competitors (optional)",
        placeholder="Animalz, Siege Media, Grow and Convert"
    )

    st.divider()
    st.caption("Queries per run: 10")

    st.divider()

# ── API Key Manager ───────────────────────────────────────────────────────

    with st.expander("Manage API Keys"):
        st.caption("Keys are saved to your .env file and persist across sessions.")

        key_fields = {
            "GROQ_API_KEY":        "Groq (Llama3 + Mixtral) - Free",
            "PERPLEXITY_API_KEY":  "Perplexity - $5 free credit",
            "GEMINI_API_KEY":      "Gemini - 1500 free calls/day",
            "OPENAI_API_KEY":      "OpenAI / ChatGPT - Paid",
            "ANTHROPIC_API_KEY":   "Claude / Anthropic - Paid",
        }

        def read_env() -> dict:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            existing = {}
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            existing[k.strip()] = v.strip()
            return existing

        def write_env(data: dict):
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            with open(env_path, "w") as f:
                for k, v in data.items():
                    f.write(f"{k}={v}\n")

        updated_keys = {}
        keys_to_remove = []

        for env_key, label in key_fields.items():
            current = os.getenv(env_key, "")
            has_key = bool(current and current.strip() and "paste_your" not in current.lower())
            masked = current[:8] + "..." if has_key else ""

            col_input, col_remove = st.columns([4, 1])

            with col_input:
                new_val = st.text_input(
                    label,
                    value="",
                    placeholder=masked if masked else "Paste key here",
                    type="password",
                    key=f"key_input_{env_key}"
                )
                if new_val.strip():
                    updated_keys[env_key] = new_val.strip()

            with col_remove:
                st.write("")
                st.write("")
                if has_key:
                    if st.button("✕", key=f"remove_{env_key}", help=f"Remove {env_key}"):
                        keys_to_remove.append(env_key)

        # Handle removes
        if keys_to_remove:
            existing = read_env()
            for k in keys_to_remove:
                existing.pop(k, None)
                os.environ.pop(k, None)
            write_env(existing)
            st.success(f"Removed {len(keys_to_remove)} key(s). Refreshing...")
            st.rerun()

        # Handle saves
        if st.button("Save Keys", use_container_width=True):
            if updated_keys:
                existing = read_env()
                existing.update(updated_keys)
                write_env(existing)
                for k, v in updated_keys.items():
                    os.environ[k] = v
                st.success(f"Saved {len(updated_keys)} key(s). Refreshing...")
                st.rerun()
            else:
                st.warning("No keys entered.")

    # ── Tool Manager ──────────────────────────────────────────────────────────
    st.subheader("AI Tools")
    st.caption("Toggle which tools to use. Add API key to .env to unlock paid tools.")

    selected_tools = []

    for tool_name, tool_info in ALL_TOOLS.items():
        key_exists = check_key_exists(tool_info["requires_key"])
        is_free = tool_info["free"]

        if is_free and key_exists:
            label = f"{tool_name} ✅ Free"
            default = True
            disabled = False
        elif not is_free and key_exists:
            label = f"{tool_name} 🔑 Key found"
            default = True
            disabled = False
        elif not key_exists and not is_free:
            label = f"{tool_name} 🔒 Needs API key"
            default = False
            disabled = True
        else:
            label = f"{tool_name} ✅ Free"
            default = True
            disabled = False

        toggled = st.checkbox(
            label,
            value=default,
            disabled=disabled,
            help=tool_info["description"],
            key=f"tool_{tool_name}"
        )

        if toggled and not disabled:
            selected_tools.append(tool_name)

    if not selected_tools:
        st.warning("Select at least one tool")

    st.divider()

    run_btn = st.button(
        "Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=len(selected_tools) == 0
    )

    total_queries = 10
    total_calls = total_queries * len(selected_tools)
    st.caption(f"Queries: {total_queries} | Tools: {len(selected_tools)} | API calls: {total_calls}")

    # ── Usage Stats ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Session Usage")

    stats = get_usage_stats()
    any_usage = any(stats[t]["calls"] > 0 for t in stats if t in [
        name for name in ALL_TOOLS
    ])

    if any_usage:
        for tool_name in selected_tools:
            if tool_name in stats:
                s = stats[tool_name]
                free_limit = ALL_TOOLS[tool_name]["free_limit"]
                cost = ALL_TOOLS[tool_name]["cost_per_call"] * s["calls"]

                st.markdown(f"**{tool_name}**")
                col1, col2 = st.columns(2)
                col1.metric("Calls", s["calls"])
                col2.metric("Tokens ~", s["estimated_tokens"])

                if free_limit > 0:
                    remaining = max(0, free_limit - s["calls"])
                    pct = min(100, int((s["calls"] / free_limit) * 100))
                    st.progress(pct / 100, text=f"{remaining} calls remaining today")
                elif cost > 0:
                    st.caption(f"Estimated cost: ${cost:.4f}")

                if s["errors"] > 0:
                    st.caption(f"Errors: {s['errors']}")

        if st.button("Reset Stats", use_container_width=True):
            reset_usage_stats()
            st.rerun()
    else:
        st.caption("No usage yet this session")

# =============================================================================
# MAIN AREA
# =============================================================================

if not run_btn:
    st.subheader("How to use")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\nEnter website URL, brand name, and ICP document in the sidebar")
    with col2:
        st.info("**Step 2**\nToggle which AI tools to run. Free tools are active by default")
    with col3:
        st.info("**Step 3**\nClick Run Analysis and wait for results")

    st.divider()
    st.subheader("Tool Status")

    rows = []
    for tool_name, info in ALL_TOOLS.items():
        key_exists = check_key_exists(info["requires_key"])
        rows.append({
            "Tool": tool_name,
            "Status": "Ready" if key_exists else "Needs API Key",
            "Free": "Yes" if info["free"] else "No",
            "Cost/Call": f"${info['cost_per_call']}" if info["cost_per_call"] > 0 else "Free",
            "Free Limit": f"{info['free_limit']} calls/day" if info["free_limit"] > 0 else "N/A",
            "Description": info["description"]
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if run_btn:
    if not website_url or not brand_name or not icp_text:
        st.error("Please fill in Website URL, Brand Name, and ICP Document.")
        st.stop()

    if not selected_tools:
        st.error("Please select at least one AI tool.")
        st.stop()

    # Step 1
    st.subheader("Step 1: Parsing ICP")
    with st.spinner("Extracting structured data..."):
        try:
            icp_data = parse_and_validate(icp_text, selected_tools=selected_tools)
            st.success(f"Industries: {', '.join(icp_data['industries'])} | Geographies: {', '.join(icp_data['geographies'])}")
        except Exception as e:
            st.error(f"ICP parsing failed: {format_error_message(str(e))}")
            st.stop()

    # Step 2
    st.subheader("Step 2: Generating Queries")
    with st.spinner("Generating buyer queries..."):
        try:
            competitors_list = [c.strip() for c in competitors_input.split(",")] if competitors_input else []
            query_backends = []
            if "Groq_Llama3" in selected_tools or "Groq_Mixtral" in selected_tools:
                query_backends.append("groq")
            if "Gemini" in selected_tools:
                query_backends.append("gemini")

            if not query_backends:
                st.error("No selected tools can generate queries. Enable Groq or Gemini.")
                st.stop()

            queries = generate_all_queries(
                icp_data,
                competitors=competitors_list,
                preferred_backend=query_backends[0],
                allowed_backends=query_backends
            )
            st.success(f"Generated {len(queries)} queries")
        except Exception as e:
            st.error(f"Query generation failed: {format_error_message(str(e))}")
            st.stop()

    # Step 3
    st.subheader("Step 3: Running AI Tools")
    all_results = []
    total_calls = len(queries) * len(selected_tools)
    progress_bar = st.progress(0)
    status_text = st.empty()
    call_count = 0
    exhausted_tools = set()
    exhausted_notices_shown = set()

    for i, q in enumerate(queries):
        status_text.text(f"Query {i+1}/{len(queries)}: {q['query'][:65]}...")
        active_tools = [t for t in selected_tools if t not in exhausted_tools]
        if not active_tools:
            st.warning("All selected tools are currently rate-limited or out of quota. Stopping run early.")
            break

        tool_responses = run_selected_tools(q["query"], active_tools)

        for tool_name, response_text in tool_responses.items():
            if response_text.startswith("ERROR"):
                clean_error = response_text.split("ERROR:", 1)[-1].strip()
                formatted_error = format_error_message(clean_error)

                error_low = clean_error.lower()
                is_quota_error = (
                    "429" in error_low
                    or "rate_limit" in error_low
                    or "resource_exhausted" in error_low
                    or "quota" in error_low
                )
                if is_quota_error:
                    exhausted_tools.add(tool_name)
                    if tool_name not in exhausted_notices_shown:
                        st.warning(
                            f"{tool_name} disabled for this run: {formatted_error}"
                        )
                        exhausted_notices_shown.add(tool_name)
                else:
                    st.warning(
                        f"{tool_name} error on query {i+1}: {formatted_error}"
                    )
                continue

            brand_data = detect_brands(response_text, brand_name)
            all_results.append({
                "query": q["query"],
                "category": q["category"],
                "filters": q["filters"],
                "tool": tool_name,
                "response": response_text,
                "brands_detected": brand_data
            })

            call_count += 1
            progress_bar.progress(call_count / total_calls)

    status_text.text("Done.")
    st.success(f"Completed {len(all_results)} query/tool combinations")

    # Step 4
    scores = calculate_citation_share(all_results, brand_name)

    # ==========================================================================
    # RESULTS
    # ==========================================================================

    st.divider()
    st.header("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Citation Share", f"{scores['overall_citation_share']}%")
    col2.metric("Total Mentions", f"{scores['total_mentions']} / {scores['total_queries_run']}")
    col3.metric("Position Score", f"{scores['position_score_pct']}%")

    st.subheader("Citation Share by Tool")
    tool_rows = []
    for tool, data in scores["citation_share_by_tool"].items():
        tool_rows.append({
            "Tool": tool,
            "Citation Share": f"{data['share_pct']}%",
            "Mentions": f"{data['mentions']}/{data['total_queries']}",
            "Free": "Yes" if ALL_TOOLS.get(tool, {}).get("free") else "No"
        })
    st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)

    st.subheader("Citation Share by Category")
    cat_rows = []
    for cat, data in scores["citation_share_by_category"].items():
        cat_rows.append({
            "Category": cat.replace("_", " ").title(),
            "Citation Share": f"{data['share_pct']}%",
            "Mentions": f"{data['mentions']}/{data['total_queries']}"
        })
    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)

    st.subheader("Top Competitor Brands")
    if scores["competitor_ranking"]:
        comp_df = pd.DataFrame(
            scores["competitor_ranking"][:15],
            columns=["Brand", "Mentions"]
        )
        comp_df["Share"] = comp_df["Mentions"].apply(
            lambda x: f"{round((x / scores['total_queries_run']) * 100)}%"
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.subheader("How Your Brand Was Mentioned")
    ctx = scores["context_breakdown"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended", ctx.get("recommended", 0))
    c2.metric("Mentioned", ctx.get("mentioned", 0))
    c3.metric("Warned Against", ctx.get("warned_against", 0))
    c4.metric("Not Mentioned", ctx.get("not_mentioned", 0))

    st.subheader("Session API Usage")
    stats = get_usage_stats()
    usage_rows = []
    for tool_name in selected_tools:
        if tool_name in stats:
            s = stats[tool_name]
            usage_rows.append({
                "Tool": tool_name,
                "API Calls Made": s["calls"],
                "Tokens Used ~": s["estimated_tokens"],
                "Errors": s["errors"],
                "Est. Cost": f"${ALL_TOOLS[tool_name]['cost_per_call'] * s['calls']:.4f}"
            })
    if usage_rows:
        st.dataframe(pd.DataFrame(usage_rows), use_container_width=True, hide_index=True)

    st.subheader("Full Query Results")
    rows = []
    for r in all_results:
        rows.append({
            "Category": r["category"].replace("_", " ").title(),
            "Tool": r["tool"],
            "Query": r["query"],
            "Answer": r["response"],
            "Mentioned": "Yes" if r["brands_detected"]["target_mentioned"] else "No",
            "Position": r["brands_detected"]["target_position"] or "-",
            "Context": r["brands_detected"]["target_context"],
            "Brands Found": ", ".join(r["brands_detected"]["all_brands"][:5]),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"citation_{brand_name.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )