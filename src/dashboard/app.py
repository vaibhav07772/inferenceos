"""InferenceOS — Streamlit Dashboard"""
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.router.model_router_v2 import ModelRouterV2
from src.router.model_router import RoutingStrategy

# ─────────────────────────────────────
st.set_page_config(
    page_title="InferenceOS",
    page_icon="🔥",
    layout="wide",
)

# ─────────────────────────────────────
# CSS
# ─────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #7C3AED, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header { color: #6B7280; margin-bottom: 1.5rem; }
    .tier-small { background: #D1FAE5; color: #065F46; padding: 0.2rem 0.6rem; border-radius: 0.25rem; font-weight: 600; }
    .tier-medium { background: #FEF3C7; color: #92400E; padding: 0.2rem 0.6rem; border-radius: 0.25rem; font-weight: 600; }
    .tier-large { background: #FEE2E2; color: #991B1B; padding: 0.2rem 0.6rem; border-radius: 0.25rem; font-weight: 600; }
    .cache-hit { background: #DBEAFE; color: #1E40AF; padding: 0.2rem 0.6rem; border-radius: 0.25rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────
# Header
# ─────────────────────────────────────
st.markdown('<h1 class="main-header">🔥 InferenceOS</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Production LLM Inference & Optimization Platform</p>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────
# Init router (cached)
# ─────────────────────────────────────
@st.cache_resource
def get_router():
    return ModelRouterV2(cache_enabled=True, cache_threshold=0.92)


# ─────────────────────────────────────
# Sidebar
# ─────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    strategy = st.selectbox(
        "Routing Strategy",
        [s.value for s in RoutingStrategy],
        index=1,  # BALANCED
    )

    max_tokens = st.slider("Max tokens", 50, 1000, 200)
    use_cache = st.checkbox("Use semantic cache", value=True)

    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.caption("**Tiers:**")
    st.caption("🟢 small  → gpt-oss-20b")
    st.caption("🟡 medium → qwen3.8-27b")
    st.caption("🔴 large  → gpt-oss-120b")

    st.markdown("---")
    st.caption("v0.1.0 • Phase 2")


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────
router = get_router()

# ─── Input section ───
st.markdown("### 🎯 Ask InferenceOS")

sample_queries = [
    "What is 2+2?",
    "What is the capital of France?",
    "Explain what an API is.",
    "Design a distributed system for 1M requests/sec.",
    "Analyze the trade-offs between microservices and monoliths.",
]

col1, col2 = st.columns([4, 1])
with col1:
    sample = st.selectbox("Try a sample:", ["Custom"] + sample_queries)

if sample == "Custom":
    prompt = st.text_area("Your prompt:", height=100, placeholder="Type your query...")
else:
    prompt = st.text_area("Your prompt:", value=sample, height=100)

with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    run = st.button("🚀 Run", type="primary", use_container_width=True)


# ─── Run inference ───
if run and prompt.strip():
    with st.spinner("Running inference..."):
        try:
            strategy_enum = RoutingStrategy(strategy)
            result = router.route(
                prompt=prompt,
                strategy=strategy_enum,
                max_tokens=max_tokens,
                use_cache=use_cache,
            )

            d = result.decision
            r = result.response

            st.markdown("---")
            st.markdown("### 📊 Result")

            # ─── Row 1: Tier + Cache ───
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                tier_class = f"tier-{d.tier}"
                st.markdown("**Tier**")
                st.markdown(
                    f'<span class="{tier_class}">{d.tier.upper()}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(d.model)

            with c2:
                st.markdown("**Complexity**")
                st.markdown(f"**{d.classifier_result['complexity']}**")
                st.caption(f"Score: {d.classifier_result['score']}")

            with c3:
                st.markdown("**Cache**")
                if result.cache_hit:
                    st.markdown(
                        f'<span class="cache-hit">HIT ({result.cache_similarity:.2f})</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Saved: ${result.cache_savings_usd:.6f}")
                else:
                    st.markdown("**MISS**")
                    st.caption("Fresh inference")

            with c4:
                st.markdown("**Latency**")
                st.markdown(f"**{result.total_latency_ms:.0f}ms**")
                st.caption(f"Cost: ${r['cost_usd']:.6f}")

            # ─── Response ───
            st.markdown("### 📄 Response")
            st.markdown(r.get("response", r.get("error", "No response")))

            # ─── Debug ───
            with st.expander("🔍 Full Details"):
                st.json(result.to_dict())

        except Exception as e:
            st.error(f"Error: {e}")


# ─── Landing state ───
else:
    st.markdown("---")
    st.markdown("### 🎯 How InferenceOS Works")

    cols = st.columns(4)
    with cols[0]:
        st.markdown("#### 🎯 Classify")
        st.caption("Analyzes query complexity (0-1 score)")
    with cols[1]:
        st.markdown("#### 🚦 Route")
        st.caption("Picks best model tier (small/medium/large)")
    with cols[2]:
        st.markdown("#### 💾 Cache")
        st.caption("Semantic cache hits save 100% cost")
    with cols[3]:
        st.markdown("#### 📊 Observe")
        st.caption("Prometheus metrics + cost tracking")

    st.info("👆 Enter a query above and click Run.")

    # ─── Cache stats ───
    st.markdown("---")
    st.markdown("### 📊 Cache Statistics")

    stats = router.cache_stats()

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Entries", stats.get("total_entries", 0))
    with sc2:
        st.metric("Hits", stats.get("hits", 0))
    with sc3:
        st.metric("Misses", stats.get("misses", 0))
    with sc4:
        st.metric("Hit Rate", f"{stats.get('hit_rate_pct', 0)}%")


# ─────────────────────────────────────
# Footer
# ─────────────────────────────────────
st.markdown("---")
st.caption("🔥 InferenceOS — Production LLM Inference Platform • v0.1.0")