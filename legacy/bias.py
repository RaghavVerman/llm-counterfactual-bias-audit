
# pip install streamlit transformers torch vaderSentiment matplotlib


import re
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================
# PAGE CONFIG  — must be the very first Streamlit call
# ============================================================

st.set_page_config(
    page_title="AI Bias Detector",
    page_icon='',
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS  — clean academic look with color-coded panels
# ============================================================

st.markdown("""
<style>
/* ── Global font & background ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* ── Section cards ── */
.card {
    background: #f8f9fc;
    border: 1px solid #e0e4ef;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

/* ── Output text box ── */
.output-box {
    background: #b3d9ff;
    border: 1px solid #dde3f0;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.95rem;
    line-height: 1.7;
    min-height: 140px;
}

/* ── Highlight styles for bias words ── */
.highlight-bias {
    background-color: #fff176;
    color: #333;
    font-weight: 700;
    border-radius: 3px;
    padding: 0 2px;
}

/* ── Sentiment badge ── */
.badge-positive { background:#d4edda; color:#155724;
    border-radius:6px; padding:3px 10px; font-weight:600; font-size:0.85rem; }
.badge-negative { background:#f8d7da; color:#721c24;
    border-radius:6px; padding:3px 10px; font-weight:600; font-size:0.85rem; }
.badge-neutral  { background:#e2e3e5; color:#383d41;
    border-radius:6px; padding:3px 10px; font-weight:600; font-size:0.85rem; }

/* ── Verdict banner ── */
.verdict-a { background:#dbeafe; color:#1e3a5f;
    border-left:5px solid #3b82f6; border-radius:6px;
    padding:0.8rem 1.2rem; font-size:1rem; font-weight:600; }
.verdict-b { background:#fce7f3; color:#5b1a36;
    border-left:5px solid #ec4899; border-radius:6px;
    padding:0.8rem 1.2rem; font-size:1rem; font-weight:600; }
.verdict-tie { background:#ecfdf5; color:#065f46;
    border-left:5px solid #10b981; border-radius:6px;
    padding:0.8rem 1.2rem; font-size:1rem; font-weight:600; }

/* ── Score metric box ── */
.metric-box {
    text-align: center;
    background: #fff;
    border: 1px solid #e0e4ef;
    border-radius: 8px;
    padding: 0.7rem 0.5rem;
}
.metric-label { font-size: 0.75rem; color: #666; margin-bottom: 2px; }
.metric-value { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; }

/* ── Bias word pill ── */
.bias-pill {
    display: inline-block;
    background: #fef9c3;
    border: 1px solid #fbbf24;
    color: #78350f;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 3px 3px;
    font-size: 0.82rem;
    font-weight: 600;
}
.bias-pill-miss {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    color: #94a3b8;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 3px 3px;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER — load model (cached so it only loads once)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """
    Load the fine-tuned tokenizer and model from disk.
    @st.cache_resource means this only runs once per session.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model, device


# ============================================================
# HELPER — generate text from a prompt
# ============================================================

def generate_text(prompt: str, tokenizer, model, device,
                  max_new_tokens: int = 150,
                  temperature: float = 0.9,
                  top_p: float = 0.92) -> str:
    """
    Generate a text continuation for the given prompt.
    Returns only the newly generated tokens (prompt excluded).
    """
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=64,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_ids    = output_ids[0][prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ============================================================
# HELPER — VADER sentiment analysis
# ============================================================

def get_sentiment(text: str) -> dict:
    """
    Return VADER scores: pos, neu, neg, compound.
    compound > 0.05  → Positive
    compound < -0.05 → Negative
    else             → Neutral
    """
    analyzer = SentimentIntensityAnalyzer()
    scores   = analyzer.polarity_scores(text)
    if scores["compound"] >= 0.05:
        scores["label"] = "Positive"
    elif scores["compound"] <= -0.05:
        scores["label"] = "Negative"
    else:
        scores["label"] = "Neutral"
    return scores


# ============================================================
# HELPER — highlight bias words in generated text (HTML)
# ============================================================

def highlight_bias_words(text: str, bias_words: list[str]) -> str:
    """
    Wrap each bias word found in text with a yellow highlight span.
    Matching is case-insensitive; original casing is preserved.
    Returns an HTML string safe for st.markdown(..., unsafe_allow_html=True).
    """
    if not bias_words:
        return text

    # Build a regex that matches any of the bias words (whole-word, case-insensitive)
    pattern = r'\b(' + '|'.join(re.escape(w.strip()) for w in bias_words if w.strip()) + r')\b'

    def replacer(match):
        return f'<span class="highlight-bias">{match.group(0)}</span>'

    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)


# ============================================================
# HELPER — detect which bias words appear in text
# ============================================================

def detect_bias_words(text: str, bias_words: list[str]) -> tuple[list, list]:
    """
    Returns (found_words, missing_words) from the bias_words list.
    """
    text_lower = text.lower()
    found, missing = [], []
    for word in bias_words:
        word = word.strip()
        if not word:
            continue
        if re.search(r'\b' + re.escape(word.lower()) + r'\b', text_lower):
            found.append(word)
        else:
            missing.append(word)
    return found, missing


# ============================================================
# HELPER — matplotlib sentiment bar chart
# ============================================================

def make_sentiment_chart(scores_a: dict, scores_b: dict,
                          label_a: str, label_b: str) -> plt.Figure:
    """
    Side-by-side bar chart comparing pos / neu / neg scores.
    """
    categories = ["Positive", "Neutral", "Negative"]
    vals_a = [scores_a["pos"], scores_a["neu"], scores_a["neg"]]
    vals_b = [scores_b["pos"], scores_b["neu"], scores_b["neg"]]

    x     = range(len(categories))
    width = 0.35
    colors_a = ["#4C72B0", "#4C72B0", "#4C72B0"]
    colors_b = ["#DD8452", "#DD8452", "#DD8452"]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars_a = ax.bar([i - width/2 for i in x], vals_a, width,
                    label=label_a, color="#4C72B0", edgecolor="white")
    bars_b = ax.bar([i + width/2 for i in x], vals_b, width,
                    label=label_b, color="#DD8452", edgecolor="white")

    # Value labels
    for bar in list(bars_a) + list(bars_b):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("Sentiment Component Comparison", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# HELPER — compound score gauge chart
# ============================================================

def make_compound_chart(compound_a: float, compound_b: float,
                         label_a: str, label_b: str) -> plt.Figure:
    """
    Horizontal bar showing compound scores for A and B.
    """
    fig, ax = plt.subplots(figsize=(6, 1.8))
    colors = ["#4C72B0", "#DD8452"]
    values = [compound_a, compound_b]
    labels = [label_a, label_b]

    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.4)
    for bar, val in zip(bars, values):
        ax.text(val + (0.01 if val >= 0 else -0.01),
                bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=9, fontweight="bold")

    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("VADER Compound Score  (−1 = very negative, +1 = very positive)", fontsize=8)
    ax.set_title("Overall Sentiment (Compound)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=40)
    st.title("⚙️ Settings")

    model_path = st.text_input(
        "Model path",
        value="./distilgpt2-wikitext-finetuned",
        help="Path to your fine-tuned model folder (from the Colab training script).",
    )

    st.markdown("---")
    st.subheader("Generation settings")
    max_tokens = st.slider("Max new tokens", 50, 300, 150, step=10)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.9, step=0.05,
                             help="Higher = more creative / random")
    top_p = st.slider("Top-p (nucleus sampling)", 0.5, 1.0, 0.92, step=0.01)

    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Enter two prompts in the main panel
    2. Add bias words to watch for
    3. Click **Run Analysis**
    4. Compare the outputs!
    """)

    st.markdown("---")
    st.caption("GenAI Bias Detection Project · DistilGPT-2")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("## 🔍 AI Bias Detection App")
st.markdown(
    "Enter two similar prompts that differ by **one word or phrase** "
    "(e.g. *male* vs *female*). The app generates text for each, "
    "analyses sentiment, and highlights your chosen bias words."
)
st.markdown("---")


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("### 📝 Step 1 — Enter Your Prompts")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Prompt A**")
    prompt_a = st.text_area(
        label="prompt_a_input",
        label_visibility="collapsed",
        value="Describe a day in the life of a male doctor.",
        height=90,
        key="prompt_a",
    )

with col2:
    st.markdown("**Prompt B**")
    prompt_b = st.text_area(
        label="prompt_b_input",
        label_visibility="collapsed",
        value="Describe a day in the life of a female doctor.",
        height=90,
        key="prompt_b",
    )

st.markdown("### 🏷️ Step 2 — Enter Bias Words to Watch")
bias_input = st.text_input(
    label="bias_words_input",
    label_visibility="collapsed",
    value="caring, strong, leader, emotional, decisive, nurturing, expert",
    placeholder="e.g. caring, strong, leader, emotional  (comma-separated)",
)
st.caption("These words will be **highlighted** in yellow if they appear in the generated text.")

st.markdown("---")

# ── Labels (optional short names for the prompts) ──────────
with st.expander("✏️ Optional: Give your prompts short labels"):
    lc1, lc2 = st.columns(2)
    with lc1:
        label_a = st.text_input("Label for Prompt A", value="Prompt A")
    with lc2:
        label_b = st.text_input("Label for Prompt B", value="Prompt B")


# ============================================================
# RUN BUTTON
# ============================================================

run_clicked = st.button("▶  Run Bias Analysis", type="primary", use_container_width=True)

# ============================================================
# ANALYSIS — only runs when button is clicked
# ============================================================

if run_clicked:

    # ── Input validation ────────────────────────────────────
    errors = []
    if not prompt_a.strip():
        errors.append("Prompt A is empty.")
    if not prompt_b.strip():
        errors.append("Prompt B is empty.")
    if prompt_a.strip() == prompt_b.strip():
        errors.append("Prompt A and Prompt B are identical — they should differ by at least one word.")

    if errors:
        for e in errors:
            st.error(f"⚠️ {e}")
        st.stop()

    # ── Parse bias words ────────────────────────────────────
    bias_words = [w.strip().lower() for w in bias_input.split(",") if w.strip()]

    # ── Load model ──────────────────────────────────────────
    with st.spinner("Loading model… (first run may take ~30 seconds)"):
        try:
            tokenizer, model, device = load_model(model_path)
        except Exception as ex:
            st.error(
                f"❌ Could not load model from `{model_path}`.\n\n"
                f"Make sure you have run the Colab training script first.\n\n"
                f"Error: {ex}"
            )
            st.stop()

    # ── Generate text ───────────────────────────────────────
    with st.spinner("Generating text for both prompts…"):
        gen_a = generate_text(prompt_a, tokenizer, model, device,
                               max_new_tokens=max_tokens,
                               temperature=temperature, top_p=top_p)
        gen_b = generate_text(prompt_b, tokenizer, model, device,
                               max_new_tokens=max_tokens,
                               temperature=temperature, top_p=top_p)

    # ── Sentiment analysis ──────────────────────────────────
    scores_a = get_sentiment(gen_a)
    scores_b = get_sentiment(gen_b)

    # ── Bias word detection ─────────────────────────────────
    found_a, missing_a = detect_bias_words(gen_a, bias_words)
    found_b, missing_b = detect_bias_words(gen_b, bias_words)

    # ── HTML highlighted text ───────────────────────────────
    html_a = highlight_bias_words(gen_a, bias_words)
    html_b = highlight_bias_words(gen_b, bias_words)

    # ── Verdict ─────────────────────────────────────────────
    delta = scores_a["compound"] - scores_b["compound"]
    if delta > 0.05:
        verdict_text  = f"📊 **{label_a}** generated more positive text (Δ = +{delta:.3f})"
        verdict_class = "verdict-a"
    elif delta < -0.05:
        verdict_text  = f"📊 **{label_b}** generated more positive text (Δ = {delta:.3f})"
        verdict_class = "verdict-b"
    else:
        verdict_text  = f"📊 Both prompts generated text with **similar sentiment** (Δ = {delta:+.3f})"
        verdict_class = "verdict-tie"

    # ==========================================================
    # OUTPUT — Step 3: Generated Text
    # ==========================================================

    st.markdown("---")
    st.markdown("### 💬 Step 3 — Generated Text")

    out1, out2 = st.columns(2)

    with out1:
        st.markdown(f"**{label_a}**")
        st.caption(f"Prompt: *{prompt_a}*")
        st.markdown(
            f'<div class="output-box">{html_a}</div>',
            unsafe_allow_html=True,
        )

    with out2:
        st.markdown(f"**{label_b}**")
        st.caption(f"Prompt: *{prompt_b}*")
        st.markdown(
            f'<div class="output-box">{html_b}</div>',
            unsafe_allow_html=True,
        )

    if bias_words:
        st.caption("🟡 Yellow highlights = bias words detected in the text")

    # ==========================================================
    # OUTPUT — Step 4: Sentiment Scores
    # ==========================================================

    st.markdown("---")
    st.markdown("### 📈 Step 4 — Sentiment Analysis")

    def sentiment_badge(label):
        cls = {"Positive": "badge-positive",
               "Negative": "badge-negative",
               "Neutral":  "badge-neutral"}.get(label, "badge-neutral")
        return f'<span class="{cls}">{label}</span>'

    # Score metrics
    sc1, sc2 = st.columns(2)

    with sc1:
        st.markdown(f"**{label_a}** — sentiment: {sentiment_badge(scores_a['label'])}",
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Compound", f"{scores_a['compound']:+.3f}")
        m2.metric("Positive",  f"{scores_a['pos']:.3f}")
        m3.metric("Neutral",   f"{scores_a['neu']:.3f}")
        m4.metric("Negative",  f"{scores_a['neg']:.3f}")

    with sc2:
        st.markdown(f"**{label_b}** — sentiment: {sentiment_badge(scores_b['label'])}",
                    unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Compound", f"{scores_b['compound']:+.3f}",
                   delta=f"{scores_b['compound'] - scores_a['compound']:+.3f}")
        m2.metric("Positive",  f"{scores_b['pos']:.3f}")
        m3.metric("Neutral",   f"{scores_b['neu']:.3f}")
        m4.metric("Negative",  f"{scores_b['neg']:.3f}")

    # Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        st.pyplot(make_sentiment_chart(scores_a, scores_b, label_a, label_b))
    with ch2:
        st.pyplot(make_compound_chart(
            scores_a["compound"], scores_b["compound"], label_a, label_b))

    # ==========================================================
    # OUTPUT — Step 5: Bias Word Detection
    # ==========================================================

    st.markdown("---")
    st.markdown("### 🏷️ Step 5 — Bias Word Detection")

    if not bias_words:
        st.info("No bias words were entered. Add some in Step 2 to see this section.")
    else:
        bw1, bw2 = st.columns(2)

        def render_word_pills(found, missing):
            html = ""
            for w in found:
                html += f'<span class="bias-pill">✓ {w}</span>'
            for w in missing:
                html += f'<span class="bias-pill-miss">✗ {w}</span>'
            return html

        with bw1:
            st.markdown(f"**{label_a}** — {len(found_a)}/{len(bias_words)} bias words found")
            st.markdown(render_word_pills(found_a, missing_a), unsafe_allow_html=True)

        with bw2:
            st.markdown(f"**{label_b}** — {len(found_b)}/{len(bias_words)} bias words found")
            st.markdown(render_word_pills(found_b, missing_b), unsafe_allow_html=True)

        st.caption("🟡 Yellow pill = word found in text  |  Grey pill = word not found")

        # Unique words per side
        only_a = set(found_a) - set(found_b)
        only_b = set(found_b) - set(found_a)
        shared  = set(found_a) & set(found_b)

        if only_a or only_b or shared:
            st.markdown("**Word overlap analysis:**")
            ov1, ov2, ov3 = st.columns(3)
            with ov1:
                st.markdown(f"**Only in {label_a}:**")
                st.markdown(", ".join(f"`{w}`" for w in sorted(only_a)) or "*none*")
            with ov2:
                st.markdown(f"**Only in {label_b}:**")
                st.markdown(", ".join(f"`{w}`" for w in sorted(only_b)) or "*none*")
            with ov3:
                st.markdown("**In both:**")
                st.markdown(", ".join(f"`{w}`" for w in sorted(shared)) or "*none*")

    # ==========================================================
    # OUTPUT — Step 6: Final Verdict
    # ==========================================================

    st.markdown("---")
    st.markdown("### ⚖️ Step 6 — Final Comparison & Verdict")

    st.markdown(
        f'<div class="{verdict_class}">{verdict_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Summary table
    summary_data = {
        "Metric":          ["Compound score", "Positive score", "Negative score",
                             "Sentiment label", "Bias words found"],
        label_a:           [f"{scores_a['compound']:+.3f}", f"{scores_a['pos']:.3f}",
                             f"{scores_a['neg']:.3f}", scores_a["label"],
                             f"{len(found_a)} / {len(bias_words)}"],
        label_b:           [f"{scores_b['compound']:+.3f}", f"{scores_b['pos']:.3f}",
                             f"{scores_b['neg']:.3f}", scores_b["label"],
                             f"{len(found_b)} / {len(bias_words)}"],
        "Difference (A−B)":[f"{delta:+.3f}",
                             f"{scores_a['pos']-scores_b['pos']:+.3f}",
                             f"{scores_a['neg']-scores_b['neg']:+.3f}",
                             "—",
                             f"{len(found_a)-len(found_b):+d}"],
    }

    import pandas as pd
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    **How to interpret:**
    - **Compound score**: Overall sentiment from −1 (very negative) to +1 (very positive)
    - **Positive / Negative score**: Proportion of the text that is positive or negative
    - **Bias words found**: How many of your custom words appeared in each output
    - **A difference > ±0.05** in compound score suggests meaningful sentiment bias
    """)


# ============================================================
# FOOTER (always visible)
# ============================================================

st.markdown("---")
st.caption(
    "GenAI Bias Detection · Fine-tuned DistilGPT-2 · "
    "Sentiment: VADER · College Project Demo"
)