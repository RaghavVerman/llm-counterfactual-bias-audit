"""Streamlit dashboard for the counterfactual bias audit.

    streamlit run app.py
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from bias_audit.audit import AuditConfig, metric_columns, run_audit
from bias_audit.generation import Generator
from bias_audit.lexicons import LEXICONS, _pattern
from bias_audit.prompts import AXES, TEMPLATES, build_skeletons
from bias_audit.scoring import SCORERS
from bias_audit.stats import pooled_sd

RESULTS = Path("results")
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
INK_MUTED = "#8a8984"

METRIC_INFO = {
    "regard": ("Regard", "How favourably the person is portrayed: P(positive) − P(negative)."),
    "regard_negative": ("Negative regard", "Probability the person is portrayed negatively."),
    "sentiment": ("Sentiment (VADER)", "Tone of the language, −1 to 1. Not specific to the person."),
    "toxicity": ("Toxicity", "Probability the continuation is toxic."),
    "lex_agentic": ("Agentic words", "Words like leader, confident, ambitious, per 100 words."),
    "lex_communal": ("Communal words", "Words like caring, supportive, warm, per 100 words."),
    "lex_appearance": ("Appearance words", "Words like beautiful, pretty, attractive, per 100 words."),
    "lex_family": ("Family words", "Words like wife, mother, children, home, per 100 words."),
    "lex_agentic_minus_communal": ("Agentic − communal", "Positive = more agentic than communal language."),
}
LEX_STYLE = {  # background tint + text label, so category is never colour-only
    "agentic": "#d9d4f7", "communal": "#c9eedf", "appearance": "#f8d6e3", "family": "#fbe7b5",
}

st.set_page_config(page_title="Bias Audit", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
.lex { border-radius: 3px; padding: 0 3px; color: #0b0b0b; }
.lex small { font-size: 0.65em; opacity: .7; margin-left: 2px; text-transform: uppercase; }
.gen { border-left: 3px solid var(--c); padding: .35rem .7rem; margin: .35rem 0; line-height: 1.55; }
.gen .meta { font-size: .78rem; opacity: .7; }
.prompt { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def label(metric: str) -> str:
    return METRIC_INFO.get(metric, (metric, ""))[0]


def saved_runs() -> list[Path]:
    if not RESULTS.exists():
        return []
    runs = [p for p in RESULTS.iterdir() if (p / "summary.csv").exists() and (p / "generations.csv").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def load_run(path: str, mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    p = Path(path)
    gens = pd.read_csv(p / "generations.csv")
    summary = pd.read_csv(p / "summary.csv")
    cfg = json.loads((p / "config.json").read_text()) if (p / "config.json").exists() else {}
    if "pooled_sd" not in summary.columns:  # runs saved before the column existed
        summary["pooled_sd"] = [
            pooled_sd(gens.loc[gens.group == r.group_a, r.metric].dropna(),
                      gens.loc[gens.group == r.group_b, r.metric].dropna())
            for r in summary.itertuples()
        ]
    return gens, summary, cfg


@st.cache_resource(show_spinner="Loading language model…")
def get_generator(model: str) -> Generator:
    return Generator(model)


def group_scale(groups: list[str]) -> alt.Scale:
    return alt.Scale(domain=groups, range=GROUP_COLORS[: len(groups)])


def highlight(text: str) -> str:
    """HTML-escape ``text`` and mark lexicon matches with a tint and category tag."""
    spans = []
    for cat in LEXICONS:
        spans += [(m.start(), m.end(), cat) for m in _pattern(cat).finditer(text)]
    spans.sort()
    out, pos = [], 0
    for s, e, cat in spans:
        if s < pos:
            continue  # overlapping match from another category
        out.append(html.escape(text[pos:s]))
        out.append(f'<span class="lex" style="background:{LEX_STYLE[cat]}">'
                   f'{html.escape(text[s:e])}<small>{cat[:4]}</small></span>')
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def describe(row) -> str:
    """One plain-English sentence per comparison."""
    direction = "higher" if row.diff > 0 else "lower"
    verdict = ("**significant** after correction" if row.significant
               else "not significant after correction")
    ci_note = "" if row.ci_low <= 0 <= row.ci_high else " (95% CI excludes zero)"
    return (f"**{label(row.metric)}**: *{row.group_b}* outputs score {direction} than *{row.group_a}* "
            f"by {abs(row.diff):.3f}{ci_note}; d = {row.cohens_d:+.2f}, "
            f"p<sub>Holm</sub> = {row.p_holm:.3f}, {verdict}.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Audit")
    mode = st.radio("Mode", ["Load saved run", "Run new audit"], horizontal=True,
                    label_visibility="collapsed")
    run_path: Path | None = None

    if mode == "Load saved run":
        runs = saved_runs()
        if not runs:
            st.info("No saved runs yet. Switch to **Run new audit**.")
        else:
            default = runs.index(Path(st.session_state["last_run"])) if \
                st.session_state.get("last_run") and Path(st.session_state["last_run"]) in runs else 0
            run_path = st.selectbox("Saved run", runs, index=default, format_func=lambda p: p.name)
    else:
        model = st.text_input("Model (HF id or local path)", "gpt2",
                              help="Base LMs (gpt2, ./model) or instruction models "
                                   "(e.g. Qwen/Qwen2.5-0.5B-Instruct).")
        axis = st.selectbox("Demographic axis", list(AXES), format_func=lambda a: a.replace("_", " "))
        st.caption(AXES[axis]["description"])
        families = st.multiselect("Template families", list(TEMPLATES), default=list(TEMPLATES))
        n_sk = st.slider("Counterfactual prompt sets", 4, 80, 24, step=4)
        n_samples = st.slider("Samples per prompt per group", 1, 20, 5)
        scorers = st.multiselect("Scorers", list(SCORERS), default=list(SCORERS))
        with st.expander("Sampling"):
            temperature = st.slider("Temperature", 0.3, 1.5, 0.8, 0.05)
            top_p = st.slider("Top-p", 0.5, 1.0, 0.95, 0.01)
            max_tokens = st.slider("Max new tokens", 20, 120, 40, 10)
            seed = st.number_input("Seed", 0, 10_000, 0)
        n_actual = len(build_skeletons(axis, families or None, max_skeletons=n_sk, seed=int(seed)))
        n_gen = n_actual * n_samples * len(AXES[axis]["groups"])
        st.caption(f"{n_actual} prompt sets × {len(AXES[axis]['groups'])} groups × {n_samples} samples "
                   f"= **{n_gen} generations**. CPU: roughly {max(1, n_gen // 30)} min.")
        go = st.button("Run audit", type="primary", use_container_width=True,
                       disabled=not (families and scorers and model.strip()))

        if go:
            cfg = AuditConfig(model=model.strip(), axis=axis, families=families, max_skeletons=n_sk,
                              samples_per_prompt=n_samples, max_new_tokens=max_tokens,
                              temperature=temperature, top_p=top_p, scorers=scorers, seed=int(seed))
            slug = re.sub(r"[^\w.-]+", "_", cfg.model.strip("./\\")) or "model"
            out = RESULTS / f"{slug}_{axis}_{datetime.now():%Y%m%d-%H%M%S}"
            bar = st.progress(0.0, text="Generating…")
            try:
                gen = get_generator(cfg.model)
                with st.spinner("Generating, scoring and testing…"):
                    run_audit(cfg, out, generator=gen,
                              progress=lambda d, t: bar.progress(d / t, text=f"Generating {d}/{t} prompt sets"))
            except Exception as exc:  # surface model-loading / download errors in the UI
                st.error(f"Audit failed: {exc}")
                st.stop()
            bar.empty()
            st.session_state["last_run"] = str(out)
            st.success(f"Saved to {out}")
        # Keep showing the latest run while the user explores tabs (each click reruns the script).
        if st.session_state.get("last_run") and Path(st.session_state["last_run"]).exists():
            run_path = Path(st.session_state["last_run"])

# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------

st.title("Counterfactual Bias Audit")
st.caption("Each prompt is filled once per demographic group, with everything else fixed, then sampled "
           "repeatedly. Differences are tested within prompt sets, with corrections for multiple tests.")

if run_path is None:
    st.info("Load a saved run or start a new audit from the sidebar.")
    st.stop()

gens, summary, cfg = load_run(str(run_path), run_path.stat().st_mtime)
groups = list(dict.fromkeys(gens["group"]))
metrics = [m for m in metric_columns(gens) if m in set(summary["metric"])]
gscale = group_scale(groups)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", str(cfg.get("model", gens["model"].iloc[0])))
c2.metric("Axis", str(gens["axis"].iloc[0]).replace("_", " "))
c3.metric("Generations", f"{len(gens):,}", help=f"{gens['skeleton_id'].nunique()} counterfactual prompt sets")
n_sig = int(summary["significant"].sum())
c4.metric("Significant differences", f"{n_sig} of {len(summary)}",
          help="Holm-corrected permutation test, α = 0.05")

tab_find, tab_dist, tab_pairs, tab_method = st.tabs(
    ["Findings", "Distributions", "Counterfactual examples", "Method & limits"])

# ---- Findings -------------------------------------------------------------
with tab_find:
    pairs = summary[["group_a", "group_b"]].drop_duplicates().apply(tuple, axis=1).tolist()
    pair = pairs[0] if len(pairs) == 1 else st.selectbox(
        "Comparison", pairs, format_func=lambda p: f"{p[1]} vs {p[0]}")
    s = summary[(summary.group_a == pair[0]) & (summary.group_b == pair[1])].copy()
    sd = s["pooled_sd"].replace(0, np.nan)
    s["std_diff"], s["std_lo"], s["std_hi"] = s["diff"] / sd, s["ci_low"] / sd, s["ci_high"] / sd
    s["label"] = s["metric"].map(label)
    s["status"] = np.where(s["significant"], "Significant (Holm)",
                           np.where((s.ci_low > 0) | (s.ci_high < 0), "CI excludes 0, not significant",
                                    "No evidence of difference"))
    order = s.sort_values("std_diff", key=np.abs, ascending=False)["label"].tolist()

    st.subheader(f"How *{pair[1]}* outputs differ from *{pair[0]}* outputs")
    st.caption("Standardized difference (difference ÷ pooled SD) with 95% cluster-bootstrap CI, so metrics "
               "on different scales share one axis. Right of zero = higher for "
               f"{pair[1]}. Filled markers = significant after Holm correction.")
    tip = [alt.Tooltip("label:N", title="Metric"),
           alt.Tooltip("mean_a:Q", title=f"Mean ({pair[0]})", format=".3f"),
           alt.Tooltip("mean_b:Q", title=f"Mean ({pair[1]})", format=".3f"),
           alt.Tooltip("diff:Q", title="Raw difference", format="+.3f"),
           alt.Tooltip("ci_low:Q", title="CI low", format="+.3f"),
           alt.Tooltip("ci_high:Q", title="CI high", format="+.3f"),
           alt.Tooltip("cohens_d:Q", title="Cohen's d", format="+.2f"),
           alt.Tooltip("p_holm:Q", title="p (Holm)", format=".3f"),
           alt.Tooltip("status:N", title="Result")]
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=220))
    base = alt.Chart(s)
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK_MUTED, strokeDash=[4, 3]).encode(x="x:Q")
    ci = base.mark_rule(strokeWidth=2, color="#52514e").encode(
        x=alt.X("std_lo:Q", title="Standardized difference"), x2="std_hi:Q", y=y, tooltip=tip)
    pts_sig = base.transform_filter("datum.significant").mark_point(
        size=110, strokeWidth=2, color="#2a78d6", filled=True, opacity=1).encode(
        x="std_diff:Q", y=y, tooltip=tip)
    pts_ns = base.transform_filter("!datum.significant").mark_point(
        size=110, strokeWidth=2, color="#2a78d6", filled=False).encode(
        x="std_diff:Q", y=y, tooltip=tip)
    st.altair_chart((zero + ci + pts_sig + pts_ns).properties(height=36 * len(s) + 40), use_container_width=True)

    st.markdown("#### What this says")
    for row in s.sort_values("p_holm").itertuples():
        st.markdown("- " + describe(row), unsafe_allow_html=True)
    if n_sig == 0:
        st.info("No difference survives multiple-testing correction. Differences whose CI excludes zero are "
                "worth re-testing with more prompt sets and samples.")

    with st.expander("Full table"):
        show = s[["label", "mean_a", "mean_b", "diff", "ci_low", "ci_high", "cohens_d",
                  "frac_skeletons_b_higher", "p_perm", "p_holm", "significant", "n_skeletons"]]
        st.dataframe(show.rename(columns={"label": "metric", "mean_a": f"mean {pair[0]}",
                                          "mean_b": f"mean {pair[1]}",
                                          "frac_skeletons_b_higher": f"share of sets {pair[1]} higher"}),
                     hide_index=True, use_container_width=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download summary.csv", summary.to_csv(index=False), "summary.csv", "text/csv")
    d2.download_button("Download generations.csv", gens.to_csv(index=False), "generations.csv", "text/csv")

# ---- Distributions --------------------------------------------------------
with tab_dist:
    m = st.selectbox("Metric", metrics, format_func=label, key="dist_metric")
    st.caption(METRIC_INFO.get(m, ("", ""))[1])
    left, right = st.columns(2)
    with left:
        st.markdown("**Per-sample distribution**")
        box = alt.Chart(gens).mark_boxplot(size=38, median={"color": "#0b0b0b"}, opacity=0.85).encode(
            x=alt.X("group:N", sort=groups, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{m}:Q", title=label(m)),
            color=alt.Color("group:N", scale=gscale, legend=alt.Legend(title="Group", orient="top")),
            tooltip=["group:N", alt.Tooltip(f"mean({m}):Q", format=".3f", title="Mean")])
        st.altair_chart(box.properties(height=320), use_container_width=True)
    with right:
        st.markdown("**Mean by template family**")
        fam = gens.groupby(["family", "group"], as_index=False)[m].mean()
        bars = alt.Chart(fam).mark_bar(cornerRadiusEnd=4, size=14).encode(
            y=alt.Y("family:N", title=None),
            yOffset=alt.YOffset("group:N", sort=groups),
            x=alt.X(f"{m}:Q", title=f"Mean {label(m).lower()}"),
            color=alt.Color("group:N", scale=gscale, legend=alt.Legend(title="Group", orient="top")),
            tooltip=["family:N", "group:N", alt.Tooltip(f"{m}:Q", format=".3f", title="Mean")])
        st.altair_chart(bars.properties(height=320), use_container_width=True)
    st.caption("Family breakdowns use few prompt sets each, so read them as exploratory, not as separate tests.")

# ---- Counterfactual examples ------------------------------------------------
with tab_pairs:
    m2 = st.selectbox("Rank prompt sets by gap in", metrics, format_func=label, key="pair_metric")
    gb = gens.groupby(["skeleton_id", "group"])[m2].mean().unstack("group")
    gap = (gb[groups[-1]] - gb[groups[0]]).dropna()
    ranked = gap.reindex(gap.abs().sort_values(ascending=False).index)
    first_prompt = gens.groupby("skeleton_id")["prompt"].first()
    sid = st.selectbox(
        "Prompt set (largest gap first)", ranked.index,
        format_func=lambda k: f"{ranked[k]:+.2f}  ·  {first_prompt[k]}")
    st.caption(f"Gap = mean {label(m2).lower()} for {groups[-1]} minus {groups[0]}. Highlighted words are "
               "lexicon matches: " + " ".join(
                   f'<span class="lex" style="background:{c}">{k}</span>' for k, c in LEX_STYLE.items()),
               unsafe_allow_html=True)
    cols = st.columns(len(groups))
    sub = gens[gens.skeleton_id == sid]
    for col, g, color in zip(cols, groups, GROUP_COLORS):
        rows = sub[sub.group == g]
        with col:
            st.markdown(f"**{g}** — mean {label(m2).lower()}: {rows[m2].mean():.3f}")
            st.markdown(f'<div class="prompt">{html.escape(rows.prompt.iloc[0])} …</div>', unsafe_allow_html=True)
            for r in rows.itertuples():
                scores = " · ".join(f"{label(k)} {getattr(r, k):+.2f}"
                                    for k in ("regard", "sentiment") if k in gens.columns)
                st.markdown(f'<div class="gen" style="--c:{color}">{highlight(str(r.continuation))}'
                            f'<div class="meta">{scores}</div></div>', unsafe_allow_html=True)
    st.caption("One prompt set is anecdotal. Use this view to check that the statistics reflect real "
               "differences in wording, not artefacts.")

# ---- Method ----------------------------------------------------------------
with tab_method:
    st.markdown(f"""
**Prompts.** Templates are filled with aligned terms per group (e.g. *father ↔ mother*, *James ↔ Mary*),
so each prompt set differs only in the demographic term. Families: {", ".join(TEMPLATES)}.
They follow Sheng et al. (2019, regard/occupation prompts) and Madera et al. (2009, reference letters).

**Generation.** Each prompt is sampled several times (temperature {cfg.get("temperature", "?")},
top-p {cfg.get("top_p", "?")}), trimmed to two sentences. Instruction-tuned models get the prompt inside
a short "continue this text" instruction.

**Metrics.** {"; ".join(f"*{v[0]}*: {v[1]}" for v in METRIC_INFO.values())}

**Statistics.** The effect is the mean within-prompt-set difference between groups. The p-values come
from a permutation test that shuffles group labels within each prompt set, and the 95% CIs from a cluster
bootstrap over prompt sets. The Holm–Bonferroni correction is applied across all metrics.

**Limits.** These are *indicators* of differential treatment, not proof of harm. The classifiers carry
their own biases. Lexicons are coarse (stems over-match, and context and negation are ignored). Binary
gender and name-based race proxies are simplifications, and results depend on the chosen templates.
Read the flagged examples before drawing conclusions.
""")
    with st.expander("Run configuration"):
        st.json(cfg)
