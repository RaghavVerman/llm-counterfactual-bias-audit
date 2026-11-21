"""End-to-end audit: build prompts -> generate -> score -> compare groups."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
from tqdm import tqdm

from .generation import Generator
from .prompts import AXES, build_skeletons
from .scoring import score_records
from .stats import compare_groups

MASK = "XYZ"


@dataclass
class AuditConfig:
    model: str
    axis: str = "gender"
    families: list[str] | None = None
    max_skeletons: int = 24
    samples_per_prompt: int = 5
    max_new_tokens: int = 40
    temperature: float = 0.8
    top_p: float = 0.95
    scorers: list[str] = field(default_factory=lambda: ["sentiment", "lexicon", "regard", "toxicity"])
    seed: int = 0
    n_perm: int = 10_000


def mask_term(text: str, term: str) -> str:
    return re.sub(r"\b" + re.escape(term) + r"\b", MASK, text, flags=re.IGNORECASE)


def generate_records(cfg: AuditConfig, generator: Generator | None = None,
                     progress: Callable[[int, int], None] | None = None) -> pd.DataFrame:
    """Generate every counterfactual sample. ``progress(done, total)`` is called per skeleton."""
    generator = generator or Generator(cfg.model)
    skeletons = build_skeletons(cfg.axis, cfg.families, max_skeletons=cfg.max_skeletons, seed=cfg.seed)
    groups = list(AXES[cfg.axis]["groups"])
    records = []
    for i, sk in enumerate(tqdm(skeletons, desc=f"generating ({cfg.axis})")):
        # One batch holds every group's prompt for this skeleton, n times each,
        # sampled under the same seed so groups see identical sampling settings.
        prompts = [sk.prompts[g] for g in groups for _ in range(cfg.samples_per_prompt)]
        outs = generator.generate(
            prompts, seed=cfg.seed * 100_003 + i, max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, top_p=cfg.top_p,
        )
        for j, (prompt, cont) in enumerate(zip(prompts, outs)):
            g = groups[j // cfg.samples_per_prompt]
            term = sk.terms[g]
            records.append({
                "model": cfg.model, "axis": cfg.axis, "group": g, "term": term,
                "skeleton_id": sk.skeleton_id, "family": sk.family, "template": sk.template,
                "occupation": sk.occupation, "sample": j % cfg.samples_per_prompt,
                "prompt": prompt, "continuation": cont,
                "masked_text": mask_term(f"{prompt} {cont}", term),
            })
        if progress:
            progress(i + 1, len(skeletons))
    return pd.DataFrame(records)


def metric_columns(df: pd.DataFrame) -> list[str]:
    known = ["regard", "regard_negative", "sentiment", "toxicity"]
    lex = [c for c in df.columns if c.startswith("lex_")]
    return [c for c in known if c in df.columns] + lex


def run_audit(
    cfg: AuditConfig, out_dir: str | Path | None = None,
    generator: Generator | None = None, progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = time.time()
    gen_df = generate_records(cfg, generator, progress)
    scored = pd.DataFrame(score_records(gen_df.to_dict("records"), cfg.scorers))
    # Empty continuations carry no signal; drop them rather than score them as neutral.
    scored = scored[scored["continuation"].str.strip().astype(bool)]
    summary = compare_groups(scored, metric_columns(scored), n_perm=cfg.n_perm, seed=cfg.seed)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out / "generations.csv", index=False)
        summary.to_csv(out / "summary.csv", index=False)
        meta = {**asdict(cfg), "n_generations": len(scored), "runtime_sec": round(time.time() - start, 1)}
        (out / "config.json").write_text(json.dumps(meta, indent=2))
    return scored, summary


def format_summary(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No comparisons (too few skeletons)."
    cols = ["metric", "group_a", "group_b", "mean_a", "mean_b", "diff", "ci_low", "ci_high",
            "cohens_d", "frac_skeletons_b_higher", "p_holm", "significant"]
    return summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}")
