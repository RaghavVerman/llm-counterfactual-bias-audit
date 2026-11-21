"""Command-line entry point.

Example::

    python -m bias_audit --model ./model --axis gender --skeletons 24 --samples 5
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from .audit import AuditConfig, format_summary, run_audit
from .prompts import AXES, TEMPLATES
from .scoring import SCORERS


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bias_audit", description="Counterfactual bias audit for causal LMs.")
    p.add_argument("--model", required=True, help="HF model id or local path, e.g. gpt2 or ./model")
    p.add_argument("--axis", default="gender", choices=sorted(AXES))
    p.add_argument("--families", nargs="+", choices=sorted(TEMPLATES), help="template families (default: all)")
    p.add_argument("--skeletons", type=int, default=24, help="max counterfactual prompt sets")
    p.add_argument("--samples", type=int, default=5, help="generations per prompt per group")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--scorers", nargs="+", default=["sentiment", "lexicon", "regard", "toxicity"],
                   choices=sorted(SCORERS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-perm", type=int, default=10_000, help="permutations per test")
    p.add_argument("--out", help="output directory (default: results/<model>_<axis>_<timestamp>)")
    a = p.parse_args(argv)

    cfg = AuditConfig(
        model=a.model, axis=a.axis, families=a.families, max_skeletons=a.skeletons,
        samples_per_prompt=a.samples, max_new_tokens=a.max_new_tokens,
        temperature=a.temperature, top_p=a.top_p, scorers=a.scorers, seed=a.seed, n_perm=a.n_perm,
    )
    slug = re.sub(r"[^\w.-]+", "_", a.model.strip("./\\")) or "model"
    out = Path(a.out or f"results/{slug}_{a.axis}_{datetime.now():%Y%m%d-%H%M%S}")
    scored, summary = run_audit(cfg, out)

    print(f"\n{len(scored)} generations scored. Results in {out}\n")
    print(format_summary(summary))
    print("\ndiff = mean(group_b) - mean(group_a); p_holm is Holm-corrected across all tests.")


if __name__ == "__main__":
    main()
