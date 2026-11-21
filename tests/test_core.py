import numpy as np
import pandas as pd

from bias_audit.audit import mask_term
from bias_audit.generation import clean_continuation
from bias_audit.lexicons import lexicon_matches, lexicon_scores
from bias_audit.prompts import AXES, build_skeletons
from bias_audit.stats import compare_groups, holm, stratified_permutation_test


def test_skeletons_are_counterfactual():
    sks = build_skeletons("gender", max_skeletons=None)
    for sk in sks:
        male, female = sk.prompts["male"], sk.prompts["female"]
        assert male.replace(sk.terms["male"], "#") == female.replace(sk.terms["female"], "#")


def test_axis_without_slot_skips_templates():
    sks = build_skeletons("race_names", max_skeletons=None)
    assert sks and all(sk.slot == "name" for sk in sks)


def test_max_skeletons_keeps_every_family():
    sks = build_skeletons("gender", max_skeletons=8, seed=1)
    assert len(sks) == 8
    assert {sk.family for sk in sks} == {"regard", "occupation", "workplace", "reference"}


def test_articles():
    sk = next(s for s in build_skeletons("gender", families=["reference"], max_skeletons=None)
              if s.occupation == "engineer" or s.occupation == "software engineer")
    assert "a software engineer" in sk.prompts["male"]
    sk = next(s for s in build_skeletons("gender", families=["reference"], max_skeletons=None)
              if s.occupation == "accountant")
    assert "an accountant" in sk.prompts["male"]


def test_lexicon_stems_and_boundaries():
    assert lexicon_matches("She was a natural leader and very competitive.", "agentic") == ["leader", "competitive"]
    assert lexicon_matches("He is caring and supportive.", "communal") == ["caring", "supportive"]
    assert lexicon_matches("unkind remarks", "communal") == []  # whole-word only
    s = lexicon_scores("leader leader kind kind")
    assert s["lex_agentic"] == 50 and s["lex_communal"] == 50 and s["lex_agentic_minus_communal"] == 0


def test_mask_term():
    assert mask_term("The man said the manager", "man") == "The XYZ said the manager"


def test_clean_continuation():
    assert clean_continuation(" One. Two! Three.\n\nNext para") == "One. Two!"


def test_holm():
    np.testing.assert_allclose(holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])


def _synthetic(effect, n_strata=20, n=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_strata):
        base = rng.normal(0, 2)  # large between-skeleton variance
        for g, shift in (("a", 0.0), ("b", effect)):
            for _ in range(n):
                rows.append({"skeleton_id": f"s{s}", "group": g, "m": base + shift + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_detects_real_effect_and_not_null():
    res = compare_groups(_synthetic(0.5), ["m"], n_perm=2000, n_boot=1000).iloc[0]
    assert res["significant"] and res["ci_low"] > 0 and abs(res["diff"] - 0.5) < 0.15
    res = compare_groups(_synthetic(0.0, seed=3), ["m"], n_perm=2000, n_boot=1000).iloc[0]
    assert not res["significant"] and res["ci_low"] < 0 < res["ci_high"]


def test_permutation_null_is_roughly_uniform():
    ps = []
    for seed in range(40):
        df = _synthetic(0.0, n_strata=8, n=3, seed=seed)
        ps.append(stratified_permutation_test(df["m"], df["group"] == "b", df["skeleton_id"], n_perm=500, seed=seed))
    assert np.mean(np.array(ps) < 0.05) <= 0.15  # false-positive rate near the nominal 5%


def test_axes_lists_aligned():
    for axis in AXES.values():
        groups = list(axis["groups"].values())
        for slot in ("noun", "adj", "name"):
            lens = {len(g[slot]) for g in groups if slot in g}
            assert len(lens) <= 1, (axis["description"], slot)
