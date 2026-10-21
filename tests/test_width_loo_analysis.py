"""The leave-circuits-out width test's analysis (scripts/width_loo_analysis.py)."""

import importlib.util
import sys
from math import comb
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))                 # for its `import _bootstrap`
_spec = importlib.util.spec_from_file_location("width_loo_analysis",
                                               SCRIPTS / "width_loo_analysis.py")
wla = importlib.util.module_from_spec(_spec)
sys.modules["width_loo_analysis"] = wla
_spec.loader.exec_module(wla)


def test_folds_split_the_dev_circuits_and_judge_each_run_on_the_other_fold():
    f1, f2 = wla.FOLDS["f1"], wla.FOLDS["f2"]
    assert sorted(f1 + f2) == sorted(wla.R_DEV) and not set(f1) & set(f2)
    assert wla.HELD_OUT == {"f1": f2, "f2": f1}


def test_permutation_p_complete_separation_in_one_stratum():
    # 3 treated above 3 controls: only 1 of C(6,3) relabellings is as extreme.
    obs, p, n = wla.stratified_perm_p({"a": ([5.0, 6.0, 7.0], [1.0, 2.0, 3.0])})
    assert n == comb(6, 3) and obs == 4.0 and abs(p - 1 / 20) < 1e-12


def test_permutation_exchanges_labels_only_within_strata():
    # Stratum b sits far above a with no effect. Within strata, only the observed split of a
    # is as extreme, and every relabelling of b ties, so p = 1/C(4,2). Pooling the strata
    # would instead let b's large values move into a's groups.
    groups = {"a": ([3.0, 4.0], [1.0, 2.0]), "b": ([10.0, 10.0], [10.0, 10.0])}
    obs, p, n = wla.stratified_perm_p(groups)
    assert n == comb(4, 2) ** 2 and obs == 1.0
    assert abs(p - 1 / comb(4, 2)) < 1e-12


def test_ties_count_towards_p():
    obs, p, _ = wla.stratified_perm_p({"a": ([1.0, 1.0], [1.0, 1.0])})
    assert obs == 0.0 and p == 1.0
