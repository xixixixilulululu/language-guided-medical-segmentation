"""Phase 4, Stage 0: coverage of the spatial score (independent of the orientation; no score is computed).

Definitions: configs/phase4/stage0_spec.json (committed before this program is run). QaTa-COV19 validation,
DD-CMD seed 1001 rows of the frozen Phase 2 manifest; IMG = nnU-Net seed 1001, Phase 3 rerun (job 371).
  denominator  targets eligible for C0, >= 1 location-only C4 donor (changed_attributes == "location") and
               >= 1 C6 donor (manifest only)
  eligible     not a lateral view (file name "_vp-ll_"), IMG mask nonempty, C0 location mappable, >= 1 mappable
               location-only C4 donor and >= 1 mappable C6 donor
  group        eligible if at least one of its denominator images is eligible
Read: the manifest and the Phase 2 location parse (no text is written out), the partition file, and the IMG
binary masks only to test whether they are empty. LANG masks, ground truth, scores, Dice and AUROC are not read
or computed.
Output: results/phase4_stage0_coverage.csv (table, part, item, numerator, denominator, ratio)
Usage:  python src/phase4/stage0_coverage.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase4 import guard, spatial as S  # noqa: E402

SPEC = os.path.join(REPO, "configs", "phase4", "stage0_spec.json")
DATASET, SPLIT, SEED = "QaTa-COV19", "val", 1001
IMG_MASKS = os.path.join(REPO, "runs", "phase3", SPLIT, "img", "nnunet", DATASET, f"seed{SEED}", "masks.npz")
OUT = os.path.join(REPO, "results", "phase4_stage0_coverage.csv")
REASONS = [S.LATERAL, S.IMG_EMPTY, "c0_" + S.LOC_EMPTY, "c0_" + S.LOC_UNSPECIFIED, "c0_" + S.LOC_UNKNOWN,
           "no_mappable_c4_location_only_donor", "no_mappable_c6_donor"]


def check_spec():
    """The spec must exist and agree with the constants of src/phase4/spatial.py."""
    spec = json.load(open(SPEC))
    g = spec["grid"]
    if (g["column_split"], tuple(g["row_edges"])) != (S.COL_SPLIT, S.ROW_EDGES):
        raise ValueError("configs/phase4/stage0_spec.json and src/phase4/spatial.py disagree on the grid")
    return spec


def target_table(man, loc, img_nonempty):
    """One row per denominator target: partition-free eligibility flags and counts."""
    c0 = set(man[man.condition == "C0"].sample_id)
    c4 = man[(man.condition == "C4") & (man.changed_attributes == "location")]
    c6 = man[man.condition == "C6"]
    den = sorted(c0 & set(c4.sample_id) & set(c6.sample_id))
    ok = lambda s: S.parse_location(loc.get(s))[1] is None  # noqa: E731
    rows = []
    for t in den:
        d4 = c4[c4.sample_id == t].donor_sample_id
        d6 = c6[c6.sample_id == t].donor_sample_id
        r = {"sample_id": t, "lateral": S.is_lateral(t), "img_empty": not img_nonempty[t],
             "c0_reason": S.parse_location(loc.get(t))[1], "n_c4_donors": len(d4), "n_c4_mappable": int(sum(map(ok, d4))),
             "n_c6_donors": len(d6), "n_c6_mappable": int(sum(map(ok, d6)))}
        flags = [r["lateral"], r["img_empty"], r["c0_reason"] == S.LOC_EMPTY, r["c0_reason"] == S.LOC_UNSPECIFIED,
                 r["c0_reason"] == S.LOC_UNKNOWN, r["n_c4_mappable"] == 0, r["n_c6_mappable"] == 0]
        r.update({k: bool(f) for k, f in zip(REASONS, flags)})
        r["first_reason"] = next((k for k, f in zip(REASONS, flags) if f), "")
        r["eligible"] = not any(flags)
        rows.append(r)
    return pd.DataFrame(rows)


def coverage(T):
    out = []
    add = lambda table, part, item, n, d: out.append({"table": table, "part": part, "item": item, "numerator": int(n),  # noqa: E731
                                                       "denominator": int(d), "ratio": n / d if d else np.nan})
    for part, g in [("all", T)] + list(T.groupby("partition")):
        add("coverage", part, "images", g.eligible.sum(), len(g))
        gg = g.groupby("group").eligible
        add("coverage", part, "groups (>= 1 eligible image)", gg.any().sum(), gg.ngroups)
        add("coverage_descriptive", part, "groups (all images eligible)", gg.all().sum(), gg.ngroups)
        for k in REASONS:                                                   # not exclusive
            add("unavailable_reason_any", part, k, g[k].sum(), len(g))
        for k in REASONS:                                                   # exclusive, fixed order
            add("unavailable_reason_first", part, k, (g.first_reason == k).sum(), len(g))
        for col, lab in (("n_c4_mappable", "mappable location-only C4 donors"), ("n_c6_mappable", "mappable C6 donors")):
            for v, n in g[col].value_counts().sort_index().items():
                add("donor_distribution", part, f"{lab} = {v}", n, len(g))
    return pd.DataFrame(out)


def main():
    guard.require_dev_split(SPLIT)
    guard.require_dev_path(IMG_MASKS)
    spec = check_spec()
    from phase2 import conditions as C
    from phase2 import texts as T
    man, _ = C.build(DATASET, SPLIT)
    man = man[man.eligible]
    guard.require_dev_ids(set(man.sample_id) | set(man.donor_sample_id.dropna()) - {""})
    loc = T.attributes(DATASET, SPLIT).set_index("sample_id").location.to_dict()
    z = np.load(IMG_MASKS)
    img_nonempty = {k: bool(np.unpackbits(z[k]).any()) for k in z.files}           # emptiness only
    tab = target_table(man, loc, img_nonempty)
    part = pd.read_csv(os.path.join(REPO, "results", "phase3_gate_partitions.csv"), dtype=str).set_index("sample_id")
    tab["partition"], tab["group"] = tab.sample_id.map(part.partition), tab.sample_id.map(part.group_id)
    if tab.partition.isna().any():
        raise ValueError("denominator target without a Phase 3 partition")
    cov = coverage(tab)
    cov.insert(0, "spec_version", spec["version"])
    cov.to_csv(OUT, index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(cov.drop(columns="spec_version").to_string(index=False))


if __name__ == "__main__":
    main()
