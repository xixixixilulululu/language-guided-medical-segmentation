"""Phase 4, Stage 0 supplement: how much the orientation choice can matter, from the report TEXT only.

No mask, Dice, LANG or IMG output is read and no spatial score is computed (project lead's task, 2026-11-03).
Sources: results/phase2_text_attributes.csv (Phase 2 location parse), the frozen donor manifest
(results/phase2_donor_manifest_val.csv, checked against conditions.build()), the cell lookup of the committed
configs/phase4/stage0_spec.json, the Phase 3 partitions. Validation split only (src/phase4/guard.py).

Scope ("text-eligible"): Stage 0 denominator targets (C0, >= 1 location-only C4 donor, >= 1 C6 donor) that are
not lateral views and whose C0 location is mappable. The Stage 0 eligibility also needs a nonempty IMG mask,
which cannot be known from text; those (4) images are therefore still in this scope.
Pairs: every mappable location-only C4 donor and every mappable C6 donor of a target.
Each pair (T = target cells, D = donor cells, in PATIENT coordinates left/right x upper/middle/lower) goes to the
first true class:
  a  D ⊇ T               the donor covers all target cells (hard to detect under any orientation rule)
  b  mirror(D) ⊇ T       covered only after swapping left and right (detectable by a fixed orientation, not by
                         "max of both")
  c  otherwise           detectable in principle by both rules
Output: results/phase4_stage0_mirror_counts.csv (counts only, no text)
Usage:  python src/phase4/stage0_mirror.py
Tests:  python src/phase4/test_stage0_mirror.py
"""
import json
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase4 import guard, spatial as S  # noqa: E402
from phase4.stage0_coverage import DATASET, SPEC, check_manifest  # noqa: E402

ATTR = os.path.join(REPO, "results", "phase2_text_attributes.csv")
OUT = os.path.join(REPO, "results", "phase4_stage0_mirror_counts.csv")
SCOPE = "text-eligible: Stage 0 denominator minus lateral views minus unmappable C0 (IMG emptiness not read)"


def mirror(cells):
    return frozenset(("right" if s == "left" else "left", z) for s, z in cells)


def classify(t, d):
    """'a', 'b' or 'c' for target cells t and donor cells d (sets of (patient side, zone))."""
    t, d = frozenset(t), frozenset(d)
    if d >= t:
        return "a"
    if mirror(d) >= t:
        return "b"
    return "c"


def cells_from_spec(spec):
    """{location string: frozenset of (side, zone)} for mappable entries of the committed lookup table."""
    out = {}
    for loc, v in spec["lookup"]["table_patient_side_zones"].items():
        if "unavailable" not in v:
            out[loc] = frozenset((s, z) for s, zones in v.items() for z in zones)
    return out


def pair_table(den, donors, loc, table):
    """One row per (target, donor) pair of the text-eligible targets. donors: target_id, condition, donor_id."""
    rows = []
    for t in den:
        if S.is_lateral(t) or loc.get(t) not in table:
            continue
        for r in donors[donors.target_id == t].itertuples():
            if loc.get(r.donor_id) in table:
                rows.append({"target_id": t, "condition": r.condition, "donor_id": r.donor_id,
                             "class": classify(table[loc[t]], table[loc[r.donor_id]])})
    return pd.DataFrame(rows)


def counts(P):
    out = []
    for part, g in [("all", P)] + list(P.groupby("partition")):
        for cond, h in g.groupby("condition"):
            n = h["class"].value_counts()
            out.append({"scope": SCOPE, "partition": {"GATE_FIT": "FIT", "GATE_CAL": "CAL"}.get(part, part),
                        "condition": cond, "n_targets": h.target_id.nunique(), "n_pairs": len(h),
                        "n_a": int(n.get("a", 0)), "n_b": int(n.get("b", 0)), "n_c": int(n.get("c", 0)),
                        "n_targets_with_any_b": h[h["class"] == "b"].target_id.nunique()})
    return pd.DataFrame(out)


def main():
    guard.require_dev_split("val")
    from phase2 import conditions as C
    man, _ = C.build(DATASET, "val")
    man = man[man.eligible]
    check_manifest(man)                                                    # generated == frozen donor manifest
    dm = pd.read_csv(os.path.join(REPO, "results", "phase2_donor_manifest_val.csv"), dtype=str, keep_default_na=False)
    dm = dm[dm.dataset == DATASET]
    guard.require_dev_ids(set(man.sample_id) | set(dm.donor_id))
    A = pd.read_csv(ATTR, dtype=str, keep_default_na=False)
    A = A[(A.dataset == DATASET) & (A.split == "val")]
    loc = dict(zip(A.sample_id, A.location))
    table = cells_from_spec(json.load(open(SPEC)))
    c4 = dm[(dm.condition == "C4") & (dm.changed_attributes == "location")]
    c6 = dm[dm.condition == "C6"]
    den = sorted(set(man[man.condition == "C0"].sample_id) & set(c4.target_id) & set(c6.target_id))
    donors = pd.concat([c4.assign(condition="C4_location"), c6])[["target_id", "condition", "donor_id"]]
    P = pair_table(den, donors, loc, table)
    part = pd.read_csv(os.path.join(REPO, "results", "phase3_gate_partitions.csv"), dtype=str).set_index("sample_id").partition
    P["partition"] = P.target_id.map(part)
    out = counts(P)
    out.to_csv(OUT, index=False)
    print(f"denominator {len(den)}, text-eligible targets {P.target_id.nunique()}")
    print(out.drop(columns="scope").to_string(index=False))


if __name__ == "__main__":
    main()
