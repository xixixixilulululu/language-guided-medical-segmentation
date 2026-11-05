"""Phase 4, Stage 1 (PHASE4_PLAN.md): does the spatial mismatch score separate C0 from location-only C4?

Data: QaTa-COV19 validation, GATE_FIT only, Stage 0-eligible targets; DD-CMD seed 1001 report assignments of the
frozen manifest (C0, location-only C4 donors, C6 donors); the nnU-Net seed 1001 IMG mask of the TARGET image.
No LANG mask, Dice or F1-F5 is read. A GATE_CAL id stops the program before any mask is read.
Scores (src/phase4/spatial.py spatial_score, orientation from configs/phase4/stage0_spec.json):
  C0  the target's own report location on the target's IMG mask
  C4  each mappable location-only donor's location on the target's IMG mask, averaged over the donors
  C6  the same with the C6 donors (descriptive)
  rows whose score is unavailable are excluded and counted; a target needs a C0, >= 1 C4 and >= 1 C6 score.
Statistics: images and groups (Phase 3 group_id); paired mean(C4 - C0) and mean(C6 - C0) of spatial_mismatch;
image-level AUROC = P(positive > negative) + 0.5 P(equal) with one C0 score (negative) and one donor-averaged C4
score (positive) per target; the same for C6 (descriptive); per-group mean differences (descriptive).
Decision (spec "stage1"): AUROC(C0 vs C4) >= 0.65 and mean(C4 - C0) > 0 and mean(C6 - C0) > 0.
A pre-declared sensitivity orientation in the spec ("stage1_sensitivity_orientation") gives a second, labelled
output "sensitivity, not used for the decision".
After scoring, the eligible GATE_FIT images and groups must equal results/phase4_stage0_coverage.csv (831 / 693),
else exit 8 with no output written and no score printed.
Refuses to run (exit 8) unless: the spec orientation is set (not pending) and the spec has
"orientation_decision" (decided_by, date, commit); results/phase4_stage0_decision.json exists with decision GO;
the working tree is clean and HEAD is on origin; the output files do not exist.
Usage:  python src/phase4/stage1.py        Tests: python src/phase4/test_stage1.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase4 import guard, spatial as S  # noqa: E402

SPEC = os.path.join("configs", "phase4", "stage0_spec.json")
STAGE0 = os.path.join("results", "phase4_stage0_decision.json")
OUT_SUMMARY = os.path.join("results", "phase4_stage1_summary.csv")
OUT_SCORES = os.path.join("results", "phase4_spatial_scores_val.csv")
DATASET, SPLIT, SEED = "QaTa-COV19", "val", 1001
IMG_MASKS = os.path.join("runs", "phase3", SPLIT, "img", "nnunet", DATASET, f"seed{SEED}", "masks.npz")
PREFLIGHT_EXIT = 8
AUROC_MIN = 0.65


def stop(msg, code=PREFLIGHT_EXIT):
    print(f"STAGE 1 STOPPED: {msg}", file=sys.stderr)
    sys.exit(code)


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)


def preflight(repo=REPO):
    """The spec (with the orientation decision) or stop."""
    spec = json.load(open(os.path.join(repo, SPEC)))
    o = spec.get("orientation", {}).get("value", "pending")
    if o not in S.ORIENTATIONS:
        stop(f"orientation is not set ({o!r})")
    d = spec.get("orientation_decision")
    if not isinstance(d, dict) or not all(d.get(k) for k in ("decided_by", "date", "commit")):
        stop("spec has no complete orientation_decision (decided_by, date, commit)")
    p = os.path.join(repo, STAGE0)
    if not os.path.exists(p) or json.load(open(p)).get("decision") != "GO":
        stop(f"{STAGE0} missing or not GO")
    if git(repo, "status", "--porcelain").stdout.strip():
        stop("the working tree has uncommitted changes")
    if git(repo, "merge-base", "--is-ancestor", "HEAD", "origin/main").returncode != 0:
        stop("HEAD is not on origin/main (push first)")
    for out in (OUT_SUMMARY, OUT_SCORES):
        if os.path.exists(os.path.join(repo, out)):
            stop(f"{out} exists; it is never overwritten")
    return spec


def require_fit(ids, partition):
    """Stop if any id is not GATE_FIT (called before any mask is read)."""
    bad = [i for i in ids if partition.get(i) != "GATE_FIT"]
    if bad:
        stop(f"{len(bad)} non-GATE_FIT id(s), e.g. {bad[0]}; Stage 1 uses GATE_FIT only")


COVERAGE = os.path.join("results", "phase4_stage0_coverage.csv")


def check_eligibility(T, group, repo=REPO):
    """Stop (no output, no score printed) unless the eligible GATE_FIT images and groups equal the Stage 0 coverage."""
    c = pd.read_csv(os.path.join(repo, COVERAGE))
    c = c[(c.table == "coverage") & (c.part == "GATE_FIT")].set_index("item").numerator
    want = (int(c["images"]), int(c["groups (>= 1 eligible image)"]))
    E = T[T.eligible]
    got = (len(E), E.sample_id.map(group).nunique())
    if got != want:
        stop(f"eligible GATE_FIT images / groups {got} differ from the Stage 0 coverage {want}")
    return got


def auroc(neg, pos):
    """P(pos > neg) + 0.5 P(pos == neg) over all (negative, positive) pairs."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def target_scores(targets, img_mask, loc, c4_donors, c6_donors, orientation):
    """One row per target with C0, donor-averaged C4 / C6 mismatch and counts of unavailable rows."""
    rows = []
    for t in targets:
        m = img_mask(t)
        c0 = S.spatial_score(t, m, loc.get(t), orientation)
        s4 = [S.spatial_score(t, m, loc.get(d), orientation) for d in c4_donors.get(t, [])]
        s6 = [S.spatial_score(t, m, loc.get(d), orientation) for d in c6_donors.get(t, [])]
        a4 = [s["mismatch"] for s in s4 if s["reason"] is None]
        a6 = [s["mismatch"] for s in s6 if s["reason"] is None]
        rows.append({"sample_id": t, "c0": c0["mismatch"], "c0_reason": c0["reason"] or "",
                     "c4": float(np.mean(a4)) if a4 else np.nan, "n_c4": len(a4), "n_c4_unavailable": len(s4) - len(a4),
                     "c6": float(np.mean(a6)) if a6 else np.nan, "n_c6": len(a6), "n_c6_unavailable": len(s6) - len(a6)})
    T = pd.DataFrame(rows)
    T["eligible"] = T.c0.notna() & (T.n_c4 > 0) & (T.n_c6 > 0)
    return T


def summarise(T, group, label, criterion=True):
    E = T[T.eligible]
    d4, d6 = E.c4 - E.c0, E.c6 - E.c0
    g = pd.DataFrame({"g": E.sample_id.map(group), "d4": d4, "d6": d6}).groupby("g").mean()
    a4, a6 = auroc(E.c0, E.c4), auroc(E.c0, E.c6)
    out = {"label": label, "n_targets": len(T), "n_eligible_images": len(E), "n_groups": g.shape[0],
           "n_excluded_images": int((~T.eligible).sum()), "n_c4_rows_unavailable": int(T.n_c4_unavailable.sum()),
           "n_c6_rows_unavailable": int(T.n_c6_unavailable.sum()), "mean_c4_minus_c0": float(d4.mean()),
           "mean_c6_minus_c0": float(d6.mean()), "auroc_c0_vs_c4": a4, "auroc_c0_vs_c6_descriptive": a6,
           "groups_mean_c4_minus_c0_positive": int((g.d4 > 0).sum()), "groups_mean_c6_minus_c0_positive": int((g.d6 > 0).sum()),
           "group_weighted_mean_c4_minus_c0": float(g.d4.mean()), "group_weighted_mean_c6_minus_c0": float(g.d6.mean())}
    if criterion:
        out["decision"] = "continue" if (a4 >= AUROC_MIN and d4.mean() > 0 and d6.mean() > 0) else "STOP"
    else:
        out["decision"] = "sensitivity, not used for the decision"
    return out


def main():
    spec = preflight()
    guard.require_dev_split(SPLIT)
    guard.require_dev_path(IMG_MASKS)
    from phase2 import conditions as C
    from phase4.stage0_coverage import check_manifest
    man, _ = C.build(DATASET, SPLIT)
    man = man[man.eligible]
    check_manifest(man)
    part = pd.read_csv(os.path.join(REPO, "results", "phase3_gate_partitions.csv"), dtype=str).set_index("sample_id")
    c4 = man[(man.condition == "C4") & (man.changed_attributes == "location")]
    c6 = man[man.condition == "C6"]
    den = sorted(set(man[man.condition == "C0"].sample_id) & set(c4.sample_id) & set(c6.sample_id))
    fit = [t for t in den if part.partition.get(t) == "GATE_FIT"]
    require_fit(fit, part.partition.to_dict())                          # before any mask is read
    guard.require_dev_ids(set(fit) | set(c4.donor_sample_id) | set(c6.donor_sample_id))
    attr = pd.read_csv(os.path.join(REPO, "results", "phase2_text_attributes.csv"), dtype=str, keep_default_na=False)
    attr = attr[(attr.dataset == DATASET) & (attr.split == SPLIT)]
    loc = dict(zip(attr.sample_id, attr.location))
    z = np.load(os.path.join(REPO, IMG_MASKS))
    img = lambda t: np.unpackbits(z[t])[:224 * 224].reshape(224, 224).astype(bool)  # noqa: E731
    d4 = c4.groupby("sample_id").donor_sample_id.apply(list).to_dict()
    d6 = c6.groupby("sample_id").donor_sample_id.apply(list).to_dict()
    group = part.group_id.to_dict()
    o = spec["orientation"]["value"]
    T = target_scores(fit, img, loc, d4, d6, o)
    check_eligibility(T, group)                                         # before any output or printed score
    rows = [summarise(T, group, f"primary ({o})")]
    sens = spec.get("stage1_sensitivity_orientation")
    if sens:
        rows.append(summarise(target_scores(fit, img, loc, d4, d6, sens), group, f"sensitivity ({sens})", criterion=False))
    for out, df in ((OUT_SCORES, T.drop(columns=["c0_reason"]).assign(partition="GATE_FIT", orientation=o)),
                    (OUT_SUMMARY, pd.DataFrame(rows))):
        p = os.path.join(REPO, out)
        df.to_csv(p + ".tmp", index=False, float_format="%.6g")
        os.replace(p + ".tmp", p)
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
