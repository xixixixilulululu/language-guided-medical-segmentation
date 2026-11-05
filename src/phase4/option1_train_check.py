"""Phase 4, orientation option 1: one-sided TRAINING reports versus the image half of the ground-truth lesion.

Rule: configs/phase4/option1_train_rule.json (committed before this program was written; read, never changed).
Only the QaTa-COV19 training split is read (masks and the Phase 2 location parse); validation and test are used
only as id lists to check for overlap. Output: counts only, no text and no per-image rows.

Self-checks before anything is read (exit status 7 on failure):
  - the rule file is tracked, has exactly one commit, and its current sha256 equals the committed content;
  - the rule commit is an ancestor of HEAD; the working tree is clean (git status --porcelain is empty);
  - the output file does not exist (never overwritten; written to a temporary file, then renamed).
Usage:  python src/phase4/option1_train_check.py   -> results/phase4_option1_train_orientation.csv
Tests:  python src/phase4/test_option1_train_check.py
"""
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
RULE = os.path.join("configs", "phase4", "option1_train_rule.json")
OUT = os.path.join("results", "phase4_option1_train_orientation.csv")
ATTR = os.path.join("results", "phase2_text_attributes.csv")
DATASET, SPLIT = "QaTa-COV19", "train"
COL_SPLIT = 112
Z = 1.959963984540054
PREFLIGHT_EXIT = 7
EXCL = ["lateral_view", "mask_empty", "parse_not_ok", "not_one_sided"]


def stop(msg, code=PREFLIGHT_EXIT):
    print(f"OPTION 1 CHECK STOPPED: {msg}", file=sys.stderr)
    sys.exit(code)


# ------------------------------------------------------------------------------------------------ self-checks
def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def preflight(repo=REPO, rule=RULE, out=OUT):
    """(rule commit, rule sha256) or stop."""
    if not os.path.exists(os.path.join(repo, rule)):
        stop(f"{rule} does not exist")
    commits = git(repo, "log", "--format=%H", "--", rule).stdout.split()
    if not commits:
        stop(f"{rule} is not committed")
    if len(commits) != 1:
        stop(f"{rule} was changed after its first commit ({len(commits)} commits)")
    committed = subprocess.run(["git", "-C", repo, "show", f"{commits[0]}:{rule}"], capture_output=True).stdout
    current = open(os.path.join(repo, rule), "rb").read()
    sha = hashlib.sha256(current).hexdigest()
    if hashlib.sha256(committed).hexdigest() != sha:
        stop(f"{rule} differs from its committed content")
    if git(repo, "merge-base", "--is-ancestor", commits[0], "HEAD").returncode != 0:
        stop("the rule commit is not an ancestor of HEAD")
    if git(repo, "status", "--porcelain").stdout.strip():
        stop("the working tree has uncommitted changes")
    if os.path.exists(os.path.join(repo, out)):
        stop(f"{out} exists; it is never overwritten")
    return commits[0], sha


# ------------------------------------------------------------------------------------------------ definitions
def lesion_fraction(mask):
    """Share of lesion pixels in the image-right half (columns 112-223); None for an empty mask."""
    m = np.asarray(mask) > 0
    n = int(m.sum())
    return None if n == 0 else float(m[:, COL_SPLIT:].sum()) / n


def side_primary(f):
    return "image_right" if f > 0.5 else "image_left" if f < 0.5 else None


def side_robust(f):
    return "image_right" if f >= 2 / 3 else "image_left" if f <= 1 / 3 else None


def one_sided(loc):
    """'left' / 'right' if the location names only that side (any zones, 'unspecified' included), else None."""
    sides = {p.split(":", 1)[0] for p in str(loc or "").split(";") if ":" in p}
    return sides.pop() if len(sides) == 1 and sides <= {"left", "right"} else None


def agrees(report_side, image_side, convention):
    """Convention A: patient left on image right; B: patient left on image left."""
    left_on = "image_right" if convention == "A" else "image_left"
    right_on = "image_left" if convention == "A" else "image_right"
    return image_side == (left_on if report_side == "left" else right_on)


def wilson_lower(k, n, z=Z):
    if n == 0:
        return float("nan")
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def group_of(sid):
    return "sub" if sid.startswith("sub-") else "covid" if sid.startswith("covid_") else "other"


def view_of(sid):
    return "vp-ap" if "_vp-ap_" in sid else "vp-pa" if "_vp-pa_" in sid else "vp-ll" if "_vp-ll_" in sid else "no view"


# ------------------------------------------------------------------------------------------------ evaluation
def records(items):
    """items: (sample_id, parse_ok, location, mask fraction or None). -> DataFrame of report records."""
    rows = []
    for sid, ok, loc, f in items:
        reason = ("lateral_view" if "_vp-ll_" in sid else "mask_empty" if f is None else
                  "parse_not_ok" if not ok else "not_one_sided" if one_sided(loc) is None else "")
        rows.append({"sample_id": sid, "group": group_of(sid), "view": view_of(sid), "excluded": reason,
                     "report_side": one_sided(loc) if not reason else None, "f": f if not reason else None})
    return pd.DataFrame(rows)


def stats(R, sidefn):
    """Agreement statistics of one set of records under one lesion-side definition."""
    use = R[R.excluded == ""].copy()
    use["image_side"] = [sidefn(f) for f in use.f]
    sided = use[use.image_side.notna()]
    out = {"n_reports": len(R), **{f"n_excl_{k}": int((R.excluded == k).sum()) for k in EXCL},
           "n_excl_lesion_side_undetermined": int(use.image_side.isna().sum()), "n_one_sided_with_side": len(sided)}
    for c in ("A", "B"):
        k = int(sum(agrees(r, s, c) for r, s in zip(sided.report_side, sided.image_side)))
        out[f"agreement_{c}"] = k / len(sided) if len(sided) else float("nan")
        out[f"wilson_lower_{c}"] = wilson_lower(k, len(sided))
    for rs in ("left", "right"):
        f = use[use.report_side == rs].f.astype(float)
        for q in (0.1, 0.5, 0.9):
            out[f"f_right_half_q{int(q * 100)}_{rs}_reports"] = float(f.quantile(q)) if len(f) else float("nan")
    return out


def patient_votes(R, convention, patient_of):
    """sub-* only, primary definition: (patients agreeing, patients voting, patients tied)."""
    use = R[(R.group == "sub") & (R.excluded == "")].copy()
    use["image_side"] = [side_primary(f) for f in use.f]
    use = use[use.image_side.notna()]
    use["agree"] = [agrees(r, s, convention) for r, s in zip(use.report_side, use.image_side)]
    use["patient"] = use.sample_id.map(patient_of)
    agree = tie = vote = 0
    for _, g in use.groupby("patient"):
        a, d = int(g.agree.sum()), int((~g.agree).sum())
        if a == d:
            tie += 1
        else:
            vote += 1
            agree += a > d
    return agree, vote, tie


def rules_1_to_5(S):
    """S: {group label: stats}. -> (dict rule -> bool, winner or None)."""
    main = ("all", "sub", "covid")
    winner = None
    for c, o in (("A", "B"), ("B", "A")):
        if all(S[g][f"agreement_{c}"] > S[g][f"agreement_{o}"] for g in main):
            winner = c
    r = {"rule_1": winner is not None}
    r["rule_2"] = bool(winner) and all(S[g][f"agreement_{winner}"] >= 0.90 for g in main)
    r["rule_3"] = bool(winner) and all(S[g][f"wilson_lower_{winner}"] >= 0.85 for g in main)
    r["rule_4"] = bool(winner) and all(S[g][f"agreement_{winner}"] >= 0.85 for g in ("all_left_only", "all_right_only"))
    r["rule_5"] = S["sub"]["n_one_sided_with_side"] >= 30 and S["covid"]["n_one_sided_with_side"] >= 30
    return r, winner


def group_stats(R, sidefn):
    S = {"all": stats(R, sidefn)}
    for g in ("sub", "covid", "other"):
        S[g] = stats(R[R.group == g], sidefn)
    for side in ("left", "right"):
        S[f"all_{side}_only"] = stats(R[R.report_side == side], sidefn)
    for v in ("vp-ap", "vp-pa", "no view"):
        S[f"sub_view_{v}"] = stats(R[(R.group == "sub") & (R.view == v)], sidefn)
    return S


def evaluate(R, patient_of):
    """All numbers and the rule outcome. -> (rows for the CSV, summary dict)."""
    P, Q = group_stats(R, side_primary), group_stats(R, side_robust)
    rp, win = rules_1_to_5(P)
    rq, win_r = rules_1_to_5(Q)
    rules = dict(rp)
    rules["rule_6"] = bool(win) and win_r == win and all(rp[k] == rq[k] for k in rp)
    agree, vote, tie = patient_votes(R, win, patient_of) if win else (0, 0, 0)
    rules["rule_7"] = bool(win) and vote > 0 and agree / vote >= 0.90
    result = "insufficient sample" if not rules["rule_5"] else "pass" if all(rules.values()) else "fail"
    summary = {**{k: bool(v) for k, v in rules.items()}, "winning_convention": win or "",
               "winning_convention_robust": win_r or "", "robust_rules_1_to_5": "/".join(str(bool(rq[k])) for k in rq),
               "patients_voting": vote, "patients_agreeing": agree, "patients_tied_excluded": tie,
               "patient_agreement": agree / vote if vote else float("nan"), "result": result}
    rows = [{"definition": d, "group": g, **s} for d, S in (("primary", P), ("robust", Q)) for g, s in S.items()]
    return rows, summary


# ------------------------------------------------------------------------------------------------ main
def main():
    commit, sha = preflight()
    from lgms import data
    from phase2 import texts as T
    from phase4 import guard
    guard.require_dev_split(SPLIT)
    train, val, test = (set(data.split_ids(DATASET, s)) for s in ("train", "val", "test"))
    if train & val or train & test:
        stop(f"training ids overlap validation ({len(train & val)}) or test ({len(train & test)}) ids")
    A = pd.read_csv(os.path.join(REPO, ATTR), dtype=str, keep_default_na=False)
    A = A[(A.dataset == DATASET) & (A.split == SPLIT)]
    if set(A.sample_id) != train or A.sample_id.duplicated().any():
        stop("the attribute rows of the training split do not match the training ids one to one")
    bad = (A.parse_ok != "True").mean()
    if bad > 0.05:
        stop(f"parse_ok is false for {bad:.1%} of the training rows (> 5%)")
    items = []
    for sid, ok, loc in zip(A.sample_id, A.parse_ok, A.location):
        if sid not in train or sid in val or sid in test:
            stop(f"id outside the training split: {sid}")
        path = data.mask_path(DATASET, SPLIT, sid)
        guard.require_dev_path(path)
        items.append((sid, ok == "True", loc, lesion_fraction(data.read_mask(DATASET, SPLIT, sid))))
    R = records(items)
    rows, summary = evaluate(R, lambda s: T.groups(DATASET, s)[0])
    meta = {"run_time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "rule_commit": commit, "rule_sha256": sha, "parse_not_ok_share_all_rows": bad}
    out = pd.DataFrame([{"row": "group", **r} for r in rows] + [{"row": "summary", **summary}])
    for k, v in meta.items():
        out[k] = v
    path = os.path.join(REPO, OUT)
    if os.path.exists(path):
        stop(f"{OUT} exists; it is never overwritten")
    tmp = path + ".tmp"
    out.to_csv(tmp, index=False, float_format="%.6g")
    os.replace(tmp, path)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
