"""Phase 4: vertical text-to-image check on the TRAINING split (upper-only / lower-only reports vs lesion rows).

Rule: configs/phase4/vertical_train_rule.json (committed before running; read, never changed). Training split
only (masks and the Phase 2 location parse); validation and test are used only as id lists for the overlap check.
Self-checks as in src/phase4/option1_train_check.py (exit 7): rule tracked with one commit, unchanged, an
ancestor of HEAD; clean working tree; output absent (temporary file, then rename).
Output: results/phase4_vertical_train_check.csv (counts only; no text, no per-image rows).
Usage:  python src/phase4/vertical_train_check.py      Tests: python src/phase4/test_vertical_train_check.py
"""
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase4 import option1_train_check as O  # noqa: E402

RULE = os.path.join("configs", "phase4", "vertical_train_rule.json")
OUT = os.path.join("results", "phase4_vertical_train_check.csv")
MID_ROW = 112
BANDS = {"upper_only": (0, 75), "lower_only": (149, 224)}
EXCL = ["lateral_view", "mask_empty", "parse_not_ok", "not_in_a_group"]


def group_of_location(loc):
    """'upper_only' / 'lower_only' if every named side has exactly that one zone, else None."""
    parts = [p for p in str(loc or "").split(";") if ":" in p]
    if not parts:
        return None
    zones = {p.split(":", 1)[1] for p in parts}
    sides = [p.split(":", 1)[0] for p in parts]
    if any(s not in ("left", "right") for s in sides) or len(zones) != 1:
        return None
    return {"upper": "upper_only", "lower": "lower_only"}.get(zones.pop())


def centroid_and_band(mask, group):
    """(mean row of lesion pixels, share of lesion pixels in the stated band) or (None, None) for an empty mask."""
    m = np.asarray(mask) > 0
    rows = np.nonzero(m)[0]
    if rows.size == 0:
        return None, None
    lo, hi = BANDS.get(group, (0, 0))
    return float(rows.mean()), float(((rows >= lo) & (rows < hi)).mean())


def records(items):
    """items: (sample_id, parse_ok, location, mask). -> DataFrame (one row per report)."""
    out = []
    for sid, ok, loc, mask in items:
        g = group_of_location(loc)
        c, b = centroid_and_band(mask, g)
        reason = ("lateral_view" if "_vp-ll_" in sid else "mask_empty" if c is None else
                  "parse_not_ok" if not ok else "not_in_a_group" if g is None else "")
        agree = None if reason else ((c < MID_ROW) if g == "upper_only" else (c >= MID_ROW))
        out.append({"excluded": reason, "group": None if reason else g, "centroid": None if reason else c,
                    "band_share": None if reason else b, "agree": agree})
    return pd.DataFrame(out)


def evaluate(R):
    rows = []
    use = R[R.excluded == ""]
    for g in ("upper_only", "lower_only", "combined"):
        h = use if g == "combined" else use[use.group == g]
        r = {"row": "group", "group": g, "n": len(h), "n_agree": int(h.agree.sum()) if len(h) else 0,
             "agreement": float(h.agree.mean()) if len(h) else float("nan")}
        if g != "combined":
            c = h.centroid.astype(float)
            r.update({f"centroid_row_q{q}": float(c.quantile(q / 100)) if len(h) else float("nan") for q in (10, 50, 90)})
            r["median_band_share"] = float(h.band_share.astype(float).median()) if len(h) else float("nan")
        rows.append(r)
    S = {r["group"]: r for r in rows}
    rule = {"rule_1": S["combined"]["agreement"] >= 0.90,
            "rule_2": S["upper_only"]["agreement"] >= 0.90 and S["lower_only"]["agreement"] >= 0.90,
            "rule_3": S["upper_only"]["n"] >= 30 and S["lower_only"]["n"] >= 30}
    rule = {k: bool(v) for k, v in rule.items()}
    result = "insufficient sample" if not rule["rule_3"] else "pass" if all(rule.values()) else "fail"
    summary = {"row": "summary", "n_reports": len(R), **{f"n_excl_{k}": int((R.excluded == k).sum()) for k in EXCL},
               **rule, "result": result}
    return rows, summary


def main():
    commit, sha = O.preflight(rule=RULE, out=OUT)
    from lgms import data
    from phase4 import guard
    guard.require_dev_split("train")
    train = data.split_ids(O.DATASET, "train")
    val, test = set(data.split_ids(O.DATASET, "val")), set(data.split_ids(O.DATASET, "test"))
    if set(train) & (val | test):
        O.stop("training ids overlap validation or test ids")
    A = pd.read_csv(os.path.join(REPO, O.ATTR), dtype=str, keep_default_na=False)
    A = A[(A.dataset == O.DATASET) & (A.split == "train")]
    if set(A.sample_id) != set(train) or A.sample_id.duplicated().any():
        O.stop("the attribute rows of the training split do not match the training ids one to one")
    bad = (A.parse_ok != "True").mean()
    if bad > 0.05:
        O.stop(f"parse_ok is false for {bad:.1%} of the training rows (> 5%)")
    items = []
    for sid, ok, loc in zip(A.sample_id, A.parse_ok, A.location):
        path = data.mask_path(O.DATASET, "train", sid)
        guard.require_dev_path(path)
        items.append((sid, ok == "True", loc, data.read_mask(O.DATASET, "train", sid)))
    rows, summary = evaluate(records(items))
    meta = {"run_time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "rule_commit": commit, "rule_sha256": sha}
    out = pd.DataFrame(rows + [summary])
    for k, v in meta.items():
        out[k] = v
    path = os.path.join(REPO, OUT)
    if os.path.exists(path):
        O.stop(f"{OUT} exists; it is never overwritten")
    out.to_csv(path + ".tmp", index=False, float_format="%.6g")
    os.replace(path + ".tmp", path)
    print(json.dumps(summary, indent=1))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
