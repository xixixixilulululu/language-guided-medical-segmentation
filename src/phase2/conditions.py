"""Phase 2, Task 2.2: the nine text conditions C0, C1, C2, C3a, C3b, C3c, C4, C5, C6 for one split.

Rules (reports/PHASE2_PLAN.md, Task 2.2; decisions of 2026-10-27 in reports/PHASE2_SETUP_PLAN.md, section 9):
  C0   the original paired text, unchanged.
  C1   the empty string "" for every method (D8); each method's normal pathway processes it (LViT-T and
       RecLMIS pad it with the official " EOF XXX" rule, DD-CMD gets [CLS] [SEP] + padding).
  C2   a fixed-template rewrite from the parsed values (D9), keeping the order scope -> count -> location and
       every value (laterality word, count, zones per side, "no infection"). Typos are not carried over.
       Unavailable when any attribute is unknown.
  C3a  delete the scope word only ("Bilateral pulmonary infection" -> "Pulmonary infection"); unavailable for
       "No" texts, whose deletion would invert the meaning (D10).
  C3b  delete the whole count clause (", two infected areas").
  C3c  delete the whole location clause (", lower left lung and lower right lung"); unavailable for "No"
       texts, which have none.
       C3 deletions edit the ORIGINAL text (nothing else changes); the residual cues are recorded.
  C4-6 a real report of another case of the same dataset and split whose canonical (scope, count, location)
       differs in exactly 1 / 2 / 3 attributes; all three known for target and donor. Donors from the
       target's patient / volume are excluded (QaTa `covid_*`: only the case itself, D1/D11; MosMed bjorke:
       the pooled bjorke unit, D2). Identical donor texts (after whitespace normalisation) count once (the
       representative donor case is the one with the smallest hash). Up to 3 distinct donor texts per target
       and condition, chosen by the smallest sha256 of "salt|dataset|split|target|condition|donor" (D12),
       before any model output exists; the same manifest is used for every method and seed.
Rows with eligible = False keep an exclusion_reason and an empty counterfactual_text.

Test split: only with the Phase 2 lock open (texts.read_texts checks it before the test file is read).

Usage:  python src/phase2/conditions.py --split val      -> results/phase2_text_conditions.csv
        python src/phase2/conditions.py --c2-review     -> 20 train + 20 val C2 pairs (fixed seed) to stdout
"""
import argparse
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from lgms import data  # noqa: E402
from phase2 import texts as T  # noqa: E402

SALT = "phase2-donor-v1"                     # D12, recorded in PHASE2_PROTOCOL.md before any run
N_DONORS = 3
CONDITIONS = ["C0", "C1", "C2", "C3a", "C3b", "C3c", "C4", "C5", "C6"]
DISTANCE = {"C4": 1, "C5": 2, "C6": 3}
ATTRS = ["scope", "count", "location"]
C2_REVIEW_SEED = 0
COLUMNS = ["dataset", "split", "sample_id", "condition", "variant_id", "original_text", "counterfactual_text",
           "donor_sample_id", "donor_group_id", "attribute_distance", "changed_attributes", "eligible",
           "exclusion_reason", "residual_cue", "text_hash"]


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ws(text):
    """Whitespace normalisation used for donor de-duplication."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------------------------
# C2: template rewrite
# ---------------------------------------------------------------------------------------------
def _side_phrase(side, zones):
    if zones == "unspecified":
        return f"the {side} lung"
    if zones == "all":
        return f"all of the {side} lung"
    return f"the {' and '.join(zones.split('+'))} part of the {side} lung"


def c2_text(scope, count, location):
    """Equivalent wording from the parsed values (all three must be known)."""
    head = f"{scope} lung infection"
    areas = f"{T.WORD[count]} area{'' if count == 1 else 's'} of infection"
    if location == "none":
        return f"{head}, {areas}."
    sides = dict(p.split(":") for p in location.split(";"))
    loc = " and ".join(_side_phrase(s, sides[s]) for s in T.SIDES if s in sides)
    return f"{head}, {areas}, in {loc}."


# ---------------------------------------------------------------------------------------------
# C3: explicit deletions on the original text
# ---------------------------------------------------------------------------------------------
SCOPE_DEL = re.compile(r"^\s*(Bilateral|Unilateral)\s+pulmonary", re.I)
COUNT_DEL = re.compile(r"\s*,\s*[A-Za-z]+?\s*infected\s+areas?(?=\s*(,|\.|$))")
LOC_DEL = re.compile(r"(infected\s+areas?)\s*,.*$", re.S)


def sides_named(location):
    return 0 if location in (None, "none") else location.count(":")


def c3(cond, raw, a):
    """(text or None, reason if unavailable, residual cue)."""
    if cond == "C3a":
        if a["scope"] is None:
            return None, "scope unknown", ""
        if a["scope"] == "No":
            return None, "No-infection text: deleting 'No' would invert the meaning (D10)", ""
        if len(SCOPE_DEL.findall(raw)) != 1:
            return None, "scope wording not deletable by the fixed rule", ""
        out = SCOPE_DEL.sub("Pulmonary", raw, count=1)
        n = sides_named(a["location"])
        cue = {2: "both sides named (implies bilateral)", 1: "one side named (implies unilateral)"}.get(n, "")
        return out, "", cue
    if cond == "C3b":
        if a["count"] is None:
            return None, "count unknown", ""
        if len(COUNT_DEL.findall(raw)) != 1:
            return None, "count clause not deletable by the fixed rule", ""
        out = COUNT_DEL.sub("", raw, count=1)
        cue = {"Bilateral": "Bilateral implies at least two areas", "No": "No infection implies zero areas"}.get(
            a["scope"], "")
        return out, "", cue
    if cond == "C3c":
        if a["location"] is None:
            return None, "location unknown", ""
        if a["location"] == "none":
            return None, "no location clause (No-infection text)", ""
        if len(LOC_DEL.findall(raw)) != 1:
            return None, "location clause not deletable by the fixed rule", ""
        out = LOC_DEL.sub(r"\1.", raw, count=1)
        cue = {"Bilateral": "Bilateral implies both lungs", "Unilateral": "Unilateral implies one lung"}.get(
            a["scope"], "")
        return out, "", cue
    raise ValueError(cond)


# ---------------------------------------------------------------------------------------------
# C4-C6: real donor reports
# ---------------------------------------------------------------------------------------------
def donor_table(att):
    """Arrays used by donor_candidates (one entry per case of the split, in `att` order)."""
    grp = [T.groups(d, i)[4] for i, d in zip(att.index, att.dataset)]
    return {"ids": np.array(att.index), "ok": att.parse_ok.to_numpy(bool), "group": np.array(grp, dtype=object),
            "attrs": [att[k].astype(str).to_numpy() for k in ATTRS], "text": att.raw_text.to_numpy()}


def donor_candidates(tab, pos):
    """{distance: [(donor_id, donor_group, donor text)]} for the target at position `pos` of the table."""
    dist = sum((a != a[pos]).astype(int) for a in tab["attrs"])
    keep = tab["ok"].copy()
    keep[pos] = False                                                   # never the target itself
    if tab["group"][pos] is not None:                                   # nor its patient / volume
        keep &= tab["group"] != tab["group"][pos]
    return {k: [(tab["ids"][j], tab["group"][j], tab["text"][j]) for j in np.flatnonzero(keep & (dist == k))]
            for k in (1, 2, 3)}


def choose_donors(dataset, split, target, cond, cands):
    """Up to N_DONORS distinct donor texts, chosen by the fixed hash (D12)."""
    best = {}                                                          # whitespace-normalised text -> (hash, donor)
    for did, dgroup, text in cands:
        h = sha(f"{SALT}|{dataset}|{split}|{target}|{cond}|{did}")
        k = ws(text)
        if k not in best or h < best[k][0]:
            best[k] = (h, did, dgroup, text)
    return [v[1:] for v in sorted(best.values())[:N_DONORS]], len(best)


def changed(a, b):
    return "+".join(k for k in ATTRS if a[k] != b[k])


def build(dataset, split, view=None):
    """All condition rows of one dataset and split (COLUMNS), plus the per-target donor counts."""
    att = T.attributes(dataset, split, view).set_index("sample_id")
    for k in ATTRS:                                                     # unknown attributes as None (not NaN)
        att[k] = att[k].astype(object).where(att[k].notna(), None)
    rows, counts = [], []

    def row(sid, cond, variant, text, eligible, reason="", donor="", dgroup="", dist="", chg="", cue=""):
        rows.append({"dataset": dataset, "split": split, "sample_id": sid, "condition": cond, "variant_id": variant,
                     "original_text": att.at[sid, "raw_text"], "counterfactual_text": text if eligible else "",
                     "donor_sample_id": donor, "donor_group_id": dgroup or "", "attribute_distance": dist,
                     "changed_attributes": chg, "eligible": eligible, "exclusion_reason": reason,
                     "residual_cue": cue, "text_hash": sha(text) if eligible else ""})

    tab = donor_table(att)
    for pos, (sid, a) in enumerate(att.iterrows()):
        raw = a.raw_text
        row(sid, "C0", "0", raw, True)
        row(sid, "C1", "0", "", True)
        if a.parse_ok:
            row(sid, "C2", "0", c2_text(a.scope, int(a["count"]), a.location), True)
        else:
            row(sid, "C2", "0", "", False, "not all attributes parsed: no faithful rewrite")
        for cond in ("C3a", "C3b", "C3c"):
            out, reason, cue = c3(cond, raw, a)
            row(sid, cond, "0", out or "", out is not None, reason, cue=cue)
        cands = donor_candidates(tab, pos) if a.parse_ok else {1: [], 2: [], 3: []}
        for cond, dist in DISTANCE.items():
            chosen, n_avail = choose_donors(dataset, split, sid, cond, cands[dist])
            counts.append({"dataset": dataset, "split": split, "sample_id": sid, "condition": cond,
                           "distinct_donor_texts": n_avail, "chosen": len(chosen)})
            if not a.parse_ok:
                row(sid, cond, "0", "", False, "target attributes not all known")
            elif not chosen:
                row(sid, cond, "0", "", False, f"no eligible donor at distance {dist}")
            for v, (did, dgroup, text) in enumerate(chosen, 1):
                row(sid, cond, f"d{v}", text, True, donor=did, dgroup=dgroup, dist=dist,
                    chg=changed(a, att.loc[did]))
    # the C1 hash is of the empty string itself
    df = pd.DataFrame(rows)[COLUMNS]
    df.loc[df.condition == "C1", "text_hash"] = sha("")
    return df, pd.DataFrame(counts)


def c2_review(n=20, seed=C2_REVIEW_SEED):
    """C2 review sample: n distinct parsed texts from train and n from val (both datasets pooled), drawn with
    numpy default_rng(seed) from the texts sorted by dataset and text; returns (split, dataset, original, C2)."""
    out = []
    for split in ("train", "val"):
        att = pd.concat([T.attributes(ds, split) for ds in data.DATASETS])
        att = att[att.parse_ok].drop_duplicates("raw_text").sort_values(["dataset", "raw_text"])
        pick = np.random.default_rng(seed).choice(len(att), size=min(n, len(att)), replace=False)
        for _, a in att.iloc[sorted(pick)].iterrows():
            out.append((split, a.dataset, a.raw_text, c2_text(a.scope, int(a["count"]), a.location)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", default=os.path.join(REPO, "results", "phase2_text_conditions.csv"))
    ap.add_argument("--c2-review", action="store_true")
    args = ap.parse_args()
    if args.c2_review:
        for split, ds, raw, new in c2_review():
            print(f"{split}\t{ds}\t{raw}\t{new}")
        return
    parts = [build(ds, args.split) for ds in data.DATASETS]
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    counts = pd.concat([p[1] for p in parts], ignore_index=True)
    df.to_csv(args.out, index=False)
    counts.to_csv(args.out.replace(".csv", "_donor_counts.csv"), index=False)
    for ds, g in df.groupby("dataset", sort=False):
        el = g[g.eligible].groupby("condition").sample_id.nunique().reindex(CONDITIONS).fillna(0).astype(int)
        print(f"{ds} {args.split}: eligible target images per condition: {el.to_dict()}")
    print("written:", os.path.relpath(args.out, REPO))


if __name__ == "__main__":
    main()
