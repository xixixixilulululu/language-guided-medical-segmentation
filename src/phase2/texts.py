"""Phase 2, Task 2.1: read the benchmark texts, parse scope / count / location, attach patient / volume groups.

Texts are read ONLY through the Phase 2 text view (src/phase2/lock.py). For the test split the Phase 2 lock
must be open: read_texts() calls lock.require_open() before the test text file is touched.

Text template of both datasets:  "<Scope> pulmonary infection, <count> infected area(s), <location>."
Attributes (decisions of 2026-10-27, reports/PHASE2_SETUP_PLAN.md section 9):
  scope     the stated laterality / presence word: Bilateral | Unilateral | No (MosMed only). Nothing else
            (type, severity) occurs in these texts, so nothing else is mixed in.
  count     number of infected areas, 0-9 (word -> integer).
  location  canonical zones per side, e.g. "left:lower;right:all". Zones are upper / middle / lower; the full
            set "upper middle lower" equals "all" (D4); a side named without a zone is "unspecified"
            (MosMed "left lung and middle right lung"); "none" for "No pulmonary infection" texts (D7).
Normalisation before parsing (D5): spaces and commas, a trailing (double) full stop, the elision
"<z> left and <z> right lung", and the fixed table TYPOS of unambiguous typos. Anything else that does not
fit (e.g. "higher", "two one infected area", "all lung") leaves that attribute UNKNOWN (empty), never zero.
Inconsistencies are kept as written and flagged in parse_note (D6: "Unilateral" with both sides named).
Masks, predictions and LLMs are never used to fill an attribute.

Groups (D1, D2): QaTa patient_id = BIDS subject `sub-S...` of the file name (naming convention of
BIMCV-COVID19+, no separate patient list); `covid_*` images have no patient id and form their own unit
(flag "patient unverified"). MosMed volume_id = file name without the final `_<slice>` (Phase 0 source
mapping); `bjorke_*` slices carry no volume id and form one pooled unit "bjorke_pooled".

Usage (train + val only until the Phase 2 lock is open):
    python src/phase2/texts.py --splits train val      -> results/phase2_text_attributes.csv
"""
import argparse
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from lgms import data  # noqa: E402
from phase2 import lock  # noqa: E402

NUM = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
WORD = {v: k for k, v in NUM.items()}
# Fixed table of unambiguous typos found in the train/val texts (D5); applied word by word after spacing
TYPOS = {"lowerl": "lower", "lowoer": "lower", "lwoer": "lower", "ower": "lower", "lowe": "lower",
         "midldle": "middle", "midddle": "middle", "middlr": "middle", "mddle": "middle", "midde": "middle",
         "iddle": "middle", "alll": "all", "uppper": "upper", "uooer": "upper", "upperl": "upper", "lift": "left",
         "ung": "lung", "thrr": "three", "upperlower": "upper lower", "middlelower": "middle lower",
         "lowerright": "lower right", "rightlung": "right lung", "leflung": "left lung",
         "oneinfected": "one infected", "threeinfected": "three infected", "andupper": "and upper"}
ZONE_ORDER = ["upper", "middle", "lower"]
SIDES = ["left", "right"]
SCOPE_RE = re.compile(r"^(Bilateral|Unilateral|No) pulmonary infection(?:,|$)")
COUNT_RE = re.compile(r"(?:^|, )(\w+) infected areas?(?=,|$)")
LOC_RE = re.compile(r" infected areas?, (.+)$")
COLUMNS = ["dataset", "split", "sample_id", "patient_id", "volume_id", "unit_id", "unit_type", "raw_text",
           "normalised_text", "scope", "count", "location", "parse_ok", "parse_note"]


# ---------------------------------------------------------------------------------------------
# Reading (through the text view; test only with the lock open)
# ---------------------------------------------------------------------------------------------
def text_key(dataset, sample_id):
    """xlsx key of a sample: the mask file name (QaTa masks are called mask_<image>.png)."""
    return f"mask_{sample_id}.png" if dataset == "QaTa-COV19" else f"{sample_id}.png"


def read_texts(dataset, split, view=None, repo=None):
    """{sample_id: raw text} of one split, in split order. Test: the Phase 2 lock is checked first."""
    if split == "test":
        lock.require_open(repo or lock.REPO, what=f"reading the {dataset} test texts")
    path = lock.text_file(dataset, split, view)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: not in the Phase 2 text view (run `python src/phase2/lock.py make-view`)")
    df = pd.read_excel(path)
    if df.Image.duplicated().any():
        raise ValueError(f"{dataset}/{split}: duplicate keys in the text file")
    by_key = dict(zip(df.Image.astype(str), df.Description.astype(str)))
    out = {}
    for i in data.split_ids(dataset, split):
        t = by_key.get(text_key(dataset, i))
        if t is None or not t.strip() or t == "nan":
            raise KeyError(f"{dataset}/{split}: no text for {i}")
        out[i] = t
    return out


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------
def normalise(text):
    """(normalised text, notes): spacing, commas, trailing full stop(s), typo table; nothing else."""
    notes = []
    t = text.strip()
    if t.endswith(".."):
        notes.append("double full stop")
    t = t.rstrip(".").strip()
    spaced = re.sub(r"\s+", " ", re.sub(r"\s*,\s*", ", ", t))
    if spaced != t:
        notes.append("spacing")
    words = []
    for w in spaced.split(" "):
        if w in TYPOS:
            notes.append(f"typo {w}->{TYPOS[w]}")
        words.append(TYPOS.get(w, w))
    return " ".join(words), notes


def canonical_zones(zones):
    """Zone words of one side -> canonical string (D4: the full set is 'all'; none -> 'unspecified')."""
    if not zones:
        return "unspecified"
    if "all" in zones:
        return "all" if set(zones) == {"all"} else None                      # 'all' mixed with zones: unclear
    if len(set(zones)) != len(zones) or any(z not in ZONE_ORDER for z in zones):
        return None
    if set(zones) == set(ZONE_ORDER):
        return "all"
    return "+".join(z for z in ZONE_ORDER if z in zones)


def parse_location(clause):
    """'lower left lung and all right lung' -> ('left:lower;right:all', None) or (None, reason)."""
    clause = re.sub(r"\bleft and\b", "left lung and", clause)                   # elision "<z> left and <z> right lung"
    sides = {}
    for part in clause.split(" and "):
        w = part.split()
        if len(w) < 2 or w[-1] != "lung" or w[-2] not in SIDES:
            return None, f"location part not understood: {part!r}"
        z = canonical_zones(w[:-2])
        if z is None:
            return None, f"zones not understood: {part!r}"
        if w[-2] in sides:
            return None, f"side named twice: {clause!r}"
        sides[w[-2]] = z
    return ";".join(f"{s}:{sides[s]}" for s in SIDES if s in sides), None


def parse(text):
    """dict(scope, count, location, parse_ok, parse_note, normalised_text); unknown attributes are None."""
    n, notes = normalise(text)
    m = SCOPE_RE.match(n)
    scope = m.group(1) if m else None
    if scope is None:
        notes.append("scope unknown")
    m = COUNT_RE.search(n)
    count = NUM.get(m.group(1)) if m else None
    if count is None:
        notes.append("count unknown")
    m = LOC_RE.search(n)
    if m:
        location, why = parse_location(m.group(1))
        if why:
            notes.append(why)
    elif scope == "No" and count == 0 and re.search(r" infected areas?$", n):
        location = "none"                                                        # D7: no infection, no location
    else:
        location = None
        notes.append("location clause missing")
    if location is None and "location" not in " ".join(notes):
        notes.append("location unknown")
    if scope and location and location not in ("none",):                         # D6: flag, keep as written
        n_sides = location.count(":")
        if scope == "Unilateral" and n_sides == 2:
            notes.append("inconsistent: Unilateral but both sides named")
        if scope == "Bilateral" and n_sides == 1:
            notes.append("inconsistent: Bilateral but one side named")
    if scope == "No" and count not in (None, 0):
        notes.append("inconsistent: No infection with a non-zero count")
    ok = scope is not None and count is not None and location is not None
    return {"scope": scope, "count": count, "location": location, "parse_ok": ok,
            "parse_note": "; ".join(notes), "normalised_text": n}


def is_scope_conflict(note):
    """D6 sensitivity check: the MosMed images whose stated scope contradicts the sides named."""
    return "inconsistent: Unilateral but both sides named" in (note or "")


# ---------------------------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------------------------
def groups(dataset, sample_id):
    """(patient_id, volume_id, unit_id, unit_type, donor_group): donor_group None = only the case itself is
    excluded as a donor (no verified group)."""
    if dataset == "QaTa-COV19":
        m = re.match(r"(sub-S\d+)_ses-", sample_id)
        if m:
            return m.group(1), "", m.group(1), "patient", m.group(1)
        return "", "", sample_id, "image (patient unverified)", None
    if sample_id.startswith("bjorke_"):
        return "", "bjorke_pooled", "bjorke_pooled", "pooled source (no volume id)", "bjorke_pooled"
    v = re.sub(r"_\d+$", "", sample_id)
    return "", v, v, "volume", v


def attributes(dataset, split, view=None):
    """DataFrame with COLUMNS for one dataset and split."""
    rows = []
    for sid, raw in read_texts(dataset, split, view).items():
        p = parse(raw)
        pid, vid, uid, utype, _ = groups(dataset, sid)
        rows.append({"dataset": dataset, "split": split, "sample_id": sid, "patient_id": pid, "volume_id": vid,
                     "unit_id": uid, "unit_type": utype, "raw_text": raw, **p})
    df = pd.DataFrame(rows)[COLUMNS]
    df["count"] = df["count"].astype("Int64")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--out", default=os.path.join(REPO, "results", "phase2_text_attributes.csv"))
    args = ap.parse_args()
    df = pd.concat([attributes(ds, sp) for ds in data.DATASETS for sp in args.splits], ignore_index=True)
    df.to_csv(args.out, index=False)
    for (ds, sp), g in df.groupby(["dataset", "split"], sort=False):
        print(f"{ds} {sp}: {len(g)} images, parsed {int(g.parse_ok.sum())} ({100 * g.parse_ok.mean():.2f}%), "
              f"distinct texts {g.raw_text.nunique()}")
    print("written:", os.path.relpath(args.out, REPO))


if __name__ == "__main__":
    main()
