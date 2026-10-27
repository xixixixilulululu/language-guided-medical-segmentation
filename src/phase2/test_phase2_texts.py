"""Tests of the Phase 2 text parser (texts.py) and condition generator (conditions.py), on synthetic texts and
on the validation split. No GPU, no test data.

  parser      typical texts, typo table, spacing and elision, D4 (all = upper middle lower), D6 flag,
              D7 ("No" text -> location "none"), ambiguous wording -> that attribute unknown (never zero)
  groups      QaTa sub-S subject / covid_* own unit (D1); MosMed volume prefix / bjorke pooled (D2)
  C2          rewrite keeps every parsed value (re-parsing the template wording gives the same values)
  C3          each deletion removes exactly its clause; C3a and C3c unavailable for "No" texts (D10)
  C4-C6       on the validation split: distances and changed attributes are correct, donors are never the
              target or its group, at most 3 distinct donor texts, the manifest is identical when built twice
              (deterministic) and independent of the row order of the text file
Usage:  python src/phase2/test_phase2_texts.py      (exit status 1 if any check fails)
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2 import conditions as C  # noqa: E402
from phase2 import texts as T  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def attrs(text):
    p = T.parse(text)
    return p["scope"], p["count"], p["location"]


def parse_template(t):
    """Read the C2 template back into (scope, count, location) for the round-trip check."""
    m = re.match(r"^(Bilateral|Unilateral|No) lung infection, (\w+) areas? of infection(?:, in (.+))?\.$", t)
    scope, count = m.group(1), T.NUM[m.group(2)]
    if not m.group(3):
        return scope, count, "none"
    sides = {}
    for part in re.split(r" and (?=the |all of)", m.group(3)):
        side = re.search(r"(left|right) lung", part).group(1)
        if part.startswith("all of"):
            z = "all"
        elif re.match(r"the (left|right) lung", part):
            z = "unspecified"
        else:
            z = "+".join(re.match(r"the (.+) part of", part).group(1).split(" and "))
        sides[side] = z
    return scope, count, ";".join(f"{s}:{sides[s]}" for s in T.SIDES if s in sides)


def main():
    # --- parser ---
    cases = [
        ("Bilateral pulmonary infection, two infected areas, all left lung and all right lung.",
         ("Bilateral", 2, "left:all;right:all")),
        ("Unilateral pulmonary infection, one infected area, lower right lung.", ("Unilateral", 1, "right:lower")),
        ("Bilateral pulmonary infection,two infected areas,lower left and lower right lung.",
         ("Bilateral", 2, "left:lower;right:lower")),                                      # spacing + elision
        ("Unilateral pulmonary infection, oneinfected areas, lower left lung.", ("Unilateral", 1, "left:lower")),
        ("Bilateral pulmonary infection, two infected areas, midldle left lung and lowerl right lung.",
         ("Bilateral", 2, "left:middle;right:lower")),                                      # typo table
        ("Bilateral pulmonary infection, three infected areas, upper middle lower left lung and lower right lung.",
         ("Bilateral", 3, "left:all;right:lower")),                                         # D4
        ("Bilateral pulmonary infection, nine infected areas, all left lung and all right lung..",
         ("Bilateral", 9, "left:all;right:all")),                                           # double full stop
        ("Unilateral pulmonary infection, one infected areas, left lung and middle right lung.",
         ("Unilateral", 1, "left:unspecified;right:middle")),                               # D6, kept as written
        ("No pulmonary infection, zero infected areas.", ("No", 0, "none")),               # D7
        ("Bilateral pulmonary infection, two infected areas, higher left lung and lower right lung.",
         ("Bilateral", 2, None)),                                                           # ambiguous -> unknown
        ("Bilateral pulmonary infection, two one infected area, lower left lung and lower right lung.",
         ("Bilateral", None, "left:lower;right:lower")),                                    # ambiguous count
        ("Bilateral pulmonary infection, two infected areas, all lung and middle lower right lung.",
         ("Bilateral", 2, None)),
    ]
    for text, want in cases:
        got = attrs(text)
        check(f"parse {text[:60]!r}", got == want, f"got {got}")
    check("D6 flag", T.is_scope_conflict(T.parse(cases[7][0])["parse_note"]))
    check("unknown is None, never 0", T.parse(cases[10][0])["count"] is None and not T.parse(cases[10][0])["parse_ok"])

    # --- groups ---
    check("group QaTa sub-S", T.groups("QaTa-COV19", "sub-S09681_ses-E24806_run-1_bp-chest_vp-ap_dx")[4] == "sub-S09681")
    check("group QaTa covid_* own unit, no donor group", T.groups("QaTa-COV19", "covid_12")[2:5:2] == ("covid_12", None))
    check("group MosMed volume", T.groups("MosMedData+", "Jun_coronacases_case8_171")[1] == "Jun_coronacases_case8")
    check("group MosMed bjorke pooled", T.groups("MosMedData+", "bjorke_91")[4] == "bjorke_pooled")

    # --- C2 round trip ---
    for text, want in cases:
        if None in want:
            continue
        t2 = C.c2_text(*want)
        check(f"C2 keeps all values: {t2[:70]!r}", parse_template(t2) == want, f"got {parse_template(t2)}")

    # --- C3 deletions ---
    a = {"scope": "Bilateral", "count": 2, "location": "left:lower;right:lower"}
    raw = "Bilateral pulmonary infection,two infected areas, lower left lung and lower right lung."
    check("C3a deletes the scope word only",
          C.c3("C3a", raw, a)[0] == "Pulmonary infection,two infected areas, lower left lung and lower right lung.")
    check("C3b deletes the count clause",
          C.c3("C3b", raw, a)[0] == "Bilateral pulmonary infection, lower left lung and lower right lung.")
    check("C3c deletes the location clause", C.c3("C3c", raw, a)[0] == "Bilateral pulmonary infection,two infected areas.")
    no = {"scope": "No", "count": 0, "location": "none"}
    raw_no = "No pulmonary infection, zero infected areas."
    check("C3a unavailable for No texts", C.c3("C3a", raw_no, no)[0] is None)
    check("C3c unavailable for No texts", C.c3("C3c", raw_no, no)[0] is None)
    check("C3b on a No text", C.c3("C3b", raw_no, no)[0] == "No pulmonary infection.")

    # --- donors on the validation split ---
    for ds in ("QaTa-COV19", "MosMedData+"):
        df, _ = C.build(ds, "val")
        att = T.attributes(ds, "val").set_index("sample_id")
        don = df[df.condition.isin(["C4", "C5", "C6"]) & df.eligible]
        bad = 0
        for r in don.itertuples():
            t, d = att.loc[r.sample_id], att.loc[r.donor_sample_id]
            dist = sum(str(t[k]) != str(d[k]) for k in C.ATTRS)
            g_t, g_d = T.groups(ds, r.sample_id)[4], T.groups(ds, r.donor_sample_id)[4]
            bad += (dist != C.DISTANCE[r.condition] or r.donor_sample_id == r.sample_id
                    or (g_t is not None and g_t == g_d) or r.counterfactual_text != d.raw_text
                    or r.changed_attributes != C.changed(t, d))
        check(f"{ds}: donor distance / group / text / changed attributes correct ({len(don)} donor rows)", bad == 0,
              f"{bad} bad")
        per = don.groupby(["sample_id", "condition"])
        check(f"{ds}: at most 3 donors, all with distinct texts",
              per.size().max() <= 3 and (per.counterfactual_text.apply(lambda s: s.map(C.ws).nunique()) == per.size()).all())
        df2, _ = C.build(ds, "val")
        check(f"{ds}: manifest identical when built twice", df.equals(df2))
        # row-order independence: shuffle the text file rows (simulated through the attribute table order)
        orig = T.attributes
        T.attributes = lambda d, s, v=None: orig(d, s, v).sample(frac=1, random_state=3)
        df3, _ = C.build(ds, "val")
        T.attributes = orig
        key = ["sample_id", "condition", "variant_id"]
        same = df.sort_values(key).reset_index(drop=True).equals(df3.sort_values(key).reset_index(drop=True))
        check(f"{ds}: manifest independent of the row order", same)
    print("ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
