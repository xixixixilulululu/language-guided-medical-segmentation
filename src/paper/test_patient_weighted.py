"""Self-check of src/paper/make_patient_weighted.py against the committed results (run after the generator).

  rows        per model x strategy: C0 2113 x 3 rows, C4-C6 2081 x 3 x 3 rows
  groups      phase2 per-case unit == phase3 per-case group on all 2113 test images
  image       image-weighted Dice / HD95 / empty recomputed from the same per-case data == phase5_test_results.csv (1e-5)
  D_C0        group-weighted C0 Dice on the common set == phase5_mixed_reliability.csv D_C0 per strategy (1e-5)
  pairwise    group-weighted LR-Gate - Fixed Average and LR-Gate - SDC-Fallback == phase5_pairwise_results.csv (1e-5)
  fallback    SDC-Fallback fallback rate (image-weighted) == phase5_test_results.csv fallback_rate (1e-5)
  B1          image-weighted ratios rounded to 0.1% == reports/PHASE3_RESULTS.md section 4.3 (RECOVERY block)
  B2          image-weighted harm == phase2_severity_by_attribute_test.csv (1e-5)
  files       written files equal the generated text / rows; each table has 15 rows matching value_rounded
Usage:  python src/paper/test_patient_weighted.py      (exit status 1 if any check fails)
"""
import csv
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_patient_weighted as W  # noqa: E402

TOL = 1e-5
FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def rows_of(rel):
    with open(W.path(rel), newline="") as f:
        return list(csv.DictReader(f))


R = W.compute()
L = R["L"]
M = list(W.NAME.values())
STR5 = W.STRATS

# ---- 1. row counts
n = L.groupby(["method", "strategy", "condition"]).size()
ok = all(n[(m, s, c)] == (2113 * 3 if c == "C0" else 2081 * 3 * 3) for m in M for s in STR5 for c in W.CONDS)
check("rows: C0 2113x3, C4-C6 2081x3x3 per model x strategy", ok)

# ---- 2. unit == group
p2 = pd.read_csv(W.path(W.SRC2), dtype=str, keep_default_na=False)
u = p2[p2.dataset == W.DATASET].drop_duplicates("sample_id").set_index("sample_id").unit
g = L.drop_duplicates("sample_id").set_index("sample_id").group
check("phase2 unit == phase3 group on all test images", len(g) == 2113 and u.reindex(g.index).eq(g).all() and len(u) == 2113,
      f"{len(g)} images, {g.nunique()} groups")

# ---- 3. image-weighted reproduction of phase5_test_results.csv
P5 = W.read_p5_results()
worst = 0.0
for k in W.METRICS:
    iw = W.image_weighted(L[L.strategy.isin(STR5)], k)
    for (m, c, s), v in iw.items():
        worst = max(worst, abs(v - float(P5[(m, c, s)][k])))
check("image-weighted dice / hd95 / empty == phase5_test_results.csv", worst <= TOL, f"max |diff| {worst:.2e}")
cnt = W.counts(L)
ok = all((int(cnt.loc[(m, c, s)].n_images), int(cnt.loc[(m, c, s)].n_groups)) ==
         (int(P5[(m, c, s)]["n_images"]), int(P5[(m, c, s)]["n_groups"])) for m in M for s in STR5 for c in W.CONDS)
check("n_images / n_groups == phase5_test_results.csv (C0 2113/474, C4-C6 2081/472)", ok)

# ---- 4. D_C0 on the common set
S = W.common_set(L)
gS = W.group_weighted(L[L.sample_id.isin(S) & (L.condition == "C0")], "dice")
mix = rows_of("results/phase5_mixed_reliability.csv")
ref = {(r["method"], r["strategy"]): float(r["D_C0"]) for r in mix}
worst = max(abs(gS[(m, "C0", s)] - ref[(m, s)]) for m in M for s in STR5)
check("group-weighted C0 Dice on S == phase5_mixed_reliability D_C0 (5 strategies)", worst <= TOL, f"max |diff| {worst:.2e}")
check("common set S has 2081 images / 472 groups", len(S) == 2081 and L[L.sample_id.isin(S)].group.nunique() == 472)

# ---- 5. paired group-weighted differences
gw = W.group_weighted(L, "dice")
pw = [r for r in rows_of("results/phase5_pairwise_results.csv") if r["kind"] == "main" and r["dataset"] == W.DATASET]
worst, nchk = 0.0, 0
for r in pw:
    a, b = r["comparison"].split(" - ")
    worst = max(worst, abs((gw[(r["method"], r["condition"], a)] - gw[(r["method"], r["condition"], b)]) - float(r["delta"])))
    nchk += 1
check("group-weighted LR-Gate - Fixed Average / - SDC-Fallback == phase5_pairwise_results.csv", nchk == 18 and worst <= TOL,
      f"{nchk} rows (C4-C6 incl. C6), max |diff| {worst:.2e}")

# ---- 6. fallback rate
fr = W.image_weighted(L[L.strategy == "SDC-Fallback"], "fallback")
worst = max(abs(v - float(P5[(m, c, s)]["fallback_rate"])) for (m, c, s), v in fr.items())
check("SDC-Fallback fallback rate == phase5_test_results.csv", len(fr) == 12 and worst <= TOL, f"max |diff| {worst:.2e}")

# ---- 7. B1 image-weighted against the Phase 3 report
txt = open(W.path("reports/PHASE3_RESULTS.md"), encoding="utf-8").read()
# only the table between the RECOVERY markers; the column is found by its header name, not by position
block = txt.split("<!-- BEGIN RECOVERY -->")[1].split("<!-- END RECOVERY -->")[0]
tab = [[x.strip() for x in ln.strip().strip("|").split("|")] for ln in block.splitlines() if ln.strip().startswith("|")]
hdr, data = tab[0], [r for r in tab[2:]]            # tab[1] is the |---| separator row
check("RECOVERY table has a 'LR-Gate recovered' column", "LR-Gate recovered" in hdr, f"header {hdr}")
check("RECOVERY table has exactly 9 data rows", len(data) == 9, f"{len(data)} rows")
ci = hdr.index("LR-Gate recovered") if "LR-Gate recovered" in hdr else None
im, ic = hdr.index("Method"), hdr.index("Condition")
rep = {(r[im], r[ic]): r[ci] for r in data} if ci is not None else {}
b1 = [r for r in R["b1"] if r["section"] == "B1"]
ok = len(rep) == 9 and all(f"{100 * float(r['value_image_weighted']):.1f}%" == rep[(r["method"], r["condition"])] for r in b1)
check("B1 image-weighted shares == PHASE3_RESULTS.md 4.3 (0.1%)", ok,
      f"min {min(float(r['value_image_weighted']) for r in b1):.4f}, max {max(float(r['value_image_weighted']) for r in b1):.4f}")

# ---- 8. B2 image-weighted against phase2_severity_by_attribute_test.csv
sev = {(r["method"], r["condition"], r["changed_attributes"]): r for r in rows_of("results/phase2_severity_by_attribute_test.csv")
       if r["dataset"] == W.DATASET}
ok, worst = len(R["b2"]) == len(sev) == 21, 0.0
for r in R["b2"]:
    s = sev[(r["method"], r["condition"], r["changed_attributes"])]
    ok &= (int(s["n_donor_pairs"]), int(s["n_targets"])) == (r["n_donor_pairs"], r["n_images"])
    worst = max(worst, abs(float(r["value_image_weighted"]) - float(s["mean_harm"])))
check("B2 image-weighted harm, n_donor_pairs, n_targets == phase2_severity_by_attribute_test.csv", ok and worst <= TOL,
      f"{len(R['b2'])} rows, max |diff| {worst:.2e}")

# ---- 9. written files
for rel, key in ((W.OUT_ALL, "tex_all"), (W.OUT_COMMON, "tex_common"), (W.OUT_CHECKS, "checks_md")):
    check(f"{rel} equals the generated text", open(W.path(rel)).read() == R[key])
num = rows_of(W.OUT_NUM)
check("patient_weighted_numbers.csv equals the generated rows (75 = 15 x 5)",
      num == [{k: str(v) for k, v in r.items()} for r in R["numbers"]] and len(num) == 75)
supp = rows_of(W.OUT_SUPP)
check("patient_weighted_supplement.csv equals the generated rows",
      supp == [{k: str(v) for k, v in r.items()} for r in R["b1"] + R["b2"]])
look = {(r["method"], r["strategy"], r["item"]): r["value_rounded"] for r in num}
for rel, c0 in ((W.OUT_ALL, "c0_dice_all"), (W.OUT_COMMON, "c0_dice_common")):
    body = [ln for ln in open(W.path(rel)).read().splitlines()
            if ln.endswith(r"\\") and not ln.startswith(("Model", " & & C0"))]
    ok, model = len(body) == 15, None
    for ln in body:
        c = [x.strip() for x in ln[:-2].split("&")]
        model = c[0] or model
        ok &= c[2:] == [look[(model, c[1], q)] for q in (c0, "c6_dice", "c6_hd95", "c6_empty_pct")]
    check(f"{os.path.basename(rel)}: 15 rows, values == value_rounded", ok)

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
