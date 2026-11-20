"""Patient (group)-weighted Table II and supporting numbers for the EMBC paper, from existing per-case results.

No model is run and no existing file is changed. QaTa-COV19 test only; IMG is the frozen nnU-Net (Phase 3).
Numeric fields are read as strings and converted with float() (as Phase 5 read_taus()).

Inputs (read only)
  results/phase3_test_per_case.csv.gz   one row per (method, seed, sample_id, condition, variant_id) x strategy, with
                                        group; LANG, IMG, LR-Gate, ORACLE used here (6 decimals)
  results/phase5_test_per_case.csv.gz   sdc_fallback (0/1) and fixed_average_* (full precision); joined to the phase3
                                        LANG rows on (method, seed, sample_id, condition, variant_id), one to one
  results/phase2_test_per_case.csv.gz   LANG per-case Dice with the patient unit (B2, as src/phase2/analyze_phase2_extra.py)
  results/phase2_donor_manifest_test.csv  changed_attributes per (target, condition, donor variant) (B2)
Strategies
  LANG, IMG, LR-Gate: phase3 rows. SDC-Fallback: the LANG row if sdc_fallback == 0, the IMG row if 1
  (reports/PHASE5_PROTOCOL_FREEZE.md section 1). Fixed Average: phase5 fixed_average_* metrics.
Aggregation
  Group-weighted (as the Phase 5 paired differences, PHASE5_PROTOCOL_FREEZE.md section 5): donors within image and
  seed -> seeds within image -> images within group (patient) -> groups equally weighted.
  Image-weighted (only to reproduce results/phase5_test_results.csv): per seed donors within image -> images; mean
  over the three seeds. empty is 0/1 per row and averaged the same way; reported in %.
Outputs
  paper/tables/table2_mitigation_tabular_patient_all.tex     C0 Dice on all 2113 images (474 groups)
  paper/tables/table2_mitigation_tabular_patient_common.tex  C0 Dice on the C4-C6 common set S (2081 images, 472 groups)
  paper/patient_weighted_numbers.csv     one row per number of the two tables
  paper/patient_weighted_checks.md       statements (a)-(e) re-checked with group weighting
  paper/patient_weighted_supplement.csv  B1: LR-Gate share of the ORACLE headroom, (LR-Gate - LANG) / (ORACLE - LANG)
                                         per model x C4-C6 (definition of reports/PHASE3_RESULTS.md section 4.3);
                                         B2: harm by changed attributes (definition of analyze_phase2_extra.py
                                         severity_by_attribute), image- and group-weighted
Usage (login node; reads ~60 MB of gzip CSV, no GPU):  python src/paper/make_patient_weighted.py
"""
import csv
import os
from itertools import combinations

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET, PART = "QaTa-COV19", "test"
SRC3 = "results/phase3_test_per_case.csv.gz"
SRC5 = "results/phase5_test_per_case.csv.gz"
SRC2 = "results/phase2_test_per_case.csv.gz"
SRCM = "results/phase2_donor_manifest_test.csv"
OUT_ALL = "paper/tables/table2_mitigation_tabular_patient_all.tex"
OUT_COMMON = "paper/tables/table2_mitigation_tabular_patient_common.tex"
OUT_NUM = "paper/patient_weighted_numbers.csv"
OUT_CHECKS = "paper/patient_weighted_checks.md"
OUT_SUPP = "paper/patient_weighted_supplement.csv"

CODES = ["lvit", "reclmis", "ddcmd"]
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
STRATS = ["LANG", "IMG", "LR-Gate", "SDC-Fallback", "Fixed Average"]
CONDS = ["C0", "C4", "C5", "C6"]
WRONG = ["C4", "C5", "C6"]
KEYS = ["method", "seed", "sample_id", "condition", "variant_id"]
METRICS = ["dice", "hd95", "empty"]
NUM_COLS = ["item", "method", "strategy", "condition", "image_set", "n_images", "n_groups", "value_full",
            "value_rounded", "source_files"]


def path(rel):
    return os.path.join(REPO, rel)


def to_float(s):
    """Strings -> float with Python float() (no pandas float parsing)."""
    return s.map(float)


# ---------------------------------------------------------------------------------------------- data
def load_long():
    """Long table: KEYS, group, strategy, dice, hd95, empty, fallback (SDC-Fallback only) for C0 and C4-C6."""
    p3 = pd.read_csv(path(SRC3), dtype=str, keep_default_na=False)
    p3 = p3[(p3.dataset == DATASET) & (p3.part == PART) & p3.condition.isin(CONDS)
            & p3.strategy.isin(["LANG", "IMG", "LR-Gate", "ORACLE"])]
    p3 = p3[KEYS + ["group", "strategy"] + METRICS].copy()
    for k in METRICS:
        p3[k] = to_float(p3[k])
    p5 = pd.read_csv(path(SRC5), dtype=str, keep_default_na=False)
    p5 = p5[(p5.dataset == DATASET) & p5.condition.isin(CONDS)]
    lang = p3[p3.strategy == "LANG"].drop(columns="strategy")
    img = p3[p3.strategy == "IMG"].drop(columns="strategy")
    # join phase5 rows to the phase3 LANG rows (keys unique on both sides, every row matched)
    j = lang.merge(p5, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if (j._merge != "both").any():
        raise SystemExit("phase5 and phase3 per-case rows do not match one to one")
    j = j.merge(img, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_img"))
    fb = to_float(j.sdc_fallback)
    if not fb.isin([0.0, 1.0]).all() or j.dice_img.isna().any():
        raise SystemExit("unexpected sdc_fallback values or missing IMG rows")
    # SDC-Fallback: IMG row where the rule falls back, LANG row otherwise
    sdc = j[KEYS + ["group"]].copy()
    for k in METRICS:
        sdc[k] = np.where(fb == 1.0, j[f"{k}_img"], j[k])
    sdc["strategy"], sdc["fallback"] = "SDC-Fallback", fb.to_numpy()
    fa = j[KEYS + ["group"]].copy()
    for k in METRICS:
        fa[k] = to_float(j[f"fixed_average_{k}"])
    fa["strategy"] = "Fixed Average"
    L = pd.concat([p3, sdc, fa], ignore_index=True)
    L["method"] = L.method.map(NAME)
    return L


def group_weighted(L, value):
    """Per (method, condition, strategy): donors -> seeds -> images -> groups (equal weight)."""
    k = ["method", "condition", "strategy"]
    v = L.groupby(k + ["group", "sample_id", "seed"])[value].mean()
    v = v.groupby(k + ["group", "sample_id"]).mean()
    v = v.groupby(k + ["group"]).mean()
    return v.groupby(k).mean()


def image_weighted(L, value):
    """Per (method, condition, strategy): per seed donors -> images; then the mean of the three seed means."""
    k = ["method", "condition", "strategy"]
    v = L.groupby(k + ["seed", "sample_id"])[value].mean()
    v = v.groupby(k + ["seed"]).mean()
    return v.groupby(k).mean()


def counts(L):
    k = ["method", "condition", "strategy"]
    return L.groupby(k).agg(n_images=("sample_id", "nunique"), n_groups=("group", "nunique"))


def common_set(L):
    """S: images eligible for C4, C5 and C6."""
    return set.intersection(*[set(L[L.condition == c].sample_id) for c in WRONG])


# ---------------------------------------------------------------------------------------------- A: tables
def table_values(L):
    """{(model, strategy, quantity): (value in display units, n_images, n_groups, condition, image_set)}."""
    S = common_set(L)
    LS = L[L.sample_id.isin(S)]
    gw = {k: group_weighted(L, k) for k in METRICS}
    gw_c0s = group_weighted(LS[LS.condition == "C0"], "dice")
    n, nS = counts(L), counts(LS[LS.condition == "C0"])
    out = {}
    for m in NAME.values():
        for s in STRATS:
            out[(m, s, "c0_dice_all")] = (100 * gw["dice"][(m, "C0", s)], *n.loc[(m, "C0", s)], "C0", "all")
            out[(m, s, "c0_dice_common")] = (100 * gw_c0s[(m, "C0", s)], *nS.loc[(m, "C0", s)], "C0", "common_S")
            out[(m, s, "c6_dice")] = (100 * gw["dice"][(m, "C6", s)], *n.loc[(m, "C6", s)], "C6", "common_S")
            out[(m, s, "c6_hd95")] = (gw["hd95"][(m, "C6", s)], *n.loc[(m, "C6", s)], "C6", "common_S")
            out[(m, s, "c6_empty_pct")] = (100 * gw["empty"][(m, "C6", s)], *n.loc[(m, "C6", s)], "C6", "common_S")
    return out


def tex(V, c0_key):
    """tabular only, \\hline rules, two header rows, model name on its first row, 2 decimals."""
    lines = [r"\begin{tabular}{llrrrr}", r"\hline",
             r" & & C0 & \multicolumn{3}{c}{C6} \\", r"Model & Strategy & Dice & Dice & HD95 & Empty (\%) \\", r"\hline"]
    for i, m in enumerate(NAME.values()):
        if i:
            lines.append(r"\hline")
        for j, s in enumerate(STRATS):
            cells = [f"{V[(m, s, q)][0]:.2f}" for q in (c0_key, "c6_dice", "c6_hd95", "c6_empty_pct")]
            lines.append(" & ".join([m if j == 0 else "", s] + cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def number_rows(V):
    src = {"LANG": SRC3, "IMG": SRC3, "LR-Gate": SRC3, "SDC-Fallback": f"{SRC3};{SRC5}", "Fixed Average": f"{SRC5};{SRC3}"}
    rows = []
    for (m, s, q), (v, ni, ng, c, iset) in V.items():
        rows.append(dict(zip(NUM_COLS, [q, m, s, c, iset, int(ni), int(ng), repr(float(v)), f"{v:.2f}", src[s]])))
    return rows


# ---------------------------------------------------------------------------------------------- A4: statements
def read_p5_results():
    rows = [r for r in csv.DictReader(open(path("results/phase5_test_results.csv"))) if r["dataset"] == DATASET]
    return {(r["method"], r["condition"], r["strategy"]): r for r in rows}


def order_flips(new, old):
    """Strategy pairs whose Dice order differs between two {strategy: value} maps."""
    return [(a, b) for a, b in combinations(STRATS, 2) if np.sign(new[a] - new[b]) != np.sign(old[a] - old[b])]


def statements(V):
    P5 = read_p5_results()
    M = list(NAME.values())
    f = lambda x: f"{x:.2f}"  # noqa: E731
    out = ["# Patient-weighted re-check of the Table II statements",
           "",
           "Generated by `src/paper/make_patient_weighted.py`. Group (patient)-weighted values in Dice points, HD95 px,",
           "empty %; comparisons use full precision. QaTa-COV19 test; IMG = nnU-Net. C0 'all' = 2113 images / 474",
           "groups; C0 'common' and C6 = the C4-C6 common set (2081 images / 472 groups). Image-weighted references are",
           "read from `results/phase5_test_results.csv` (C0 on 2113 images).", ""]
    res = {}
    # (a) LANG above IMG under C0
    for key, lab in (("c0_dice_all", "all"), ("c0_dice_common", "common")):
        ok = all(V[(m, "LANG", key)][0] > V[(m, "IMG", key)][0] for m in M)
        res[f"a_{lab}"] = ok
        det = "; ".join(f"{m}: LANG {f(V[(m, 'LANG', key)][0])} vs IMG {f(V[(m, 'IMG', key)][0])}" for m in M)
        out.append(f"- **(a) C0 ({lab}): LANG Dice > IMG Dice for all three models: {ok}.** {det}.")
    # (b) IMG above Fixed Average under C6
    ok = all(V[(m, "IMG", "c6_dice")][0] > V[(m, "Fixed Average", "c6_dice")][0] for m in M)
    res["b"] = ok
    out.append(f"- **(b) C6: IMG Dice > Fixed Average Dice for all three models: {ok}.** " + "; ".join(
        f"{m}: IMG {f(V[(m, 'IMG', 'c6_dice')][0])} vs Fixed Average {f(V[(m, 'Fixed Average', 'c6_dice')][0])}" for m in M) + ".")
    # (c) C6 ranking by HD95 (ascending) equals ranking by Dice (descending)
    lines, okc = [], True
    for m in M:
        rd = sorted(STRATS, key=lambda s: -V[(m, s, "c6_dice")][0])
        rh = sorted(STRATS, key=lambda s: V[(m, s, "c6_hd95")][0])
        okc &= rd == rh
        lines.append(f"  - {m}: Dice (high to low) {' > '.join(rd)}; HD95 (low to high) {' < '.join(rh)}; same: {rd == rh}")
    res["c"] = okc
    out.append(f"- **(c) C6: the HD95 ranking of the five strategies equals the Dice ranking for every model: {okc}.**")
    out += lines
    # (d) Fixed Average HD95 below LANG under C6
    ok = all(V[(m, "Fixed Average", "c6_hd95")][0] < V[(m, "LANG", "c6_hd95")][0] for m in M)
    res["d"] = ok
    out.append(f"- **(d) C6: Fixed Average HD95 < LANG HD95 for all three models: {ok}.** " + "; ".join(
        f"{m}: Fixed Average {f(V[(m, 'Fixed Average', 'c6_hd95')][0])} vs LANG {f(V[(m, 'LANG', 'c6_hd95')][0])}" for m in M) + ".")
    # (e) pairwise Dice order, group- vs image-weighted
    flips = []
    for lab, key, cond in (("C0 (all)", "c0_dice_all", "C0"), ("C0 (common)", "c0_dice_common", "C0"), ("C6", "c6_dice", "C6")):
        for m in M:
            new = {s: V[(m, s, key)][0] for s in STRATS}
            old = {s: 100 * float(P5[(m, cond, s)]["dice"]) for s in STRATS}
            for a, b in order_flips(new, old):
                flips.append(f"  - {lab}, {m}: {a} vs {b}: image-weighted {f(old[a])} vs {f(old[b])}; "
                             f"group-weighted {f(new[a])} vs {f(new[b])}")
    res["e_changed"] = bool(flips)
    out.append(f"- **(e) Pairwise Dice order of the five strategies changed (group- vs image-weighted, 10 pairs x 3 "
               f"models per row set): {bool(flips)}; {len(flips)} change(s).**")
    out += flips if flips else ["  - none"]
    return "\n".join(out) + "\n", res


# ---------------------------------------------------------------------------------------------- B
def b1(L):
    """(LR-Gate - LANG) / (ORACLE - LANG) per model x C4-C6, group- and image-weighted."""
    gw, iw, n = group_weighted(L, "dice"), image_weighted(L, "dice"), counts(L)
    rows = []
    for m in NAME.values():
        for c in WRONG:
            r = {}
            for lab, T in (("image", iw), ("group", gw)):
                r[lab] = (T[(m, c, "LR-Gate")] - T[(m, c, "LANG")]) / (T[(m, c, "ORACLE")] - T[(m, c, "LANG")])
            ni, ng = n.loc[(m, c, "LANG")]
            rows.append({"section": "B1", "method": m, "condition": c, "changed_attributes": "", "n_donor_pairs": "",
                         "n_images": int(ni), "n_groups": int(ng), "quantity": "LR-Gate share of ORACLE headroom",
                         "value_image_weighted": repr(float(r["image"])), "value_group_weighted": repr(float(r["group"])),
                         "value_group_rounded": f"{100 * r['group']:.1f}%"})
    base = list(rows)                      # the nine model x condition rows (summary rows are appended below)
    for lab in ("image", "group"):
        v = [float(r[f"value_{lab}_weighted"]) for r in base]
        for q, x in (("min", min(v)), ("max", max(v))):
            rows.append({"section": "B1_summary", "method": "all", "condition": "C4-C6", "changed_attributes": "",
                         "n_donor_pairs": "", "n_images": "", "n_groups": "", "quantity": f"{q} ({lab}-weighted)",
                         "value_image_weighted": repr(x) if lab == "image" else "",
                         "value_group_weighted": repr(x) if lab == "group" else "", "value_group_rounded": f"{100 * x:.1f}%"})
    return rows


def b2():
    """Harm D(C0) - D(Cx, donor) by changed attributes (severity_by_attribute logic, QaTa only), with groups."""
    pc = pd.read_csv(path(SRC2), dtype=str, keep_default_na=False)
    pc = pc[pc.dataset == DATASET].copy()
    pc["dice"] = to_float(pc.dice)
    man = pd.read_csv(path(SRCM), dtype=str, keep_default_na=False)
    man = man[man.dataset == DATASET]
    S_ = set.intersection(*[set(man[man.condition == c].target_id) for c in WRONG])
    unit = pc.drop_duplicates("sample_id").set_index("sample_id").unit
    rows = []
    for code in CODES:
        p = pc[pc.method == code]
        c0 = p[p.condition == "C0"].groupby("sample_id").dice.mean()                       # seed mean
        x = p[p.condition.isin(WRONG) & p.sample_id.isin(S_)]
        x = x.groupby(["condition", "variant_id", "sample_id"]).dice.mean().reset_index()  # seed mean per donor
        x = x.merge(man.rename(columns={"target_id": "sample_id", "variant": "variant_id"}),
                    on=["condition", "variant_id", "sample_id"], how="left", validate="one_to_one")
        if x.changed_attributes.isna().any() or (x.changed_attributes == "").any():
            raise SystemExit(f"{code}: per-case donor rows not in the manifest")
        x["harm"] = c0.reindex(x.sample_id).values - x.dice
        for (c, ch), g in x.groupby(["condition", "changed_attributes"]):
            per_target = g.groupby("sample_id").harm.mean()                                # donors within target
            per_group = per_target.groupby(unit.reindex(per_target.index).values).mean()  # targets within group
            rows.append({"section": "B2", "method": NAME[code], "condition": c, "changed_attributes": ch,
                         "n_donor_pairs": len(g), "n_images": len(per_target), "n_groups": len(per_group),
                         "quantity": "mean harm (Dice)", "value_image_weighted": repr(float(per_target.mean())),
                         "value_group_weighted": repr(float(per_group.mean())),
                         "value_group_rounded": f"{100 * per_group.mean():.2f}"})
    return rows


# ---------------------------------------------------------------------------------------------- main
def compute():
    L = load_long()
    V = table_values(L)
    md, res = statements(V)
    return {"L": L, "V": V, "tex_all": tex(V, "c0_dice_all"), "tex_common": tex(V, "c0_dice_common"),
            "numbers": number_rows(V), "checks_md": md, "checks": res, "b1": b1(L), "b2": b2()}


def write_csv(rel, rows):
    with open(path(rel), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    R = compute()
    os.makedirs(path("paper/tables"), exist_ok=True)
    for rel, txt in ((OUT_ALL, R["tex_all"]), (OUT_COMMON, R["tex_common"]), (OUT_CHECKS, R["checks_md"])):
        with open(path(rel), "w") as f:
            f.write(txt)
    write_csv(OUT_NUM, R["numbers"])
    write_csv(OUT_SUPP, R["b1"] + R["b2"])
    print(R["checks_md"])
    print("written:", OUT_ALL, OUT_COMMON, OUT_NUM, OUT_CHECKS, OUT_SUPP)


if __name__ == "__main__":
    main()
