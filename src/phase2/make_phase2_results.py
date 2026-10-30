"""Fill the generated tables of reports/PHASE2_RESULTS.md from the Phase 2 TEST result files (no text).

Tables (between <!-- BEGIN NAME --> and <!-- END NAME -->; a bare <!-- NAME --> is replaced on the first run):
  RECONCILE   results/phase2_test_c0_reconciliation.csv (P2-2)
  MAIN        results/phase2_main_results_test.csv: Dice mean +- SD over the 3 per-seed image means, in points
  PAIRWISE    results/phase2_pairwise_results_test.csv: the 11 comparisons, unit-weighted delta, 95% t-CI, Holm
  VALTEST     validation vs test unit-weighted deltas of the 11 comparisons (phase2_pairwise_results_val.csv)
  SEVERITY    results/phase2_severity_results_test.csv (Task 2.6, attribute distance on S)
  ATTR        results/phase2_severity_by_attribute_test.csv (harm by changed-attribute combination)
  SUSCEPT     results/phase2_susceptibility_test.csv (Task 2.7)
  C2CHANGE    results/phase2_c2_change_test.csv (C0 vs C2 change)
  D6          results/phase2_d6_sensitivity_test.csv (MosMed without the scope-conflict images)
The Phase 3 go / no-go rule files of the test split are listed in the report but never tabulated here.
Usage:  python src/phase2/make_phase2_results.py
"""
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = lambda f: os.path.join(REPO, "results", f)  # noqa: E731
DOC = os.path.join(REPO, "reports", "PHASE2_RESULTS.md")
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
CONDS = ["C0", "C1", "C2", "C3a", "C3b", "C3c", "C4", "C5", "C6"]


def put(text, name, table):
    block = f"<!-- BEGIN {name} -->\n{table}\n<!-- END {name} -->"
    pat = re.compile(rf"<!-- BEGIN {name} -->.*?<!-- END {name} -->", re.S)
    if pat.search(text):
        return pat.sub(lambda _: block, text)
    return text.replace(f"<!-- {name} -->", block)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main():
    c = pd.read_csv(R("phase2_test_c0_reconciliation.csv"))
    rec = pd.DataFrame({"Method": c.method.map(NAME), "Dataset": c.dataset, "Seed": c.seed, "n": c.n,
                        "Cases differing": c.cases_differing, "Max px": c.max_pixels_differing,
                        "Empty flips": c.empty_flips, "|Δ mean Dice|": c.abs_mean_dice_difference.map("{:.2e}".format),
                        "Max case |ΔDice|": c.max_abs_case_dice_difference.map("{:.6f}".format),
                        "Max case |ΔHD95|": c.max_abs_case_hd95_difference.map("{:.4f}".format),
                        "Within P2-2": c.within_rule})
    M = pd.read_csv(R("phase2_main_results_test.csv"))
    parts = []
    for ds, g in M.groupby("dataset", sort=False):
        rows = []
        for m, h in g.groupby("method", sort=False):
            h = h.set_index("condition")
            rows.append({"Method": m, **{k: f"{100 * h.dice_mean[k]:.2f} ± {100 * h.dice_sd[k]:.2f}" for k in CONDS}})
        n = g.groupby("condition").n_images.first()
        parts.append(f"**{ds}** ({g.n_units.max()} units; eligible images: "
                     + ", ".join(f"{k} {n[k]}" for k in CONDS) + ")\n\n" + md(pd.DataFrame(rows)))
    P = pd.read_csv(R("phase2_pairwise_results_test.csv"))
    pw = []
    for ds, g in P.groupby("dataset", sort=False):
        t = pd.DataFrame({"Method": g.method, "Comparison": g.comparison, "Images": g.n_images, "Units": g.n_units,
                          "Δ image-weighted": (100 * g.image_weighted_delta).map("{:+.2f}".format),
                          "Δ unit-weighted": (100 * g.unit_weighted_delta).map("{:+.2f}".format),
                          "95% CI": [f"[{100 * a:+.2f}, {100 * b:+.2f}]" for a, b in zip(g.ci95_low, g.ci95_high)],
                          "p raw": g.p_raw.map("{:.2g}".format), "p Holm": g.p_holm.map("{:.2g}".format),
                          "Holm < 0.05": g["significant_holm_0.05"],
                          "Exploratory": g.exploratory})
        pw.append(f"**{ds}**\n\n" + md(t))
    k = ["dataset", "method", "comparison"]
    V = pd.read_csv(R("phase2_pairwise_results_val.csv"))[k + ["unit_weighted_delta", "significant_holm_0.05"]]
    VT = V.merge(P[k + ["unit_weighted_delta", "significant_holm_0.05"]], on=k, suffixes=("_val", "_test"))
    vt = []
    for ds, g in VT.groupby("dataset", sort=False):
        same = (g.unit_weighted_delta_val > 0) == (g.unit_weighted_delta_test > 0)
        same_h = g["significant_holm_0.05_val"] == g["significant_holm_0.05_test"]
        t = pd.DataFrame({"Method": g.method, "Comparison": g.comparison,
                          "Validation Δ": (100 * g.unit_weighted_delta_val).map("{:+.2f}".format),
                          "Holm < 0.05 (val)": g["significant_holm_0.05_val"],
                          "Test Δ": (100 * g.unit_weighted_delta_test).map("{:+.2f}".format),
                          "Holm < 0.05 (test)": g["significant_holm_0.05_test"],
                          "Same sign": same, "Same Holm decision": same_h})
        vt.append(f"**{ds}**: same sign in {int(same.sum())} of {len(g)}; same Holm decision in "
                  f"{int(same_h.sum())} of {len(g)} (validation {int(g['significant_holm_0.05_val'].sum())}, "
                  f"test {int(g['significant_holm_0.05_test'].sum())} with Holm p < 0.05)\n\n" + md(t))
    S6 = pd.read_csv(R("phase2_severity_results_test.csv"))
    pt = lambda x: f"{100 * x:+.2f}"  # noqa: E731
    sev = pd.DataFrame({"Dataset": S6.dataset, "Method": S6.method, "Images in S": S6.n_images_S,
                        "Units": S6.n_units_S, "Coverage": S6.coverage_S.map("{:.1%}".format),
                        "C0": (100 * S6.mean_C0).map("{:.2f}".format), "C4": (100 * S6.mean_C4).map("{:.2f}".format),
                        "C5": (100 * S6.mean_C5).map("{:.2f}".format), "C6": (100 * S6.mean_C6).map("{:.2f}".format),
                        **{f"C0−{c} unit [95% CI]": [f"{pt(r[f'unit_delta_C0-{c}'])} [{pt(r[f'unit_ci95_low_C0-{c}'])}, "
                                                    f"{pt(r[f'unit_ci95_high_C0-{c}'])}]" for _, r in S6.iterrows()]
                           for c in ("C4", "C5", "C6")}})
    A = pd.read_csv(R("phase2_severity_by_attribute_test.csv"))
    attr = A.pivot_table(index=["dataset", "condition", "changed_attributes", "n_donor_pairs", "n_targets"],
                         columns="method", values="mean_harm", sort=False).reset_index()
    attr = pd.DataFrame({"Dataset": attr.dataset, "Condition": attr.condition, "Changed": attr.changed_attributes,
                         "Donor pairs": attr.n_donor_pairs, "Targets": attr.n_targets,
                         **{m: attr[m].map(pt) for m in ("LViT-T", "RecLMIS", "DD-CMD")}})
    U = pd.read_csv(R("phase2_susceptibility_test.csv"))
    um, up = U[U.kind == "method"], U[U.kind == "pair"]
    sus_m = pd.DataFrame({"Dataset": um.dataset, "Method": um.method_a, "Images": um.n_images, "Units": um.n_units,
                          "C0": (100 * um.C0).map("{:.2f}".format), "C0−C1": um["C0-C1"].map(pt),
                          "C1−C6": um["C1-C6"].map(pt), "H = C0−C6": um.H.map(pt)})
    sus_p = pd.DataFrame({"Dataset": up.dataset, "Pair": up.method_a + " − " + up.method_b,
                          "ΔH image": up.image_delta_H.map(pt), "ΔH unit [95% CI]":
                          [f"{pt(a)} [{pt(b)}, {pt(c)}]" for a, b, c in zip(up.unit_delta_H, up.unit_ci95_low, up.unit_ci95_high)]})
    C2 = pd.read_csv(R("phase2_c2_change_test.csv"))
    c2 = pd.DataFrame({"Dataset": C2.dataset, "Method": C2.method, "Images": C2.n_images,
                       "Mean |ΔDice|": (100 * C2.mean_abs_dDice).map("{:.2f}".format),
                       "Mean signed C0−C2": C2.mean_signed_dDice.map(pt),
                       "Masks changed": C2.frac_mask_changed.map("{:.1%}".format),
                       "Changed > 4 px (DD-CMD)": C2.frac_mask_changed_gt4px.map(lambda x: "" if pd.isna(x) else f"{x:.1%}"),
                       "Empty flips (per seed)": C2.empty_flips_per_seed})
    D6 = pd.read_csv(R("phase2_d6_sensitivity_test.csv"))
    d6 = pd.DataFrame({"Method": D6.method, "Comparison": D6.comparison,
                       "All images (273) Δ unit": D6.unit_weighted_delta_all.map(pt),
                       f"Without the {int(D6.d6_images_excluded.iloc[0])} D6 images Δ unit": D6.unit_weighted_delta_without_d6.map(pt)})
    s = open(DOC).read()
    for name, t in (("VALTEST", "\n\n".join(vt)), ("SEVERITY", md(sev)), ("ATTR", md(attr)),
                    ("SUSCEPT", md(sus_m) + "\n\n" + md(sus_p)), ("C2CHANGE", md(c2)), ("D6", md(d6))):
        if f"<!-- {name} -->" in s or f"<!-- BEGIN {name} -->" in s:
            s = put(s, name, t)
    s = put(s, "RECONCILE", md(rec))
    s = put(s, "MAIN", "\n\n".join(parts))
    s = put(s, "PAIRWISE", "\n\n".join(pw))
    open(DOC, "w").write(s)
    print("tables written")


if __name__ == "__main__":
    main()
