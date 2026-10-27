"""Fill reports/templates/PHASE2_VALIDATION_SUMMARY_template.md with tables and numbers read from the validation CSVs.

Every number in reports/PHASE2_VALIDATION_SUMMARY.md comes from this script: tables are written as Markdown from the
results files, and the numbers quoted in the text are template fields ({{N:name}}) computed here. A field that is
not defined stops the script, so no number is typed by hand. Validation only; no test file is read. The condition
examples (section 2) are taken from the validation manifest for one target chosen by a fixed rule (the eligible
QaTa target with the smallest sha256 of "phase2-example-v1|<id>" among the targets with all nine conditions).

Usage:  python src/phase2/make_phase2_tables.py
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
R = lambda f: os.path.join(REPO, "results", f)  # noqa: E731
TEMPLATE = os.path.join(REPO, "reports", "templates", "PHASE2_VALIDATION_SUMMARY_template.md")
OUT = os.path.join(REPO, "reports", "PHASE2_VALIDATION_SUMMARY.md")
DATASETS = ["QaTa-COV19", "MosMedData+"]
METHODS = ["LViT-T", "RecLMIS", "DD-CMD"]
CONDS = ["C0", "C1", "C2", "C3a", "C3b", "C3c", "C4", "C5", "C6"]
COMPS = ["C0-C1", "C0-C2", "C0-C3a", "C0-C3b", "C0-C3c", "C0-C4", "C0-C5", "C0-C6", "C1-C4", "C1-C5", "C1-C6"]


def pts(x, sign=True):
    """[0, 1] Dice -> points with two decimals."""
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{100 * x:+.2f}" if sign else f"{100 * x:.2f}"


def pval(p):
    if p is None or not np.isfinite(p):
        return "NA"
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def load():
    return {k: pd.read_csv(R(f)) for k, f in {
        "main": "phase2_main_results_val.csv", "pw": "phase2_pairwise_results_val.csv",
        "d6": "phase2_d6_sensitivity_val.csv", "rule": "phase2_phase3_rule_val.csv",
        "donor": "phase2_phase3_per_donor_val.csv", "sev": "phase2_severity_results_val.csv",
        "att": "phase2_severity_by_attribute_val.csv", "sus": "phase2_susceptibility_val.csv",
        "c2": "phase2_c2_change_val.csv", "audit": "phase2_effective_input_val_audit.csv"}.items()}


def tables(D):
    T, N = {}, {}
    main, pw = D["main"], D["pw"]
    # coverage: images / units per condition (identical for the three methods; LViT-T rows)
    cov = main[main.method == "LViT-T"].pivot_table(index="condition", columns="dataset", values=["n_images", "n_units"])
    T["coverage"] = md(pd.DataFrame({"Condition": CONDS,
                                     **{f"{ds} images": [int(cov.loc[c, ("n_images", ds)]) for c in CONDS] for ds in DATASETS},
                                     **{f"{ds} units": [int(cov.loc[c, ("n_units", ds)]) for c in CONDS] for ds in DATASETS}}))
    a = D["audit"]
    g = a.groupby(["dataset", "method"]).agg(rows=("rows", "sum"), raw=("raw_text_changed", "sum"),
                                             unchanged=("raw_changed_but_input_unchanged", "sum")).reset_index()
    T["audit"] = md(g.rename(columns={"dataset": "Dataset", "method": "Method", "rows": "Condition rows",
                                      "raw": "Text changed", "unchanged": "Text changed, model input unchanged"}))
    N["audit_rows"] = f"{int(a.rows.sum()):,}"
    N["audit_unchanged"] = str(int(a.raw_changed_but_input_unchanged.sum()))
    lw = a[(a.method == "LViT-T") & (a.condition == "C4")]
    for ds in DATASETS:
        r = lw[lw.dataset == ds].iloc[0]
        tag = "q" if ds == "QaTa-COV19" else "m"
        N[f"lvit_c4_window_same_{tag}"] = f"{int(r.rows - r.lvit_window_changed)} of {int(r.rows)}"
    # main table (Task 2.4)
    rows = []
    for ds in DATASETS:
        for m in METHODS:
            t = main[(main.dataset == ds) & (main.method == m)].set_index("condition")
            rows.append([ds, m + (" (batch-24 training)" if ds == "MosMedData+" and m == "LViT-T" else "")]
                        + [f"{pts(t.loc[c, 'dice_mean'], False)} ± {pts(t.loc[c, 'dice_sd'], False)}" for c in CONDS])
    T["main"] = md(pd.DataFrame(rows, columns=["Dataset", "Method"] + CONDS))
    # pairwise per dataset (Task 2.5)
    for ds in DATASETS:
        t = pw[pw.dataset == ds]
        rows = []
        for m in METHODS:
            for c in COMPS:
                r = t[(t.method == m) & (t.comparison == c)].iloc[0]
                rows.append([m, c, int(r.n_images), int(r.n_units),
                             f"{pts(r.unit_weighted_delta)} [{pts(r.ci95_low)}, {pts(r.ci95_high)}]",
                             pval(r.p_holm), "yes" if r["significant_holm_0.05"] else "no", pts(r.image_weighted_delta)])
        T[f"pairwise_{ds}"] = md(pd.DataFrame(rows, columns=["Method", "Comparison", "Images", "Units",
                                                             "Unit-weighted Δ [95% t-CI]", "Holm p", "Holm < 0.05",
                                                             "Image-weighted Δ"]))
    q = pw[pw.dataset == "QaTa-COV19"].set_index(["method", "comparison"])
    mo = pw[pw.dataset == "MosMedData+"].set_index(["method", "comparison"])
    rng = lambda t, c: f"{pts(t.xs(c, level='comparison').unit_weighted_delta.min())} to {pts(t.xs(c, level='comparison').unit_weighted_delta.max())}"  # noqa: E731
    for c in COMPS:
        N[f"q_{c}"] = rng(q, c)
        N[f"m_{c}"] = rng(mo, c)
    N["q_sig"] = str(int(pw[(pw.dataset == "QaTa-COV19")]["significant_holm_0.05"].sum()))
    N["m_sig"] = str(int(pw[(pw.dataset == "MosMedData+")]["significant_holm_0.05"].sum()))
    N["m_c1c6_neg"] = str(int((mo.xs("C1-C6", level="comparison").unit_weighted_delta < 0).sum()))
    # C2
    c2 = D["c2"]
    rows = []
    for ds in DATASETS:
        for m in METHODS:
            r = c2[(c2.dataset == ds) & (c2.method == m)].iloc[0]
            rows.append([ds, m, int(r.n_images), f"{pts(r.mean_abs_dDice, False)} ± {pts(r.mean_abs_dDice_sd, False)}",
                         f"{pts(r.mean_signed_dDice)} ± {pts(r.mean_signed_dDice_sd, False)}",
                         f"{100 * r.frac_mask_changed:.1f} ± {100 * r.frac_mask_changed_sd:.1f}",
                         "NA" if not np.isfinite(r.frac_mask_changed_gt4px) else f"{100 * r.frac_mask_changed_gt4px:.1f} ± {100 * r.frac_mask_changed_gt4px_sd:.1f}",
                         f"{int(r.empty_flips_total)} ({r.empty_flips_per_seed})"])
    T["c2"] = md(pd.DataFrame(rows, columns=["Dataset", "Method", "Images", "Mean |D(C0) − D(C2)| (points)",
                                             "Mean D(C0) − D(C2) (points)", "Images with a changed mask (%)",
                                             "Changed by > 4 px (%) (DD-CMD)", "Empty ↔ non-empty flips (seeds 1001/1002/1003)"]))
    N["c2_abs_range"] = f"{pts(c2.mean_abs_dDice.min(), False)} to {pts(c2.mean_abs_dDice.max(), False)}"
    qc = c2[c2.dataset == "QaTa-COV19"]
    N["c2_q_changed"] = f"{100 * qc.frac_mask_changed.min():.1f}–{100 * qc.frac_mask_changed.max():.1f}"
    mc = c2[c2.dataset == "MosMedData+"]
    N["c2_m_changed"] = f"{100 * mc.frac_mask_changed.min():.1f}–{100 * mc.frac_mask_changed.max():.1f}"
    dd = c2[c2.method == "DD-CMD"].set_index("dataset")
    N["c2_dd_gt4_q"] = f"{100 * dd.loc['QaTa-COV19'].frac_mask_changed_gt4px:.1f}"
    N["c2_dd_gt4_m"] = f"{100 * dd.loc['MosMedData+'].frac_mask_changed_gt4px:.1f}"
    N["c2_flips"] = str(int(c2.empty_flips_total.sum()))
    # severity (Task 2.6)
    sev = D["sev"]
    rows = []
    for r in sev.itertuples():
        rec = sev.loc[r.Index]
        rows.append([r.dataset, r.method, int(r.n_images_S), int(r.n_units_S), f"{100 * r.coverage_S:.1f}",
                     pts(r.mean_C0, False), pts(r.mean_C1, False), pts(r.mean_C4, False), pts(r.mean_C5, False),
                     pts(r.mean_C6, False)]
                    + [f"{pts(rec[f'image_delta_C0-{c}'])} / {pts(rec[f'unit_delta_C0-{c}'])} "
                       f"[{pts(rec[f'unit_ci95_low_C0-{c}'])}, {pts(rec[f'unit_ci95_high_C0-{c}'])}]" for c in ("C4", "C5", "C6")])
    T["severity"] = md(pd.DataFrame(rows, columns=["Dataset", "Method", "S images", "S units", "S coverage (%)",
                                                   "C0", "C1", "C4", "C5", "C6",
                                                   "C0−C4 image / unit [95% t-CI]", "C0−C5 image / unit [95% t-CI]",
                                                   "C0−C6 image / unit [95% t-CI]"]))
    N["S_same"] = "yes" if sev.S_same_as_method_eligible.all() else "no"
    for ds, tag in (("QaTa-COV19", "q"), ("MosMedData+", "m")):
        r = sev[sev.dataset == ds].iloc[0]
        N[f"S_{tag}"] = f"{int(r.n_images_S)} of {int(r.n_val_images)} images ({100 * r.coverage_S:.1f}%), {int(r.n_units_S)} units"
    qs = sev[sev.dataset == "QaTa-COV19"]
    N["q_c5_gt_c4"] = str(int((qs.mean_C5 > qs.mean_C4).sum()))
    att = D["att"].copy()
    att["mean_harm"] = att.mean_harm.map(lambda x: pts(x))
    T["attribute"] = md(att.rename(columns={"dataset": "Dataset", "method": "Method", "condition": "Condition",
                                            "changed_attributes": "Changed attributes", "n_donor_pairs": "Donor pairs",
                                            "n_targets": "Target images", "mean_harm": "Mean harm D(C0) − D(Cx) (points)"}))
    qa = D["att"][(D["att"].dataset == "QaTa-COV19") & (D["att"].condition == "C4")]
    loc = qa[qa.changed_attributes == "location"]
    N["q_c4_location_pairs"] = f"{int(loc.n_donor_pairs.iloc[0])} of {int(qa.groupby('method').n_donor_pairs.sum().iloc[0])}"
    N["q_c4_location_harm"] = f"{pts(loc.mean_harm.min())} to {pts(loc.mean_harm.max())}"
    # susceptibility (Task 2.7)
    sus = D["sus"]
    m_ = sus[sus.kind == "method"]
    T["sus_methods"] = md(pd.DataFrame([[r.dataset, r.method_a, int(r.n_images), int(r.n_units), pts(r.C0, False),
                                         pts(r["C0-C1"]), pts(r["C1-C6"]), pts(r.H)] for _, r in m_.iterrows()],
                                       columns=["Dataset", "Method", "Images", "Units", "C0", "C0−C1", "C1−C6",
                                                "H = C0−C6"]))
    p_ = sus[sus.kind == "pair"]
    T["sus_pairs"] = md(pd.DataFrame([[r.dataset, f"{r.method_a} − {r.method_b}", int(r.n_images), int(r.n_units),
                                       pts(r.image_delta_H), f"{pts(r.unit_delta_H)} [{pts(r.unit_ci95_low)}, {pts(r.unit_ci95_high)}]"]
                                      for _, r in p_.iterrows()],
                                     columns=["Dataset", "H_A − H_B", "Images", "Units", "Image-weighted",
                                              "Unit-weighted [95% t-CI]"]))
    qp = p_[p_.dataset == "QaTa-COV19"].set_index(["method_a", "method_b"])
    for (a_, b_), tag in ((("LViT-T", "RecLMIS"), "lr"), (("LViT-T", "DD-CMD"), "ld"), (("RecLMIS", "DD-CMD"), "rd")):
        r = qp.loc[(a_, b_)]
        N[f"sus_q_{tag}"] = f"{pts(r.unit_delta_H)} [{pts(r.unit_ci95_low)}, {pts(r.unit_ci95_high)}]"
    # Phase 3 rule (section 5.1)
    rule = D["rule"]
    rows = []
    for (ds, m, c), g in rule.groupby(["dataset", "method", "comparison"], sort=False):
        g = g.set_index("seed")
        rows.append([ds, m, c, f"{g.threshold.iloc[0] * 100:.0f}"]
                    + [f"{pts(g.loc[s, 'delta_img'])} / {pts(g.loc[s, 'delta_unit'])}" for s in (1001, 1002, 1003)]
                    + [g.classification.iloc[0]])
    T["phase3"] = md(pd.DataFrame(rows, columns=["Dataset", "Method", "Comparison", "Threshold (points)",
                                                 "Seed 1001 img / unit", "Seed 1002 img / unit", "Seed 1003 img / unit",
                                                 "Classification"]))
    cls = rule.drop_duplicates(["dataset", "method", "comparison"])
    N["n_repeated"] = str(int((cls.classification == "repeated signal").sum()))
    N["n_unstable"] = str(int((cls.classification == "unstable signal").sum()))
    N["n_none"] = str(int((cls.classification == "no signal").sum()))
    rep = cls[cls.classification == "repeated signal"]
    N["repeated_list"] = "; ".join(f"{r.dataset} {r.method} {r.comparison}" for r in rep.itertuples())
    md_ = rep.groupby(["dataset", "method"]).size()
    N["outcome"] = ("basis for Phase 3 (repeated signal in: " + ", ".join(f"{d} {m}" for d, m in md_.index) + ")"
                    if len(md_) else "no-go")
    # per-donor check
    dn = D["donor"]
    thr = {"C0-C5": 0.02, "C0-C6": 0.02, "C1-C5": 0.01, "C1-C6": 0.01}
    rows = []
    for (ds, m, c), g in dn.groupby(["dataset", "method", "comparison"], sort=False):
        k = cls[(cls.dataset == ds) & (cls.method == m) & (cls.comparison == c)].classification.iloc[0]
        rows.append([ds, m, c, k, f"{pts(g.delta_img.min())} to {pts(g.delta_img.max())}",
                     f"{int((g.delta_img >= thr[c]).sum())} of {len(g)}"])
    T["donor"] = md(pd.DataFrame(rows, columns=["Dataset", "Method", "Comparison", "Classification",
                                                "Per-donor image-weighted Δ (range over 3 seeds × 3 donors)",
                                                "Seed × donor cells at or above threshold"]))
    rd = dn.merge(cls[["dataset", "method", "comparison", "classification"]], on=["dataset", "method", "comparison"])
    rd = rd[rd.classification == "repeated signal"]
    N["donor_rep_min"] = pts(rd.delta_img.min())
    N["donor_rep_cells"] = f"{int(sum((r.delta_img >= thr[r.comparison]) for r in rd.itertuples()))} of {len(rd)}"
    # D6 (pre-specified sensitivity check)
    d6 = D["d6"].copy()
    d6["diff"] = (d6.unit_weighted_delta_without_d6 - d6.unit_weighted_delta_all).abs()
    T["d6"] = md(pd.DataFrame([[r.method, r.comparison, int(r.n_images_all), int(r.n_images_without_d6),
                                pts(r.image_weighted_delta_all), pts(r.image_weighted_delta_without_d6),
                                pts(r.unit_weighted_delta_all), pts(r.unit_weighted_delta_without_d6)] for r in d6.itertuples()],
                              columns=["Method", "Comparison", "Images (all)", "Images (without D6)", "Image Δ all",
                                       "Image Δ without D6", "Unit Δ all", "Unit Δ without D6"]))
    N["d6_excluded"] = str(int(d6.d6_images_excluded.iloc[0]))
    N["d6_maxdiff"] = pts(d6["diff"].max(), False)
    # appendix: supplementary main and pairwise metrics
    rows = []
    for ds in DATASETS:
        for m in METHODS:
            for c in CONDS:
                r = main[(main.dataset == ds) & (main.method == m) & (main.condition == c)].iloc[0]
                rows.append([ds, m, c, f"{pts(r.miou_mean, False)} ± {pts(r.miou_sd, False)}",
                             f"{r.hd95_mean:.2f} ± {r.hd95_sd:.2f}", f"{r.assd_mean:.2f} ± {r.assd_sd:.2f}",
                             f"{100 * r.empty_mean:.2f}", pts(r.dice_seed1001, False), pts(r.dice_seed1002, False),
                             pts(r.dice_seed1003, False)])
    T["app_main"] = md(pd.DataFrame(rows, columns=["Dataset", "Method", "Condition", "mIoU (%)", "HD95 (px)",
                                                   "ASSD (px)", "Empty (%)", "Dice s1001", "Dice s1002", "Dice s1003"]))
    rows = []
    for r in pw.itertuples():
        rows.append([r.dataset, r.method, r.comparison,
                     f"{pts(r.miou_unit_weighted_delta)} [{pts(r.miou_boot_ci95_low)}, {pts(r.miou_boot_ci95_high)}]",
                     f"{r.hd95_unit_weighted_delta:+.2f} [{r.hd95_boot_ci95_low:+.2f}, {r.hd95_boot_ci95_high:+.2f}]",
                     f"{r.assd_unit_weighted_delta:+.2f} [{r.assd_boot_ci95_low:+.2f}, {r.assd_boot_ci95_high:+.2f}]"])
    T["app_pw"] = md(pd.DataFrame(rows, columns=["Dataset", "Method", "Comparison", "ΔmIoU (points) [bootstrap CI]",
                                                 "ΔHD95 (px) [bootstrap CI]", "ΔASSD (px) [bootstrap CI]"]))
    # case selection
    sel = json.load(open(R("phase2_val_case_selection.json")))
    T["cases"] = md(pd.DataFrame([[ds, f"q{int(p['quantile'] * 100)}", f"`{p['sample_id']}`", pts(p["harm"])]
                                  for ds in DATASETS for p in sel[ds]],
                                 columns=["Dataset", "Position", "Sample id", "Harm C0−C6 (points, mean of 3 methods × 3 seeds × donors)"]))
    # checkpoints C0 (validation) per method, for section 2
    c0 = main[main.condition == "C0"]
    N["c0_q"] = ", ".join(f"{r.method} {pts(r.dice_mean, False)}" for r in c0[c0.dataset == "QaTa-COV19"].itertuples())
    N["c0_m"] = ", ".join(f"{r.method} {pts(r.dice_mean, False)}" for r in c0[c0.dataset == "MosMedData+"].itertuples())
    return T, N


def example_texts():
    """Condition texts of one QaTa validation target chosen by a fixed rule (see the module docstring)."""
    from phase2 import conditions as C
    man, _ = C.build("QaTa-COV19", "val")
    el = man[man.eligible]
    full = el.groupby("sample_id").condition.nunique()
    ids = sorted(full[full == 9].index, key=lambda s: hashlib.sha256(f"phase2-example-v1|{s}".encode()).hexdigest())
    sid = ids[0]
    g = el[el.sample_id == sid].set_index(["condition", "variant_id"])
    ex = {"ex_id": sid}
    for c in ["C0", "C1", "C2", "C3a", "C3b", "C3c"]:
        ex[f"ex_{c}"] = g.loc[(c, "0")].counterfactual_text or "(empty string)"
    for c in ["C4", "C5", "C6"]:
        r = g.loc[(c, "d1")]
        ex[f"ex_{c}"] = f"{r.counterfactual_text} (donor `{r.donor_sample_id}`; changed: {r.changed_attributes})"
    return ex


def main():
    D = load()
    T, N = tables(D)
    N.update(example_texts())
    text = open(TEMPLATE).read()
    for key in re.findall(r"\{\{T:(\w[\w+\-]*)\}\}", text):
        text = text.replace(f"{{{{T:{key}}}}}", T[key])
    for key in re.findall(r"\{\{N:(\w[\w+\-]*)\}\}", text):
        text = text.replace(f"{{{{N:{key}}}}}", N[key])
    left = re.findall(r"\{\{[TN]:[^}]*\}\}", text)
    if left:
        raise SystemExit(f"unfilled fields: {left}")
    with open(OUT, "w") as f:
        f.write(text)
    print("written:", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
