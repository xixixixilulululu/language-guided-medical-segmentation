"""Fill the generated tables of reports/PHASE2_RESULTS.md from the Phase 2 TEST result files (no text).

Tables (between <!-- BEGIN NAME --> and <!-- END NAME -->; a bare <!-- NAME --> is replaced on the first run):
  RECONCILE   results/phase2_test_c0_reconciliation.csv (P2-2)
  MAIN        results/phase2_main_results_test.csv: Dice mean +- SD over the 3 per-seed image means, in points
  PAIRWISE    results/phase2_pairwise_results_test.csv: the 11 comparisons, unit-weighted delta, 95% t-CI, Holm
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
    s = open(DOC).read()
    s = put(s, "RECONCILE", md(rec))
    s = put(s, "MAIN", "\n\n".join(parts))
    s = put(s, "PAIRWISE", "\n\n".join(pw))
    open(DOC, "w").write(s)
    print("tables written")


if __name__ == "__main__":
    main()
