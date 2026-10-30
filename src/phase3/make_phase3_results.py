"""Fill the generated tables of reports/PHASE3_RESULTS.md from the Phase 3 TEST result files (descriptive only).

Tables (between <!-- BEGIN NAME --> and <!-- END NAME -->; a bare <!-- NAME --> is replaced on the first run):
  QATA       Task 3.5, QaTa test: image-weighted Dice of LANG / IMG / D-Gate / LR-Gate / NULL (C1) / ORACLE and the
             image-weighted fallback rates, per method x condition (results/phase3_test_results.csv)
  RECOVERY   C4-C6: share of the ORACLE headroom recovered, (gate - LANG) / (ORACLE - LANG), from the same table;
             IMG - gate for reference. Descriptive, no test.
  IMGREF     IMG alone as reference: IMG - LANG, IMG - D-Gate, IMG - LR-Gate (image-weighted)
  PRIMARY    Task 3.7 primary and secondary families (results/phase3_pairwise_results.csv)
  MARGIN     C0 / C2 gate - LANG with the 95% CI and the -0.005 margin; validation GATE_CAL / GATE_FIT for comparison
             (results/phase3_pairwise_results_val.csv)
  C3         C1 and C3a-c: gate - LANG and LR-Gate - D-Gate (descriptive)
  ROUTING    beneficial / harmful switches and the net gain (results/phase3_test_routing_quality.csv)
  MOSMED     MosMedData+ exploratory: Dice, ORACLE headroom, fallback rates, groups
  COST       Task 3.8, job 391 (results/phase3_computational_cost.csv)
  LRCOEF     LR coefficients per model x seed (results/phase3_lr_artifacts.json)
Usage:  python src/phase3/make_phase3_results.py
"""
import json
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = lambda f: os.path.join(REPO, "results", f)  # noqa: E731
DOC = os.path.join(REPO, "reports", "PHASE3_RESULTS.md")
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
METHODS = ["LViT-T", "RecLMIS", "DD-CMD"]
CONDS = ["C0", "C1", "C2", "C3a", "C3b", "C3c", "C4", "C5", "C6"]
QATA = "QaTa-COV19"
f2 = "{:.2f}".format
sg = "{:+.2f}".format


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


def results():
    T = pd.read_csv(R("phase3_test_results.csv"))
    T["method"] = T.method.map(NAME)
    return T


def wide(T, ds):
    """{(method, condition): {strategy: dice, 'sw_<strategy>': fallback rate}} in points / percent."""
    out = {}
    for (m, c), g in T[T.dataset == ds].groupby(["method", "condition"]):
        g = g.set_index("strategy")
        out[(m, c)] = {**{s: 100 * g.dice[s] for s in g.index},
                       **{f"sw_{s}": 100 * g.fallback_rate[s] for s in ("D-Gate", "LR-Gate", "ORACLE")},
                       "n_images": int(g.n_images["LANG"]), "n_groups": int(g.n_groups["LANG"])}
    return out


def t_qata(W):
    rows = []
    for m in METHODS:
        for c in CONDS:
            w = W[(m, c)]
            rows.append({"Method": m, "Condition": c, "Images": w["n_images"], "LANG": f2(w["LANG"]),
                         "IMG": f2(w["IMG"]), "D-Gate": f2(w["D-Gate"]), "LR-Gate": f2(w["LR-Gate"]),
                         "NULL (C1)": f2(w["NULL (C1)"]), "ORACLE": f2(w["ORACLE"]),
                         "D-Gate switch %": f"{w['sw_D-Gate']:.1f}", "LR-Gate switch %": f"{w['sw_LR-Gate']:.1f}",
                         "ORACLE switch %": f"{w['sw_ORACLE']:.1f}"})
    return md(pd.DataFrame(rows))


def recovery(W):
    rows = []
    for m in METHODS:
        for c in ("C4", "C5", "C6"):
            w = W[(m, c)]
            head = w["ORACLE"] - w["LANG"]
            rows.append({"Method": m, "Condition": c, "ORACLE − LANG": f2(head),
                         "D-Gate − LANG": sg(w["D-Gate"] - w["LANG"]), "LR-Gate − LANG": sg(w["LR-Gate"] - w["LANG"]),
                         "D-Gate recovered": f"{100 * (w['D-Gate'] - w['LANG']) / head:.1f}%",
                         "LR-Gate recovered": f"{100 * (w['LR-Gate'] - w['LANG']) / head:.1f}%",
                         "IMG − LR-Gate": sg(w["IMG"] - w["LR-Gate"])})
    return md(pd.DataFrame(rows))


def imgref(W):
    rows = []
    for m in METHODS:
        for c in ("C0", "C2", "C1", "C4", "C5", "C6"):
            w = W[(m, c)]
            rows.append({"Method": m, "Condition": c, "IMG": f2(w["IMG"]), "IMG − LANG": sg(w["IMG"] - w["LANG"]),
                         "IMG − D-Gate": sg(w["IMG"] - w["D-Gate"]), "IMG − LR-Gate": sg(w["IMG"] - w["LR-Gate"])})
    return md(pd.DataFrame(rows))


def ci(r):
    return f"{100 * r.delta:+.2f} [{100 * r.ci95_low:+.2f}, {100 * r.ci95_high:+.2f}]"


def primary(P):
    q = P[(P.dataset == QATA) & P.family.isin(["primary", "secondary"])]
    rows = [{"Family": r.family, "Comparison": r.comparison, "Method": r.method, "Condition": r.condition,
             "Images": r.n_images, "Groups": r.n_groups, "Δ [95% CI]": ci(r), "p raw": f"{r.p_raw:.2g}",
             "p Holm": f"{r.p_holm:.2g}", "Holm < 0.05": bool(r.p_holm < 0.05)} for r in q.itertuples()]
    return md(pd.DataFrame(rows))


def margin(P, V):
    q = P[(P.dataset == QATA) & P.margin.notna()]
    v = V[(V.dataset == QATA) & V.margin.notna()].set_index(["part", "comparison", "method", "condition"]).delta
    rows = [{"Comparison": r.comparison, "Method": r.method, "Condition": r.condition,
             "Test Δ [95% CI]": ci(r), "CI lower bound ≥ −0.50": bool(r.ci95_low >= -0.005),
             "GATE_CAL Δ": sg(100 * v[("GATE_CAL", r.comparison, r.method, r.condition)]),
             "GATE_FIT Δ": sg(100 * v[("GATE_FIT", r.comparison, r.method, r.condition)])} for r in q.itertuples()]
    return md(pd.DataFrame(rows))


def c3(P):
    q = P[(P.dataset == QATA) & P.condition.isin(["C1", "C3a", "C3b", "C3c"])]
    rows = []
    for (m, c), g in q.groupby(["method", "condition"], sort=False):
        g = g.set_index("comparison")
        rows.append({"Method": m, "Condition": c, "D-Gate − LANG": ci(g.loc["D-Gate - LANG"]),
                     "LR-Gate − LANG": ci(g.loc["LR-Gate - LANG"]), "LR-Gate − D-Gate": ci(g.loc["LR-Gate - D-Gate"])})
    rows.sort(key=lambda r: (METHODS.index(r["Method"]), ["C1", "C3a", "C3b", "C3c"].index(r["Condition"])))
    return md(pd.DataFrame(rows))


def routing():
    Q = pd.read_csv(R("phase3_test_routing_quality.csv"))
    Q = Q[Q.dataset == QATA].copy()
    Q["method"] = Q.method.map(NAME)
    rows = []
    for gate in ("D-Gate", "LR-Gate"):
        for m in METHODS:
            for c in CONDS:
                r = Q[(Q.gate == gate) & (Q.method == m) & (Q.condition == c)].iloc[0]
                rows.append({"Gate": gate, "Method": m, "Condition": c, "Rows": r.n_rows, "Switched": r.n_switched,
                             "Switch % (rows)": f"{100 * r.switch_rate:.2f}", "Beneficial": r.n_beneficial,
                             "Harmful": r.n_harmful, "Neutral": r.n_neutral,
                             "Beneficial % of switched": f"{100 * r.pct_beneficial_of_switched:.1f}",
                             "Harmful % of switched": f"{100 * r.pct_harmful_of_switched:.1f}",
                             "Net gain (image-weighted)": sg(100 * r.net_dice_gain_image_weighted)})
    return md(pd.DataFrame(rows))


def mosmed(W, P):
    q = P[P.dataset == "MosMedData+"].set_index(["comparison", "method", "condition"])
    rows = []
    for m in METHODS:
        for c in CONDS:
            w = W[(m, c)]
            rows.append({"Method": m, "Condition": c, "Images": w["n_images"], "Units": w["n_groups"],
                         "LANG": f2(w["LANG"]), "IMG": f2(w["IMG"]), "D-Gate": f2(w["D-Gate"]),
                         "LR-Gate": f2(w["LR-Gate"]), "ORACLE − LANG": f2(w["ORACLE"] - w["LANG"]),
                         "D switch %": f"{w['sw_D-Gate']:.2f}", "LR switch %": f"{w['sw_LR-Gate']:.2f}",
                         "LR-Gate − LANG (units) [95% CI]": ci(q.loc[("LR-Gate - LANG", m, c)])})
    return md(pd.DataFrame(rows))


def cost():
    C = pd.read_csv(R("phase3_computational_cost.csv"))
    rows = [{"Method": r.method, "Mode": r.mode, "Mean ms": f"{r.mean_ms:.2f}", "Median ms": f"{r.median_ms:.2f}",
             "p95 ms": f"{r.p95_ms:.2f}", "Max allocated MB": f"{r.max_allocated_mb:.0f}",
             "Max reserved MB": f"{r.max_reserved_mb:.0f}"} for r in C.itertuples()]
    return md(pd.DataFrame(rows))


def lrcoef():
    a = json.load(open(R("phase3_lr_artifacts.json")))
    rows = []
    for m in METHODS:
        for s in (1001, 1002, 1003):
            x = a[f"{m}_seed{s}"]
            rows.append({"Model × seed": f"{m} {s}", **{f"F{i + 1}": f"{v:+.3f}" for i, v in enumerate(x["coef"])},
                         "Intercept": f"{x['intercept']:+.3f}"})
    return md(pd.DataFrame(rows))


def main():
    T = results()
    P = pd.read_csv(R("phase3_pairwise_results.csv"))
    V = pd.read_csv(R("phase3_pairwise_results_val.csv"))
    Wq, Wm = wide(T, QATA), wide(T, "MosMedData+")
    s = open(DOC).read()
    for name, t in (("QATA", t_qata(Wq)), ("RECOVERY", recovery(Wq)), ("IMGREF", imgref(Wq)), ("PRIMARY", primary(P)),
                    ("MARGIN", margin(P, V)), ("C3", c3(P)), ("ROUTING", routing()), ("MOSMED", mosmed(Wm, P)),
                    ("COST", cost()), ("LRCOEF", lrcoef())):
        s = put(s, name, t)
    open(DOC, "w").write(s)
    print("tables written")


if __name__ == "__main__":
    main()
