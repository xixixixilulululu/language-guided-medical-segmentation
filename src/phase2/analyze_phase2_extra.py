"""Phase 2, additional VALIDATION analyses for the supervisor's summary (Tasks 2.6, 2.7, C2 change, figures).

Split: validation by default (draft stage: refused once the lock is open). --split test (after the freeze, Phase 2
lock open; draft test pipeline of reports/PHASE2_PROTOCOL_FREEZE.md) runs the same steps on the test files and
writes the same outputs with "test" in place of "val" in every file name. Existing results files and the outputs
of src/phase2/analyze_phase2.py are not changed. Inputs (validation names):
  results/phase2_val_per_case.csv.gz             per-case metrics of every condition pass (analyze_phase2.py)
  results/phase2_donor_manifest_val.csv          donor manifest without text (step A of this script)
  results/phase2_pairwise_results_val.csv        for figures 1 and 3
  runs/phase2/val/<method>/<dataset>/seed<k>/masks_<cond>_<variant>.npz   scored masks (C2 change, cases)
  runs/phase2/val/effective_input_<method>_<dataset>.csv                  effective-input audit of the benchmark
Aggregation as in analyze_phase2.py: metrics are averaged over donors within image and seed, then over seeds.
Units: patient / CT volume (D1, D2); unit-weighted means with a 95% t-CI over unit differences (Phase 0
paired_t). MosMed has 11 validation units: every MosMed interval is exploratory. Nothing here is a test.

Steps and outputs (results/, reports/figures/):
  A  phase2_donor_manifest_val.csv               dataset, target_id, condition, variant, donor_id, changed_attributes
  B  phase2_severity_results_val.csv             Task 2.6 on S = targets with C4, C5 and C6 donors
     phase2_severity_by_attribute_val.csv        harm by changed-attribute combination (descriptive)
  C  phase2_susceptibility_val.csv               Task 2.7 on the C6-eligible images
  D  phase2_c2_change_val.csv, phase2_c2_mask_change_val.csv.gz   C0 vs C2: |dDice|, mask change, flips
     phase2_effective_input_val_audit.csv        audit of the benchmark manifest: text changed / input unchanged
  E  reports/figures/phase2_val_fig1..fig4.png, phase2_val_cases_<dataset>.png, phase2_val_case_selection.json

Usage:  python src/phase2/analyze_phase2_extra.py [--split test]       (all steps, foreground, CPU)
Tests:  python src/phase2/test_analyze_phase2_extra.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

METHODS = ["lvit", "reclmis", "ddcmd"]
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
DATASETS = ["QaTa-COV19", "MosMedData+"]
SEEDS = [1001, 1002, 1003]
MISMATCH = ["C4", "C5", "C6"]
PAIRS = [("lvit", "reclmis"), ("lvit", "ddcmd"), ("reclmis", "ddcmd")]
# Phase 1 figure style (src/analyze_phase1.py): method colours, ink, grid, ground-truth contour
COLORS = {"lvit": "#8e44c9", "reclmis": "#d6407f", "ddcmd": "#8a7a12"}
MARKERS = {"lvit": "^", "reclmis": "v", "ddcmd": "P"}
INK, INK2, GRID, GT_COLOR = "#0b0b0b", "#52514e", "#d9d8d4", "#f2c200"
CASE_QUANTILES = [0.10, 0.30, 0.50, 0.70, 0.90]
CASE_SEED, CASE_DONOR = 1001, "d1"
SPLIT = "val"                                   # set by main(); "test" only with the Phase 2 lock open
RUNS = os.path.join(REPO, "runs", "phase2", SPLIT)
RES = os.path.join(REPO, "results")
FIG = os.path.join(REPO, "reports", "figures")


def S():
    """Phase 0 statistics (paired_t), imported unchanged."""
    import analyze_phase0
    return analyze_phase0


def unit_ci(unit_diffs):
    """Mean of unit differences and its 95% t-CI (NaN when fewer than 2 units or no variance)."""
    u = np.asarray(unit_diffs, float)
    if len(u) < 2 or u.std(ddof=1) == 0:
        return float(u.mean()) if len(u) else np.nan, np.nan, np.nan
    t = S().paired_t(u, np.zeros(len(u)))
    return t["delta"], t["ci_low"], t["ci_high"]


# ---------------------------------------------------------------------------------------------
# A. Donor manifest (no text)
# ---------------------------------------------------------------------------------------------
def donor_manifest():
    from phase2 import conditions as C
    rows = []
    for ds in DATASETS:
        man, _ = C.build(ds, SPLIT)
        d = man[man.condition.isin(MISMATCH) & man.eligible]
        rows.append(pd.DataFrame({"dataset": ds, "target_id": d.sample_id, "condition": d.condition,
                                  "variant": d.variant_id, "donor_id": d.donor_sample_id,
                                  "changed_attributes": d.changed_attributes}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------
def load_per_case(path):
    pc = pd.read_csv(path, dtype={"sample_id": str, "variant_id": str, "donor_sample_id": str}, keep_default_na=False)
    pc["empty"] = pc["empty"].astype(str).eq("True")
    return pc


def image_means(pc, metric="dice"):
    """{(dataset, method): DataFrame sample_id x condition}: donor mean within seed, then seed mean; + unit."""
    per_seed = pc.groupby(["dataset", "method", "seed", "condition", "sample_id", "unit"])[metric].mean()
    img = per_seed.groupby(["dataset", "method", "condition", "sample_id", "unit"]).mean().reset_index()
    return {k: g.pivot_table(index=["sample_id", "unit"], columns="condition", values=metric)
            for k, g in img.groupby(["dataset", "method"])}


def paired_on(w, a, b, ids):
    """Image differences a - b on the image set `ids`, and their unit means."""
    sub = w[w.index.get_level_values("sample_id").isin(ids)]
    d = (sub[a] - sub[b]).dropna()
    return d, d.groupby(level="unit").mean()


# ---------------------------------------------------------------------------------------------
# B. Task 2.6: mismatch extent on S
# ---------------------------------------------------------------------------------------------
def severity(pc, man):
    W = image_means(pc)
    rows = []
    for ds in DATASETS:
        m_ds = man[man.dataset == ds]
        sets = [set(m_ds[m_ds.condition == c].target_id) for c in MISMATCH]
        S_ = set.intersection(*sets)
        n_val = pc[(pc.dataset == ds) & (pc.condition == "C0")].sample_id.nunique()
        for m in METHODS:
            w = W.get((ds, m))
            if w is None:
                continue
            have = set.intersection(*[set(w[c].dropna().index.get_level_values("sample_id")) for c in MISMATCH])
            sub = w[w.index.get_level_values("sample_id").isin(S_)]
            r = {"dataset": ds, "method": NAME[m], "n_images_S": len(S_), "n_units_S": sub.index.get_level_values("unit").nunique(),
                 "n_val_images": n_val, "coverage_S": len(S_) / n_val, "S_same_as_method_eligible": have == S_}
            for c in ["C0", "C1", "C4", "C5", "C6"]:
                r[f"mean_{c}"] = float(sub[c].mean())
            for c in MISMATCH:
                d, u = paired_on(w, "C0", c, S_)
                r[f"image_delta_C0-{c}"] = float(d.mean())
                r[f"unit_delta_C0-{c}"], r[f"unit_ci95_low_C0-{c}"], r[f"unit_ci95_high_C0-{c}"] = unit_ci(u)
            rows.append(r)
    return pd.DataFrame(rows)


def severity_by_attribute(pc, man):
    """Harm D(C0) - D(Cx, donor) per (target, donor), seed-averaged, grouped by changed attributes on S."""
    rows = []
    for ds in DATASETS:
        m_ds = man[man.dataset == ds]
        S_ = set.intersection(*[set(m_ds[m_ds.condition == c].target_id) for c in MISMATCH])
        for m in METHODS:
            p = pc[(pc.dataset == ds) & (pc.method == m)]
            if p.empty:
                continue
            c0 = p[p.condition == "C0"].groupby("sample_id").dice.mean()                     # seed mean
            x = p[p.condition.isin(MISMATCH) & p.sample_id.isin(S_)]
            x = x.groupby(["condition", "variant_id", "sample_id"]).dice.mean().reset_index()  # seed mean per donor
            x = x.merge(m_ds.rename(columns={"target_id": "sample_id", "variant": "variant_id"}),
                        on=["condition", "variant_id", "sample_id"], how="left")
            if x.changed_attributes.isna().any():
                raise ValueError(f"{ds} {m}: per-case donor rows not in the manifest")
            x["harm"] = c0.reindex(x.sample_id).values - x.dice
            for (c, ch), g in x.groupby(["condition", "changed_attributes"]):
                per_target = g.groupby("sample_id").harm.mean()
                rows.append({"dataset": ds, "method": NAME[m], "condition": c, "changed_attributes": ch,
                             "n_donor_pairs": len(g), "n_targets": len(per_target),
                             "mean_harm": float(per_target.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# C. Task 2.7: susceptibility on the C6-eligible images
# ---------------------------------------------------------------------------------------------
def susceptibility(pc, man):
    W = image_means(pc)
    rows = []
    for ds in DATASETS:
        ids = set(man[(man.dataset == ds) & (man.condition == "C6")].target_id)
        H = {}
        for m in METHODS:
            w = W.get((ds, m))
            if w is None:
                continue
            sub = w[w.index.get_level_values("sample_id").isin(ids)]
            H[m] = (sub["C0"] - sub["C6"]).dropna()
            rows.append({"dataset": ds, "kind": "method", "method_a": NAME[m], "method_b": "", "n_images": len(H[m]),
                         "n_units": sub.index.get_level_values("unit").nunique(),
                         "C0": float(sub["C0"].mean()), "C0-C1": float((sub["C0"] - sub["C1"]).mean()),
                         "C1-C6": float((sub["C1"] - sub["C6"]).mean()), "H": float(H[m].mean())})
        for a, b in PAIRS:
            if a not in H or b not in H:
                continue
            d = (H[a] - H[b]).dropna()
            u = d.groupby(level="unit").mean()
            ud, lo, hi = unit_ci(u)
            rows.append({"dataset": ds, "kind": "pair", "method_a": NAME[a], "method_b": NAME[b], "n_images": len(d),
                         "n_units": len(u), "image_delta_H": float(d.mean()), "unit_delta_H": ud,
                         "unit_ci95_low": lo, "unit_ci95_high": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# D. C0 vs C2
# ---------------------------------------------------------------------------------------------
def unpack(npz, key):
    return np.unpackbits(npz[key])[:224 * 224].reshape(224, 224).astype(bool)


def c2_mask_change(runs=None):
    runs = runs or RUNS
    rows = []
    for m in METHODS:
        for ds in DATASETS:
            for s in SEEDS:
                d = os.path.join(runs, m, ds, f"seed{s}")
                a, b = np.load(os.path.join(d, "masks_C0_0.npz")), np.load(os.path.join(d, "masks_C2_0.npz"))
                for sid in b.files:
                    x, y = unpack(a, sid), unpack(b, sid)
                    rows.append({"dataset": ds, "method": NAME[m], "seed": s, "sample_id": sid,
                                 "pixels_changed": int((x != y).sum()), "empty_C0": not x.any(), "empty_C2": not y.any()})
    return pd.DataFrame(rows)


def c2_change(pc, mc):
    p = pc[pc.condition.isin(["C0", "C2"])].pivot_table(index=["dataset", "method", "seed", "sample_id"],
                                                        columns="condition", values="dice").dropna()
    p["abs_d"] = (p["C0"] - p["C2"]).abs()
    p["signed_d"] = p["C0"] - p["C2"]
    p = p.reset_index()
    p["method"] = p.method.map(NAME)
    rows = []
    for (ds, m), g in p.groupby(["dataset", "method"], sort=False):
        k = mc[(mc.dataset == ds) & (mc.method == m)]
        per_seed_abs = g.groupby("seed").abs_d.mean()
        per_seed_signed = g.groupby("seed").signed_d.mean()
        ch = k.groupby("seed").pixels_changed.apply(lambda x: (x >= 1).mean())
        gt4 = k.groupby("seed").pixels_changed.apply(lambda x: (x > 4).mean())
        flips = k.groupby("seed").apply(lambda x: int((x.empty_C0 != x.empty_C2).sum()))
        r = {"dataset": ds, "method": m, "n_images": g.sample_id.nunique(),
             "mean_abs_dDice": per_seed_abs.mean(), "mean_abs_dDice_sd": per_seed_abs.std(ddof=1),
             "mean_signed_dDice": per_seed_signed.mean(), "mean_signed_dDice_sd": per_seed_signed.std(ddof=1),
             "frac_mask_changed": ch.mean(), "frac_mask_changed_sd": ch.std(ddof=1),
             "frac_mask_changed_gt4px": gt4.mean() if m == "DD-CMD" else np.nan,
             "frac_mask_changed_gt4px_sd": gt4.std(ddof=1) if m == "DD-CMD" else np.nan,
             "empty_flips_total": int(flips.sum()), "empty_flips_per_seed": "/".join(str(int(flips[s])) for s in SEEDS)}
        for s in SEEDS:
            r[f"mean_abs_dDice_seed{s}"] = per_seed_abs[s]
            r[f"frac_mask_changed_seed{s}"] = ch[s]
        rows.append(r)
    return pd.DataFrame(rows)


def audit_summary(runs=None):
    runs = runs or RUNS
    rows = []
    for f in sorted(glob.glob(os.path.join(runs, "effective_input_*.csv"))):
        a = pd.read_csv(f)
        for (ds, c), g in a.groupby(["dataset", "condition"], sort=False):
            rows.append({"dataset": ds, "method": NAME[a.method.iloc[0]], "condition": c, "rows": len(g),
                         "raw_text_changed": int(g.raw_text_changed.sum()),
                         "raw_changed_but_input_unchanged": int((g.raw_text_changed & ~g.model_input_changed).sum()),
                         "lvit_window_changed": int(g.lvit_window_changed.sum()) if "lvit_window_changed" in g else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# E. Figures
# ---------------------------------------------------------------------------------------------
def _style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_effects(pw, comparisons, path, title, note):
    """Unit-weighted delta (Dice points) with 95% t-CI per comparison and method; one panel per dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(comparisons) + 1.9), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        t = pw[(pw.dataset == ds) & pw.comparison.isin(comparisons)]
        for k, m in enumerate(METHODS):
            r = t[t.method == NAME[m]].set_index("comparison").reindex(comparisons)
            y = np.arange(len(comparisons)) + (k - 1) * 0.22
            x = r.unit_weighted_delta * 100
            ax.errorbar(x, y, xerr=[x - r.ci95_low * 100, r.ci95_high * 100 - x], fmt=MARKERS[m], color=COLORS[m],
                        ms=7, lw=1.6, capsize=0, mec="white", mew=0.8, label=NAME[m])
        n = t.groupby("comparison").n_images.first().reindex(comparisons)
        ax.set_yticks(np.arange(len(comparisons)), [f"{c}  (n={int(v)})" for c, v in zip(comparisons, n)])
        ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=1)
        _style(ax)
        ax.set_xlabel("Unit-weighted Δ Dice (points), 95% t-CI", color=INK2, fontsize=9)
        ax.set_title(ds + ("  (exploratory: 11 units)" if ds == "MosMedData+" else ""), fontsize=10, color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=9, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_extent(sev, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ds in zip(axes, DATASETS):
        t = sev[sev.dataset == ds]
        for m in METHODS:
            r = t[t.method == NAME[m]].iloc[0]
            ax.plot([1, 2, 3], [r.mean_C4 * 100, r.mean_C5 * 100, r.mean_C6 * 100], marker=MARKERS[m], ms=8,
                    color=COLORS[m], lw=2, mec="white", mew=0.8, label=NAME[m])
            ax.axhline(r.mean_C0 * 100, color=COLORS[m], ls="--", lw=1.2)
            ax.axhline(r.mean_C1 * 100, color=COLORS[m], ls=":", lw=1.4)
        ax.set_xticks([1, 2, 3], ["1 attribute\n(C4)", "2 attributes\n(C5)", "3 attributes\n(C6)"])
        _style(ax, "y")
        ax.set_ylabel("Mean Dice (%) on S", color=INK2, fontsize=9)
        ax.set_title(f"{ds}  (S = {int(t.n_images_S.iloc[0])} images, {int(t.n_units_S.iloc[0])} units)"
                     + ("  exploratory" if ds == "MosMedData+" else ""), fontsize=10, color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=9, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 0.995))
    fig.text(0.01, 0.01, "Lines: mean Dice of the mismatched report (donor and seed averaged). Dashed: C0 of the "
             "same method; dotted: C1. Attribute distance is not a validated severity scale.", fontsize=8, color=INK2)
    fig.suptitle("Mismatch extent on the common set S (C4, C5 and C6 all available)", x=0.01, ha="left", fontsize=11,
                 color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def unit_effects(pc, man):
    """Per-unit C0-C1 (all images) and C0-C6 (C6-eligible images), donor and seed averaged."""
    W = image_means(pc)
    rows = []
    for (ds, m), w in W.items():
        ids6 = set(man[(man.dataset == ds) & (man.condition == "C6")].target_id)
        for comp, ids in (("C0-C1", set(w.index.get_level_values("sample_id"))), ("C0-C6", ids6)):
            a, b = comp.split("-")
            _, u = paired_on(w, a, b, ids)
            rows += [{"dataset": ds, "method": NAME[m], "comparison": comp, "unit": k, "delta": v} for k, v in u.items()]
    return pd.DataFrame(rows)


def fig_units(ue, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, DATASETS):
        pos = 0
        ticks, labels = [], []
        for comp in ("C0-C1", "C0-C6"):
            for m in METHODS:
                v = ue[(ue.dataset == ds) & (ue.method == NAME[m]) & (ue.comparison == comp)].delta.values * 100
                if ds == "QaTa-COV19":
                    ax.boxplot(v, positions=[pos], widths=0.6, showfliers=False, medianprops=dict(color=INK),
                               boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
                    ax.scatter(pos + rng.uniform(-0.22, 0.22, len(v)), v, s=4, color=COLORS[m], alpha=0.35, lw=0)
                else:
                    ax.scatter(pos + rng.uniform(-0.15, 0.15, len(v)), v, s=36, color=COLORS[m], marker=MARKERS[m],
                               edgecolors="white", linewidths=0.8, zorder=3)
                ticks.append(pos)
                labels.append(f"{NAME[m]}\n{comp}")
                pos += 1
            pos += 0.6
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(ticks, labels, fontsize=8)
        _style(ax, "y")
        ax.set_ylabel("Δ Dice per unit (points)", color=INK2, fontsize=9)
        n = ue[ue.dataset == ds].groupby("comparison").unit.nunique()
        units = f"C0−C1: {n.get('C0-C1', 0)} units, C0−C6: {n.get('C0-C6', 0)} units"
        ax.set_title(f"{ds}\n({units}" + ("; all shown, exploratory)" if ds == "MosMedData+"
                                          else "; box: quartiles, whiskers 1.5 IQR)"), fontsize=10, color=INK)
    fig.suptitle("Per-unit effects: C0−C1 (all images) and C0−C6 (C6-eligible images)", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def select_cases(pc, man, dataset):
    """C6-eligible images sorted by harm C0-C6 (mean over methods, seeds, donors); quantile positions
    round(q (n - 1)); ties broken by sample id."""
    p = pc[pc.dataset == dataset]
    ids = set(man[(man.dataset == dataset) & (man.condition == "C6")].target_id)
    c0 = p[p.condition == "C0"].groupby(["method", "seed", "sample_id"]).dice.mean()
    c6 = p[(p.condition == "C6") & p.sample_id.isin(ids)].groupby(["method", "seed", "sample_id"]).dice.mean()
    harm = (c0.reindex(c6.index) - c6).groupby("sample_id").mean()
    order = harm.reset_index().sort_values(["dice", "sample_id"]).reset_index(drop=True)
    return [{"quantile": q, "sample_id": order.sample_id[int(round(q * (len(order) - 1)))],
             "harm": float(order.dice[int(round(q * (len(order) - 1)))])} for q in CASE_QUANTILES]


def fig_cases(pc, man, dataset, picks, path, texts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lgms import data
    cols = ["Image + GT"] + [f"{NAME[m]}\n{c}" for m in METHODS for c in ("C0", "C1", "C6 (d1)")]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(1.55 * len(cols), 2.45 * len(picks)))
    d1 = man[(man.dataset == dataset) & (man.condition == "C6") & (man.variant == CASE_DONOR)].set_index("target_id")
    p = pc[(pc.dataset == dataset) & (pc.seed == CASE_SEED)].set_index(["method", "condition", "variant_id", "sample_id"]).dice
    masks = {(m, c, v): np.load(os.path.join(RUNS, m, dataset, f"seed{CASE_SEED}", f"masks_{c}_{v}.npz"))
             for m in METHODS for c, v in (("C0", "0"), ("C1", "0"), ("C6", CASE_DONOR))}
    for r, pk in enumerate(picks):
        sid = pk["sample_id"]
        img = data.read_image(dataset, SPLIT, sid)
        gt = data.read_mask(dataset, SPLIT, sid) > 0
        for c in range(len(cols)):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.contour(gt.astype(float), levels=[0.5], colors=[GT_COLOR], linestyles="--", linewidths=1.1)
            if c > 0:
                m = METHODS[(c - 1) // 3]
                cond, var = [("C0", "0"), ("C1", "0"), ("C6", CASE_DONOR)][(c - 1) % 3]
                pred = unpack(masks[(m, cond, var)], sid)
                ov = np.zeros((224, 224, 4))
                rgb = matplotlib.colors.to_rgb(COLORS[m])
                ov[pred] = (*rgb, 0.45)
                ax.imshow(ov)
                ax.text(3, 220, f"Dice {p[(m, cond, var, sid)] * 100:.1f}", color="white", fontsize=7,
                        va="bottom", bbox=dict(facecolor="black", alpha=0.55, pad=1.2, linewidth=0))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=8, color=INK)
        axes[r, 0].set_ylabel(f"q{int(pk['quantile'] * 100)}\nharm {pk['harm'] * 100:.1f}", fontsize=8, color=INK)
        donor = d1.loc[sid].donor_id
        axes[r, 0].text(0, -0.04, f"{sid} | C0: {texts[sid]}\nC6 donor d1 ({donor}): {texts[donor]}",
                        transform=axes[r, 0].transAxes, va="top", ha="left", fontsize=6.8, color=INK2)
    fig.suptitle(f"{dataset} validation: cases at the 10/30/50/70/90% positions of the C0−C6 harm (mean over 3 methods, "
                 f"3 seeds, donors); drawn: seed {CASE_SEED}, donor {CASE_DONOR}; GT yellow dashed",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.94, bottom=0.04, hspace=0.38, wspace=0.04)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    global SPLIT, RUNS
    SPLIT = ap.parse_args().split
    RUNS = os.path.join(REPO, "runs", "phase2", SPLIT)
    from phase2 import lock
    if SPLIT == "test":
        lock.require_open(what="Phase 2 test analysis (extra)")
    else:
        ok, _, _ = lock.check(fetch=False)
        if ok:                                 # the validation run is for the draft stage only
            sys.exit("Phase 2 lock is open: the validation run is not meant to run after the freeze")
    sp, label = SPLIT, {"val": "validation", "test": "test"}[SPLIT]
    man = donor_manifest()
    man.to_csv(os.path.join(RES, f"phase2_donor_manifest_{sp}.csv"), index=False)
    pc = load_per_case(os.path.join(RES, f"phase2_{sp}_per_case.csv.gz"))
    man = pd.read_csv(os.path.join(RES, f"phase2_donor_manifest_{sp}.csv"), dtype=str)
    sev = severity(pc, man)
    sev.to_csv(os.path.join(RES, f"phase2_severity_results_{sp}.csv"), index=False, float_format="%.6g")
    severity_by_attribute(pc, man).to_csv(os.path.join(RES, f"phase2_severity_by_attribute_{sp}.csv"), index=False,
                                          float_format="%.6g")
    susceptibility(pc, man).to_csv(os.path.join(RES, f"phase2_susceptibility_{sp}.csv"), index=False, float_format="%.6g")
    mc = c2_mask_change()
    mc.to_csv(os.path.join(RES, f"phase2_c2_mask_change_{sp}.csv.gz"), index=False)
    c2_change(pc, mc).to_csv(os.path.join(RES, f"phase2_c2_change_{sp}.csv"), index=False, float_format="%.6g")
    audit_summary().to_csv(os.path.join(RES, f"phase2_effective_input_{sp}_audit.csv"), index=False)
    pw = pd.read_csv(os.path.join(RES, f"phase2_pairwise_results_{sp}.csv"))
    fig_effects(pw, ["C0-C1", "C0-C2", "C0-C3a", "C0-C3b", "C0-C3c"], os.path.join(FIG, f"phase2_{sp}_fig1_c0_c1_c3.png"),
                "Null, equivalent and incomplete text: unit-weighted C0 − Cx (" + label + ")",
                "Positive = lower Dice under the changed text. n = images in the paired set. Units: patients (QaTa), "
                "CT volumes (MosMed).")
    fig_extent(sev, os.path.join(FIG, f"phase2_{sp}_fig2_extent.png"))
    fig_effects(pw, ["C1-C4", "C1-C5", "C1-C6"], os.path.join(FIG, f"phase2_{sp}_fig3_wrong_vs_null.png"),
                "Wrong text versus no text: unit-weighted C1 − Cx (" + label + ")",
                "Positive = the mismatched report gives lower Dice than the empty text (C1).")
    fig_units(unit_effects(pc, man), os.path.join(FIG, f"phase2_{sp}_fig4_units.png"))
    from phase2 import texts as T
    sel = {}
    for ds in DATASETS:
        sel[ds] = select_cases(pc, man, ds)
        tx = T.read_texts(ds, SPLIT)
        fig_cases(pc, man, ds, sel[ds], os.path.join(FIG, f"phase2_{sp}_cases_{ds.replace('+', 'plus')}.png"), tx)
    with open(os.path.join(RES, f"phase2_{sp}_case_selection.json"), "w") as f:
        json.dump(sel, f, indent=2)
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
