"""EMBC paper assets: Fig. 1, Fig. 2 and Table II from the frozen result files (no model is run, no new analysis).

Every number is read from committed CSVs of QaTa-COV19 test (HEAD a8eb1ed). Numeric fields are read as strings and
converted with float() (pandas' default float parser was found to be 1 ulp off in Phase 5, read_taus()).

Fig. 1  Effect of unreliable text on overlap and boundary quality (results/phase2_pairwise_results_test.csv,
        dataset == "QaTa-COV19", comparisons C0-C1, C0-C3c, C0-C4, C0-C5, C0-C6; LViT-T, RecLMIS, DD-CMD).
        Paired patient-weighted differences: image differences C0 - condition, averaged within patient, patients
        equally weighted (every QaTa test image has unit type "patient"; C0-C1: 2113 images / 474 patients, the
        other comparisons: 2081 images / 472 patients).
        (a) Dice drop in points = 100 * unit_weighted_delta (C0 Dice - condition Dice; positive = worse), error
            bars = 100 * [ci95_low, ci95_high] (95% t-interval over patients).
        (b) HD95 increase in px = -hd95_unit_weighted_delta (the file stores C0 - condition; the figure shows
            condition HD95 - C0 HD95; positive = worse), error bars = [-hd95_boot_ci95_high, -hd95_boot_ci95_low]
            (95% paired patient bootstrap interval, 10,000 resamples, Phase 2).
Fig. 2  Mixed-reliability trade-off (results/phase5_mixed_reliability.csv): E(p) = (1 - p) D(C0) + p D_wrong,
        D_wrong = mean of D(C4), D(C5), D(C6), group-weighted on the common C4/C5/C6 targets, for LANG, IMG (nnU-Net)
        and Fixed Average; y = 100 * expected_dice at p = 0, 0.01, ..., 1. The shaded band is the range of p where
        Fixed Average is above BOTH LANG and IMG (not above every Phase 5 strategy: e.g. for RecLMIS SDC-Fallback
        is higher than Fixed Average near p = 0.11-0.12). Its ends are nnunet_p_lo / nnunet_p_hi of
        results/phase5_part2_backbone_consistency.csv; they are recomputed here in closed form from D_C0 and D_wrong
        and must agree within 1e-6. Both files store 10 significant digits, so differences of order 1e-9 are
        rounding; the check is there to catch a wrong row or column (which would differ by more than 1e-3), and the
        paper shows p to 3 decimals. results/phase5_crossover_summary.csv is not used.
Table II Main mitigation strategies (results/phase5_test_results.csv): LANG, IMG, LR-Gate, SDC-Fallback, Fixed
        Average per model; C0 Dice, C6 Dice, C6 HD95 (px), C6 empty rate (%). Image-weighted means (donors within
        image, then images per seed), averaged over the three seeds. Dice and empty * 100; 2 decimals.

paper/paper_numbers.csv has one row per number used in a figure or the table: raw_value is the string in the source
CSV, transform what was applied, displayed_value the value at the precision used in the paper (the figures plot the
unrounded transformed value). For Fig. 2 only the curve end points (p = 0 and p = 1) and the band ends are listed.

Usage (login node is fine: reads small CSVs and draws two figures):  python src/paper/make_paper_assets.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------------------------- layout parameters
FIG_WIDTH_IN = 3.487          # IEEEtran conference \columnwidth = 21 pc = 252 pt = 3.487 in
FIG1_HEIGHT_IN = 1.95
FIG2_HEIGHT_IN = 2.0
DPI_PNG = 300
FONT_PT = 7                   # smallest text at final size
LABEL_PT = 7.5

# ---------------------------------------------------------------------------------------------- inputs and outputs
DATASET = "QaTa-COV19"
SRC_PAIR = "results/phase2_pairwise_results_test.csv"
SRC_P5 = "results/phase5_test_results.csv"
SRC_MIX = "results/phase5_mixed_reliability.csv"
SRC_CONS = "results/phase5_part2_backbone_consistency.csv"
OUT_FIG1 = "paper/figures/fig1_text_reliability"
OUT_FIG2 = "paper/figures/fig2_mixed_reliability"
OUT_TAB2 = "paper/tables/table2_mitigation_tabular.tex"
OUT_NUM = "paper/paper_numbers.csv"

MODELS = ["LViT-T", "RecLMIS", "DD-CMD"]
FIG1_CONDS = ["C1", "C3c", "C4", "C5", "C6"]
TAB2_STRATS = ["LANG", "IMG", "LR-Gate", "SDC-Fallback", "Fixed Average"]
FIG2_STRATS = ["LANG", "IMG", "Fixed Average"]
RANGE_TOL = 1e-6

# Okabe-Ito colours. Fig. 1 models and Fig. 2 strategies use disjoint colours on purpose.
MODEL_STYLE = {"LViT-T": dict(color="#0072B2", marker="o"),
               "RecLMIS": dict(color="#D55E00", marker="s"),
               "DD-CMD": dict(color="#009E73", marker="^")}
STRAT_STYLE = {"Fixed Average": dict(color="#000000", linestyle="-", linewidth=1.8),
               "LANG": dict(color="#E69F00", linestyle="--", linewidth=1.3),
               "IMG": dict(color="#CC79A7", linestyle=":", linewidth=1.4)}
BAND_COLOR, BAND_ALPHA = "#BBBBBB", 0.4
OFFSET = 0.22                 # horizontal offset of the three models within a condition

NUM_COLS = ["item", "method", "condition_or_comparison", "strategy", "quantity", "source_file", "source_column",
            "raw_value", "transform", "displayed_value"]


def style():
    """Serif (STIX) text, TrueType (Type 42) fonts in PDF/PS, sizes at final print size."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": FONT_PT, "axes.labelsize": LABEL_PT, "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.markersize": 3.5,
    })


def read_rows(rel):
    """All rows of a CSV as dicts of raw strings (no float parsing by pandas)."""
    with open(os.path.join(REPO, rel), newline="") as f:
        return list(csv.DictReader(f))


def one(rows, **key):
    """The single row whose fields equal key (stops if there is not exactly one)."""
    hit = [r for r in rows if all(r[k] == v for k, v in key.items())]
    if len(hit) != 1:
        raise SystemExit(f"expected exactly one row for {key}, found {len(hit)}")
    return hit[0]


def num(item, method, cond, strat, quantity, src, col, raw, transform, shown):
    return dict(zip(NUM_COLS, [item, method, cond, strat, quantity, src, col, raw, transform, shown]))


# ---------------------------------------------------------------------------------------------- data
def fig1_data():
    """{(model, cond): {'dice': (v, lo, hi), 'hd95': (v, lo, hi)}} in display units, and the number records."""
    rows = [r for r in read_rows(SRC_PAIR) if r["dataset"] == DATASET]
    out, recs = {}, []
    for m in MODELS:
        for c in FIG1_CONDS:
            comp = f"C0-{c}"
            r = one(rows, method=m, comparison=comp)
            dice = tuple(100 * float(r[k]) for k in ("unit_weighted_delta", "ci95_low", "ci95_high"))
            hd = (-float(r["hd95_unit_weighted_delta"]), -float(r["hd95_boot_ci95_high"]), -float(r["hd95_boot_ci95_low"]))
            if not (dice[1] <= dice[0] <= dice[2] and hd[1] <= hd[0] <= hd[2]):
                raise SystemExit(f"interval does not contain the estimate: {m} {comp}")
            out[(m, c)] = {"dice": dice, "hd95": hd, "n": (r["n_images"], r["n_units"])}
            for q, col, v in zip(("dice_drop", "dice_drop_ci95_low", "dice_drop_ci95_high"),
                                 ("unit_weighted_delta", "ci95_low", "ci95_high"), dice):
                recs.append(num("fig1a", m, comp, "LANG", q, SRC_PAIR, col, r[col], "*100", f"{v:.2f}"))
            for q, col, v in zip(("hd95_increase", "hd95_increase_ci95_low", "hd95_increase_ci95_high"),
                                 ("hd95_unit_weighted_delta", "hd95_boot_ci95_high", "hd95_boot_ci95_low"), hd):
                recs.append(num("fig1b", m, comp, "LANG", q, SRC_PAIR, col, r[col], "negate", f"{v:.2f}"))
    return out, recs


def better_interval(dc, dw):
    """Closed-form {p in [0, 1]: dc + p (dw - dc) > 0} as (lo, hi), or None (as src/phase5/core.py)."""
    slope = dw - dc
    if slope == 0:
        return (0.0, 1.0) if dc > 0 else None
    p = -dc / slope
    lo, hi = (max(0.0, p), 1.0) if slope > 0 else (0.0, min(1.0, p))
    return (lo, hi) if lo < hi else None


def fig2_data():
    """Curves (p, 100 * E(p)) per model x strategy, the shaded band per model, and the number records."""
    rows = read_rows(SRC_MIX)
    cons = read_rows(SRC_CONS)
    curves, bands, recs = {}, {}, []
    for m in MODELS:
        d = {}
        for s in FIG2_STRATS:
            rs = sorted((r for r in rows if r["method"] == m and r["strategy"] == s), key=lambda r: float(r["p"]))
            if len(rs) != 101 or [round(float(r["p"]), 2) for r in rs] != [i / 100 for i in range(101)]:
                raise SystemExit(f"{m} {s}: expected p = 0, 0.01, ..., 1")
            if len({(r["D_C0"], r["D_wrong"]) for r in rs}) != 1:
                raise SystemExit(f"{m} {s}: D_C0 / D_wrong not constant")
            curves[(m, s)] = ([float(r["p"]) for r in rs], [100 * float(r["expected_dice"]) for r in rs])
            d[s] = (float(rs[0]["D_C0"]), float(rs[0]["D_wrong"]))
            for r in (rs[0], rs[-1]):
                v = 100 * float(r["expected_dice"])
                recs.append(num("fig2_endpoint", m, f"p={r['p']}", s, "expected_dice", SRC_MIX, "expected_dice",
                                r["expected_dice"], "*100", f"{v:.2f}"))
        # closed-form band: Fixed Average above LANG and above IMG
        ivs = [better_interval(d["Fixed Average"][0] - d[b][0], d["Fixed Average"][1] - d[b][1]) for b in ("LANG", "IMG")]
        if None in ivs or max(ivs[0][0], ivs[1][0]) >= min(ivs[0][1], ivs[1][1]):
            raise SystemExit(f"{m}: no closed-form range where Fixed Average beats both")
        mine = (max(ivs[0][0], ivs[1][0]), min(ivs[0][1], ivs[1][1]))
        c = one(cons, method=m)
        band = (float(c["nnunet_p_lo"]), float(c["nnunet_p_hi"]))
        if max(abs(mine[0] - band[0]), abs(mine[1] - band[1])) >= RANGE_TOL:
            raise SystemExit(f"{m}: band {band} differs from the closed form {mine} by >= {RANGE_TOL}; stop")
        bands[m] = band
        for q, col, v in (("band_lo", "nnunet_p_lo", band[0]), ("band_hi", "nnunet_p_hi", band[1])):
            recs.append(num("fig2_range", m, "equal C4/C5/C6", "Fixed Average", q, SRC_CONS, col, c[col], "none", f"{v:.3f}"))
    return curves, bands, recs


def table2_data():
    """Table II rows [(model, strategy, [4 display strings])] and the number records."""
    rows = [r for r in read_rows(SRC_P5) if r["dataset"] == DATASET]
    out, recs = [], []
    cols = [("C0", "dice", "C0 Dice", 100), ("C6", "dice", "C6 Dice", 100), ("C6", "hd95", "C6 HD95", 1),
            ("C6", "empty", "C6 Empty (%)", 100)]
    for m in MODELS:
        for s in TAB2_STRATS:
            cells = []
            for cond, col, q, f in cols:
                r = one(rows, method=m, strategy=s, condition=cond)
                shown = f"{f * float(r[col]):.2f}"
                cells.append(shown)
                recs.append(num("table2", m, cond, s, q, SRC_P5, col, r[col], "*100" if f == 100 else "none", shown))
            out.append((m, s, cells))
    return out, recs


# ---------------------------------------------------------------------------------------------- outputs
def make_fig1(data):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG1_HEIGHT_IN))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.2, top=0.8, wspace=0.42)
    for ax, key, ylab, lab in ((axes[0], "dice", "Dice drop (points)", "(a)"),
                               (axes[1], "hd95", "HD95 increase (px)", "(b)")):
        ax.axhline(0, color="0.5", linewidth=0.6, zorder=0)
        for j, m in enumerate(MODELS):
            xs = [i + (j - 1) * OFFSET for i in range(len(FIG1_CONDS))]
            v = [data[(m, c)][key] for c in FIG1_CONDS]
            ax.errorbar(xs, [x[0] for x in v], yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]],
                        linestyle="none", capsize=1.5, elinewidth=0.7, capthick=0.7, label=m,
                        markeredgewidth=0, **MODEL_STYLE[m])
        ax.set_xticks(range(len(FIG1_CONDS)))
        ax.set_xticklabels(FIG1_CONDS)
        ax.set_xlim(-0.6, len(FIG1_CONDS) - 0.4)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Text condition")
        ax.set_ylabel(ylab)
        ax.text(0.0, 1.03, lab, transform=ax.transAxes, ha="left", va="bottom", fontsize=LABEL_PT)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3,
               columnspacing=1.2)
    return fig


def make_fig2(curves, bands):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG2_HEIGHT_IN), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.2, top=0.78, wspace=0.12)
    for k, (ax, m) in enumerate(zip(axes, MODELS)):
        lo, hi = bands[m]
        ax.axvspan(lo, hi, color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0, zorder=0)
        for s in ("LANG", "IMG", "Fixed Average"):     # Fixed Average drawn last (on top)
            p, y = curves[(m, s)]
            ax.plot(p, y, label=s, **STRAT_STYLE[s])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.text(0.0, 1.03, f"({'abc'[k]}) {m}", transform=ax.transAxes, ha="left", va="bottom", fontsize=LABEL_PT)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Expected Dice (points)")
    fig.text(0.5 * (0.13 + 0.985), 0.03, "Assumed mismatch proportion $p$", ha="center", va="bottom",
             fontsize=LABEL_PT)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(s) for s in FIG2_STRATS]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6, handletextpad=0.4, columnspacing=1.2)
    return fig


def text_sizes(fig):
    """Font sizes of every visible, non-empty text object in the figure (for the >= 7 pt check)."""
    return [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def table2_tex(rows):
    """Only the tabular environment; plain \\hline rules, model name on its first row only, % escaped."""
    lines = [r"\begin{tabular}{llrrrr}", r"\hline",
             r"Model & Strategy & C0 Dice & C6 Dice & C6 HD95 & C6 Empty (\%) \\", r"\hline"]
    for i, (m, s, cells) in enumerate(rows):
        first = i == 0 or rows[i - 1][0] != m
        if first and i > 0:
            lines.append(r"\hline")
        lines.append(" & ".join([m if first else "", s] + cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def save(fig, stem):
    path = os.path.join(REPO, stem)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf")                 # no bbox_inches="tight": the page is exactly the figure size
    fig.savefig(path + ".png", dpi=DPI_PNG)


def build():
    """Everything in memory: figures, table text, number records (used by main() and the self-check)."""
    style()
    f1, r1 = fig1_data()
    curves, bands, r2 = fig2_data()
    t2, r3 = table2_data()
    return {"fig1": make_fig1(f1), "fig2": make_fig2(curves, bands), "fig1_data": f1, "bands": bands,
            "table2_rows": t2, "table2_tex": table2_tex(t2), "numbers": r1 + r2 + r3}


def main():
    b = build()
    for k in ("fig1", "fig2"):
        small = [s for s in text_sizes(b[k]) if s < 7]
        if small:
            raise SystemExit(f"{k}: text below 7 pt: {small}")
    save(b["fig1"], OUT_FIG1)
    save(b["fig2"], OUT_FIG2)
    os.makedirs(os.path.join(REPO, os.path.dirname(OUT_TAB2)), exist_ok=True)
    with open(os.path.join(REPO, OUT_TAB2), "w") as f:
        f.write(b["table2_tex"])
    with open(os.path.join(REPO, OUT_NUM), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NUM_COLS)
        w.writeheader()
        w.writerows(b["numbers"])
    print(f"written: {OUT_FIG1}.pdf/.png, {OUT_FIG2}.pdf/.png, {OUT_TAB2}, {OUT_NUM} ({len(b['numbers'])} rows)")


if __name__ == "__main__":
    main()
