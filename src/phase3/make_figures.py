"""Phase 3 test figures (PHASE3_PLAN Task 3.9) -> reports/figures/phase3/. Descriptive; nothing is fitted or tested.

  fig1_qata_dice.png        QaTa Dice under C0 / C2 versus C4 / C5 / C6 for LANG, IMG, D-Gate, LR-Gate
                            (image-weighted mean of the 3 per-seed means, error bar = SD over seeds)
  fig2_switches.png         fallback (switch) rate split into beneficial / harmful / neutral switches, all nine
                            conditions, both gates (rows; results/phase3_test_routing_quality.csv)
  fig3_paired_lr.png        group-weighted LR-Gate - LANG and LR-Gate - D-Gate with 95% t-CIs, all nine conditions;
                            the C3c diagnostic marked (results/phase3_pairwise_results.csv)
  fig4_mosmed.png           MosMedData+ (exploratory, 9 units): ORACLE headroom and gate - LANG with unit CIs
  fig5_cases.png            real QaTa test cases chosen by the rule below
Case rule (QaTa test, seed 1001, LR-Gate, rows of all three language models pooled; per row the Dice of LANG and
IMG from results/phase3_test_per_case.csv.gz, gain = Dice(IMG) - Dice(LANG); ties: method order LViT-T, RecLMIS,
DD-CMD, then sample id, then donor variant):
  A  C6, switched to IMG, largest gain        (beneficial switch)
  B  C6, switched to IMG, smallest gain       (harmful switch)
  C  C0, switched to IMG, smallest gain       (harmful switch under the original text)
  D  C6, not switched, largest gain           (missed opportunity: the ORACLE would have switched)
Each case: the image, the ground truth (yellow dashed contour on every panel), the LANG and the IMG mask; every
mask panel is labelled with its Dice and HD95; the gate's choice is framed.
Usage:  python src/phase3/make_figures.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
R = lambda f: os.path.join(REPO, "results", f)  # noqa: E731
OUT = os.path.join(REPO, "reports", "figures", "phase3")
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
CODE = {v: k for k, v in NAME.items()}
METHODS = ["LViT-T", "RecLMIS", "DD-CMD"]
CONDS = ["C0", "C1", "C2", "C3a", "C3b", "C3c", "C4", "C5", "C6"]
QATA, MOS = "QaTa-COV19", "MosMedData+"
CASE_SEED = 1001
# Phase 1 / 2 figure style: ink, grid, ground-truth contour; strategy colours
INK, INK2, GRID, GT_COLOR = "#0b0b0b", "#52514e", "#d9d8d4", "#f2c200"
SCOL = {"LANG": "#3b6fb6", "IMG": "#5a5a5a", "D-Gate": "#d98b2b", "LR-Gate": "#2f9e6e", "ORACLE": "#b9b6ae"}
BEN, HARM, NEU = "#2f9e6e", "#c8463d", "#b9b6ae"
MASK = {"LANG": "#3b6fb6", "IMG": "#d6407f"}                  # case-figure overlays (IMG grey would be invisible)


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def results():
    T = pd.read_csv(R("phase3_test_results.csv"))
    T["method"] = T.method.map(NAME)
    return T


def fig1(T):
    plt = plt_()
    conds = ["C0", "C2", "C4", "C5", "C6"]
    strat = ["LANG", "IMG", "D-Gate", "LR-Gate"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, m in zip(axes, METHODS):
        g = T[(T.dataset == QATA) & (T.method == m)].set_index(["condition", "strategy"])
        for j, s in enumerate(strat):
            x = np.arange(len(conds)) + (j - 1.5) * 0.17
            y = [100 * g.dice[(c, s)] for c in conds]
            e = [100 * g.dice_sd[(c, s)] for c in conds]
            ax.errorbar(x, y, yerr=e, fmt="o", ms=5, color=SCOL[s], capsize=2, lw=1, label=s)
        ax.axvline(1.5, color=INK2, lw=0.8, ls=":")
        ax.set_xticks(range(len(conds)), conds)
        ax.set_title(m, fontsize=10, color=INK)
        ax.text(0.5, 0.02, "original / equivalent", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5, color=INK2)
        ax.text(3.0, 0.02, "real-report mismatch", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5, color=INK2)
        style(ax)
    axes[0].set_ylabel("Dice (points)", color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.08))
    fig.suptitle("QaTa-COV19 test: Dice under C0 / C2 versus C4 / C5 / C6 (image-weighted, mean ± SD over 3 seeds)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig1_qata_dice.png"), dpi=160)
    plt.close(fig)


def fig2():
    plt = plt_()
    Q = pd.read_csv(R("phase3_test_routing_quality.csv"))
    Q = Q[Q.dataset == QATA].copy()
    Q["method"] = Q.method.map(NAME)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), sharey=True)
    for i, gate in enumerate(("D-Gate", "LR-Gate")):
        for j, m in enumerate(METHODS):
            ax = axes[i, j]
            g = Q[(Q.gate == gate) & (Q.method == m)].set_index("condition").loc[CONDS]
            b, h, n = (100 * g[c] / g.n_rows for c in ("n_beneficial", "n_harmful", "n_neutral"))
            x = np.arange(len(CONDS))
            ax.bar(x, b, color=BEN, label="beneficial")
            ax.bar(x, h, bottom=b, color=HARM, label="harmful")
            ax.bar(x, n, bottom=b + h, color=NEU, label="neutral (|ΔDice| ≤ 1e-6)")
            ax.set_xticks(x, CONDS, fontsize=8)
            ax.set_title(f"{gate} · {m}", fontsize=9.5, color=INK)
            style(ax)
        axes[i, 0].set_ylabel("switched rows (% of rows)", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("QaTa-COV19 test: fallback rate split into beneficial / harmful switches (row-weighted)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig2_switches.png"), dpi=160)
    plt.close(fig)


def fig3():
    plt = plt_()
    P = pd.read_csv(R("phase3_pairwise_results.csv"))
    P = P[P.dataset == QATA]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, m in zip(axes, METHODS):
        for j, (comp, col, lab) in enumerate((("LR-Gate - LANG", SCOL["LR-Gate"], "LR-Gate − LANG"),
                                              ("LR-Gate - D-Gate", SCOL["D-Gate"], "LR-Gate − D-Gate"))):
            g = P[(P.method == m) & (P.comparison == comp)].set_index("condition").loc[CONDS]
            x = np.arange(len(CONDS)) + (j - 0.5) * 0.25
            y = 100 * g.delta
            ax.errorbar(x, y, yerr=[y - 100 * g.ci95_low, 100 * g.ci95_high - y], fmt="o", ms=4.5, color=col,
                        capsize=2, lw=1, label=lab)
        k = CONDS.index("C3c")
        ax.axvspan(k - 0.45, k + 0.45, color="#efe9d8", zorder=0)
        ax.text(k, 0.97, "C3c\n(withheld)", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(CONDS)), CONDS, fontsize=8)
        ax.set_title(m, fontsize=10, color=INK)
        style(ax)
    axes[0].set_ylabel("paired Δ Dice (points)", color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.88))
    fig.suptitle("QaTa-COV19 test: paired LR-Gate improvements, group-weighted (472–474 patient groups), 95% t-CI. "
                 "Inference only for C4–C6; others descriptive", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig3_paired_lr.png"), dpi=160)
    plt.close(fig)


def fig4(T):
    plt = plt_()
    P = pd.read_csv(R("phase3_pairwise_results.csv"))
    P = P[P.dataset == MOS]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, m in zip(axes, METHODS):
        g = T[(T.dataset == MOS) & (T.method == m)].set_index(["condition", "strategy"]).dice
        x = np.arange(len(CONDS))
        ax.bar(x, [100 * (g[(c, "ORACLE")] - g[(c, "LANG")]) for c in CONDS], color=SCOL["ORACLE"], width=0.7,
               label="ORACLE − LANG (headroom)")
        for j, (comp, col, lab) in enumerate((("D-Gate - LANG", SCOL["D-Gate"], "D-Gate − LANG"),
                                              ("LR-Gate - LANG", SCOL["LR-Gate"], "LR-Gate − LANG"))):
            q = P[(P.method == m) & (P.comparison == comp)].set_index("condition").loc[CONDS]
            y = 100 * q.delta
            ax.errorbar(x + (j - 0.5) * 0.22, y, yerr=[y - 100 * q.ci95_low, 100 * q.ci95_high - y], fmt="o", ms=4,
                        color=col, capsize=2, lw=1, label=lab + " (unit CI)")
        ax.axhline(0, color=INK2, lw=0.8)
        n = T[(T.dataset == MOS) & (T.method == m) & (T.strategy == "LANG")].n_groups.max()
        ax.set_xticks(x, CONDS, fontsize=8)
        ax.set_title(f"{m}  ({n} units)", fontsize=10, color=INK)
        style(ax)
    axes[0].set_ylabel("Δ Dice (points)", color=INK)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("MosMedData+ test, exploratory: QaTa gates applied unchanged; headroom image-weighted, gate − LANG "
                 "unit-weighted with 95% t-CI (9 CT-volume units)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig4_mosmed.png"), dpi=160)
    plt.close(fig)


def select_cases():
    """The four cases of the documented rule (see the module docstring)."""
    S = pd.read_csv(R("phase3_test_per_case.csv.gz"), dtype={"sample_id": str, "variant_id": str})
    S = S[(S.dataset == QATA) & (S.seed == CASE_SEED) & S.strategy.isin(["LANG", "IMG", "LR-Gate"])]
    k = ["method", "sample_id", "condition", "variant_id"]
    W = S.pivot_table(index=k, columns="strategy", values="dice").reset_index()
    sw = S[S.strategy == "LR-Gate"].set_index(k).switch
    W["switched"] = W.set_index(k).index.map(sw).astype(float) > 0.5
    W["gain"] = W.IMG - W.LANG
    W["m_order"] = W.method.map({c: i for i, c in enumerate(CODE[m] for m in METHODS)})
    hd = S.pivot_table(index=k, columns="strategy", values="hd95")
    rules = [("A", "C6, switched, largest gain (beneficial)", "C6", True, False),
             ("B", "C6, switched, smallest gain (harmful)", "C6", True, True),
             ("C", "C0, switched, smallest gain (harmful)", "C0", True, True),
             ("D", "C6, not switched, largest gain (missed)", "C6", False, False)]
    picks = []
    for tag, label, cond, switched, smallest in rules:
        c = W[(W.condition == cond) & (W.switched == switched)]
        c = c.sort_values(["gain", "m_order", "sample_id", "variant_id"], ascending=[smallest, True, True, True])
        r = c.iloc[0]
        key = (r.method, r.sample_id, r.condition, r.variant_id)
        picks.append({"case": tag, "rule": label, "method": NAME[r.method], "sample_id": r.sample_id,
                      "condition": r.condition, "variant_id": r.variant_id, "switched": bool(switched),
                      "dice_lang": float(r.LANG), "dice_img": float(r.IMG), "hd95_lang": float(hd.loc[key, "LANG"]),
                      "hd95_img": float(hd.loc[key, "IMG"]), "n_candidates": int(len(c))})
    return picks


def unpack(npz, key):
    return np.unpackbits(npz[key])[:224 * 224].reshape(224, 224).astype(bool)


def fig5(picks):
    plt = plt_()
    import matplotlib
    from lgms import data
    fig, axes = plt.subplots(len(picks), 4, figsize=(8.6, 2.45 * len(picks)))
    img_masks = np.load(os.path.join(REPO, "runs", "phase3", "test", "img", "nnunet", QATA, f"seed{CASE_SEED}", "masks.npz"))
    for r, pk in enumerate(picks):
        sid = pk["sample_id"]
        lang_masks = np.load(os.path.join(REPO, "runs", "phase2", "test", CODE[pk["method"]], QATA, f"seed{CASE_SEED}",
                                          f"masks_{pk['condition']}_{pk['variant_id']}.npz"))
        img = data.read_image(QATA, "test", sid)
        gt = data.read_mask(QATA, "test", sid) > 0
        panels = [("Image", None, None), ("Ground truth", gt, None),
                  (f"LANG ({pk['method']}, {pk['condition']}{'' if pk['variant_id'] == '0' else ' ' + pk['variant_id']})",
                   unpack(lang_masks, sid), (pk["dice_lang"], pk["hd95_lang"])),
                  ("IMG (nnU-Net)", unpack(img_masks, sid), (pk["dice_img"], pk["hd95_img"]))]
        chosen = 3 if pk["switched"] else 2
        for c, (title, mask, sc) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if c > 0:
                ax.contour(gt.astype(float), levels=[0.5], colors=[GT_COLOR], linestyles="--", linewidths=1.1)
            if mask is not None and c >= 2:
                ov = np.zeros((224, 224, 4))
                ov[mask] = (*matplotlib.colors.to_rgb(MASK["LANG"] if c == 2 else MASK["IMG"]), 0.45)
                ax.imshow(ov)
                ax.text(3, 220, f"Dice {100 * sc[0]:.1f} | HD95 {sc[1]:.1f}", color="white", fontsize=6.8, va="bottom",
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.2, linewidth=0))
            if c == chosen:
                for side in ax.spines.values():
                    side.set_edgecolor(SCOL["LR-Gate"]); side.set_linewidth(3)
                title += "  ← LR-Gate"
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=7.5, color=INK)
        axes[r, 0].set_ylabel(f"{pk['case']}: {pk['rule']}", fontsize=7.5, color=INK)
        axes[r, 0].text(0, -0.05, sid, transform=axes[r, 0].transAxes, va="top", fontsize=6, color=INK2)
    fig.suptitle(f"QaTa-COV19 test, seed {CASE_SEED}, LR-Gate: cases by the documented rule (GT yellow dashed; "
                 "the gate's choice framed)", x=0.01, ha="left", fontsize=9, color=INK)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.03, hspace=0.32, wspace=0.04)
    fig.savefig(os.path.join(OUT, "fig5_cases.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    T = results()
    fig1(T)
    fig2()
    fig3()
    fig4(T)
    picks = select_cases()
    with open(os.path.join(OUT, "case_selection.json"), "w") as f:
        json.dump(picks, f, indent=2)
    fig5(picks)
    print(json.dumps(picks, indent=1))


if __name__ == "__main__":
    main()
