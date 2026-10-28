"""Phase 3, Tasks 3.3 (D-Gate) and 3.4 (LR-Gate): fitting on QaTa GATE_FIT and threshold calibration on GATE_CAL.

Inputs: results/phase3_reliability_features.csv.gz (F1-F5 only) and results/phase3_feature_labels.csv.gz (Dice,
gain, y, group, partition). Only QaTa rows of the conditions C0, C2, C4, C5, C6 are used here; C1, C3a-C3c and
MosMed are never read for fitting or calibration, and no gate outcome is computed for them (Task 3.5 later).

Calibration rule (PHASE3_PLAN Task 3.3; PHASE3_SETUP.md section 5, items 2-5), per language model x seed:
  Delta(c, t) = mean over groups of [mean over images of [mean over donors of gain x 1{score > t}]], computed as a
  weighted sum (row weight = 1 / (#groups x #images in the group x #donor rows of the image)) on
    C0, C2: all eligible GATE_CAL images;   C4, C5, C6: S_CAL = GATE_CAL targets eligible for C4, C5 and C6.
  objective = [Delta(C4) + Delta(C5) + Delta(C6)] / 3; feasible: all five Delta >= -0.005.
  Candidates: {-inf, +inf} and every distinct score of the GATE_CAL rows of C0, C2, C4, C5, C6; rule score > t.
  Choose the feasible candidate with the largest objective, ties (<= 1e-12) to the higher threshold; if the best
  objective is <= 0, choose +inf (never fall back).
D-Gate score: F1.  LR-Gate score: decision_function of the fitted LR (item 5).
LR (Task 3.4; items 6, 7, 18): GATE_FIT rows of C0, C2, C4, C5, C6; y = gain > 0.005; weights: each group 1/|G_c|
within condition c, split equally over its images and then over an image's donor rows, each condition x 1/5,
rescaled to mean 1; StandardScaler and LogisticRegression(L2, C=1.0, lbfgs, max_iter=2000) with these weights.
One class only -> "LR-Gate unsupported" for that model x seed.

Outputs: results/phase3_dgate_calibration.csv, results/phase3_lr_calibration.csv (chosen thresholds + the C0/C2
losses, C4-C6 gains and fallback rates on GATE_CAL), *_curve.csv.gz (every candidate), results/phase3_lr_artifacts.json,
runs/phase3/<method>/seed<k>/gate_artifacts/{dgate.json, lr.json}.
Usage:  python src/phase3/gates.py
"""
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = lambda f: os.path.join(REPO, "results", f)  # noqa: E731
METHODS = ["lvit", "reclmis", "ddcmd"]
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
SEEDS = [1001, 1002, 1003]
CONDS = ["C0", "C2", "C4", "C5", "C6"]
MISMATCH = ["C4", "C5", "C6"]
FEATS = ["F1", "F2", "F3", "F4", "F5"]
MARGIN, TIE = -0.005, 1e-12
KEYS = ["dataset", "method", "seed", "sample_id", "condition", "variant_id", "donor_sample_id"]


def nested_weights(df):
    """Row weights that turn a weighted sum into: donors within image -> images within group -> groups equally."""
    n_groups = df.group.nunique()
    n_img_in_group = df.groupby("group").sample_id.transform("nunique")
    n_rows_in_img = df.groupby("sample_id").sample_id.transform("size")
    return 1.0 / (n_groups * n_img_in_group * n_rows_in_img)


def calibration_sets(cal):
    """{condition: rows} with C0 / C2 on all eligible CAL images and C4-C6 on S_CAL."""
    s_cal = set.intersection(*[set(cal[cal.condition == c].sample_id) for c in MISMATCH])
    return {c: cal[(cal.condition == c) & (cal.sample_id.isin(s_cal) if c in MISMATCH else True)].copy() for c in CONDS}


def calibrate(cal, score_col):
    """(curve DataFrame over all candidates, chosen row). cal: CAL rows of one model x seed with score, gain."""
    sets = calibration_sets(cal)
    cand = np.unique(np.concatenate([[-np.inf, np.inf], cal[score_col].to_numpy()]))
    curve = pd.DataFrame({"threshold": cand})
    for c, d in sets.items():
        w = nested_weights(d).to_numpy() * d.gain.to_numpy()
        s = d[score_col].to_numpy()
        order = np.argsort(s)
        ss, cw = s[order], np.cumsum(w[order][::-1])[::-1]               # cw[k] = sum of w over ranks >= k
        cnt = np.arange(len(ss), 0, -1)
        k = np.searchsorted(ss, cand, side="right")                        # first rank with score > t
        curve[f"delta_{c}"] = np.where(k < len(ss), cw[np.minimum(k, len(ss) - 1)], 0.0)
        curve[f"fallback_rate_{c}"] = np.where(k < len(ss), cnt[np.minimum(k, len(ss) - 1)], 0) / len(ss)
        curve[f"n_rows_{c}"] = len(ss)
    curve["objective"] = curve[[f"delta_{c}" for c in MISMATCH]].mean(axis=1)
    curve["feasible"] = (curve[[f"delta_{c}" for c in CONDS]] >= MARGIN).all(axis=1)
    f = curve[curve.feasible]
    best = f.objective.max()
    if best <= 0:
        chosen = curve[curve.threshold == np.inf].iloc[0]
        why = "never fall back (no feasible threshold with a positive objective)"
    else:
        chosen = f[f.objective >= best - TIE].sort_values("threshold").iloc[-1]   # ties -> higher threshold
        why = "feasible maximum"
    return curve, chosen, why


def lr_weights(fit):
    """Item 6: groups balanced within condition, images within group, donor rows within image; conditions equal;
    rescaled to mean 1."""
    w = pd.Series(0.0, index=fit.index)
    for c in CONDS:
        d = fit[fit.condition == c]
        if len(d):
            w[d.index] = nested_weights(d) / len(CONDS)
    return w * len(w) / w.sum()


def fit_lr(fit):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y = fit.y.to_numpy()
    info = {"n_rows": int(len(fit)), "n_positive": int(y.sum()), "n_negative": int(len(y) - y.sum())}
    if len(np.unique(y)) < 2:
        return None, None, {**info, "status": "LR-Gate unsupported (one class in GATE_FIT)"}
    w = lr_weights(fit).to_numpy()
    sc = StandardScaler().fit(fit[FEATS].to_numpy(), sample_weight=w)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)       # L2 = scikit-learn default (l1_ratio 0)
    lr.fit(sc.transform(fit[FEATS].to_numpy()), y, sample_weight=w)
    return sc, lr, {**info, "status": "fitted", "n_iter": int(lr.n_iter_[0]),
                    "weight_sum": float(w.sum()), "weight_mean": float(w.mean())}


def summary_row(method, seed, gate, chosen, why, n_cand):
    r = {"method": NAME[method], "seed": seed, "gate": gate, "threshold": chosen.threshold, "choice": why,
         "n_candidates": n_cand, "objective": chosen.objective}
    for c in CONDS:
        r[f"delta_{c}"] = chosen[f"delta_{c}"]
        r[f"fallback_rate_{c}"] = chosen[f"fallback_rate_{c}"]
        r[f"n_rows_{c}"] = int(chosen[f"n_rows_{c}"])
    return r


def main():
    F = pd.read_csv(R("phase3_reliability_features.csv.gz"), dtype={"sample_id": str, "variant_id": str,
                                                                    "donor_sample_id": str}, keep_default_na=False)
    L = pd.read_csv(R("phase3_feature_labels.csv.gz"), dtype={"sample_id": str, "variant_id": str,
                                                              "donor_sample_id": str}, keep_default_na=False)
    D = F.merge(L, on=KEYS, validate="one_to_one")
    D = D[(D.dataset == "QaTa-COV19") & D.condition.isin(CONDS)]          # C1, C3 and MosMed are not used here
    drows, lrows, dcurves, lcurves, artifacts = [], [], [], [], {}
    for m in METHODS:
        for s in SEEDS:
            d = D[(D.method == m) & (D.seed == s)]
            fit, cal = d[d.partition == "GATE_FIT"].copy(), d[d.partition == "GATE_CAL"].copy()
            out = os.path.join(REPO, "runs", "phase3", m, f"seed{s}", "gate_artifacts")
            os.makedirs(out, exist_ok=True)
            # D-Gate
            curve, ch, why = calibrate(cal, "F1")
            drows.append(summary_row(m, s, "D-Gate", ch, why, len(curve)))
            dcurves.append(curve.assign(method=NAME[m], seed=s))
            json.dump({"gate": "D-Gate", "feature": "F1", "rule": "F1 > threshold -> IMG", "threshold": float(ch.threshold)},
                      open(os.path.join(out, "dgate.json"), "w"), indent=2)
            # LR-Gate
            sc, lr, info = fit_lr(fit)
            key = f"{NAME[m]}_seed{s}"
            if lr is None:
                lrows.append({"method": NAME[m], "seed": s, "gate": "LR-Gate", **info})
                artifacts[key] = info
                continue
            cal["lr_score"] = lr.decision_function(sc.transform(cal[FEATS].to_numpy()))
            curve, ch, why = calibrate(cal, "lr_score")
            row = summary_row(m, s, "LR-Gate", ch, why, len(curve))
            from sklearn.metrics import average_precision_score, roc_auc_score
            if cal.y.nunique() == 2:                                       # diagnostic only (item 13)
                row["cal_auroc"] = roc_auc_score(cal.y, cal.lr_score)
                row["cal_average_precision"] = average_precision_score(cal.y, cal.lr_score)
            lrows.append({**row, **info})
            lcurves.append(curve.assign(method=NAME[m], seed=s))
            art = {**info, "features": FEATS, "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
                   "coef": lr.coef_.ravel().tolist(), "intercept": float(lr.intercept_[0]),
                   "penalty": "L2 (scikit-learn default), C = 1.0, lbfgs, max_iter 2000",
                   "sklearn": __import__("sklearn").__version__, "score": "decision_function (log-odds)", "rule": "score > threshold -> IMG",
                   "threshold": float(ch.threshold)}
            artifacts[key] = art
            json.dump(art, open(os.path.join(out, "lr.json"), "w"), indent=2)
    pd.DataFrame(drows).to_csv(R("phase3_dgate_calibration.csv"), index=False, float_format="%.6g")
    pd.DataFrame(lrows).to_csv(R("phase3_lr_calibration.csv"), index=False, float_format="%.6g")
    pd.concat(dcurves).to_csv(R("phase3_dgate_calibration_curve.csv.gz"), index=False, float_format="%.6g")
    if lcurves:
        pd.concat(lcurves).to_csv(R("phase3_lr_calibration_curve.csv.gz"), index=False, float_format="%.6g")
    json.dump(artifacts, open(R("phase3_lr_artifacts.json"), "w"), indent=2)
    cols = ["method", "seed", "threshold", "choice", "delta_C0", "delta_C2", "delta_C4", "delta_C5", "delta_C6",
            "fallback_rate_C0", "fallback_rate_C2", "fallback_rate_C4", "fallback_rate_C5", "fallback_rate_C6"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("D-Gate\n", pd.DataFrame(drows)[cols].round(4).to_string(index=False))
        print("LR-Gate\n", pd.DataFrame(lrows)[[c for c in cols + ["n_positive", "n_negative", "status"]
                                               if c in pd.DataFrame(lrows)]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
