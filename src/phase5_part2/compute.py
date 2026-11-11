"""Phase 5 Part 2, Tasks 5.2.3-5.2.5 on the QaTa TEST split: U-Net as the IMG reference (post-hoc sensitivity).

Rules: configs/phase5_part2/spec.json; freeze: reports/PHASE5_PART2_PROTOCOL_FREEZE.md; inputs:
configs/phase5_part2/artifact_manifest.tsv. Before anything of the test split is read (exit 10 otherwise): Phase 2
lock open; spec, freeze document and manifest each committed once and unchanged, ancestors of HEAD; clean tree;
HEAD on origin/main; every manifest file has its recorded sha256; result files absent (the run log is append-only).
Strategies: LANG, U-Net IMG, U-Net Fixed Average (P_avg = 0.5 P_lang + 0.5 P_img in float64, P_avg > 0.5).
The nnU-Net half of the backbone table is read from committed Phase 5 files (gain over LANG computed here from
the committed per-row results, checked against results/phase5_test_results.csv).
Usage (Slurm job slurm/phase5_part2_test.slurm):  python src/phase5_part2/compute.py
"""
import datetime
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase5 import core, guards  # noqa: E402

DATASET = "QaTa-COV19"
N_ROWS = 206595
COUNTS = {"C0": (2113, 474), "C2": (2113, 474), "C4": (2081, 472), "C5": (2081, 472), "C6": (2081, 472)}
SPEC = os.path.join("configs", "phase5_part2", "spec.json")
FREEZE = os.path.join("reports", "PHASE5_PART2_PROTOCOL_FREEZE.md")
MANIFEST = os.path.join("configs", "phase5_part2", "artifact_manifest.tsv")
R = lambda f: os.path.join("results", f)  # noqa: E731
OUTS = {"results": R("phase5_part2_unet_test_results.csv"), "pairwise": R("phase5_part2_unet_pairwise.csv"),
        "mixed": R("phase5_part2_unet_mixed_reliability.csv"), "crossover": R("phase5_part2_unet_crossovers.csv"),
        "consistency": R("phase5_part2_backbone_consistency.csv"), "per_case": R("phase5_part2_unet_test_per_case.csv.gz"),
        "log": R("phase5_part2_test_run_log.tsv")}
IMG, FA = "U-Net IMG", "U-Net Fixed Average"
STRATS = ["LANG", IMG, FA]
LABEL = "post-hoc sensitivity analysis, descriptive (test split seen in Phases 2, 3 and 5)"
CONCLUSION = {True: "Conclusion: the simple-fusion mitigation finding is not specific to nnU-Net.",
              False: "Conclusion: the mitigation effect is backbone-dependent and the manuscript claim must be limited accordingly."}
_GT = {}


# ------------------------------------------------------------------------------------------------ rules
def row_consistent(ci_lows, p_range):
    """Draft judging rule (fixed in the freeze document): all three C4-C6 lower bounds of Fixed Average - LANG > 0
    and a non-empty range of p (width > 0) where Fixed Average beats both LANG and IMG."""
    return bool(len(ci_lows) == 3 and all(x > 0 for x in ci_lows) and p_range is not None and p_range[1] - p_range[0] > 0)


def overall(unet_rows_consistent):
    return CONCLUSION[bool(unet_rows_consistent) and all(unet_rows_consistent)]


def parse_range(text):
    """(lo, hi) from the Phase 5 better_than_both_p_range text (e.g. 'np.float64(0.07) - np.float64(0.16)', '0.0 - ...')."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or str(text).strip() in ("", "none"):
        return None
    v = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text).replace("np.float64", ""))]
    if len(v) != 2:
        raise ValueError(f"unexpected range text {text!r}")
    return (v[0], v[1])


def check_counts(D):
    if len(D) != N_ROWS:
        guards.stop(f"{len(D)} rows, expected {N_ROWS}")
    for c, (n_img, n_grp) in COUNTS.items():
        d = D[D.condition == c]
        got = (d.sample_id.nunique(), d.group.nunique())
        if got != (n_img, n_grp):
            guards.stop(f"{c}: {got[0]} images / {got[1]} groups, expected {n_img} / {n_grp}")


def result_outputs():
    return [v for k, v in OUTS.items() if k != "log"]


# ------------------------------------------------------------------------------------------------ data
def preflight(repo=guards.REPO):
    for f in (SPEC, FREEZE, MANIFEST):
        guards.committed_unchanged(repo, f)
    guards.clean_tree(repo)
    guards.on_origin(repo)
    guards.outputs_absent(repo, result_outputs())
    m = pd.read_csv(os.path.join(repo, MANIFEST), sep="\t", dtype=str)
    for path, sha in zip(m.path, m.sha256):
        full = os.path.expanduser(path) if path.startswith("~") else os.path.join(repo, path)
        if not os.path.exists(full) or guards.sha256_file(full) != sha:
            guards.stop(f"manifest file missing or changed: {path}")
    return len(m)


def init_worker():
    from lgms import data
    ids, _, masks = data.load_split_arrays(DATASET, "test")
    _GT.clear()
    _GT.update(zip(ids, masks))


def unet_dir(seed):
    return os.path.join(REPO, "runs", "phase3", "test", "img", "unet", DATASET, f"seed{seed}")


def test_rows(args):
    """U-Net Fixed Average metrics of one model x seed, with the archived-mask checks."""
    m, s = args
    from phase3 import features as Fe
    pimg, mimg = np.load(os.path.join(unet_dir(s), "probs.npz")), np.load(os.path.join(unet_dir(s), "masks.npz"))
    img_ok, rows = set(), []
    for cond, var, fcsv, fnpz in Fe.passes(m, DATASET, s, "test"):
        if cond not in core.CONDS:
            continue
        pc = pd.read_csv(fcsv, dtype={"sample_id": str, "variant_id": str}, keep_default_na=False)
        pl, ml = np.load(fnpz), np.load(os.path.join(os.path.dirname(fnpz), f"masks_{cond}_{var}.npz"))
        for r in pc.itertuples():
            i = r.sample_id
            if not np.array_equal(np.packbits(pl[i] > 0.5), ml[i]):
                raise RuntimeError(f"LANG archived mask differs: {m} {s} {cond} {var} {i}")
            if i not in img_ok:
                if not np.array_equal(np.packbits(pimg[i] > 0.5), mimg[i]):
                    raise RuntimeError(f"U-Net archived mask differs: {s} {i}")
                img_ok.add(i)
            fa = core.score_mask(core.fixed_average(pl[i], pimg[i])[1], _GT[i])
            rows.append((DATASET, m, s, i, cond, var, r.donor_sample_id) + tuple(fa[k] for k in core.METRICS))
    return rows


def base_rows():
    """Keys, group, LANG and U-Net IMG metrics of the 206,595 test rows."""
    dt = {"sample_id": str, "variant_id": str, "donor_sample_id": str}
    L = pd.read_csv(os.path.join(REPO, R("phase3_feature_labels_test.csv.gz")), dtype=dt, keep_default_na=False)
    L = L[(L.dataset == DATASET) & L.condition.isin(core.CONDS)][core.KEYS + ["group"]]
    p2 = pd.read_csv(os.path.join(REPO, R("phase2_test_per_case.csv.gz")), dtype=dt, keep_default_na=False)
    lang = p2[core.KEYS + core.METRICS].rename(columns={k: f"lang_{k}" for k in core.METRICS})
    D = L.merge(lang, on=core.KEYS, how="left", validate="one_to_one")
    img = pd.concat([pd.read_csv(os.path.join(unet_dir(s), "per_case.csv"), dtype={"sample_id": str}) for s in core.SEEDS])
    img = img[["seed", "sample_id"] + core.METRICS].rename(columns={k: f"img_{k}" for k in core.METRICS})
    D = D.merge(img, on=["seed", "sample_id"], how="left", validate="many_to_one")
    for p in ("lang", "img"):
        D[f"{p}_empty"] = D[f"{p}_empty"].astype(str).eq("True").astype(float)
    if D[[f"lang_{k}" for k in core.METRICS] + [f"img_{k}" for k in core.METRICS]].isna().any().any():
        guards.stop("LANG or U-Net IMG metrics missing for some rows")
    return D


def long_table(D, cols):
    """Strategy rows with the columns expected by evaluate_gates.image_weighted / stats_phase3.group_differences."""
    base = D[["dataset", "group", "method", "seed", "sample_id", "condition", "variant_id"]].assign(part="test")
    out = []
    for st, pre in cols.items():
        t = base.copy()
        t["strategy"], t["switch"] = st, np.nan
        for k in core.METRICS:
            t[k] = D[f"{pre}_{k}"].to_numpy()
        out.append(t)
    return pd.concat(out, ignore_index=True)


def paired(S, specs):
    from phase3 import stats_phase3 as P
    rows = []
    for (a, b), conds, kind in specs:
        for (ds, part, m, c), (u, n_img) in P.group_differences(S, a, b).items():
            if c in conds:
                r = P.t_over_groups(u.values)
                rows.append({"comparison": f"{a} - {b}", "kind": kind, "method": core.NAME[m], "condition": c,
                             "n_images": n_img, "n_groups": r["n_groups"], "delta": r["delta"], "ci95_low": r["ci95_low"],
                             "ci95_high": r["ci95_high"], "label": LABEL})
    return pd.DataFrame(rows)


def mixed(S, strategies, fa):
    common = set.intersection(*[set(S[S.condition == c].sample_id) for c in core.WRONG])
    P = S[S.sample_id.isin(common) & S.condition.isin(["C0"] + core.WRONG)]
    if (len(common), P.group.nunique()) != (2081, 472):
        guards.stop(f"common population {len(common)} / {P.group.nunique()}, expected 2081 / 472")
    Dm = {st: core.group_weighted_mean(P[P.strategy == st]) for st in strategies}
    curves, cross, ranges = [], [], {}
    for m in core.METHODS:
        d = {st: {c: float(Dm[st][(m, c)]) for c in ["C0"] + core.WRONG} for st in strategies}
        for st in strategies:
            dw = float(np.mean([d[st][c] for c in core.WRONG]))
            for p in np.round(np.arange(101) / 100, 2):
                curves.append({"method": core.NAME[m], "strategy": st, "p": p, "D_C0": d[st]["C0"], "D_wrong": dw,
                               "expected_dice": core.expected(d[st]["C0"], dw, p)})
        for mix, cs in (("equal C4/C5/C6", core.WRONG), ("C4 only", ["C4"]), ("C5 only", ["C5"]), ("C6 only", ["C6"])):
            ivs = []
            for b in ("LANG", IMG):
                dc = d[fa]["C0"] - d[b]["C0"]
                dw = float(np.mean([d[fa][c] for c in cs]) - np.mean([d[b][c] for c in cs]))
                ps = core.crossover(dc, dw)
                ivs.append(core.better_interval(dc, dw))
                cross.append({"method": core.NAME[m], "strategy": fa, "versus": b, "mixture": mix, "delta_clean": dc,
                              "delta_wrong": dw, "p_star": "none" if ps is None else repr(float(ps))})
            both = core.intersect(*ivs)
            both = None if both is None else (float(both[0]), float(both[1]))
            cross.append({"method": core.NAME[m], "strategy": fa, "versus": "both LANG and IMG", "mixture": mix,
                          "p_lo": "none" if both is None else repr(both[0]), "p_hi": "none" if both is None else repr(both[1]),
                          "p_width": 0.0 if both is None else both[1] - both[0]})
            if mix == "equal C4/C5/C6":
                ranges[core.NAME[m]] = both
    return pd.DataFrame(curves), pd.DataFrame(cross), ranges


def nnunet_half(D):
    """nnU-Net Fixed Average - LANG under C4-C6 from committed Phase 5 per-row results (checked against Phase 5)."""
    dt = {"sample_id": str, "variant_id": str, "donor_sample_id": str}
    pc = pd.read_csv(os.path.join(REPO, R("phase5_test_per_case.csv.gz")), dtype=dt, keep_default_na=False)
    pc = pc[core.KEYS + [f"fixed_average_{k}" for k in core.METRICS]].rename(
        columns={f"fixed_average_{k}": f"nnfa_{k}" for k in core.METRICS})
    X = D.merge(pc, on=core.KEYS, how="left", validate="one_to_one")
    if X[[f"nnfa_{k}" for k in core.METRICS]].isna().any().any():
        guards.stop("Phase 5 per-row Fixed Average missing for some rows")
    S = long_table(X, {"LANG": "lang", "Fixed Average": "nnfa"})
    from phase3 import evaluate_gates as E
    iw = E.image_weighted(S).set_index(["method", "condition", "strategy"]).dice
    p5 = pd.read_csv(os.path.join(REPO, R("phase5_test_results.csv"))).set_index(["method", "condition", "strategy"]).dice
    for m in core.METHODS:
        for c in core.CONDS:
            mine = iw[(m, c, "Fixed Average")] - iw[(m, c, "LANG")]
            ref = p5[(core.NAME[m], c, "Fixed Average")] - p5[(core.NAME[m], c, "LANG")]
            if abs(mine - ref) > 1e-6:
                guards.stop(f"nnU-Net check failed: {m} {c} {mine} vs {ref}")
    return paired(S, [(("Fixed Average", "LANG"), core.WRONG, "nnU-Net gain")])


def consistency(res, pw, ranges, nn_pw):
    p5 = pd.read_csv(os.path.join(REPO, R("phase5_test_results.csv")))
    xs = pd.read_csv(os.path.join(REPO, R("phase5_crossover_summary.csv")), dtype=str, keep_default_na=False)
    rows, unet_ok = [], []
    for m in core.METHODS:
        name = core.NAME[m]
        nn_clean = float(p5[(p5.method == name) & (p5.condition == "C0") & (p5.strategy == "IMG")].dice.iloc[0])
        un_clean = float(res[(res.method == m) & (res.condition == "C0") & (res.strategy == IMG)].dice.iloc[0])
        r = {"method": name, "nnunet_img_clean_dice": nn_clean, "unet_img_clean_dice": un_clean}
        lows = {"nnunet": [], "unet": []}
        for half, t, comp in (("nnunet", nn_pw, "Fixed Average - LANG"), ("unet", pw, f"{FA} - LANG")):
            for c in core.WRONG:
                q = t[(t.method == name) & (t.condition == c) & (t.comparison == comp)].iloc[0]
                r[f"{half}_fa_minus_lang_{c}"], r[f"{half}_ci95_low_{c}"], r[f"{half}_ci95_high_{c}"] = q.delta, q.ci95_low, q.ci95_high
                lows[half].append(q.ci95_low)
        nn_rng = parse_range(xs[(xs.method == name) & (xs.strategy == "Fixed Average") & (xs.versus == "both LANG and IMG")
                                & (xs.mixture == "equal C4/C5/C6")].better_than_both_p_range.iloc[0])
        un_rng = ranges[name]
        for half, rng in (("nnunet", nn_rng), ("unet", un_rng)):
            r[f"{half}_p_lo"], r[f"{half}_p_hi"] = (np.nan, np.nan) if rng is None else rng
            r[f"{half}_p_width"] = 0.0 if rng is None else rng[1] - rng[0]
        r["nnunet_row"] = "consistent" if row_consistent(lows["nnunet"], nn_rng) else "not consistent"
        ok = row_consistent(lows["unet"], un_rng)
        r["unet_row"] = "consistent" if ok else "not consistent"
        unet_ok.append(ok)
        rows.append(r)
    out = pd.DataFrame(rows)
    out["qualitative_conclusion"] = overall(unet_ok)
    return out


def log(event, msg=""):
    p = os.path.join(REPO, OUTS["log"])
    new = not os.path.exists(p)
    with open(p, "a") as f:
        if new:
            f.write("event\ttime_utc\tjob_id\thead\tnote\n")
        f.write(f"{event}\t{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}\t"
                f"{os.environ.get('SLURM_JOB_ID', 'none')}\t{guards.git(REPO, 'rev-parse', 'HEAD').stdout.strip()}\t{msg}\n")


def main():
    from phase2 import lock
    lock.require_open(what="Phase 5 Part 2 test computation")
    n = preflight()
    log("start", f"manifest files verified: {n}")
    D = base_rows()
    check_counts(D)
    with Pool(9, initializer=init_worker) as pool:
        parts = pool.map(test_rows, [(m, s) for m in core.METHODS for s in core.SEEDS])
    X = pd.DataFrame([x for p in parts for x in p], columns=core.KEYS + [f"fa_{k}" for k in core.METRICS])
    D = D.merge(X, on=core.KEYS, how="left", validate="one_to_one")
    if D.fa_dice.isna().any() or len(D) != N_ROWS:
        guards.stop("Fixed Average rows do not match the test rows one to one")
    S = long_table(D, {"LANG": "lang", IMG: "img", FA: "fa"})
    from phase3 import evaluate_gates as E
    res = E.image_weighted(S)
    pw = paired(S, [((FA, "LANG"), core.WRONG, "main"), ((FA, IMG), core.WRONG, "main"),
                    ((FA, "LANG"), ["C0", "C2"], "clean"), ((IMG, "LANG"), ["C0", "C2"], "clean")])
    curves, cross, ranges = mixed(S, STRATS, FA)
    nn_pw = nnunet_half(D)
    cons = consistency(res, pw, ranges, nn_pw)
    pc = D[core.KEYS].copy()
    for k in core.METRICS:
        pc[f"unet_fixed_average_{k}"] = D[f"fa_{k}"].to_numpy()
    out = {"results": res.assign(method=res.method.map(core.NAME), label=LABEL), "pairwise": pd.concat([pw, nn_pw]),
           "mixed": curves, "crossover": cross, "consistency": cons}
    for k, df in out.items():
        guards.write_atomic(df, REPO, OUTS[k], float_format="%.10g")
    guards.write_atomic(pc, REPO, OUTS["per_case"], float_format="%.17g")
    log("end", f"rows {len(D)}")
    print(cons.T.to_string())


if __name__ == "__main__":
    main()
