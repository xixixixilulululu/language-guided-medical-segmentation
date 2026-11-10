"""Phase 5, Tasks 5.3-5.4 on the QaTa TEST split, run once after the freeze (post-hoc supplementary comparison).

Rules: configs/phase5/spec.json; freeze: reports/PHASE5_PROTOCOL_FREEZE.md; inputs: configs/phase5/artifact_manifest.tsv.
Before anything of the test split is read (exit 10 otherwise): Phase 2 lock open; spec, freeze document, manifest
and SDC calibration each committed once and unchanged, ancestors of HEAD; clean tree; HEAD on origin/main; every
manifest file has its recorded sha256; outputs absent. Then: 206,595 rows asserted; every LANG / IMG archived mask
equals packbits(P > 0.5); SDC-Fallback and Fixed Average added to the Phase 3 strategies; tables, paired
differences (95% t-CI, no p), compute summary, mixed reliability and exact crossovers.
Usage (Slurm job slurm/phase5_test.slurm):  python src/phase5/compute_test_split.py
"""
import datetime
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase5 import core, guards  # noqa: E402

DATASET = "QaTa-COV19"
N_ROWS = 206595
FREEZE = os.path.join("reports", "PHASE5_PROTOCOL_FREEZE.md")
MANIFEST = os.path.join("configs", "phase5", "artifact_manifest.tsv")
CALIB = os.path.join("results", "phase5_sdc_calibration.csv")
COST = os.path.join("results", "phase3_computational_cost.csv")
R = lambda f: os.path.join("results", f)  # noqa: E731
OUTS = {"results": R("phase5_test_results.csv"), "pairwise": R("phase5_pairwise_results.csv"),
        "compute": R("phase5_compute_summary.csv"), "mixed": R("phase5_mixed_reliability.csv"),
        "crossover": R("phase5_crossover_summary.csv"), "per_case": R("phase5_test_per_case.csv.gz"),
        "log": R("phase5_test_run_log.tsv")}
STRATS = ["LANG", "IMG", "D-Gate", "LR-Gate", "SDC-Fallback", "Fixed Average"]
MIXED = ["LANG", "IMG", "LR-Gate", "SDC-Fallback", "Fixed Average"]          # the plan's five (spec "mixed_reliability")
POST_HOC = "post-hoc supplementary, descriptive (test split seen in Phases 2-3)"
_GT = {}                                                # filled in every worker by init_worker()


def init_worker():
    """Pool initializer: each worker reads the test ground truth itself (no reliance on fork)."""
    from lgms import data
    ids, _, masks = data.load_split_arrays(DATASET, "test")
    _GT.clear()
    _GT.update(zip(ids, masks))


def preflight(repo=guards.REPO):
    for f in (guards.SPEC, FREEZE, MANIFEST, CALIB):
        guards.committed_unchanged(repo, f)
    guards.clean_tree(repo)
    guards.on_origin(repo)
    guards.outputs_absent(repo, OUTS.values())
    m = pd.read_csv(os.path.join(repo, MANIFEST), sep="\t", dtype=str)
    for path, sha in zip(m.path, m.sha256):
        full = os.path.expanduser(path) if path.startswith("~") else os.path.join(repo, path)
        if not os.path.exists(full) or guards.sha256_file(full) != sha:
            guards.stop(f"manifest file missing or changed: {path}")
    return len(m)


def test_rows(args):
    """Per-row SDC and Fixed Average of one model x seed (test), with the archived-mask checks."""
    m, s = args
    from phase3 import features as Fe
    di = os.path.join(REPO, "runs", "phase3", "test", "img", "nnunet", DATASET, f"seed{s}")
    pimg, mimg = np.load(os.path.join(di, "probs.npz")), np.load(os.path.join(di, "masks.npz"))
    img_ok = {}
    rows = []
    for cond, var, fcsv, fnpz in Fe.passes(m, DATASET, s, "test"):
        if cond not in core.CONDS:
            continue
        pc = pd.read_csv(fcsv, dtype={"sample_id": str, "variant_id": str}, keep_default_na=False)
        pl = np.load(fnpz)
        ml = np.load(os.path.join(os.path.dirname(fnpz), f"masks_{cond}_{var}.npz"))
        for r in pc.itertuples():
            i = r.sample_id
            p_l, p_i = pl[i], pimg[i]
            if not np.array_equal(np.packbits(p_l > 0.5), ml[i]):
                raise RuntimeError(f"LANG archived mask differs: {m} {s} {cond} {var} {i}")
            if i not in img_ok:
                if not np.array_equal(np.packbits(p_i > 0.5), mimg[i]):
                    raise RuntimeError(f"IMG archived mask differs: {s} {i}")
                img_ok[i] = True
            _, b_avg = core.fixed_average(p_l, p_i)
            fa = core.score_mask(b_avg, _GT[i])
            rows.append((DATASET, m, s, i, cond, var, r.donor_sample_id, core.sdc(p_l),
                         fa["dice"], fa["miou"], fa["hd95"], fa["assd"], fa["empty"]))
    return rows


def build_strategies(D, taus):
    """Long table of the six strategies (metrics of the selected or new mask) on the same rows."""
    from phase3 import evaluate_gates as E
    S = E.strategy_metrics(D)
    S = S[S.strategy.isin(["LANG", "IMG", "D-Gate", "LR-Gate"])]
    base = D[["dataset", "part", "group", "method", "seed", "sample_id", "condition", "variant_id"]]
    tau = np.array([taus[(m, s)] for m, s in zip(D.method, D.seed)])
    sw = core.sdc_switch(D.sdc.to_numpy(), tau)
    t = base.copy()
    t["strategy"], t["switch"] = "SDC-Fallback", sw.astype(float)
    for k in core.METRICS:
        t[k] = np.where(sw, D[f"img_{k}"], D[f"lang_{k}"])
    f = base.copy()
    f["strategy"], f["switch"] = "Fixed Average", np.nan
    for k in core.METRICS:
        f[k] = D[f"fa_{k}"].to_numpy()
    return pd.concat([S, t, f], ignore_index=True), sw


def pairwise(S):
    from phase3 import stats_phase3 as P
    rows = []
    specs = [(("LR-Gate", "SDC-Fallback"), core.WRONG, "main"), (("LR-Gate", "Fixed Average"), core.WRONG, "main")]
    specs += [((x, "LANG"), ["C0", "C2"], "clean_cost") for x in ("IMG", "D-Gate", "LR-Gate", "SDC-Fallback", "Fixed Average")]
    for (a, b), conds, kind in specs:
        for (ds, part, m, c), (u, n_img) in P.group_differences(S, a, b).items():
            if c not in conds:
                continue
            r = P.t_over_groups(u.values)
            row = {"comparison": f"{a} - {b}", "kind": kind, "dataset": ds, "method": core.NAME[m], "condition": c,
                   "n_images": n_img, "n_groups": r["n_groups"], "delta": r["delta"], "ci95_low": r["ci95_low"],
                   "ci95_high": r["ci95_high"], "label": POST_HOC}
            if kind == "clean_cost":
                row["margin"] = -0.005
                row["ci95_low_above_margin"] = bool(r["ci95_low"] >= -0.005)
            rows.append(row)
    return pd.DataFrame(rows)


def mixed(S):
    """Expected Dice curves and exact crossovers on the common C4/C5/C6 population."""
    common = set.intersection(*[set(S[S.condition == c].sample_id) for c in core.WRONG])
    P = S[S.sample_id.isin(common) & S.condition.isin(["C0"] + core.WRONG)]
    if (len(common), P.group.nunique()) != (2081, 472):
        guards.stop(f"common population {len(common)} images / {P.group.nunique()} groups, expected 2081 / 472")
    Dm = {st: core.group_weighted_mean(P[P.strategy == st]) for st in MIXED}
    curves, cross = [], []
    grid = np.round(np.arange(101) / 100, 2)
    for m in core.METHODS:
        d = {st: {c: float(Dm[st][(m, c)]) for c in ["C0"] + core.WRONG} for st in MIXED}
        for st in MIXED:
            dw = np.mean([d[st][c] for c in core.WRONG])
            for p in grid:
                curves.append({"method": core.NAME[m], "strategy": st, "p": p, "D_C0": d[st]["C0"], "D_wrong": dw,
                               "expected_dice": core.expected(d[st]["C0"], dw, p)})
        for a in ("LR-Gate", "SDC-Fallback", "Fixed Average"):
            for mix, cs in (("equal C4/C5/C6", core.WRONG), ("C4 only", ["C4"]), ("C5 only", ["C5"]), ("C6 only", ["C6"])):
                ivs = []
                for b in ("LANG", "IMG"):
                    dc = d[a]["C0"] - d[b]["C0"]
                    dw = np.mean([d[a][c] for c in cs]) - np.mean([d[b][c] for c in cs])
                    ps = core.crossover(dc, dw)
                    ivs.append(core.better_interval(dc, dw))
                    cross.append({"method": core.NAME[m], "strategy": a, "versus": b, "mixture": mix, "delta_clean": dc,
                                  "delta_wrong": dw, "p_star": "none" if ps is None else repr(ps)})
                both = core.intersect(*ivs) if mix == "equal C4/C5/C6" else None
                cross.append({"method": core.NAME[m], "strategy": a, "versus": "both LANG and IMG", "mixture": mix,
                              "better_than_both_p_range": ("" if mix != "equal C4/C5/C6" else
                                                           "none" if both is None else f"{both[0]!r} - {both[1]!r}")})
    return pd.DataFrame(curves), pd.DataFrame(cross)


def compute_summary(res):
    c = pd.read_csv(os.path.join(guards.REPO, COST)).set_index(["method", "mode"]).mean_ms
    img_ms = float(c[("nnU-Net", "IMG only")])
    rows = []
    for m in core.METHODS:
        lang_ms = float(c[(core.NAME[m], "LANG only")])
        for cond in core.CONDS:
            fr = float(res[(res.method == m) & (res.condition == cond) & (res.strategy == "SDC-Fallback")].fallback_rate.iloc[0])
            for st, ms in (("LANG", lang_ms), ("IMG", img_ms), ("D-Gate", lang_ms + img_ms), ("LR-Gate", lang_ms + img_ms),
                           ("Fixed Average", lang_ms + img_ms), ("SDC-Fallback", lang_ms + fr * img_ms)):
                rows.append({"method": core.NAME[m], "condition": cond, "strategy": st, "approx_forward_ms": ms,
                             "lang_ms": lang_ms, "img_ms": img_ms, "sdc_fallback_rate": fr if st == "SDC-Fallback" else np.nan,
                             "note": "approximation: job 391 mean_ms (validation, 64 QaTa images, seed 1001, C0); score overhead ignored"})
    return pd.DataFrame(rows)


def log(event, msg=""):
    p = os.path.join(guards.REPO, OUTS["log"])
    new = not os.path.exists(p)
    with open(p, "a") as f:
        if new:
            f.write("event\ttime_utc\tjob_id\thead\tnote\n")
        f.write(f"{event}\t{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M:%S}\t"
                f"{os.environ.get('SLURM_JOB_ID', 'none')}\t{guards.git(guards.REPO, 'rev-parse', 'HEAD').stdout.strip()}\t{msg}\n")


def main():
    from phase2 import lock
    lock.require_open(what="Phase 5 test computation")
    n_manifest = preflight()
    log("start", f"manifest files verified: {n_manifest}")
    from phase3 import evaluate_gates as E
    D, arch = E.load_rows("test")
    if arch != "nnunet":
        guards.stop(f"IMG architecture {arch}")
    D = E.apply_gates(D)
    D = D[(D.dataset == DATASET) & D.condition.isin(core.CONDS)].copy()
    if len(D) != N_ROWS:
        guards.stop(f"{len(D)} test rows, expected {N_ROWS}")
    with Pool(9, initializer=init_worker) as pool:
        parts = pool.map(test_rows, [(m, s) for m in core.METHODS for s in core.SEEDS])
    X = pd.DataFrame([x for p in parts for x in p], columns=core.KEYS + ["sdc"] + [f"fa_{k}" for k in core.METRICS])
    D = D.merge(X, on=core.KEYS, how="left", validate="one_to_one")
    if D.sdc.isna().any() or len(D) != N_ROWS:
        guards.stop("SDC / Fixed Average rows do not match the test rows one to one")
    cal = pd.read_csv(os.path.join(guards.REPO, CALIB))
    code = {v: k for k, v in core.NAME.items()}
    taus = {(code[r.model], int(r.seed)): float(r.threshold) for r in cal.itertuples()}
    S, sw = build_strategies(D, taus)
    res = E.image_weighted(S)
    res = res[res.strategy.isin(STRATS)]
    out = {"results": res.assign(method=res.method.map(core.NAME), label=POST_HOC),
           "pairwise": pairwise(S), "compute": compute_summary(res)}
    out["mixed"], out["crossover"] = mixed(S)
    pc = D[core.KEYS].copy()
    pc["sdc"], pc["sdc_fallback"] = D.sdc.to_numpy(), sw.astype(int)
    for k in core.METRICS:
        pc[f"fixed_average_{k}"] = D[f"fa_{k}"].to_numpy()
    out["per_case"] = pc
    for k in ("results", "pairwise", "compute", "mixed", "crossover"):
        guards.write_atomic(out[k], guards.REPO, OUTS[k], float_format="%.10g")
    guards.write_atomic(out["per_case"], guards.REPO, OUTS["per_case"], float_format="%.17g")   # full precision (SDC < tau)
    log("end", f"rows {len(D)}")
    print(out["crossover"].to_string(index=False))


if __name__ == "__main__":
    main()
