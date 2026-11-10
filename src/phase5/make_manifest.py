"""Phase 5, Task 5.2: sha256 manifest of every input the Phase 5 programs read (run before the freeze is signed).

Inputs: validation and test LANG probability maps, archived masks and per-case files of C0, C2, C4, C5, C6 (rows
and paths as src/phase3/features.py passes()); IMG nnU-Net probabilities, masks and per-case files; the QaTa test
ground-truth masks and id list (lgms.data, as in the Phase 2 test inference); the existing tables in
configs/phase5/spec.json "inputs" and the SDC calibration outputs; and the CODE of the test run: src/phase5/*.py and
every in-repository module that the Phase 5 programs import (collected by importing them, lazy imports included),
slurm/phase5_test.slurm and the environment scripts it sources. Any later change is caught by the pre-check of
src/phase5/compute_test_split.py.
Output: configs/phase5/artifact_manifest.tsv (path, sha256, bytes); never overwritten.
Usage (Slurm job slurm/phase5_manifest.slurm):  python src/phase5/make_manifest.py
"""
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase5 import core, guards  # noqa: E402

OUT = os.path.join("configs", "phase5", "artifact_manifest.tsv")
TABLES = ["phase3_reliability_features.csv.gz", "phase3_feature_labels.csv.gz", "phase3_reliability_features_test.csv.gz",
          "phase3_feature_labels_test.csv.gz", "phase2_test_per_case.csv.gz", "phase3_img_test_per_case.csv.gz",
          "phase3_dgate_thresholds.json", "phase3_lr_artifacts.json", "phase3_fallback_selection.csv",
          "phase3_gate_partitions.csv", "phase3_computational_cost.csv", "phase5_sdc_calibration.csv", "phase5_sdc_val.csv.gz"]


def rel(path):
    """Repository-relative path, or '~/...' for data outside the repository."""
    a = os.path.abspath(path)
    if a.startswith(REPO + os.sep):
        return os.path.relpath(a, REPO)
    home = os.path.expanduser("~")
    return "~" + a[len(home):] if a.startswith(home) else a


CODE_MODULES = ["phase5.compute_test_split", "phase5.calibrate_sdc", "phase5.make_manifest", "phase5.make_figures",
                "phase5.core", "phase5.guards", "phase3.features", "phase3.gates", "phase3.evaluate_gates",
                "phase3.stats_phase3", "phase2.lock", "phase4.guard", "lgms.data", "lgms.metrics",
                "analyze_phase0"]                                         # analyze_phase0: lazy import of phase3 code
SLURM = ["slurm/phase5_test.slurm", "slurm/env_phase1.sh", "slurm/env.sh", "slurm/notify_trap.sh"]


def code_files():
    """Repository source files of the imported modules (and of everything they import), plus the job scripts."""
    import glob
    import importlib
    for m in CODE_MODULES:
        importlib.import_module(m)
    src = os.path.join(REPO, "src") + os.sep
    out = {os.path.abspath(f) for f in glob.glob(os.path.join(REPO, "src", "phase5", "*.py"))}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and os.path.abspath(f).startswith(src) and f.endswith(".py"):
            out.add(os.path.abspath(f))
    return sorted(out) + [os.path.join(REPO, s) for s in SLURM]


def files():
    from lgms import data
    from phase3 import features as Fe
    out = code_files() + [os.path.join(REPO, "results", t) for t in TABLES]
    for split in ("val", "test"):
        for m in core.METHODS:
            for s in core.SEEDS:
                for cond, var, fcsv, fnpz in Fe.passes(m, "QaTa-COV19", s, split):
                    if cond in core.CONDS:
                        out += [fcsv, fnpz, os.path.join(os.path.dirname(fnpz), f"masks_{cond}_{var}.npz")]
        for s in core.SEEDS:
            d = os.path.join(REPO, "runs", "phase3", split, "img", "nnunet", "QaTa-COV19", f"seed{s}")
            out += [os.path.join(d, f) for f in ("probs.npz", "masks.npz", "per_case.csv")]
    root = data.common_root()
    out.append(os.path.join(root, "QaTa-COV19", "test.txt"))
    out += [data.mask_path("QaTa-COV19", "test", i) for i in data.split_ids("QaTa-COV19", "test")]
    return out


def main():
    guards.outputs_absent(REPO, [OUT])
    rows = []
    for f in files():
        if not os.path.exists(f):
            guards.stop(f"input missing: {f}")
        rows.append({"path": rel(f), "sha256": guards.sha256_file(f), "bytes": os.path.getsize(f)})
    m = pd.DataFrame(rows)
    if m.path.duplicated().any():
        guards.stop("duplicate manifest paths")
    p = os.path.join(REPO, OUT)
    m.to_csv(p + ".tmp", sep="\t", index=False)
    os.replace(p + ".tmp", p)
    print(f"{len(m)} files, {m.bytes.sum() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
