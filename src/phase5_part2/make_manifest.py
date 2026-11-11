"""Phase 5 Part 2, Task 5.2.2: sha256 manifest of every input of the U-Net test computation (data and code).

Data: test LANG probability maps, archived masks and per-case files of C0, C2, C4-C6 (src/phase3/features.py
passes()); the U-Net test probs.npz, masks.npz, per_case.csv and reconcile_phase0.csv of the three seeds; the QaTa
test ground-truth masks and id list (lgms.data, under slurm/env_phase1.sh); the committed tables read by
src/phase5_part2/compute.py. Code: src/phase5_part2/*.py and every in-repository module imported, the test job and
its environment scripts. Output: configs/phase5_part2/artifact_manifest.tsv (path, sha256, bytes); never overwritten.
Usage (Slurm job slurm/phase5_part2_manifest.slurm):  python src/phase5_part2/make_manifest.py
"""
import glob
import importlib
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase5 import core, guards  # noqa: E402

OUT = os.path.join("configs", "phase5_part2", "artifact_manifest.tsv")
TABLES = ["phase3_feature_labels_test.csv.gz", "phase2_test_per_case.csv.gz", "phase5_test_per_case.csv.gz",
          "phase5_test_results.csv", "phase5_crossover_summary.csv", "phase5_part2_unet_img_test_per_case.csv.gz"]
CODE_MODULES = ["phase5_part2.compute", "phase5_part2.reconcile_unet", "phase5_part2.make_manifest", "phase5.core",
                "phase5.guards", "phase3.features", "phase3.evaluate_gates", "phase3.stats_phase3", "phase3.infer_img",
                "phase2.lock", "phase2.infer", "lgms.data", "lgms.metrics", "analyze_phase0"]
SLURM = ["slurm/phase5_part2_test.slurm", "slurm/phase5_part2_unet_infer.slurm", "slurm/env_phase1.sh", "slurm/env.sh",
         "slurm/notify_trap.sh"]


def rel(path):
    a = os.path.abspath(path)
    if a.startswith(REPO + os.sep):
        return os.path.relpath(a, REPO)
    home = os.path.expanduser("~")
    return "~" + a[len(home):] if a.startswith(home) else a


def code_files():
    for m in CODE_MODULES:
        importlib.import_module(m)
    src = os.path.join(REPO, "src") + os.sep
    out = {os.path.abspath(f) for f in glob.glob(os.path.join(REPO, "src", "phase5_part2", "*.py"))}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if f and os.path.abspath(f).startswith(src) and f.endswith(".py"):
            out.add(os.path.abspath(f))
    return sorted(out) + [os.path.join(REPO, s) for s in SLURM]


def files():
    from lgms import data
    from phase3 import features as Fe
    out = code_files() + [os.path.join(REPO, "results", t) for t in TABLES]
    for m in core.METHODS:
        for s in core.SEEDS:
            for cond, var, fcsv, fnpz in Fe.passes(m, "QaTa-COV19", s, "test"):
                if cond in core.CONDS:
                    out += [fcsv, fnpz, os.path.join(os.path.dirname(fnpz), f"masks_{cond}_{var}.npz")]
    for s in core.SEEDS:
        d = os.path.join(REPO, "runs", "phase3", "test", "img", "unet", "QaTa-COV19", f"seed{s}")
        out += [os.path.join(d, f) for f in ("probs.npz", "masks.npz", "per_case.csv", "reconcile_phase0.csv")]
    out.append(os.path.join(data.common_root(), "QaTa-COV19", "test.txt"))
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
