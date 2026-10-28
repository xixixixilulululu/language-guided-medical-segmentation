"""Fill the generated tables of the freeze drafts from the result files (rerun after any amendment).

reports/PHASE3_PROTOCOL_FREEZE.md: IMG checkpoints, partitions, D-Gate thresholds, LR-Gates, Phase 3 file sha256
reports/PHASE2_PROTOCOL_FREEZE.md: Phase 2 file sha256
Each table sits between <!-- BEGIN NAME --> and <!-- END NAME --> (a bare <!-- NAME --> is replaced on the first
run). The sha256 tables are snapshots of the working tree (src/phase2/lock.py manifest); the freeze record
configs/phase2_frozen/phase2_lock.json is authoritative. The two freeze documents are not listed in their own
tables (a file cannot contain its own hash).
Usage:  python src/phase3/make_freeze_tables.py
"""
import json
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
R = lambda f: os.path.join(REPO, "results", f)  # noqa: E731
KEYS = [f"{m}_seed{s}" for m in ("LViT-T", "RecLMIS", "DD-CMD") for s in (1001, 1002, 1003)]


def put(text, name, table):
    block = f"<!-- BEGIN {name} -->\n{table}\n<!-- END {name} -->"
    pat = re.compile(rf"<!-- BEGIN {name} -->.*?<!-- END {name} -->", re.S)
    if pat.search(text):
        return pat.sub(lambda _: block, text)
    if f"<!-- {name} -->" not in text:
        raise KeyError(name)
    return text.replace(f"<!-- {name} -->", block)


def md(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def sha_table(files, skip):
    from phase2 import lock
    rows = [{"file": f"`{p}`", "sha256": f"`{d}`"} for p, d in lock.manifest(files=files) if p not in skip]
    return md(rows, ["file", "sha256"])


def main():
    from phase2 import lock
    ck = pd.read_csv(R("phase3_checkpoints.tsv"), sep="\t", dtype=str)
    img = ck[(ck.role == "IMG") & (ck.model == "nnunet")]
    t_img = md([{"dataset": r.dataset, "seed": r.seed, "checkpoint": f"`{r.path}`", "sha256": f"`{r.sha256}`"}
                for r in img.itertuples()], ["dataset", "seed", "checkpoint", "sha256"])
    p = pd.read_csv(R("phase3_gate_partitions.csv"), dtype=str)
    t_part = md([{"partition": k, "groups": g.group_id.nunique(), "images": len(g)} for k, g in p.groupby("partition")],
                ["partition", "groups", "images"])
    dg = json.load(open(R("phase3_dgate_thresholds.json")))
    dc = pd.read_csv(R("phase3_dgate_calibration.csv"))
    choice = {f"{r.method}_seed{r.seed}": r.choice for r in dc.itertuples()}
    t_d = md([{"model × seed": k.replace("_seed", " "), "t_D (F1 >)": repr(dg[k]["threshold"]), "choice": choice[k]}
              for k in KEYS], ["model × seed", "t_D (F1 >)", "choice"])
    art = json.load(open(R("phase3_lr_artifacts.json")))
    lc = pd.read_csv(R("phase3_lr_calibration.csv"))
    lchoice = {f"{r.method}_seed{r.seed}": r.choice for r in lc.itertuples()}
    f = lambda v: ", ".join(f"{x:.6g}" for x in v)  # noqa: E731
    t_lr = md([{"model × seed": k.replace("_seed", " "), "t_LR (score >)": repr(art[k]["threshold"]),
                "choice": lchoice[k], "intercept": f"{art[k]['intercept']:.6g}", "coef F1–F5": f(art[k]["coef"]),
                "scaler mean": f(art[k]["scaler_mean"]), "scaler scale": f(art[k]["scaler_scale"])} for k in KEYS],
              ["model × seed", "t_LR (score >)", "choice", "intercept", "coef F1–F5", "scaler mean", "scaler scale"])
    t_lr += ("\n\nCoefficients and scaler values are shown to 6 significant digits; the JSON file holds them at "
             "full precision and is the one the gates read.")
    skip = {lock.PHASE2_FREEZE, lock.PHASE3_FREEZE}
    p3 = os.path.join(REPO, lock.PHASE3_FREEZE)
    s = open(p3).read()
    for name, t in (("IMGCKPT", t_img), ("PARTITIONS", t_part), ("DGATE", t_d), ("LR", t_lr),
                    ("MANIFEST-PHASE3", sha_table(lock.PHASE3_FILES, skip))):
        s = put(s, name, t)
    open(p3, "w").write(s)
    p2 = os.path.join(REPO, lock.PHASE2_FREEZE)
    s = put(open(p2).read(), "MANIFEST-PHASE2", sha_table(lock.PHASE2_FILES + [lock.DECISION], skip))
    open(p2, "w").write(s)
    print("tables written")


if __name__ == "__main__":
    main()
