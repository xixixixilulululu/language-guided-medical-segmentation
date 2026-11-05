"""Tests of src/phase4/stage1.py with synthetic data and temporary git repositories (no real data is read).

  auroc       C4 always above C0 -> 1; identical scores -> 0.5; reversed -> 0; ties count one half
  scores      donor averaging, unavailable rows excluded and counted, eligibility
  GATE_FIT    a GATE_CAL id stops the program
  self-checks refused: orientation pending; no orientation_decision; Stage 0 decision missing / not GO; dirty
              working tree; HEAD not pushed; output present
Usage:  python src/phase4/test_stage1.py      (exit status 1 if any check fails)
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4 import stage1 as St  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def refused(fn, *a, code=St.PREFLIGHT_EXIT):
    try:
        fn(*a)
    except SystemExit as e:
        return e.code == code
    return False


check("AUROC = 1 when every C4 score is above every C0 score", St.auroc([0.1, 0.2, 0.3], [0.5, 0.6, 0.9]) == 1.0)
check("AUROC = 0.5 for identical scores", St.auroc([0.2, 0.4, 0.4], [0.2, 0.4, 0.4]) == 0.5 and St.auroc([0.3] * 4, [0.3] * 4) == 0.5)
check("AUROC = 0 when reversed", St.auroc([0.9, 0.8], [0.1, 0.2]) == 0.0)
check("AUROC ties count one half", St.auroc([0.5], [0.5, 0.6]) == 0.75)

m = np.zeros((224, 224), bool); m[10, 10] = True                   # image-left upper
loc = {"t1": "right:upper", "dA": "left:upper", "dB": "right:all", "dU": "left:all;right:unspecified", "lat": "left:upper"}
T = St.target_scores(["t1"], lambda t: m, loc, {"t1": ["dA", "dB", "dU"]}, {"t1": ["dA"]}, "patient_left_on_image_right")
r = T.iloc[0]
check("C0 mismatch 0 when the report matches the IMG cell", r.c0 == 0.0)
check("C4 = mean over available donors (1 and 0) = 0.5, one unavailable counted", r.c4 == 0.5 and r.n_c4 == 2 and r.n_c4_unavailable == 1)
check("C6 single donor mismatch 1", r.c6 == 1.0 and bool(r.eligible))
T2 = St.target_scores(["t1"], lambda t: np.zeros((224, 224), bool), loc, {"t1": ["dA"]}, {"t1": ["dA"]}, "patient_left_on_image_right")
check("empty IMG -> target not eligible", not bool(T2.iloc[0].eligible) and T2.iloc[0].c0_reason == "img_empty")
check("GATE_CAL id stops before any mask is read", refused(St.require_fit, ["a", "b"], {"a": "GATE_FIT", "b": "GATE_CAL"}))
check("GATE_FIT ids pass", not refused(St.require_fit, ["a"], {"a": "GATE_FIT"}))
s = St.summarise(St.target_scores(["t1"], lambda t: m, loc, {"t1": ["dA"]}, {"t1": ["dA"]}, "patient_left_on_image_right"),
                 {"t1": "g1"}, "x")
check("summary: decision rule applied (AUROC 1, both differences > 0 -> continue)", s["decision"] == "continue" and s["auroc_c0_vs_c4"] == 1.0)


def repo(spec_orientation="patient_left_on_image_left", decision=True, stage0="GO", push=True):
    base = tempfile.mkdtemp()
    origin, d = os.path.join(base, "o.git"), os.path.join(base, "r")
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", origin], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", d], check=True)
    run = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True, check=True)  # noqa: E731
    run("config", "user.email", "t@t"); run("config", "user.name", "t"); run("remote", "add", "origin", origin)
    os.makedirs(os.path.join(d, "configs", "phase4")); os.makedirs(os.path.join(d, "results"))
    spec = {"orientation": {"value": spec_orientation}}
    if decision:
        spec["orientation_decision"] = {"decided_by": "supervisor", "date": "2026-11-10", "commit": "abc"}
    json.dump(spec, open(os.path.join(d, St.SPEC), "w"))
    if stage0:
        json.dump({"decision": stage0}, open(os.path.join(d, St.STAGE0), "w"))
    run("add", "-A"); run("commit", "-q", "-m", "x")
    if push:
        run("push", "-q", "origin", "main")
    run("fetch", "-q", "origin")
    return d, run


d, run = repo()
check("all conditions met -> runs", not refused(St.preflight, d))
check("refused: orientation pending", refused(St.preflight, repo(spec_orientation="pending supervisor decision")[0]))
check("refused: no orientation_decision", refused(St.preflight, repo(decision=False)[0]))
check("refused: Stage 0 decision file missing", refused(St.preflight, repo(stage0=None)[0]))
check("refused: Stage 0 decision STOP", refused(St.preflight, repo(stage0="STOP")[0]))
d, run = repo(); open(os.path.join(d, "x.txt"), "w").write("x")
check("refused: dirty working tree", refused(St.preflight, d))
d, run = repo(push=False)
check("refused: HEAD not pushed to origin", refused(St.preflight, d))
d, run = repo(); open(os.path.join(d, St.OUT_SUMMARY), "w").write("x"); run("add", "-A"); run("commit", "-qm", "o"); run("push", "-q", "origin", "main")
check("refused: output present", refused(St.preflight, d))

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
