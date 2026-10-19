"""Tests of the submission decision of src/submit_next_p1.py (no job is submitted, squeue is not called).

Ending jobs (state CG) count neither as waiting nor for the 2-job limit; pending jobs (PD) are waiting.
Row choice (pick_next, with a synthetic plan / log / run state): file order, --day / --night, continuation
segments first, a run is never submitted twice.
Usage:  python src/test_submit_next_p1.py      (exit status 1 if any check fails)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import submit_next_p1 as S  # noqa: E402

CASES = [  # (jobs as (id, state, name), submission allowed?)
    ([], True),
    ([("1", "R", "a")], True),
    ([("1", "CG", "a")], True),                                   # only an ending job
    ([("1", "R", "a"), ("2", "CG", "b")], True),                  # running + ending: still room for 1 waiting
    ([("1", "PD", "a")], False),                                  # a job is already waiting
    ([("1", "R", "a"), ("2", "PD", "b")], False),                 # 1 running + 1 waiting = limit
    ([("1", "CG", "a"), ("2", "R", "b"), ("3", "PD", "c")], False),
    ([("1", "R", "a"), ("2", "R", "b")], False),                  # 2 active jobs
]


def row(m, ds, seed, tag="-"):
    return {"method": m, "dataset": ds, "seed": seed, "tag": tag}


PLAN = [row("lvit", "QaTa-COV19", "1001"), row("ddcmd", "MosMedData+", "1001"), row("reclmis", "QaTa-COV19", "1001", "night"),
        row("lvit", "MosMedData+", "1001")]


def pick_cases():
    """(description, log keys, queued job names, runs with segments {key: [segments]}, dataset, expected)"""
    seg_end = [{"start_epoch": 1}, {"last_epoch": 160, "ended": "segment"}]
    return [
        ("file order", ["lvit|QaTa-COV19|1001"], [], {}, None, ("ddcmd", None)),
        ("--night skips MosMed rows", ["lvit|QaTa-COV19|1001"], [], {}, "QaTa-COV19", ("reclmis", None)),
        ("--day", ["lvit|QaTa-COV19|1001", "ddcmd|MosMedData+|1001"], [], {}, "MosMedData+", ("lvit", None)),
        ("segment 2 first, even with --day", ["lvit|QaTa-COV19|1001"], [], {"lvit|QaTa-COV19|1001": seg_end},
         "MosMedData+", ("lvit", 2)),
        ("segment 2 also after a kill by the 12 h limit (no end line), with --night", ["lvit|QaTa-COV19|1001"], [],
         {"lvit|QaTa-COV19|1001": [{"start_epoch": 1}]}, "QaTa-COV19", ("lvit", 2)),
        ("segment 3 after two guarded segments", ["lvit|QaTa-COV19|1001"], [],
         {"lvit|QaTa-COV19|1001": seg_end + [{"start_epoch": 160}, {"last_epoch": 320, "ended": "segment"}]},
         None, ("lvit", 3)),
        ("a finished run is not continued", ["lvit|QaTa-COV19|1001"], [],
         {"lvit|QaTa-COV19|1001": [{"start_epoch": 1}, {"last_epoch": 161, "ended": "early_stopping"}]},
         "QaTa-COV19", ("reclmis", None)),
        ("no segment while that run is queued", ["lvit|QaTa-COV19|1001"], ["p1-lvit-QaTa-1001"],
         {"lvit|QaTa-COV19|1001": seg_end}, None, ("ddcmd", None)),
        ("a run already queued under another row is not submitted twice",
         ["lvit|QaTa-COV19|1001", "ddcmd|MosMedData+|1001"], ["p1-reclmis-QaTa-1001"], {}, "QaTa-COV19", (None, None)),
        ("a run that has started is not submitted twice",
         ["lvit|QaTa-COV19|1001", "ddcmd|MosMedData+|1001"], [], {"reclmis|QaTa-COV19|1001": [{"start_epoch": 1}]},
         "QaTa-COV19", (None, None)),
    ]


def main():
    failed = 0
    for jobs, allowed in CASES:
        got = S.refusal(jobs) is None
        print(f"[{'ok' if got == allowed else 'FAIL'}] {[j[1] for j in jobs]} -> {'allowed' if got else 'refused'}")
        failed += got != allowed
    for desc, keys, queued, segs, ds, (want_m, want_seg) in pick_cases():
        S.segments = lambda r, segs=segs: segs.get(f"{r['method']}|{r['dataset']}|{r['seed']}", [])
        r, seg = S.pick_next(PLAN, [{"key": k} for k in keys], [("9", "R", n) for n in queued], ds)
        ok = (r["method"] if r else None) == want_m and seg == want_seg
        print(f"[{'ok' if ok else 'FAIL'}] pick: {desc} -> {S.label(r, seg) if r else 'nothing'}")
        failed += not ok
    print("ALL CHECKS PASSED" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
