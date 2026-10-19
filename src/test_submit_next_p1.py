"""Tests of the submission decision of src/submit_next_p1.py (no job is submitted, squeue is not called).

Ending jobs (state CG) count neither as waiting nor for the 2-job limit; pending jobs (PD) are waiting.
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


def main():
    failed = 0
    for jobs, allowed in CASES:
        got = S.refusal(jobs) is None
        print(f"[{'ok' if got == allowed else 'FAIL'}] {[j[1] for j in jobs]} -> {'allowed' if got else 'refused'}")
        failed += got != allowed
    print("ALL CHECKS PASSED" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
