"""Test of the pre-specified non-finite rule in src/phase1/train_reclmis.py (CPU, stub model, no data).

A stub model with one parameter is "trained" for 4 steps; step 3 returns a NaN CCL loss. train_one_epoch must
raise NonFiniteLoss at step 3 naming loss_ccl, the parameter must carry exactly the updates of steps 1-2 (no
update from step 3), and the statistics must count 2 normal steps and 1 non-finite step.

Usage:  python src/phase1/test_nonfinite_reclmis.py      (exit status 1 if any check fails)
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import train_reclmis as T  # noqa: E402


class Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))
        self.aux = True


def main():
    model = Stub()
    calls = {"n": 0}

    def fake_forward(m, sample, device):
        calls["n"] += 1
        pred = torch.sigmoid(m.w * torch.ones(1, 1, 2, 2))
        aux = {k: m.w * 0 + 1.0 for k in T.AUX_LOSSES}           # finite auxiliary losses ...
        if calls["n"] == 3:
            aux["loss_ccl"] = m.w * float("nan")                 # ... except CCL at step 3
        return pred, torch.ones(1, 2, 2), aux

    T.forward_batch = fake_forward
    conf = types.SimpleNamespace(loss_weight={"loss_criterion": 5, "loss_ccl": 0.2, "loss_text_rec": 1, "loss_img_rec": 1})
    crit = lambda p, m: ((p - m.unsqueeze(1)) ** 2).mean()          # noqa: E731
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    stats = {"steps": 0, "rec_loss_zero_steps": 0, "non_finite_steps": 0}
    after = []
    orig_step = opt.step
    opt.step = lambda *a, **k: (orig_step(*a, **k), after.append(model.w.item()))[0]
    ok = True
    try:
        T.train_one_epoch(model, [({}, ["a"])] * 4, crit, opt, conf, "cpu", stats)
        print("[FAIL] no exception"); ok = False
    except T.NonFiniteLoss as e:
        c1 = e.step == 3 and e.terms == ["loss_ccl"]
        c2 = len(after) == 2 and model.w.item() == after[-1]
        c3 = stats["steps"] == 2 and stats["non_finite_steps"] == 1
        for name, c in [("stops at step 3, term loss_ccl", c1), ("no update from the non-finite step", c2),
                        ("statistics: 2 steps, 1 non-finite step", c3)]:
            print(f"[{'ok' if c else 'FAIL'}] {name}")
            ok &= c
    print("ALL CHECKS PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
