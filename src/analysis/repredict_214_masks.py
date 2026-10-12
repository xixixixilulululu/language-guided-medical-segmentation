"""One-off repair of job 214 (nnU-Net QaTa-COV19 seed 1001): re-predict the test set once to complete the
predicted test masks. Decided on 2026-10-12; run exactly once, whatever the outcome.

Background:
- Job 214 was stopped by the partition time limit while writing the predicted test masks.
- 1217 of 2113 masks had been written; the last of them was truncated to 0 bytes and is counted with the
  missing ones.
- per_case.csv and summary.json were complete.
- Masks regenerated from the float16 test_probs.npz differ from the scored float32 predictions in 221
  cases (boundary pixels at the 0.5 threshold).

Procedure (prints only counts and checksums, never a score):
  1. Check that the sha256 of eval_test/per_case.csv and summary.json equal the values recorded before
     the job (repredict_214/sha256_before.txt).
  2. Predict the test set once with the already selected checkpoint (selection.json, not re-selected),
     using exactly the original inference: nnunet_select.make_predictor / predict.
     - Settings: tile step 0.5, Gaussian weighting, mirroring TTA, lesion probability in float32.
     - Masks: metrics.binarize (prob > 0.5), as in runner.final_evaluation.
     - New masks go to repredict_214/masks/. Nothing is scored, saved as probabilities, or overwritten.
  3. Check 1: the new masks of the 1217 original cases vs. the masks job 214 wrote itself, pixel by pixel.
  4. Check 2: the new masks of the 896 missing cases, Dice vs. per_case.csv (|difference| < 1e-6).
  5. Only if both checks have zero mismatches: copy those 896 new masks into eval_test/masks/.
     The 1217 original masks are never touched. Otherwise nothing is copied: the float16-derived masks stay.
  6. Re-check both sha256 values.
The outcome is written to repredict_214/result.json.

Usage (GPU, inside slurm/repredict_214.slurm):  python src/analysis/repredict_214_masks.py
"""
import hashlib
import json
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from lgms import data, metrics  # noqa: E402
import nnunet_select as ns  # noqa: E402  (the original selection / final-prediction code)

RUN = os.path.join(REPO, "runs", "nnunet", "QaTa-COV19", "seed1001")
WORK = os.path.join(RUN, "repredict_214")
MODEL_DIR = os.path.join(REPO, "runs", "nnunet", "nnUNet_results_seed1001", "Dataset501_QaTaCOV19",
                         "nnUNetTrainerLGMS__nnUNetPlans__2d")
RAW = os.path.join(os.environ["nnUNet_raw"], "Dataset501_QaTaCOV19", "imagesTs")


def sha256s():
    return {n: hashlib.sha256(open(os.path.join(RUN, "eval_test", n), "rb").read()).hexdigest()
            for n in ["per_case.csv", "summary.json"]}


def main():
    result = {}
    recorded = {line.split()[1].split("/")[-1]: line.split()[0]
                for line in open(os.path.join(WORK, "sha256_before.txt")) if line.strip()}
    before = sha256s()
    result["sha256_before_matches_record"] = before == recorded
    if before != recorded:
        raise SystemExit("per_case.csv / summary.json changed since they were recorded -- stopping")

    ids = data.split_ids("QaTa-COV19", "test")
    orig_ids = open(os.path.join(WORK, "original_ids.txt")).read().split()
    missing_ids = open(os.path.join(WORK, "missing_ids.txt")).read().split()
    assert len(orig_ids) == 1217 and len(missing_ids) == 896 and sorted(orig_ids + missing_ids) == sorted(ids)

    # 2. one prediction with the selected checkpoint and the original inference code
    best = json.load(open(os.path.join(RUN, "selection.json")))["best_checkpoint"]
    result["checkpoint"] = best
    predictor = ns.make_predictor(MODEL_DIR, best)                      # mirroring TTA on, as originally
    probs = ns.predict(predictor, [os.path.join(RAW, f"{i}_0000.png") for i in ids], True, 4)
    new_dir = os.path.join(WORK, "masks")
    os.makedirs(new_dir, exist_ok=True)
    new = {}
    for i, p in zip(ids, probs.astype(np.float32)):                   # same as runner.final_evaluation
        new[i] = metrics.binarize(p)
        cv2.imwrite(os.path.join(new_dir, f"{i}.png"), new[i].astype(np.uint8) * 255)

    # 3. check 1: new vs. original masks of job 214, pixel by pixel
    official = os.path.join(RUN, "eval_test", "masks")
    mism1 = sum(not np.array_equal(new[i], cv2.imread(os.path.join(official, f"{i}.png"), cv2.IMREAD_GRAYSCALE) > 0)
                for i in orig_ids)
    result["check1_original_cases"] = len(orig_ids)
    result["check1_mismatching_cases"] = int(mism1)

    # 4. check 2: new masks of the missing cases vs. the scored per-case Dice
    pc = pd.read_csv(os.path.join(RUN, "eval_test", "per_case.csv"), dtype={"sample_id": str}).set_index("sample_id")
    mism2 = sum(abs(metrics.dice_score(new[i], data.read_mask("QaTa-COV19", "test", i) > 0) - pc.loc[i, "dice"]) >= 1e-6
                for i in missing_ids)
    result["check2_missing_cases"] = len(missing_ids)
    result["check2_mismatching_cases"] = int(mism2)

    # 5. copy only if both checks are fully consistent
    ok = mism1 == 0 and mism2 == 0
    if ok:
        for i in missing_ids:
            shutil.copyfile(os.path.join(new_dir, f"{i}.png"), os.path.join(official, f"{i}.png"))
    result["copied_896_masks"] = bool(ok)
    result["outcome"] = ("both checks consistent: the 896 re-predicted masks were copied into eval_test/masks"
                         if ok else "inconsistent: nothing copied; the float16-derived masks remain (option 2)")

    # 6. the scored files must be unchanged
    after = sha256s()
    result["sha256_after_unchanged"] = after == before
    json.dump(result, open(os.path.join(WORK, "result.json"), "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
