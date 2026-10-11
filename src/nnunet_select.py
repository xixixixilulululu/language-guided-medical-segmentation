"""nnU-Net checkpoint selection on the full validation set, then test prediction (Phase 0).

For one finished nnU-Net run (fold 0 of nnUNetTrainerLGMS, 2d):
  1. every snapshot checkpoint_ep<epoch>.pth (every 50 epochs, see src/nnunet_ext/nnUNetTrainerLGMS.py)
     predicts the complete LViT validation set with nnU-Net's own predictor (sliding window, Gaussian
     weighting, mirroring TTA on -- nnU-Net defaults);
  2. the snapshot with the highest mean per-case validation Dice of the common evaluator is selected
     (ties -> earlier epoch); the test set is not touched in this step;
  3. the selected snapshot predicts the validation and test sets with lesion probabilities, which are
     scored by the common evaluator (prob > 0.5; identical to nnU-Net's argmax for two classes).

Outputs in --out: selection.csv (epoch, val_dice), selection.json, val/test probs npz, eval_val/, eval_test/.

Usage (inside a Slurm job; nnUNet_* and nnUNet_extTrainer environment variables set):
    python src/nnunet_select.py --dataset QaTa-COV19 --model-dir <nnUNet_results>/Dataset501_QaTaCOV19/
        nnUNetTrainerLGMS__nnUNetPlans__2d --out runs/nnunet/QaTa-COV19/seed1001
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from lgms import data, metrics, runner  # noqa: E402
from prepare_nnunet import NNUNET_IDS  # noqa: E402


def make_predictor(model_dir, checkpoint, use_mirroring=True):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=use_mirroring,
                        perform_everything_on_device=True, device=torch.device("cuda"), allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_dir, use_folds=(0,), checkpoint_name=checkpoint)
    return p


def predict(predictor, image_files, with_probs, n_proc):
    """Lesion probability (or 0/1 segmentation) maps (N, 224, 224), in the order of image_files."""
    out = predictor.predict_from_files([[f] for f in image_files], None, save_probabilities=with_probs,
                                       overwrite=True, num_processes_preprocessing=n_proc,
                                       num_processes_segmentation_export=n_proc)
    if with_probs:
        # each result: (segmentation (1, H, W), probabilities (C, 1, H, W)); keep P(lesion)
        return np.stack([r[1][1, 0] for r in out]).astype(np.float32)
    return np.stack([r[0] for r in out]).astype(np.float32).reshape(len(image_files), 224, 224)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=data.DATASETS)
    ap.add_argument("--model-dir", required=True, help="folder that contains fold_0/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset-val", type=int, help="first N validation cases (quick checks only)")
    ap.add_argument("--skip-test", action="store_true", help="quick checks: selection only")
    ap.add_argument("--n-proc", type=int, default=4)
    args = ap.parse_args()

    log = runner.make_logger(args.out)
    raw = os.path.join(os.environ["nnUNet_raw"], NNUNET_IDS[args.dataset][1])
    files = {"val": lambda i: os.path.join(raw, "imagesTr", f"{i}_0000.png"),
             "test": lambda i: os.path.join(raw, "imagesTs", f"{i}_0000.png")}
    val_ids = data.split_ids(args.dataset, "val")[: args.subset_val]
    val_gt = np.stack([data.read_mask(args.dataset, "val", i) for i in val_ids])

    # 1-2. validation Dice of every snapshot
    snaps = sorted(glob.glob(os.path.join(args.model_dir, "fold_0", "checkpoint_ep*.pth")))
    assert snaps, f"no snapshots in {args.model_dir}/fold_0"
    rows = []
    for s in snaps:
        epoch = int(re.search(r"ep(\d+)", s).group(1))
        with runner.Timer() as t:
            seg = predict(make_predictor(args.model_dir, os.path.basename(s)),
                          [files["val"](i) for i in val_ids], False, args.n_proc)
            dice = float(metrics.dice_many(seg, val_gt).mean())
        rows.append({"epoch": epoch, "checkpoint": os.path.basename(s), "val_dice": dice, "seconds": t.seconds})
        log.info(f"snapshot epoch {epoch:4d}: val Dice {dice:.4f} ({t.seconds:.0f} s, {len(val_ids)} cases)")
    sel = pd.DataFrame(rows)
    sel.to_csv(os.path.join(args.out, "selection.csv"), index=False, float_format="%.6f")
    best = sel.sort_values(["val_dice", "epoch"], ascending=[False, True]).iloc[0]
    runner.save_json({"best_epoch": int(best.epoch), "best_checkpoint": best.checkpoint,
                      "best_val_dice": float(best.val_dice), "n_snapshots": len(sel),
                      "n_val_cases": len(val_ids)}, os.path.join(args.out, "selection.json"))
    log.info(f"selected {best.checkpoint} (val Dice {best.val_dice:.4f})")
    if args.skip_test:
        return

    # 3. final predictions of the selected snapshot, scored by the common evaluator
    predictor = make_predictor(args.model_dir, best.checkpoint)
    for split in ["val", "test"]:
        ids = data.split_ids(args.dataset, split)
        probs = predict(predictor, [files[split](i) for i in ids], True, args.n_proc)
        runner.final_evaluation(args.out, args.dataset, split, ids, probs, log)


if __name__ == "__main__":
    main()
