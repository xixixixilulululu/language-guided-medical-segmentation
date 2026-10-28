"""Phase 3, Task 3.1A: inference-only rerun of the image-only models (IMG) on the VALIDATION split, float32.

Why: the Phase 0 validation probabilities (runs/{unet,nnunet}/<ds>/seed<k>/val_probs.npz) are float16 archives,
and thresholding them does not reproduce the scored masks exactly (up to 11 pixels per case, see
reports/PHASE3_SETUP.md). Phase 3 needs float32 probabilities that match their scored masks, so both are
re-created here in one pass, with the same frozen checkpoint and the Phase 0 inference settings:
  U-Net   runs/unet/<ds>/seed<k>/best_model.pt; src/train_unet.py (UNetDataset, build_model, predict),
          eval mode, batch val_batch_size, sigmoid foreground probability, no TTA
  nnU-Net the selected snapshot of runs/nnunet/<ds>/seed<k>/selection.json in
          runs/nnunet/nnUNet_results_seed<k>/<nnU-Net dataset>/nnUNetTrainerLGMS__nnUNetPlans__2d/fold_0;
          src/nnunet_select.py (make_predictor, predict): sliding window, Gaussian weighting, mirroring TTA
          on (nnU-Net defaults, as in Phase 0), softmax probability of the lesion class
Scoring: the common evaluator's case function (prob > 0.5). New probabilities, masks and scores are saved
together; old scores are never attached to new masks. Reconciliation with the Phase 0 archive (per-case Dice
and the scored mask PNGs) is written as a reproducibility record only.

Validation only: the test split is not accepted (Task 3.6: test after both protocols are frozen).

Outputs: runs/phase3/val/img/<arch>/<dataset>/seed<k>/
  per_case.csv (dataset, split, arch, checkpoint_sha256, seed, sample_id, dice, miou, hd95, assd, empty),
  masks.npz (packed bits), probs.npz (float32), reconcile_phase0.csv, passes.tsv (Slurm job id)

Usage (inside a Slurm job):  python src/phase3/infer_img.py --arch unet --dataset QaTa-COV19
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
SEEDS = [1001, 1002, 1003]
NNUNET_DS = {"QaTa-COV19": "Dataset501_QaTaCOV19", "MosMedData+": "Dataset502_MosMedDataPlus"}
OUT = os.path.join(REPO, "runs", "phase3", "val", "img")
CKPTS = os.path.join(REPO, "results", "phase3_checkpoints.tsv")


def recorded_checkpoint(arch, dataset, seed):
    t = pd.read_csv(CKPTS, sep="\t", dtype=str)
    r = t[(t.role == "IMG") & (t.model == arch) & (t.dataset == dataset) & (t.seed == str(seed))]
    if len(r) != 1:
        raise KeyError(f"{arch} {dataset} {seed} not in {CKPTS}")
    return os.path.join(REPO, r.path.iloc[0]), r.sha256.iloc[0]


def predict_unet(path, dataset, ids, images, masks):
    import torch
    from torch.utils.data import DataLoader
    import train_unet as U
    from lgms import runner
    cfg = runner.load_config(os.path.join(REPO, "configs", "unet.yaml"))
    model = U.build_model(cfg)
    model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    model = model.to(torch.device("cuda")).eval()
    loader = DataLoader(U.UNetDataset(ids, images, masks, train=False), batch_size=cfg["val_batch_size"],
                        shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    with torch.no_grad():
        out_ids, probs = U.predict(model, loader, torch.device("cuda"))
    if list(out_ids) != list(ids):
        raise RuntimeError("U-Net prediction order differs from the ids")
    return probs.astype(np.float32)


def predict_nnunet(path, dataset, ids):
    import nnunet_select as N
    raw = os.path.join(os.environ["nnUNet_raw"], NNUNET_DS[dataset])
    files = [os.path.join(raw, "imagesTr", f"{i}_0000.png") for i in ids]
    model_dir, checkpoint = os.path.dirname(os.path.dirname(path)), os.path.basename(path)
    return N.predict(N.make_predictor(model_dir, checkpoint), files, True, 4).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=["unet", "nnunet"])
    ap.add_argument("--dataset", required=True, choices=list(NNUNET_DS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    import cv2
    from lgms import data, runner
    from phase2.infer import score, sha_file
    ids, images, masks = data.load_split_arrays(args.dataset, "val")
    masks_by_id = dict(zip(ids, masks))
    for seed in args.seeds:
        out = os.path.join(OUT, args.arch, args.dataset, f"seed{seed}")
        if os.path.exists(os.path.join(out, "per_case.csv")):
            print(f"{args.arch} {args.dataset} {seed}: done earlier, skipped", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        path, rec = recorded_checkpoint(args.arch, args.dataset, seed)
        if sha_file(path) != rec:
            raise RuntimeError(f"{path}: sha256 differs from {CKPTS}")
        runner.set_seed(seed, True)
        t0 = time.time()
        probs = (predict_unet(path, args.dataset, ids, images, masks) if args.arch == "unet"
                 else predict_nnunet(path, args.dataset, ids))
        m = score(ids, probs, masks_by_id)
        np.savez_compressed(os.path.join(out, "masks.npz"), **{i: np.packbits(p > 0.5) for i, p in zip(ids, probs)})
        np.savez_compressed(os.path.join(out, "probs.npz"), **dict(zip(ids, probs)))
        # reproducibility record against the Phase 0 archive (scores are NOT taken from it)
        p0 = os.path.join(REPO, "runs", args.arch, args.dataset, f"seed{seed}", "eval_val")
        old = pd.read_csv(os.path.join(p0, "per_case.csv")).set_index("sample_id")
        rec_rows = [{"sample_id": i, "dice_phase3": d, "dice_phase0": old.dice[i], "abs_dice_diff": abs(d - old.dice[i]),
                     "mask_pixels_differing": int(((cv2.imread(os.path.join(p0, "masks", f"{i}.png"), 0) > 0) != (p > 0.5)).sum())}
                    for i, d, p in zip(ids, m.dice, probs)]
        pd.DataFrame(rec_rows).to_csv(os.path.join(out, "reconcile_phase0.csv"), index=False, float_format="%.6f")
        m.insert(0, "seed", seed)
        m.insert(0, "checkpoint_sha256", rec)
        m.insert(0, "arch", args.arch)
        m.insert(0, "split", "val")
        m.insert(0, "dataset", args.dataset)
        m.to_csv(os.path.join(out, "per_case.csv"), index=False, float_format="%.6f")   # written last
        with open(os.path.join(out, "passes.tsv"), "a") as f:
            f.write(f"{os.environ.get('SLURM_JOB_ID', 'none')}\t{time.strftime('%Y-%m-%d %H:%M:%S')}\t{len(ids)}\n")
        r = pd.DataFrame(rec_rows)
        print(json.dumps({"arch": args.arch, "dataset": args.dataset, "seed": seed, "mean_dice": float(m.dice.mean()),
                          "phase0_mean_dice": float(old.dice.mean()), "max_abs_dice_diff": float(r.abs_dice_diff.max()),
                          "cases_mask_differs": int((r.mask_pixels_differing > 0).sum()),
                          "max_mask_pixels_differing": int(r.mask_pixels_differing.max()),
                          "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
