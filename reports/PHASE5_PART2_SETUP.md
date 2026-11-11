# Phase 5 Part 2 — Setup: frozen U-Net test predictions (Task 5.2.1)

Status: 2026-11-11. Plan: [PHASE5_PART2_PLAN.md](PHASE5_PART2_PLAN.md) (commit c8854eb); rules:
`configs/phase5_part2/spec.json` (commit 90a7bd2). Post-hoc sensitivity analysis on a test split already examined
in Phases 2, 3 and 5. Inference only; no model was trained.

## 1. Checkpoint and configuration per seed

| Seed | Checkpoint (results/phase3_checkpoints.tsv rows 20, 22, 24) | sha256 (checked by infer_img.py) | Phase 0 configuration |
|---|---|---|---|
| 1001 | `runs/unet/QaTa-COV19/seed1001/best_model.pt` | `557be758ffd879a59fade3ab04d730f98c82a08cbfd61cc219d6195b38fe399f` | `runs/unet/QaTa-COV19/seed1001/config_used.yaml` |
| 1002 | `runs/unet/QaTa-COV19/seed1002/best_model.pt` | `c5edf80f57e27bc38e528e5911a12b60b3c412df824059a3dc885cb9f8939f10` | `runs/unet/QaTa-COV19/seed1002/config_used.yaml` |
| 1003 | `runs/unet/QaTa-COV19/seed1003/best_model.pt` | `63e0b695a20b213c5554528572f8dc09199fa8b3d9d421b6a17c8c9b00aa9565` | `runs/unet/QaTa-COV19/seed1003/config_used.yaml` |

The settings come from `configs/unet.yaml` (`deterministic: true`, line 58). For every seed, each key of
`configs/unet.yaml` has the same parsed value in the Phase 0 `config_used.yaml`; the files differ only in number
notation (e.g. `3.0e-4` / `0.0003`) and in the run fields that Phase 0 added (`dataset`, `seed`, `lr_used`,
`subset_train`, `subset_val`).

## 2. Inference command and commit

- Slurm job **479** (`slurm/phase5_part2_unet_infer.slurm`), 2026-11-11 11:38:24–11:41:10 EST, HEAD 90a7bd2;
  Phase 2 lock open (frozen commit b593d7f, fetch ok).
- `python src/phase3/infer_img.py --arch unet --dataset QaTa-COV19 --seeds 1001 1002 1003 --split test`
  (unchanged program): eval mode, `no_grad`, `UNetDataset(train=False)`, the model's sigmoid output, no TTA;
  then `python src/phase5_part2/reconcile_unet.py`.
- Data root: `LGMS_COMMON_ROOT = ~/datasets/phase1/common224_trainval` (slurm/env_phase1.sh). Its QaTa-COV19 test
  folder and id list are symbolic links made by `src/phase1/analysis_lock.py` `link_test` (lines 119–126);
  `readlink -f` gives:
  - `~/datasets/phase1/common224_trainval/QaTa-COV19/test` → `/mnt/cv_data/users/hahaxixi/datasets/common224/QaTa-COV19/test`
  - `~/datasets/phase1/common224_trainval/QaTa-COV19/test.txt` → `/mnt/cv_data/users/hahaxixi/datasets/common224/QaTa-COV19/test.txt`

  i.e. the same files as the Phase 0 root `~/datasets/common224` (slurm/env.sh).
- Test log: `infer_img.py` appended six rows (start / end per seed, job 479, lock commit b593d7f, origin/main
  90a7bd2, fetch ok) to `results/phase2_test_eval_log.tsv`; their wording "Phase 3 IMG test inference" /
  `phase3/infer_img.py` comes from the reused program. The rows are committed together with this document.

## 3. Probability and mask paths

`runs/phase3/test/img/unet/QaTa-COV19/seed<k>/`: `probs.npz` (float32), `masks.npz` (packbits of P > 0.5),
`per_case.csv`, `reconcile_phase0.csv`, `passes.tsv` (job 479, 2113 images per seed). These files are not in git
(`runs/` is ignored); their sha256 is listed in section 5.

## 4. Reconciliation with the Phase 0 archive (rule of the spec, deterministic tier of P2-2)

Each mask P > 0.5 was compared pixel by pixel with `runs/unet/QaTa-COV19/seed<k>/eval_test/masks/<id>.png`
(written from float32 probabilities, `src/lgms/runner.py` lines 87 and 93), and the per-case Dice with the archived
`per_case.csv` (6 stored decimals); the stored `masks.npz` was checked against packbits(P > 0.5).

| Seed | Images | Images with a mask difference | Max pixel difference | Mean \|ΔDice\| | Max \|ΔDice\| | masks.npz = packbits(P > 0.5) |
|---|---:|---:|---:|---:|---:|---|
| 1001 | 2113 | 0 | 0 | 0 | 0 | True |
| 1002 | 2113 | 0 | 0 | 0 | 0 | True |
| 1003 | 2113 | 0 | 0 | 0 | 0 | True |

**The rule is met for all three seeds: the U-Net test masks reproduce the Phase 0 archive exactly.** (The
`max_abs_dice_diff` of about 5e-7 printed by `infer_img.py` compares its unrounded Dice with the 6-decimal archive;
both rounded per-case files agree exactly.) Per-case metrics with the reconciliation columns:
`results/phase5_part2_unet_img_test_per_case.csv.gz`.

## 5. Artifact hashes

| File | sha256 |
|---|---|
| `runs/phase3/test/img/unet/QaTa-COV19/seed1001/probs.npz` | `abc38e14d8aa14924bc4738ca161891fc1a31ce30643f95920639e30e3edd202` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1001/masks.npz` | `9d23d4c61cda748918b5e5f976733af5f6b9383abb3e68c06e7b373bacf4981b` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1001/per_case.csv` | `7fd958b5e4ac33fdf482a8a09c7f97700331731f22ce968e8a89fe5c62446dbf` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1001/reconcile_phase0.csv` | `90dffd7fefe275c86f81237e0907044ba3e27156a7ad23b6a0d613860e204fee` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1002/probs.npz` | `1dff2290eeea09672bf97e2706261595c10c5392dab6410bf2e3fc092ff233ee` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1002/masks.npz` | `72659306840cbe1f894c1115ac1ed4f7e0d3e3bf7dbcca750593d2b0116fe917` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1002/per_case.csv` | `2f47709d092c92d130b6f90dcd7e275ba3f3bad8221543825a15d9857d4cfc9e` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1002/reconcile_phase0.csv` | `1b29218f5384a621c5a032bf5ee8237af1700f0f23ab15ce62d19aaf68ad0bac` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1003/probs.npz` | `5c494590e4993930abf380f4474473c5334be894ec25d6d1a700eb131b32cf06` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1003/masks.npz` | `e877fe7fd74b30a3d26f9627dadd84241397b06607db58ac74a41239427158c4` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1003/per_case.csv` | `dc3525caededd4ede790f4bf2cb89d04a90d9c6276712ae9dcade8dadf769c73` |
| `runs/phase3/test/img/unet/QaTa-COV19/seed1003/reconcile_phase0.csv` | `60363c70bb3d8cb01eeaba8ccf0651182673cf54fa812e8e29c18025d58459cb` |
| `results/phase5_part2_unet_img_test_per_case.csv.gz` | `e9cc5e2eb130a6139da76dc606b88c0cd642a656559c0606df636173ee90b345` |
