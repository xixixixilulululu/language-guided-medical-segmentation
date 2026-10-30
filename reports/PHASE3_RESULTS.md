# Phase 3 — Test results (frozen protocol)

Status: 2026-10-30. **Run record and checks only; the results and conclusions sections are not written yet.**
Protocol: [PHASE3_PROTOCOL_FREEZE.md](PHASE3_PROTOCOL_FREEZE.md), frozen in commit b593d7f (record
`configs/phase2_frozen/phase2_lock.json`). This file is not part of the freeze record. Gates as fixed at
b58cdf6; nothing was fitted, calibrated or selected on test.

## 1. Test run record

- Job 394 (`slurm/phase3_test.slurm`), 2026-10-30 14:38–14:46 EDT, one run: lock check OPEN; nnU-Net test
  inference (2 datasets × 3 seeds); `export_per_case.py`, `features.py`, `evaluate_gates.py`, `stats_phase3.py`,
  all with `--split test`.
- `results/phase2_test_eval_log.tsv`: 6 start and 6 end entries of `phase3/infer_img.py`, job 394.
- Inputs: the Phase 2 test benchmark (job 392; [PHASE2_RESULTS.md](PHASE2_RESULTS.md)), P_lang of every
  condition and donor (amendment A1).

## 2. Checks

- **Rows.** 319,230 feature / label / decision rows (QaTa 282,375; MosMed 36,855), equal to the rows of
  `results/phase2_test_per_case.csv.gz`.
- **nnU-Net test inference vs the Phase 0 test archive** (reproducibility record only; Phase 3 uses the new
  scores; job 394 log, `runs/phase3/test/img/nnunet/<ds>/seed<k>/reconcile_phase0.csv`):

| Dataset | Seed | Mean Dice (new) | Mean Dice (Phase 0) | Max case \|ΔDice\| | Cases with a mask difference | Max pixels |
|---|---|---:|---:|---:|---:|---:|
| QaTa-COV19 | 1001 | 79.8076 | 79.8076 | 0.0060 | 636 / 2113 | 9 |
| QaTa-COV19 | 1002 | 79.5355 | 79.5358 | 0.0040 | 497 / 2113 | 4 |
| QaTa-COV19 | 1003 | 79.8293 | 79.8291 | 0.0033 | 702 / 2113 | 10 |
| MosMedData+ | 1001 | 71.2861 | 71.2850 | 0.0031 | 23 / 273 | 2 |
| MosMedData+ | 1002 | 71.9562 | 71.9574 | 0.0020 | 17 / 273 | 3 |
| MosMedData+ | 1003 | 72.9503 | 72.9516 | 0.0180 | 19 / 273 | 1 |

  The same kind of difference as on validation (PHASE3_SETUP.md section 6): the Phase 0 archive is float16.

## 3. Result files

- `phase3_test_results.csv` (Task 3.5, image-weighted), `phase3_test_per_case.csv.gz` (every row × strategy),
  `phase3_test_paired_differences.csv`, `phase3_test_fallback_rates.csv`, `phase3_test_routing_quality.csv`,
  `phase3_test_mosmed_exploratory.csv`
- `phase3_pairwise_results.csv` (Task 3.7)
- for independent checking: `phase3_test_gate_decisions.csv.gz` (keys, F1, LR score, both switches; full
  precision), `phase3_img_test_per_case.csv.gz` (IMG test per-case scores),
  `phase3_reliability_features_test.csv.gz`, `phase3_feature_labels_test.csv.gz`

## 4. Results

*(not written yet)*

## 5. Conclusions

*(not written yet)*
