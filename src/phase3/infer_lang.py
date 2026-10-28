"""Phase 3, Task 3.1A: inference-only rerun of the language-guided models for the conditions without probability
maps (C2, C3a, C3b, C3c) on the VALIDATION split, float32 probabilities + masks + scores saved together.

Phase 2 kept float32 probabilities only for C0, C1, C4-C6 (D13). Phase 3 needs probability maps for every
condition it uses, so C2 and C3a-C3c are re-inferred with the same frozen checkpoints, the same Phase 2
manifest (src/phase2/conditions.py, unchanged; same texts, same eligible cases) and the same inference path
(src/phase2/infer.py: Adapter, passes, score; imported unchanged). Nothing in runs/phase2 is modified.
The Phase 3 analysis uses these NEW scores for C2 / C3a-C3c. Pixel differences to the Phase 2 masks of the same
passes are recorded for reproducibility only (DD-CMD is not bit-reproducible across processes; protocol 4.8).

Validation only: the test split is not accepted (Task 3.6).

Outputs: runs/phase3/val/lang/<method>/<dataset>/seed<k>/
  per_case_<cond>_0.csv, masks_<cond>_0.npz, probs_<cond>_0.npz (float32), phase2_mask_diff_<cond>.csv, passes.tsv

Usage (inside a Slurm job):  python src/phase3/infer_lang.py --method lvit --dataset QaTa-COV19
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
CONDITIONS = ["C2", "C3a", "C3b", "C3c"]
OUT = os.path.join(REPO, "runs", "phase3", "val", "lang")
P2 = os.path.join(REPO, "runs", "phase2", "val")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["lvit", "reclmis", "ddcmd"])
    ap.add_argument("--dataset", required=True, choices=["QaTa-COV19", "MosMedData+"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    args = ap.parse_args()
    from lgms import data
    from phase2 import conditions as C
    from phase2 import infer as I
    manifest, _ = C.build(args.dataset, "val")
    plan = [p for p in I.passes(manifest) if p[0] in CONDITIONS]
    c0_texts = next(t for c, _, t, _ in I.passes(manifest) if c == "C0")
    ids_all, images, masks = data.load_split_arrays(args.dataset, "val")
    index = {s: k for k, s in enumerate(ids_all)}
    masks_by_id = dict(zip(ids_all, masks))
    _, cfg_path, _ = I.checkpoint(args.method, args.dataset, args.seeds[0])
    ad = I.Adapter(args.method, args.dataset, cfg_path)
    ad.prepare([t for _, _, texts, _ in plan for t in texts.values()] + list(c0_texts.values()))
    for seed in args.seeds:
        ckpt, cfg_path, rec = I.checkpoint(args.method, args.dataset, seed)
        if I.sha_file(ckpt) != rec:
            raise RuntimeError(f"{ckpt}: sha256 differs from the Phase 2 record")
        out = os.path.join(OUT, args.method, args.dataset, f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        meta = {"dataset": args.dataset, "split": "val", "method": args.method,
                "config": os.path.relpath(cfg_path, REPO), "checkpoint_sha256": rec, "seed": seed}
        model = None
        for cond, variant, texts, donors in plan:
            if os.path.exists(os.path.join(out, f"per_case_{cond}_{variant}.csv")):
                print(f"seed {seed} {cond}: done earlier, skipped", flush=True)
                continue
            if model is None:
                model = ad.load(ckpt)
            n_input = sum(ad.model_input(texts[s]) != ad.model_input(c0_texts[s]) for s in texts)
            n_raw = sum(texts[s] != c0_texts[s] for s in texts)
            if n_input != n_raw:
                raise RuntimeError(f"{cond}: model input changed in {n_input} cases, text in {n_raw}")
            ids = [s for s in ids_all if s in texts]
            sel = [index[s] for s in ids]
            ad.D.runner.set_seed(seed, ad.cfg["deterministic"])
            ids, probs, targets = ad.predict(model, ids, images[sel], masks[sel], texts)
            if not all((targets[k] == masks_by_id[s]).all() for k, s in enumerate(ids)):
                raise RuntimeError("targets differ from the common masks (pairing error)")
            m = I.score(ids, probs, masks_by_id)
            I.save_pass(out, cond, variant, meta, ids, probs, m, donors, keep_probs=True)
            I.log_pass(out, "phase3_rerun", cond, variant, len(ids), f"model_input_changed={n_input}/{len(ids)}")
            old = np.load(os.path.join(P2, args.method, args.dataset, f"seed{seed}", f"masks_{cond}_{variant}.npz"))
            diff = [int((np.unpackbits(old[i])[:224 * 224].reshape(224, 224).astype(bool) != (p > 0.5)).sum())
                    for i, p in zip(ids, probs)]
            pd.DataFrame({"sample_id": ids, "pixels_differing_from_phase2": diff}).to_csv(
                os.path.join(out, f"phase2_mask_diff_{cond}.csv"), index=False)
            print(f"seed {seed} {cond}: {len(ids)} cases, mean Dice {m.dice.mean():.4f}, cases differing from "
                  f"Phase 2 masks {sum(d > 0 for d in diff)}, max pixels {max(diff)}", flush=True)
        model = None


if __name__ == "__main__":
    main()
