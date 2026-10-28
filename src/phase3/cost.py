"""Phase 3, Task 3.8: computational cost per image / report on a fixed validation subset (one GPU job).

Subset (fixed): the first 64 QaTa-COV19 validation sample ids in sorted order; the original text (C0); seed 1001
checkpoints. For each language model, four modes are timed per image, end to end, after 3 untimed warm-up images:
  LANG only              text processing (LViT-T: BERT word vectors, computed per image; RecLMIS / DD-CMD: tokenizer,
                         inside the model) + forward pass of the frozen language-guided model
  IMG only               nnU-Net for one image: nnU-Net's single-array predictor (preprocessing, sliding window,
                         Gaussian weighting, mirroring TTA), i.e. the Phase 0 settings without the batch file loader
  LANG + IMG + D-Gate    both predictions + F1 + threshold
  LANG + IMG + LR-Gate   both predictions + F1-F5 + scaler / LR score + threshold
Timing: time.perf_counter with torch.cuda.synchronize() around every image; batch size 1; data-loader workers 0.
Peak GPU memory: torch.cuda.max_memory_allocated after resetting the peak at the start of each mode; only the
models of the mode are resident (LANG only: language model; IMG only: nnU-Net; gated modes: both).
Model loading is reported separately (not part of the per-image latency). IMG caching across donor reports is not
used: every image runs both models. Validation only; no gate, threshold or model is changed.
Check: the single-image nnU-Net probabilities are compared with the Phase 3 rerun of the same images.

Usage (inside a Slurm job):  python src/phase3/cost.py   -> results/phase3_computational_cost.csv
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
DATASET, SEED, N_IMAGES, WARMUP = "QaTa-COV19", 1001, 64, 3
METHODS = ["lvit", "reclmis", "ddcmd"]
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}


def sync():
    import torch
    torch.cuda.synchronize()


def nnunet_single(predictor, path):
    """One image through nnU-Net's single-array predictor; returns the lesion probability (224, 224)."""
    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
    img, props = NaturalImage2DIO().read_images([path])
    _, probs = predictor.predict_single_npy_array(img, props, None, None, True)
    return np.asarray(probs[1, 0], dtype=np.float32)


def main():
    import torch
    from lgms import data
    from phase2 import conditions as C
    from phase2 import infer as I
    import nnunet_select as N
    from phase3.features import features
    dg = json.load(open(os.path.join(REPO, "results", "phase3_dgate_thresholds.json")))
    art = json.load(open(os.path.join(REPO, "results", "phase3_lr_artifacts.json")))
    ck = pd.read_csv(os.path.join(REPO, "results", "phase3_checkpoints.tsv"), sep="\t", dtype=str)
    ids_all, images, masks = data.load_split_arrays(DATASET, "val")
    ids = sorted(ids_all)[:N_IMAGES]
    pos = {s: k for k, s in enumerate(ids_all)}
    man, _ = C.build(DATASET, "val")
    c0 = dict(zip(man[man.condition == "C0"].sample_id, man[man.condition == "C0"].counterfactual_text))
    nn = ck[(ck.role == "IMG") & (ck.model == "nnunet") & (ck.dataset == DATASET) & (ck.seed == str(SEED))].path.iloc[0]
    nn_path = os.path.join(REPO, nn)
    raw = os.path.join(os.environ["nnUNet_raw"], "Dataset501_QaTaCOV19", "imagesTr")
    ref = np.load(os.path.join(REPO, "runs", "phase3", "val", "img", "nnunet", DATASET, f"seed{SEED}", "probs.npz"))
    rows = []
    gpu = torch.cuda.get_device_name(0)

    def timed(mode, method, fn, loads):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        lat = []
        for k, s in enumerate(ids[:WARMUP] + ids):
            sync()
            t0 = time.perf_counter()
            fn(s)
            sync()
            if k >= WARMUP:
                lat.append(1000 * (time.perf_counter() - t0))
        lat = np.array(lat)
        rows.append({"method": NAME.get(method, method), "mode": mode, "n_images": len(lat), "gpu": gpu,
                     "mean_ms": lat.mean(), "median_ms": np.median(lat), "p95_ms": np.quantile(lat, 0.95),
                     "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20, **loads})
        print(json.dumps(rows[-1]), flush=True)

    # IMG only (no language model resident)
    t0 = time.perf_counter()
    predictor = N.make_predictor(os.path.dirname(os.path.dirname(nn_path)), os.path.basename(nn_path))
    nn_load = time.perf_counter() - t0
    diffs = [float(np.abs(nnunet_single(predictor, os.path.join(raw, f"{s}_0000.png")) - ref[s]).max()) for s in ids[:8]]
    print(f"single-image nnU-Net vs rerun probabilities (8 images): max |diff| {max(diffs):.3g}", flush=True)
    timed("IMG only", "nnU-Net", lambda s: nnunet_single(predictor, os.path.join(raw, f"{s}_0000.png")),
          {"model_load_s": nn_load, "nnunet_single_vs_rerun_max_abs_diff": max(diffs)})
    for m in METHODS:
        _, cfg, _ = I.checkpoint(m, DATASET, SEED)
        ckpt = I.checkpoint(m, DATASET, SEED)[0]
        t0 = time.perf_counter()
        ad = I.Adapter(m, DATASET, cfg)
        ad.cfg = {**ad.cfg, "num_workers": 0}
        model = ad.load(ckpt)
        load = time.perf_counter() - t0
        t, a = dg[f"{NAME[m]}_seed{SEED}"]["threshold"], art[f"{NAME[m]}_seed{SEED}"]

        def lang(s):
            if m == "lvit":
                ad.vectors.clear()                        # BERT features are computed per image (not cached)
            ad.prepare([c0[s]])
            k = pos[s]
            return ad.predict(model, [s], images[k:k + 1], masks[k:k + 1], {s: c0[s]})[1][0]

        def gated(s, lr):
            pl = lang(s)
            pi = nnunet_single(predictor, os.path.join(raw, f"{s}_0000.png"))
            if not lr:
                f1 = float(np.abs(pl.astype(np.float64) - pi.astype(np.float64)).mean())
                return pi if f1 > t else pl
            f = np.array(features(pl, pi))
            score = float(((f - np.array(a["scaler_mean"])) / np.array(a["scaler_scale"])) @ np.array(a["coef"]) + a["intercept"])
            return pi if score > a["threshold"] else pl

        timed("LANG + IMG + D-Gate", m, lambda s: gated(s, False), {"model_load_s": load})
        timed("LANG + IMG + LR-Gate", m, lambda s: gated(s, True), {"model_load_s": load})
        del predictor
        torch.cuda.empty_cache()
        timed("LANG only", m, lang, {"model_load_s": load})
        predictor = N.make_predictor(os.path.dirname(os.path.dirname(nn_path)), os.path.basename(nn_path))
        del model, ad
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(os.path.join(REPO, "results", "phase3_computational_cost.csv"), index=False,
                              float_format="%.4g")


if __name__ == "__main__":
    main()
