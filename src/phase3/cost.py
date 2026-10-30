"""Phase 3, Task 3.8: computational cost per image on a fixed validation subset; ONE method x mode per process.

Subset (fixed): the first 64 QaTa-COV19 validation sample ids in sorted order; the original text (C0); seed 1001
checkpoints. Modes, end to end per image (batch size 1, data-loader workers 0):
  lang    LANG only             text processing (LViT-T: BERT word vectors, computed per image; RecLMIS / DD-CMD:
                                tokenizer, inside the model) + forward pass of the frozen language-guided model
  img     IMG only              nnU-Net for one image: nnU-Net's single-array predictor (preprocessing, sliding
                                window, Gaussian weighting, mirroring TTA), i.e. the Phase 0 settings without the
                                batch file loader
  dgate   LANG + IMG + D-Gate   both predictions + F1 + threshold
  lrgate  LANG + IMG + LR-Gate  both predictions + F1-F5 + scaler / LR score + threshold
Measurement (revised 2026-10-30 after job 379, PHASE3_SETUP.md section 10):
  - one process per method x mode; only the models of that mode are loaded (lang: language model; img: nnU-Net;
    gates: both models and the gate parameters);
  - warm-up: the first 3 subset images through the complete path (inference + features + gate), then
    torch.cuda.synchronize() and torch.cuda.reset_peak_memory_stats();
  - then the 64 images, each timed with time.perf_counter and torch.cuda.synchronize() before and after;
  - latency: mean, median, p95 (ms); GPU memory: max_memory_allocated and max_memory_reserved (MB) over the 64.
Model loading is reported separately. IMG caching across donor reports is not used. Validation only; no gate,
threshold or model is changed. Check (img mode): single-image nnU-Net probabilities vs the Phase 3 rerun.

Usage (inside a Slurm job):  python src/phase3/cost.py --method lvit --mode dgate
                             python src/phase3/cost.py --merge     -> results/phase3_computational_cost.csv
One row per process is written to runs/phase3/cost/<method>_<mode>.json.
"""
import argparse
import glob
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
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD", "nnunet": "nnU-Net"}
MODE = {"lang": "LANG only", "img": "IMG only", "dgate": "LANG + IMG + D-Gate", "lrgate": "LANG + IMG + LR-Gate"}
OUT = os.path.join(REPO, "runs", "phase3", "cost")


def sync():
    import torch
    torch.cuda.synchronize()


def nnunet_single(predictor, path):
    """One image through nnU-Net's single-array predictor; returns the lesion probability (224, 224)."""
    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
    img, props = NaturalImage2DIO().read_images([path])
    _, probs = predictor.predict_single_npy_array(img, props, None, None, True)
    return np.asarray(probs[1, 0], dtype=np.float32)


def run(method, mode):
    import torch
    from lgms import data
    ids_all, images, masks = data.load_split_arrays(DATASET, "val")
    ids = sorted(ids_all)[:N_IMAGES]
    pos = {s: k for k, s in enumerate(ids_all)}
    raw = os.path.join(os.environ["nnUNet_raw"], "Dataset501_QaTaCOV19", "imagesTr")
    extra, load_s = {}, {}
    use_lang, use_img = mode != "img", mode != "lang"

    if use_img:                                                  # nnU-Net, seed 1001 (results/phase3_checkpoints.tsv)
        import nnunet_select as N
        ck = pd.read_csv(os.path.join(REPO, "results", "phase3_checkpoints.tsv"), sep="\t", dtype=str)
        nn = os.path.join(REPO, ck[(ck.role == "IMG") & (ck.model == "nnunet") & (ck.dataset == DATASET)
                                   & (ck.seed == str(SEED))].path.iloc[0])
        t0 = time.perf_counter()
        predictor = N.make_predictor(os.path.dirname(os.path.dirname(nn)), os.path.basename(nn))
        load_s["nnunet_load_s"] = time.perf_counter() - t0
        img_fn = lambda s: nnunet_single(predictor, os.path.join(raw, f"{s}_0000.png"))  # noqa: E731
    if use_lang:
        from phase2 import conditions as C
        from phase2 import infer as I
        man, _ = C.build(DATASET, "val")
        c0 = dict(zip(man[man.condition == "C0"].sample_id, man[man.condition == "C0"].counterfactual_text))
        ckpt, cfg, _ = I.checkpoint(method, DATASET, SEED)
        t0 = time.perf_counter()
        ad = I.Adapter(method, DATASET, cfg)
        ad.cfg = {**ad.cfg, "num_workers": 0}
        model = ad.load(ckpt)
        load_s["lang_load_s"] = time.perf_counter() - t0

        def lang_fn(s):
            if method == "lvit":
                ad.vectors.clear()                               # BERT features are computed per image (not cached)
            ad.prepare([c0[s]])
            k = pos[s]
            return ad.predict(model, [s], images[k:k + 1], masks[k:k + 1], {s: c0[s]})[1][0]

    if mode == "lang":
        step = lang_fn
    elif mode == "img":
        step = img_fn
    else:
        from phase3.features import features
        if mode == "dgate":
            t = json.load(open(os.path.join(REPO, "results", "phase3_dgate_thresholds.json")))[f"{NAME[method]}_seed{SEED}"]["threshold"]

            def step(s):
                pl, pi = lang_fn(s), img_fn(s)
                f1 = float(np.abs(pl.astype(np.float64) - pi.astype(np.float64)).mean())
                return pi if f1 > t else pl
        else:
            a = json.load(open(os.path.join(REPO, "results", "phase3_lr_artifacts.json")))[f"{NAME[method]}_seed{SEED}"]
            mu, sc, coef = np.array(a["scaler_mean"]), np.array(a["scaler_scale"]), np.array(a["coef"])

            def step(s):
                pl, pi = lang_fn(s), img_fn(s)
                score = float(((np.array(features(pl, pi)) - mu) / sc) @ coef + a["intercept"])
                return pi if score > a["threshold"] else pl

    for s in ids[:WARMUP]:                                       # warm-up: complete path, not measured
        step(s)
    sync()
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for s in ids:
        sync()
        t0 = time.perf_counter()
        step(s)
        sync()
        lat.append(1000 * (time.perf_counter() - t0))
    lat = np.array(lat)
    row = {"method": NAME[method], "mode": MODE[mode], "n_images": len(lat), "n_warmup": WARMUP,
           "gpu": torch.cuda.get_device_name(0), "mean_ms": lat.mean(), "median_ms": np.median(lat),
           "p95_ms": np.quantile(lat, 0.95), "max_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
           "max_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20, **load_s,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID", "none")}
    if mode == "img":                                            # timing check only; gates use the rerun probabilities
        ref = np.load(os.path.join(REPO, "runs", "phase3", "val", "img", "nnunet", DATASET, f"seed{SEED}", "probs.npz"))
        row["nnunet_single_vs_rerun_max_abs_diff"] = max(float(np.abs(img_fn(s) - ref[s]).max()) for s in ids[:8])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{method}_{mode}.json"), "w") as f:
        json.dump(row, f, indent=1)
    print(json.dumps(row), flush=True)


def merge():
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    order = {m: i for i, m in enumerate(["nnU-Net"] + [NAME[m] for m in METHODS])}
    T = pd.DataFrame(rows)
    T = T.assign(_m=T.method.map(order), _o=T["mode"].map({v: i for i, v in enumerate(MODE.values())}))
    T = T.sort_values(["_m", "_o"]).drop(columns=["_m", "_o"])
    T.to_csv(os.path.join(REPO, "results", "phase3_computational_cost.csv"), index=False, float_format="%.4g")
    print(T.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS + ["nnunet"])
    ap.add_argument("--mode", choices=list(MODE))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge()
    if (a.mode == "img") != (a.method == "nnunet") or a.mode is None:
        ap.error("--method nnunet goes with --mode img only; language models with lang / dgate / lrgate")
    run(a.method, a.mode)


if __name__ == "__main__":
    main()
