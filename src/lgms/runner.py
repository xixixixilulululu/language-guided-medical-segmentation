"""Helpers shared by the U-Net and TransUNet training scripts (seeding, logging, final evaluation)."""
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import yaml

from . import data, metrics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def per_dataset(value, dataset):
    """Config values may be given per dataset as {dataset: value}; return the value for `dataset`."""
    return value[dataset] if isinstance(value, dict) else value


def set_seed(seed, deterministic=True):
    """Seed python, numpy and torch; deterministic cuDNN as in both official training scripts."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def make_logger(run_dir):
    """Logger writing to <run_dir>/train.log and stdout."""
    os.makedirs(run_dir, exist_ok=True)
    log = logging.getLogger(run_dir)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(os.path.join(run_dir, "train.log")), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


class EpochLog:
    """Appends one CSV row per epoch to <run_dir>/epochs.csv."""

    def __init__(self, run_dir, columns):
        self.path = os.path.join(run_dir, "epochs.csv")
        self.columns = columns
        with open(self.path, "w") as f:
            f.write(",".join(columns) + "\n")

    def write(self, **row):
        with open(self.path, "a") as f:
            f.write(",".join(f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c])
                             for c in self.columns) + "\n")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_probs(ids, probs, path):
    """Foreground probabilities of one split, float16, one array per sample id (evaluator input)."""
    np.savez_compressed(path, **{i: p.astype(np.float16) for i, p in zip(ids, probs)})


def final_evaluation(run_dir, dataset, split, ids, probs, log):
    """Score one split with the common evaluator and write per_case.csv, summary.json and masks."""
    import cv2

    sys.path.insert(0, os.path.join(REPO, "src"))
    import evaluate as ev  # the common evaluator (src/evaluate.py)

    out = os.path.join(run_dir, f"eval_{split}")
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    save_probs(ids, probs, os.path.join(run_dir, f"{split}_probs.npz"))
    preds = dict(zip(ids, probs.astype(np.float32)))
    df = ev.evaluate(dataset, split, preds)
    summary = {"dataset": dataset, "split": split, **ev.summarize(df, preds)}
    df.to_csv(os.path.join(out, "per_case.csv"), index=False, float_format="%.6f")
    save_json(summary, os.path.join(out, "summary.json"))
    for i, p in preds.items():
        cv2.imwrite(os.path.join(out, "masks", f"{i}.png"), metrics.binarize(p).astype(np.uint8) * 255)
    log.info(f"[{split}] " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}))
    return summary


class Timer:
    """Wall-clock timer that synchronises CUDA so that GPU time is included."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.time() - self.t0
