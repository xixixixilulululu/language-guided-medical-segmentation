"""Helpers shared by the three Phase 1 training drivers (train_lvit.py, train_reclmis.py, train_ddcmd.py).

* Texts: the LViT xlsx annotations, keyed by common224 sample id (one text per image, checked).
* Resume: every epoch the driver writes <run>/last.pth (model, optimizer, scheduler, epoch, best Dice and
  epoch, RNG states). A new job (= segment) started with the same --out continues from there.
* Wall-clock guard: before each epoch the driver asks `WallClock.should_stop(longest_epoch)`; if the next
  epoch might not finish inside the job budget (default 11 h 30 min of a 12 h job), the driver saves and
  exits with SEGMENT_EXIT_CODE. The Slurm script then notifies the user to submit the next segment by hand.
  Nothing here ever submits a job.
* Final evaluation: the Phase 0 common evaluator (src/lgms/runner.final_evaluation -> src/evaluate.py).
  The test split is evaluated only if BOTH locks are open (require_split): the program lock (analysis fixed
  on origin/main) and the data lock (test split linked into the Phase 1 data view).
"""
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from lgms import data, metrics, runner  # noqa: E402,F401  (re-exported for the drivers)

SEEDS = [1001, 1002, 1003]
SEGMENT_EXIT_CODE = 3            # "segment ended by the wall-clock guard, training not finished"
LOCKED_EXIT_CODE = 4             # "training finished, but the test evaluation is locked (program or data lock)"
DEFAULT_BUDGET_HOURS = 11.5      # of the 12 h Slurm limit; leaves time for saving and the final evaluation

# Raw LViT-layout folders that hold the per-split text files (written by src/prepare_lvit_datasets.py)
TEXT_ROOT = os.path.expanduser(os.environ.get("LGMS_TEXT_ROOT", "~/datasets"))
SPLIT_FOLDER = {"train": "Train", "val": "Val", "test": "Test"}


# ---------------------------------------------------------------------------------------------
# Texts
# ---------------------------------------------------------------------------------------------
def text_key(dataset, sample_id):
    """xlsx key of a sample: the mask file name (QaTa masks are called mask_<image>.png)."""
    return f"mask_{sample_id}.png" if dataset == "QaTa-COV19" else f"{sample_id}.png"


def read_texts(dataset, split, ids=None):
    """{sample_id: raw text} for one split. Every id must have exactly one non-empty text."""
    s = SPLIT_FOLDER[split]
    df = pd.read_excel(os.path.join(TEXT_ROOT, dataset, f"{s}_Folder", f"{s}_text.xlsx"))
    if df.Image.duplicated().any():
        raise ValueError(f"{dataset}/{split}: duplicate keys in the text file")
    by_key = dict(zip(df.Image.astype(str), df.Description.astype(str)))
    ids = ids if ids is not None else data.split_ids(dataset, split)
    out = {}
    for i in ids:
        t = by_key.get(text_key(dataset, i))
        if t is None or not t.strip() or t == "nan":
            raise KeyError(f"{dataset}/{split}: no text for {i}")
        out[i] = t
    return out


def lvit_pad(text):
    """Padding of LViT and RecLMIS utils.read_text: texts with < 9 words get ' EOF XXX' * (9 - count).

    The official code writes this back with a chained pandas assignment, which pandas >= 3 silently
    ignores (copy-on-write); the rule is therefore applied here explicitly.
    """
    count = len(text.split())
    return text + " EOF XXX" * (9 - count) if count < 9 else text


# ---------------------------------------------------------------------------------------------
# Seeds, RNG state, resume
# ---------------------------------------------------------------------------------------------
def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_atomic(obj, path):
    """torch.save to a temporary file, then rename: an interrupted save never leaves a broken file."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_state(path):
    # Our own checkpoints contain numpy RNG states (not plain tensors), hence weights_only=False
    return torch.load(path, map_location="cpu", weights_only=False)


class WallClock:
    """Job-time budget. The clock starts when the driver starts (a few seconds after the Slurm job)."""

    def __init__(self, budget_hours=None):
        hours = budget_hours or float(os.environ.get("LGMS_TIME_BUDGET_HOURS", DEFAULT_BUDGET_HOURS))
        self.t0, self.budget = time.time(), hours * 3600.0

    def elapsed(self):
        return time.time() - self.t0

    def should_stop(self, longest_epoch_seconds):
        """True if one more epoch (taken as twice the longest epoch so far) could pass the budget."""
        return self.elapsed() + 2.0 * longest_epoch_seconds > self.budget


class EpochLog:
    """epochs.csv, one row per epoch; on resume the existing rows are kept (cut after the resume epoch)."""

    def __init__(self, run_dir, columns, resume_epoch=0):
        self.path = os.path.join(run_dir, "epochs.csv")
        self.columns = columns
        if resume_epoch and os.path.exists(self.path):
            keep = pd.read_csv(self.path)
            keep = keep[keep.epoch <= resume_epoch]
            keep.to_csv(self.path, index=False)
        else:
            with open(self.path, "w") as f:
                f.write(",".join(columns) + "\n")

    def write(self, **row):
        with open(self.path, "a") as f:
            f.write(",".join(f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c])
                             for c in self.columns) + "\n")


def log_segment(run_dir, **info):
    """Append one line per segment (job id, start epoch, end reason) to segments.jsonl."""
    info = {"job": os.environ.get("SLURM_JOB_ID", "none"), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    with open(os.path.join(run_dir, "segments.jsonl"), "a") as f:
        f.write(json.dumps(info) + "\n")


# ---------------------------------------------------------------------------------------------
# Early stopping / selection (identical rule in all three drivers)
# ---------------------------------------------------------------------------------------------
class Selection:
    """Best validation Dice (mean per-case Dice, threshold 0.5) and early stopping.

    `first_epoch` reproduces LViT's "save only from the 6th epoch on"; RecLMIS and DD-CMD use 1.
    Early stopping as in LViT / RecLMIS: stop when epoch - best_epoch > patience (1-based epochs).
    """

    def __init__(self, patience, first_epoch=1, best_dice=0.0, best_epoch=1):
        self.patience, self.first_epoch = patience, first_epoch
        self.best_dice, self.best_epoch = best_dice, best_epoch

    def update(self, epoch, val_dice):
        """Returns True if this epoch is the new best."""
        if val_dice > self.best_dice and epoch >= self.first_epoch:
            self.best_dice, self.best_epoch = val_dice, epoch
            return True
        return False

    def stop(self, epoch):
        return epoch - self.best_epoch > self.patience

    def state(self):
        return {"best_dice": self.best_dice, "best_epoch": self.best_epoch}


TEST_EVAL_LOG = os.path.join(REPO, "results", "phase1_test_eval_log.tsv")
TEST_EVAL_LOG_COLUMNS = ["event", "time", "job_id", "method", "dataset", "seed", "analysis_commit", "origin_main",
                         "fetch"]


def log_test_eval(event, run, lock_info, path=None):
    """One line per start / end of a test evaluation (results/phase1_test_eval_log.tsv)."""
    path = path or TEST_EVAL_LOG
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = [event, time.strftime("%Y-%m-%d %H:%M:%S"), os.environ.get("SLURM_JOB_ID", "none"), run["method"],
           run["dataset"], str(run["seed"]), lock_info.get("commit", ""), lock_info.get("remote_head", ""),
           lock_info.get("fetch", "")]
    with open(path, "a") as f:
        if new:
            f.write("\t".join(TEST_EVAL_LOG_COLUMNS) + "\n")
        f.write("\t".join(row) + "\n")


def require_split(dataset, split, log=None, run=None, lock_repo=None, eval_log=None):
    """Gate before a split is evaluated. Two independent locks protect the test split:
    1. program lock (src/phase1/analysis_lock.py): the Phase 1 analysis is fixed on origin/main;
    2. data lock (2026-10-19): Phase 1 jobs read a data view (LGMS_COMMON_ROOT, slurm/env_phase1.sh) that holds
       only train and val until the analysis is fixed.
    If either is closed, stop with LOCKED_EXIT_CODE: training and the validation-selected checkpoint are
    already saved, and running the same job again later only performs the final evaluation.
    For the test split, the start of the evaluation is logged (results/phase1_test_eval_log.tsv); returns the
    lock information (for the matching "end" line)."""
    say = log.info if log else print
    lock_info = {}
    if split == "test":
        from phase1 import analysis_lock
        ok, reasons, lock_info = analysis_lock.check(lock_repo or REPO, fetch=True, fetch_required=False)
        if not ok:
            say("PROGRAM LOCK CLOSED: the Phase 1 analysis is not fixed on origin/main; test evaluation skipped: "
                + "; ".join(reasons))
            sys.exit(LOCKED_EXIT_CODE)
    path = os.path.join(data.common_root(), dataset, f"{split}.txt")
    if not os.path.exists(path):
        say(f"DATA LOCK: split '{split}' of {dataset} is not in the data view {data.common_root()}; stopping")
        sys.exit(LOCKED_EXIT_CODE)
    if split == "test":
        log_test_eval("start", run, lock_info, eval_log)
        say(f"test evaluation allowed: analysis fixed in commit {lock_info.get('commit')} "
            f"(origin/main {lock_info.get('remote_head', '')[:12]}, fetch {lock_info.get('fetch')})")
    return lock_info


def check_targets(dataset, split, ids, targets):
    """Alignment guard: the target masks produced by a method's own pipeline must equal the common224
    masks of the same ids, pixel for pixel (catches reordering, transposes, flips, wrong keys)."""
    for i, t in zip(ids, targets):
        ref = data.read_mask(dataset, split, i)
        t = np.asarray(t).reshape(ref.shape)
        if not np.array_equal(t > 0, ref > 0):
            raise AssertionError(f"{dataset}/{split}/{i}: method target differs from the common224 mask")
