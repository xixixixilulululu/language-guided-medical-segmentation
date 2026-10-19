"""Diagnostic run of DD-CMD training (quick-check job 317 hung after Lightning's sanity check, before the end
of the first training epoch). Runs src/phase1/train_ddcmd.py unchanged on a small subset, with:
  * faulthandler stack dumps of ALL threads every --dump-every seconds, in the main process AND in every
    data-loader worker (so a hang shows where each process waits);
  * one progress line per training batch (time since start);
  * variants of the data loading only (never of the model or the training):
      --workers N          number of data-loader workers (train and val), via a copy of configs/ddcmd.yaml
      --sharing file_system  torch.multiprocessing sharing strategy of the official train.py
      --spawn              DataLoader multiprocessing_context='spawn' instead of fork
Validation split only (--no-final-eval); the test split is never touched.

Usage:  python src/phase1/diag_ddcmd.py --out <dir> --workers 8 [--sharing file_system] [--spawn]
"""
import argparse
import faulthandler
import functools
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import common  # noqa: E402

T0 = time.time()


def worker_init(worker_id):
    """Data-loader worker: periodic stack dumps, then MONAI's worker_init_fn (as in train_ddcmd.py)."""
    from monai.data.utils import worker_init_fn as monai_worker_init_fn
    faulthandler.dump_traceback_later(int(os.environ.get("DIAG_DUMP_EVERY", "60")), repeat=True, file=sys.stderr)
    print(f"[diag] worker {worker_id} started (pid {os.getpid()})", flush=True)
    monai_worker_init_fn(worker_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--sharing", choices=["file_descriptor", "file_system"], default=None)
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--dataset", default="QaTa-COV19")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--dump-every", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    faulthandler.dump_traceback_later(args.dump_every, repeat=True, file=sys.stderr)   # main process

    from phase1 import train_ddcmd as T
    import torch
    if args.sharing:
        torch.multiprocessing.set_sharing_strategy(args.sharing)
    print(f"[diag] workers {args.workers} | sharing {torch.multiprocessing.get_sharing_strategy()} | "
          f"start method {'spawn' if args.spawn else 'fork (default)'} | torch threads {torch.get_num_threads()}",
          flush=True)

    # Data-loader workers: stack dumps as well, then the official MONAI worker seeding (module-level function,
    # so that it can be pickled for --spawn; the dump interval is passed through the environment)
    os.environ["DIAG_DUMP_EVERY"] = str(args.dump_every)
    T.monai_worker_init_fn = worker_init
    if args.spawn:
        T.DataLoader = functools.partial(T.DataLoader, multiprocessing_context="spawn")

    # One line per training batch
    class DiagControl(T.SegmentControl):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            print(f"[diag] epoch {trainer.current_epoch} batch {batch_idx} done, t={time.time() - T0:.0f}s", flush=True)

        def on_validation_epoch_end(self, trainer, pl_module):
            print(f"[diag] validation epoch end (sanity={trainer.sanity_checking}), t={time.time() - T0:.0f}s", flush=True)

    T.SegmentControl = DiagControl

    # Config copy with the requested number of workers (everything else unchanged)
    cfg = yaml.safe_load(open(os.path.join(common.REPO, "configs", "ddcmd.yaml")))
    cfg["num_workers"] = args.workers
    cfg_path = os.path.join(args.out, "ddcmd_diag.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    sys.argv = ["train_ddcmd.py", "--config", cfg_path, "--dataset", args.dataset, "--seed", "1001",
                "--subset-train", "64", "--subset-val", "32", "--max-epochs", str(args.epochs),
                "--no-final-eval", "--out", os.path.join(args.out, "run")]
    T.main()
    print(f"[diag] finished, t={time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
