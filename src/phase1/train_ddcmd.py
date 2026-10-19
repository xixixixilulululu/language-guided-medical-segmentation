"""Train DD-CMD on one dataset with one seed (Phase 1), then evaluate the best checkpoint.

The official LightningModule (net/creratemodel.py: model, DiceCE loss, AdamW, cosine schedule, official
step/epoch hooks) is used unchanged with pytorch_lightning 1.9.0 / torchmetrics 0.10.3 (the official pins).
The dataset re-implements utils/dataset.SegData on the common 224x224 data in memory with the official MONAI
transforms. Trainer settings as train.py. Deviations (configs/ddcmd.yaml): selection and early stopping on
the mean per-case validation Dice (D3), 8 seeded workers (D6), seeds 1001-1003.

Resumable: Lightning's own checkpoint <out>/checkpoints/last.ckpt (written at the end of every epoch) is
continued with trainer.fit(ckpt_path=...); RNG and loader-generator states are added by a callback.

Usage (inside a Slurm job):
    python src/phase1/train_ddcmd.py --dataset QaTa-COV19 --seed 1001 --out runs/ddcmd/QaTa-COV19/seed1001
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import common  # noqa: E402
from phase1.common import data, metrics, runner  # noqa: E402

sys.path.insert(0, os.path.join(common.REPO, "external", "DD-CMD"))
# NumPy 2 compatibility (decision of 2026-10-19, option A): pytorch_lightning 1.9.0 (the official pin) uses
# `torch.tensor(np.Inf)` in EarlyStopping and ModelCheckpoint, and NumPy 2.0 removed the alias `np.Inf`.
# `np.inf` is the same value (IEEE positive infinity), so restoring the alias changes nothing in training.
if not hasattr(np, "Inf"):
    np.Inf = np.inf
import pytorch_lightning as pl  # noqa: E402
from monai.data.utils import worker_init_fn as monai_worker_init_fn  # noqa: E402
from monai.transforms import (Compose, NormalizeIntensityd, RandFlipd, RandGaussianNoised,  # noqa: E402
                              RandRotate90d, RandRotated, RandZoomd, Resized, ToTensord)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import CSVLogger  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from net.creratemodel import CreateModel  # noqa: E402


def official_transforms(train, size):
    """utils/dataset.py SegData transforms without LoadImaged/EnsureChannelFirstd (arrays come channel-first)."""
    keys = ["image", "gt"]
    if train:
        return Compose([
            RandZoomd(keys=keys, min_zoom=0.95, max_zoom=1.15, mode=["bicubic", "nearest"], prob=0.3),
            RandRotated(keys=keys, range_x=[-0.3, 0.3], keep_size=True, mode=["bicubic", "nearest"], prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandRotate90d(keys=keys, prob=0.2, max_k=3),
            Resized(keys=["image"], spatial_size=size, mode="bicubic"),
            Resized(keys=["gt"], spatial_size=size, mode="nearest"),
            NormalizeIntensityd(keys=["image"], channel_wise=True),
            ToTensord(keys=["image", "gt"]),
        ])
    return Compose([
        Resized(keys=["image"], spatial_size=size, mode="bicubic"),
        Resized(keys=["gt"], spatial_size=size, mode="nearest"),
        NormalizeIntensityd(keys=["image"], channel_wise=True),
        ToTensord(keys=["image", "gt"]),
    ])


class DDCMDDataset(Dataset):
    """SegData.__getitem__ on in-memory common224 data: returns ([image, text], gt) as the official class."""

    def __init__(self, ids, images, masks, texts, tokenizer, cfg, train):
        self.ids, self.images, self.masks = ids, images, masks
        self.transform = official_transforms(train, cfg["input_size"])
        self.text = []
        for i in ids:  # tokenisation is deterministic: done once (SegData does it per item)
            tok = tokenizer(texts[i], padding="max_length", max_length=cfg["input_text_len"], truncation=True,
                            return_tensors="pt")
            self.text.append({k: v.squeeze(0) for k, v in tok.items()})

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # PILReader output: float32 grey values 0..255 (channel-first); mask 0/255 -> here already {0, 1}
        d = {"image": self.images[idx][None].astype(np.float32), "gt": self.masks[idx][None].astype(np.float32)}
        d = self.transform(d)
        image, gt = d["image"].float(), d["gt"].float()
        gt = (gt > 0.5).float()                       # SegData: gt == 255 -> 1, then > 0.5
        return [image, self.text[idx]], gt


def make_dataset(dataset, split, tokenizer, cfg, train, subset=None):
    ids, img, msk = data.load_split_arrays(dataset, split)
    if subset:
        ids, img, msk = ids[:subset], img[:subset], msk[:subset]
    return DDCMDDataset(ids, img, msk, common.read_texts(dataset, split, ids), tokenizer, cfg, train)


class DDCMDModule(CreateModel):
    """Official CreateModel + the per-case validation Dice (logged as val_case_dice) used by D3."""

    def __init__(self, args):
        super().__init__(args)  # save_hyperparameters() stores only `args` (val masks are set afterwards)
        self.val_masks = None   # (N, 224, 224) {0, 1}, set by the driver
        self._val_probs = []

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        self._val_probs.append(out["preds"][:, 0].float().cpu().numpy())
        return out

    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs)                       # official metrics / history
        probs = np.concatenate(self._val_probs)
        self._val_probs = []
        if self.trainer.sanity_checking:                            # 2 sanity batches only
            return
        dice = float(metrics.dice_many(probs, self.val_masks).mean())
        self.history[self.trainer.current_epoch]["val_case_dice"] = dice
        self.log("val_case_dice", dice, logger=True)


class SegmentControl(pl.Callback):
    """Wall-clock guard, epochs.csv and the RNG / loader-generator state in Lightning checkpoints."""

    def __init__(self, run_dir, generator, stop_after_epochs=None, resume_epoch=0):
        self.clock, self.g, self.stop_after = common.WallClock(), generator, stop_after_epochs
        self.longest, self.t0, self.segment_ended, self.start_epoch = 0.0, None, False, None
        self.elog = common.EpochLog(run_dir, ["epoch", "lr", "train_loss", "val_dice", "best_val_dice",
                                              "best_epoch", "val_MIoU_official", "epoch_seconds"],
                                    resume_epoch=resume_epoch)
        self.best_dice, self.best_epoch = 0.0, 1

    def on_train_epoch_start(self, trainer, pl_module):
        import time
        self.t0 = time.time()
        if self.start_epoch is None:
            self.start_epoch = trainer.current_epoch
        self.lr = trainer.optimizers[0].param_groups[0]["lr"]

    def on_train_epoch_end(self, trainer, pl_module):
        # Runs after validation and before ModelCheckpoint (Lightning orders checkpoint callbacks last)
        import time
        sec = time.time() - self.t0
        self.longest = max(self.longest, sec)
        e = trainer.current_epoch
        h = pl_module.history.get(e, {})
        dice = h.get("val_case_dice", float("nan"))
        if dice > self.best_dice:
            self.best_dice, self.best_epoch = dice, e + 1
        self.elog.write(epoch=e + 1, lr=self.lr, train_loss=h.get("train_loss", float("nan")), val_dice=dice,
                        best_val_dice=self.best_dice, best_epoch=self.best_epoch,
                        val_MIoU_official=h.get("val_MIoU", float("nan")), epoch_seconds=sec)
        done_here = e + 1 - self.start_epoch
        if self.clock.should_stop(self.longest) or (self.stop_after and done_here >= self.stop_after):
            if e + 1 < trainer.max_epochs and not trainer.should_stop:
                self.segment_ended = True
                trainer.should_stop = True

    def state_dict(self):
        return {"rng": common.rng_state(), "loader_generator": self.g.get_state(), "longest": self.longest,
                "best_dice": self.best_dice, "best_epoch": self.best_epoch}

    def load_state_dict(self, state):
        common.set_rng_state(state["rng"])
        self.g.set_state(state["loader_generator"])
        self.longest, self.best_dice, self.best_epoch = state["longest"], state["best_dice"], state["best_epoch"]


def module_args(cfg):
    return types.SimpleNamespace(bert_type=cfg["model"]["bert_type"], vision_type=cfg["model"]["vision_type"],
                                 lr=cfg["lr"])


@torch.no_grad()
def predict(module, loader, device, text_override=None):
    """Foreground probabilities sigmoid(logits) (N,224,224), targets, in loader order.
    text_override (quick check only): replace every text by this tokenised text dict."""
    module.eval()
    probs, targets = [], []
    for (image, text), gt in loader:
        if text_override is not None:
            text = {k: v.expand(len(image), -1) for k, v in text_override.items()}
        text = {k: v.to(device) for k, v in text.items()}
        probs.append(torch.sigmoid(module([image.to(device), text]))[:, 0].float().cpu().numpy())
        targets.append(gt[:, 0].numpy())
    return np.concatenate(probs), np.concatenate(targets)


def final_eval(args, cfg, log, best_path):
    """Final evaluation of the validation-selected checkpoint with the common evaluator.
    The test split needs both the program lock and the data lock open: otherwise stop with LOCKED_EXIT_CODE."""
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_type"], trust_remote_code=True)
    best = DDCMDModule(module_args(cfg))
    best.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"])
    best.to(device)
    for split in args.eval_splits.split(","):
        run = {"method": "ddcmd", "dataset": args.dataset, "seed": args.seed}
        lock = common.require_split(args.dataset, split, log, run=run)   # test: program lock + data lock
        ds = make_dataset(args.dataset, split, tokenizer, cfg, False)
        loader = DataLoader(ds, batch_size=cfg["val_batch_size_eval"], shuffle=False, num_workers=cfg["num_workers"])
        probs, targets = predict(best, loader, device)
        common.check_targets(args.dataset, split, ds.ids, targets)
        runner.final_evaluation(args.out, args.dataset, split, ds.ids, probs, log)
        if split == "test":
            common.log_test_eval("end", run, lock)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(common.REPO, "configs", "ddcmd.yaml"))
    ap.add_argument("--dataset", required=True, choices=data.DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-epochs", type=int, help="override (quick checks only)")
    ap.add_argument("--subset-train", type=int, help="first N training cases (quick checks only)")
    ap.add_argument("--subset-val", type=int, help="first N validation cases (quick checks only)")
    ap.add_argument("--stop-after-epochs", type=int, help="end the segment after N epochs (resume test only)")
    ap.add_argument("--no-final-eval", action="store_true")
    ap.add_argument("--eval-splits", default="val,test",
                    help="splits scored at the end (quick checks: val only, the test set is never touched)")
    args = ap.parse_args()

    cfg = runner.load_config(args.config)
    max_epochs = args.max_epochs or cfg["max_epochs"]
    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(args.out, "checkpoints")
    last_path, done_flag = os.path.join(ckpt_dir, "last.ckpt"), os.path.join(args.out, "TRAINING_FINISHED")
    if os.path.exists(done_flag):  # training done earlier: only the final evaluation is (re)run
        log = runner.make_logger(args.out)
        log.info("training already finished; final evaluation only")
        if not args.no_final_eval:
            with open(os.path.join(args.out, "training_summary.json")) as f:
                best_path = os.path.join(args.out, json.load(f)["best_model_path"])
            final_eval(args, cfg, log, best_path)
        return
    resume = last_path if os.path.exists(last_path) else None
    resume_epoch = torch.load(resume, map_location="cpu", weights_only=False)["epoch"] + 1 if resume else 0

    runner.set_seed(args.seed, cfg["deterministic"])
    log = runner.make_logger(args.out)
    if not resume:
        with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
            yaml.safe_dump({**cfg, "dataset": args.dataset, "seed": args.seed, "max_epochs_used": max_epochs,
                            "subset_train": args.subset_train, "subset_val": args.subset_val}, f, sort_keys=False)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_type"], trust_remote_code=True)
    train_ds = make_dataset(args.dataset, "train", tokenizer, cfg, True, args.subset_train)
    val_ds = make_dataset(args.dataset, "val", tokenizer, cfg, False, args.subset_val)
    g = torch.Generator()
    g.manual_seed(args.seed)
    for ds in (train_ds, val_ds):   # main-process MONAI random state (workers are re-seeded per worker)
        ds.transform.set_random_state(seed=args.seed)
    dl_train = DataLoader(train_ds, batch_size=cfg["train_batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                          pin_memory=True, worker_init_fn=monai_worker_init_fn, generator=g,
                          persistent_workers=False)
    dl_valid = DataLoader(val_ds, batch_size=cfg["valid_batch_size"], shuffle=False, num_workers=cfg["num_workers"],
                          pin_memory=True)

    module = DDCMDModule(module_args(cfg))
    module.val_masks = val_ds.masks
    control = SegmentControl(args.out, g, args.stop_after_epochs, resume_epoch)
    # Order: EarlyStopping before the wall-clock control (an early-stopped run is never marked "segment");
    # Lightning moves ModelCheckpoint to the end, so last.ckpt is written after both.
    callbacks = [
        EarlyStopping(monitor=cfg["monitor"], patience=cfg["patience"], mode="max"),
        control,
        ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor=cfg["monitor"], mode="max", save_top_k=1,
                        save_last=True, save_on_train_epoch_end=True, verbose=True),
    ]
    # D3: both early stopping and checkpoint selection must monitor the per-case validation Dice
    monitors = {type(c).__name__: c.monitor for c in callbacks if isinstance(c, (EarlyStopping, ModelCheckpoint))}
    assert set(monitors.values()) == {"val_case_dice"}, monitors
    trainer = pl.Trainer(logger=CSVLogger(save_dir=args.out, name="csv_logs"), min_epochs=cfg["min_epochs"],
                         max_epochs=max_epochs, accelerator="gpu", devices=1, callbacks=callbacks,
                         enable_progress_bar=False, log_every_n_steps=10, enable_model_summary=False,
                         precision=cfg["precision"])
    common.log_segment(args.out, start_epoch=resume_epoch + 1)
    log.info(f"DD-CMD | {args.dataset} | seed {args.seed} | train {len(train_ds)} | val {len(val_ds)} | "
             f"batch {cfg['train_batch_size']} | lr {cfg['lr']} | start epoch {resume_epoch + 1} | "
             f"PL {pl.__version__}")
    trainer.fit(module, dl_train, dl_valid, ckpt_path=resume)

    last_epoch = trainer.current_epoch  # number of completed epochs
    ended = "segment" if control.segment_ended else ("early_stopping" if last_epoch < max_epochs else "max_epochs")
    ckpt_cb = callbacks[-1]
    best_epoch = int(torch.load(ckpt_cb.best_model_path, map_location="cpu", weights_only=False)["epoch"]) + 1
    common.log_segment(args.out, last_epoch=last_epoch, ended=ended)
    runner.save_json({"best_epoch": best_epoch, "best_val_dice": float(ckpt_cb.best_model_score),
                      "last_epoch": last_epoch, "ended": ended, "monitors": monitors,
                      "best_model_path": os.path.relpath(ckpt_cb.best_model_path, args.out),
                      "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9},
                     os.path.join(args.out, "training_summary.json"))
    log.info(f"{ended} after epoch {last_epoch}; best val Dice {float(ckpt_cb.best_model_score):.4f} @ {best_epoch}")
    if ended == "segment":
        log.info("segment ended (wall-clock guard); submit the next segment by hand")
        sys.exit(common.SEGMENT_EXIT_CODE)
    open(done_flag, "w").close()
    if args.no_final_eval:
        return

    final_eval(args, cfg, log, ckpt_cb.best_model_path)


if __name__ == "__main__":
    main()
