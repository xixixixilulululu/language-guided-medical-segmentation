"""Train LViT-T on one dataset with one seed (Phase 1), then evaluate the best checkpoint.

Model, loss and LR scheduler are imported unchanged from the LViT repository (external/LViT, commit
ba90775). The data loader is the Phase 0 one (src/train_unet.py: LViT's RandomGenerator / ValGenerator on
the common 224x224 data) plus the text: the 10 x 768 BERT word vectors of the sample's padded text, read
from the cache of src/phase1/lvit_text.py. Training loop = LViT train_model.py / Train_one_epoch.py.
Settings: configs/lvit.yaml.

Resumable: an existing <out>/last.pth is continued (one Slurm job = one segment, submitted by hand).

Usage (inside a Slurm job):
    python src/phase1/train_lvit.py --dataset QaTa-COV19 --seed 1001 --out runs/lvit/QaTa-COV19/seed1001
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import common, lvit_text  # noqa: E402
from phase1.common import data, metrics, runner  # noqa: E402
import train_unet as p0  # noqa: E402  (Phase 0 augmentation: random_rot_flip, random_rotate)

sys.path.insert(0, os.path.join(common.REPO, "external", "LViT"))
_env = dict(os.environ)
import Config as lvit_config  # noqa: E402  (only get_CTranS_config is used)
os.environ.clear(); os.environ.update(_env)  # Config.py sets CUDA_VISIBLE_DEVICES / PYTHONHASHSEED on import
from nets.LViT import LViT  # noqa: E402
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE  # noqa: E402  (LViT utils.py)


class LViTDataset(Dataset):
    """Phase 0 UNetDataset + text vectors. Returns image (3,224,224) in [0,1], mask long, text (10,768)."""

    def __init__(self, ids, images, masks, texts, cache, train):
        self.ids, self.images, self.masks, self.train = ids, images, masks, train
        self.text = [cache[common.lvit_pad(texts[i])] for i in ids]      # KeyError if a text is missing

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = np.repeat(self.images[idx][..., None], 3, axis=2)
        label = self.masks[idx]
        if self.train:  # RandomGenerator.__call__
            if random.random() > 0.5:
                image, label = p0.random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = p0.random_rotate(image, label)
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return image, label, self.text[idx], self.ids[idx]


def make_dataset(dataset, split, cache, train, subset=None):
    ids, img, msk = data.load_split_arrays(dataset, split)
    if subset:
        ids, img, msk = ids[:subset], img[:subset], msk[:subset]
    return LViTDataset(ids, img, msk, common.read_texts(dataset, split, ids), cache, train)


def build_model():
    return LViT(lvit_config.get_CTranS_config(), n_channels=3, n_classes=1)


@torch.no_grad()
def predict(model, loader, device, text_override=None):
    """Foreground probabilities (N,224,224), targets and ids in loader order.
    text_override (quick check only): replace every text by this (10,768) tensor."""
    model.eval()
    probs, targets, ids = [], [], []
    for images, masks, text, names in loader:
        if text_override is not None:
            text = text_override.expand(len(images), -1, -1)
        probs.append(model(images.to(device), text.to(device))[:, 0].float().cpu().numpy())
        targets.append(masks.numpy())
        ids.extend(names)
    return ids, np.concatenate(probs), np.concatenate(targets)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """LViT Train_one_epoch.py, training branch. Returns the mean batch loss."""
    model.train()
    total, n = 0.0, 0
    for images, masks, text, _ in loader:
        images, masks, text = images.to(device), masks.to(device), text.to(device)
        preds = model(images, text)                 # sigmoid probabilities (B, 1, H, W)
        loss = criterion(preds, masks.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(images)
        n += len(images)
    return total / n


def final_eval(args, cfg, log):
    """Final evaluation of the validation-selected checkpoint (best_model.pt) with the common evaluator.
    The test split needs both the program lock and the data lock open: otherwise stop with LOCKED_EXIT_CODE."""
    device = torch.device("cuda")
    cache = lvit_text.load_cache(os.path.expanduser(cfg["text_cache"]))
    model = build_model().to(device)
    model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pt"), map_location=device)["state_dict"])
    for split in args.eval_splits.split(","):
        run = {"method": "lvit", "dataset": args.dataset, "seed": args.seed}
        lock = common.require_split(args.dataset, split, log, run=run)   # test: program lock + data lock
        ds = make_dataset(args.dataset, split, cache, False)
        loader = DataLoader(ds, batch_size=cfg["val_batch_size"], shuffle=False, num_workers=cfg["num_workers"])
        ids, probs, targets = predict(model, loader, device)
        common.check_targets(args.dataset, split, ids, targets)
        runner.final_evaluation(args.out, args.dataset, split, ids, probs, log)
        if split == "test":
            common.log_test_eval("end", run, lock)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(common.REPO, "configs", "lvit.yaml"))
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
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    lr = runner.per_dataset(cfg["optimizer"]["lr"], args.dataset)
    os.makedirs(args.out, exist_ok=True)
    last_path, best_path = os.path.join(args.out, "last.pth"), os.path.join(args.out, "best_model.pt")
    resume = common.load_state(last_path) if os.path.exists(last_path) else None
    if resume and resume.get("finished"):  # training done earlier: only the final evaluation is (re)run
        log = runner.make_logger(args.out)
        log.info("training already finished; final evaluation only")
        if not args.no_final_eval:
            final_eval(args, cfg, log)
        return

    runner.set_seed(args.seed, cfg["deterministic"])
    log = runner.make_logger(args.out)
    if not resume:
        with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
            yaml.safe_dump({**cfg, "dataset": args.dataset, "seed": args.seed, "lr_used": lr,
                            "subset_train": args.subset_train, "subset_val": args.subset_val}, f, sort_keys=False)

    device = torch.device("cuda")
    cache = lvit_text.load_cache(os.path.expanduser(cfg["text_cache"]))
    train_ds = make_dataset(args.dataset, "train", cache, True, args.subset_train)
    val_ds = make_dataset(args.dataset, "val", cache, False, args.subset_val)

    def worker_init_fn(worker_id):  # LViT train_model.py
        random.seed(args.seed + worker_id)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                              pin_memory=True, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg["val_batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model().to(device)
    criterion = WeightedDiceBCE(dice_weight=cfg["loss"]["dice_weight"], BCE_weight=cfg["loss"]["bce_weight"])
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    sc = cfg["scheduler"]
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=sc["T_0"], T_mult=sc["T_mult"], eta_min=sc["eta_min"])
    sel = common.Selection(cfg["early_stopping_patience"], cfg["first_epoch_for_best"])
    start_epoch, longest = 1, 0.0
    if resume:  # continue exactly where the previous segment ended
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        sel = common.Selection(cfg["early_stopping_patience"], cfg["first_epoch_for_best"], **resume["selection"])
        g.set_state(resume["loader_generator"])
        common.set_rng_state(resume["rng"])
        start_epoch, longest = resume["epoch"] + 1, resume["longest_epoch"]
    common.log_segment(args.out, start_epoch=start_epoch)
    log.info(f"LViT-T | {args.dataset} | seed {args.seed} | train {len(train_ds)} | val {len(val_ds)} | "
             f"batch {cfg['batch_size']} | lr {lr} | start epoch {start_epoch} | "
             f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    elog = common.EpochLog(args.out, ["epoch", "lr", "train_loss", "val_dice", "best_val_dice", "best_epoch",
                                      "train_seconds", "val_seconds"], resume_epoch=start_epoch - 1)
    clock, epoch, ended = common.WallClock(), start_epoch - 1, "max_epochs"
    for epoch in range(start_epoch, cfg["max_epochs"] + 1):
        if clock.should_stop(longest) or (args.stop_after_epochs and epoch - start_epoch >= args.stop_after_epochs):
            epoch, ended = epoch - 1, "segment"
            break
        lr_now = optimizer.param_groups[0]["lr"]
        with runner.Timer() as t_train:
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        with runner.Timer() as t_val:
            _, probs, _ = predict(model, val_loader, device)
            val_dice = float(metrics.dice_many(probs, val_ds.masks).mean())
        scheduler.step()  # LViT: once per epoch, after validation
        longest = max(longest, t_train.seconds + t_val.seconds)
        if sel.update(epoch, val_dice):
            common.save_atomic({"epoch": epoch, "state_dict": model.state_dict(), "val_dice": val_dice}, best_path)
        stop = sel.stop(epoch)
        common.save_atomic({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "selection": sel.state(), "longest_epoch": longest,
                            "loader_generator": g.get_state(), "rng": common.rng_state(), "finished": stop},
                           last_path)
        elog.write(epoch=epoch, lr=lr_now, train_loss=train_loss, val_dice=val_dice, best_val_dice=sel.best_dice,
                   best_epoch=sel.best_epoch, train_seconds=t_train.seconds, val_seconds=t_val.seconds)
        log.info(f"epoch {epoch:4d} | lr {lr_now:.2e} | loss {train_loss:.4f} | val Dice {val_dice:.4f} | "
                 f"best {sel.best_dice:.4f} @ {sel.best_epoch} | {t_train.seconds:.0f}+{t_val.seconds:.0f} s")
        if stop:
            log.info(f"early stopping at epoch {epoch} (best epoch {sel.best_epoch})")
            ended = "early_stopping"
            break

    common.log_segment(args.out, last_epoch=epoch, ended=ended)
    runner.save_json({"best_epoch": sel.best_epoch, "best_val_dice": sel.best_dice, "last_epoch": epoch,
                      "ended": ended, "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9},
                     os.path.join(args.out, "training_summary.json"))
    if ended == "segment":
        log.info(f"segment ended after epoch {epoch} (wall-clock guard); submit the next segment by hand")
        sys.exit(common.SEGMENT_EXIT_CODE)
    if ended == "max_epochs":  # mark as finished: a further job only re-runs the final evaluation
        state = common.load_state(last_path)
        state["finished"] = True
        common.save_atomic(state, last_path)
    if args.no_final_eval:
        return

    final_eval(args, cfg, log)


if __name__ == "__main__":
    main()
