"""Train RecLMIS on one dataset with one seed (Phase 1), then evaluate the best checkpoint.

Model, loss, scheduler, augmentation classes and the per-dataset settings are taken unchanged from the
RecLMIS repository (external/RecLMIS, commit d3265c8). The dataset class re-implements
Load_Dataset.ImageToImage2D.__getitem__ on the common 224x224 data in memory (grayscale repeated to the 3
channels cv2.imread returns; text padded as utils.read_text; clip.tokenize with token_len 18), then applies
the official transforms. Training loop = train_model.py / Train_one_epoch.py.
Compatibility (no change of the method): fairseq.utils.softmax provided here (fairseq cannot be built);
the reconstruction / CCL losses are checked every step (the official code turns them off silently on error).

Resumable: an existing <out>/last.pth is continued (one Slurm job = one segment, submitted by hand).

Usage (inside a Slurm job):
    python src/phase1/train_reclmis.py --dataset QaTa-COV19 --seed 1001 --out runs/reclmis/QaTa-COV19/seed1001
"""
import argparse
import importlib
import math
import os
import random
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import common  # noqa: E402
from phase1.common import data, metrics, runner  # noqa: E402

# fairseq is only used for utils.softmax (nets/transformer/mutihead_attention.py:184). fairseq's definition:
#   softmax(x, dim, onnx_trace=False) = F.softmax(x.float(), dim) if onnx_trace else F.softmax(x, dim, dtype=float32)
_fq, _fq_utils = types.ModuleType("fairseq"), types.ModuleType("fairseq.utils")
_fq_utils.softmax = lambda x, dim, onnx_trace=False: (
    F.softmax(x.float(), dim=dim) if onnx_trace else F.softmax(x, dim=dim, dtype=torch.float32))
_fq.utils = _fq_utils
sys.modules.setdefault("fairseq", _fq)
sys.modules.setdefault("fairseq.utils", _fq_utils)

sys.path.insert(0, os.path.join(common.REPO, "external", "RecLMIS"))
import clip  # noqa: E402  (OpenAI CLIP package: tokenizer)
from Load_Dataset import RandomGenerator, ValGenerator, correct_dims  # noqa: E402
from nets.RecLMIS import RecLMIS  # noqa: E402
from tools.transform import BioMedicalGaussianBlur, PhotoMetricDistortion  # noqa: E402
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE  # noqa: E402  (RecLMIS utils.py)

AUX_LOSSES = ("loss_ccl", "loss_text_rec", "loss_img_rec")


def official_config(cfg, dataset):
    """Import the official config module of the dataset and check it against configs/reclmis.yaml."""
    env = dict(os.environ)
    mod = importlib.import_module(cfg["config_module"][dataset])
    os.environ.clear(); os.environ.update(env)  # the module sets PYTHONHASHSEED on import
    expect = {"batch_size": cfg["batch_size"][dataset], "learning_rate": cfg["lr"][dataset],
              "lr": cfg["scheduler"][dataset], "epochs": cfg["max_epochs"][dataset],
              "early_stopping_patience": cfg["early_stopping_patience"][dataset],
              "loss_weight": cfg["loss_weight"], "token_len": cfg["text"]["token_len"], "optimizer": "Adam",
              "img_size": cfg["input_size"]}
    for k, v in expect.items():
        if getattr(mod, k) != v:
            raise ValueError(f"configs/reclmis.yaml {k}={v} differs from {mod.__name__}.{k}={getattr(mod, k)}")
    if mod.get_ViT_config().clip_backbone != cfg["model"]["clip_backbone"]:
        raise ValueError("clip_backbone differs from the official config")
    return mod


class RecLMISDataset(Dataset):
    """ImageToImage2D.__getitem__ on in-memory common224 data. Returns (sample dict, sample id)."""

    def __init__(self, ids, images, masks, texts, token_len, train):
        self.ids, self.images, self.masks, self.train = ids, images, masks, train
        self.tokens = []
        for i in ids:  # Load_Dataset.py:224-228 (deterministic, so computed once)
            tok = clip.tokenize(common.lvit_pad(texts[i]).split("\n"), context_length=token_len, truncate=True)
            self.tokens.append(tok.squeeze())
        self.blur, self.photo = BioMedicalGaussianBlur(prob=0.5), PhotoMetricDistortion()
        self.joint = RandomGenerator(output_size=[224, 224]) if train else ValGenerator(output_size=[224, 224])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = np.repeat(self.images[idx][..., None], 3, axis=2)        # cv2.imread: 3 identical channels
        mask = self.masks[idx].copy()                                    # {0, 1}
        image, mask = correct_dims(image, mask)
        token = self.tokens[idx]
        sample = {"image": image, "label": mask, "text_token": token, "text_mask": (token != 0).int()}
        if self.train:
            sample = self.blur.transform(sample)
            sample = self.photo.transform(sample)
        return self.joint(sample), self.ids[idx]


def make_dataset(dataset, split, token_len, train, subset=None):
    ids, img, msk = data.load_split_arrays(dataset, split)
    if subset:
        ids, img, msk = ids[:subset], img[:subset], msk[:subset]
    return RecLMISDataset(ids, img, msk, common.read_texts(dataset, split, ids), token_len, train)


def build_model(conf):
    return RecLMIS(conf, conf.get_ViT_config(), n_channels=conf.n_channels, n_classes=conf.n_labels)


def build_scheduler(conf, optimizer):
    """train_model.py:150-154."""
    if conf.lr == "cosineLR":
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    if conf.lr == "exp":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.99 ** epoch, 0.1))
    raise ValueError(conf.lr)


def forward_batch(model, sample, device):
    images, masks = sample["image"].to(device), sample["label"].to(device)
    preds, loss_dict = model(images, masks, sample["text_token"].to(device), sample["text_mask"].to(device))
    return preds, masks, loss_dict


@torch.no_grad()
def predict(model, loader, device, token_override=None):
    """Foreground probabilities (N,224,224), targets and ids in loader order.
    token_override (quick check only): replace every text by this tokenised text."""
    model.eval()
    probs, targets, ids = [], [], []
    for sample, names in loader:
        if token_override is not None:
            sample["text_token"] = token_override.expand(len(names), -1)
            sample["text_mask"] = (sample["text_token"] != 0).int()
        preds, _, _ = forward_batch(model, sample, device)
        probs.append(preds[:, 0].float().cpu().numpy())
        targets.append(sample["label"].numpy())
        ids.extend(names)
    return ids, np.concatenate(probs), np.concatenate(targets)


def train_one_epoch(model, loader, criterion, optimizer, conf, device, stats):
    """Train_one_epoch.py, training branch. Returns the mean weighted total loss and the mean of each term."""
    model.train()
    sums, n = {}, 0
    for sample, _ in loader:
        preds, masks, loss_dict = forward_batch(model, sample, device)
        loss_dict["loss_criterion"] = criterion(preds, masks.float())
        # The method's own losses must be active in every step (see module docstring)
        missing = [k for k in AUX_LOSSES if k not in loss_dict]
        if missing or not model.aux:
            raise RuntimeError(f"RecLMIS auxiliary losses switched off (missing {missing}, aux={model.aux})")
        out_loss = 0
        for k, v in loss_dict.items():
            loss_dict[k] = v.mean()
            out_loss = out_loss + conf.loss_weight[k] * loss_dict[k]
        values = {k: float(v.detach()) for k, v in loss_dict.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise RuntimeError(f"non-finite loss: {values}")
        # The official reconstructor replaces a NaN reconstruction loss by exactly 0: count such steps
        stats["rec_loss_zero_steps"] += int(values["loss_text_rec"] == 0.0 or values["loss_img_rec"] == 0.0)
        stats["steps"] += 1
        optimizer.zero_grad()
        out_loss.backward()
        optimizer.step()
        b = len(masks)
        for k, v in values.items():
            sums[k] = sums.get(k, 0.0) + v * b
        sums["total"] = sums.get("total", 0.0) + float(out_loss.detach()) * b
        n += b
    return {k: v / n for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(common.REPO, "configs", "reclmis.yaml"))
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
    conf = official_config(cfg, args.dataset)
    max_epochs = args.max_epochs or conf.epochs
    os.makedirs(args.out, exist_ok=True)
    last_path, best_path = os.path.join(args.out, "last.pth"), os.path.join(args.out, "best_model.pt")
    resume = common.load_state(last_path) if os.path.exists(last_path) else None
    if resume and resume.get("finished"):
        print("training already finished; nothing to do"); return

    runner.set_seed(args.seed, cfg["deterministic"])
    log = runner.make_logger(args.out)
    if not resume:
        with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
            yaml.safe_dump({**cfg, "dataset": args.dataset, "seed": args.seed, "official_module": conf.__name__,
                            "max_epochs_used": max_epochs, "subset_train": args.subset_train,
                            "subset_val": args.subset_val}, f, sort_keys=False)

    device = torch.device("cuda")
    train_ds = make_dataset(args.dataset, "train", conf.token_len, True, args.subset_train)
    val_ds = make_dataset(args.dataset, "val", conf.token_len, False, args.subset_val)

    def worker_init_fn(worker_id):  # train_model.py
        random.seed(args.seed + worker_id)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=conf.batch_size, shuffle=True, num_workers=cfg["num_workers"],
                              pin_memory=True, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_ds, batch_size=cfg["val_batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(conf).to(device)
    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf.learning_rate)
    scheduler = build_scheduler(conf, optimizer)
    sel = common.Selection(conf.early_stopping_patience, cfg["first_epoch_for_best"])
    stats = {"steps": 0, "rec_loss_zero_steps": 0}
    start_epoch, longest = 1, 0.0
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        sel = common.Selection(conf.early_stopping_patience, cfg["first_epoch_for_best"], **resume["selection"])
        stats = resume["stats"]
        g.set_state(resume["loader_generator"])
        common.set_rng_state(resume["rng"])
        start_epoch, longest = resume["epoch"] + 1, resume["longest_epoch"]
    common.log_segment(args.out, start_epoch=start_epoch)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"RecLMIS | {args.dataset} | {conf.__name__} | seed {args.seed} | train {len(train_ds)} | "
             f"val {len(val_ds)} | batch {conf.batch_size} | lr {conf.learning_rate} | sched {conf.lr} | "
             f"start epoch {start_epoch} | trainable params {n_train / 1e6:.2f} M")

    cols = ["epoch", "lr", "train_loss", "loss_criterion", "loss_ccl", "loss_text_rec", "loss_img_rec",
            "val_dice", "best_val_dice", "best_epoch", "train_seconds", "val_seconds"]
    elog = common.EpochLog(args.out, cols, resume_epoch=start_epoch - 1)
    clock, epoch, ended = common.WallClock(), start_epoch - 1, "max_epochs"
    for epoch in range(start_epoch, max_epochs + 1):
        if clock.should_stop(longest) or (args.stop_after_epochs and epoch - start_epoch >= args.stop_after_epochs):
            epoch, ended = epoch - 1, "segment"
            break
        lr_now = optimizer.param_groups[0]["lr"]
        with runner.Timer() as t_train:
            losses = train_one_epoch(model, train_loader, criterion, optimizer, conf, device, stats)
        with runner.Timer() as t_val:
            _, probs, _ = predict(model, val_loader, device)
            val_dice = float(metrics.dice_many(probs, val_ds.masks).mean())
        scheduler.step()  # official: once per epoch, at the end of the validation pass
        longest = max(longest, t_train.seconds + t_val.seconds)
        if sel.update(epoch, val_dice):
            common.save_atomic({"epoch": epoch, "state_dict": model.state_dict(), "val_dice": val_dice}, best_path)
        stop = sel.stop(epoch)
        common.save_atomic({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "selection": sel.state(), "stats": stats,
                            "longest_epoch": longest, "loader_generator": g.get_state(),
                            "rng": common.rng_state(), "finished": stop}, last_path)
        elog.write(epoch=epoch, lr=lr_now, train_loss=losses["total"],
                   **{k: losses[k] for k in ("loss_criterion",) + AUX_LOSSES}, val_dice=val_dice,
                   best_val_dice=sel.best_dice, best_epoch=sel.best_epoch,
                   train_seconds=t_train.seconds, val_seconds=t_val.seconds)
        log.info(f"epoch {epoch:4d} | lr {lr_now:.2e} | loss {losses['total']:.4f} (seg {losses['loss_criterion']:.4f}"
                 f" ccl {losses['loss_ccl']:.4f} trec {losses['loss_text_rec']:.4f} irec {losses['loss_img_rec']:.4f})"
                 f" | val Dice {val_dice:.4f} | best {sel.best_dice:.4f} @ {sel.best_epoch} | "
                 f"{t_train.seconds:.0f}+{t_val.seconds:.0f} s")
        if stop:
            log.info(f"early stopping at epoch {epoch} (best epoch {sel.best_epoch})")
            ended = "early_stopping"
            break

    common.log_segment(args.out, last_epoch=epoch, ended=ended)
    runner.save_json({"best_epoch": sel.best_epoch, "best_val_dice": sel.best_dice, "last_epoch": epoch,
                      "ended": ended, **stats, "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9},
                     os.path.join(args.out, "training_summary.json"))
    if ended == "segment":
        log.info(f"segment ended after epoch {epoch} (wall-clock guard); submit the next segment by hand")
        sys.exit(common.SEGMENT_EXIT_CODE)
    if ended == "max_epochs":
        state = common.load_state(last_path)
        state["finished"] = True
        common.save_atomic(state, last_path)
    if args.no_final_eval:
        return

    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    for split in args.eval_splits.split(","):
        ds = make_dataset(args.dataset, split, conf.token_len, False)
        loader = DataLoader(ds, batch_size=cfg["val_batch_size"], shuffle=False, num_workers=cfg["num_workers"])
        ids, probs, targets = predict(model, loader, device)
        common.check_targets(args.dataset, split, ids, targets)
        runner.final_evaluation(args.out, args.dataset, split, ids, probs, log)


if __name__ == "__main__":
    main()
