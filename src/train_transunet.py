"""Train TransUNet (R50-ViT-B_16) on one dataset with one seed (Phase 0), then evaluate the best epoch.

Network, pretrained-weight loading and Dice loss are imported unchanged from the official repository
(external/TransUNet, commit 02ef001). The training loop follows trainer.py (trainer_synapse) line by
line; the only addition is the per-epoch validation Dice used to keep the best epoch (agreed plan).
The Synapse data loader is replaced by the common 224x224 data with the official RandomGenerator
augmentation (datasets/dataset_synapse.py), copied below. Settings come from configs/transunet.yaml.

Run directory (`--out`): same files as src/train_unet.py.

Usage (inside a Slurm job):
    python src/train_transunet.py --dataset QaTa-COV19 --seed 1001 --out runs/transunet/QaTa-COV19/seed1001
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from scipy import ndimage
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "external", "TransUNet"))
from lgms import data, metrics, runner  # noqa: E402
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg  # noqa: E402  (official)
from networks.vit_seg_modeling import VisionTransformer as ViT_seg  # noqa: E402
from utils import DiceLoss  # noqa: E402  (official TransUNet utils.py)


# ---------------------------------------------------------------------------------------------
# Augmentation: copied from TransUNet datasets/dataset_synapse.py
# ---------------------------------------------------------------------------------------------
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class TransUNetDataset(Dataset):
    """Common data as the official RandomGenerator returns it: image float (1, 224, 224) in [0, 1],
    label long (224, 224). Images are already 224x224, so the official zoom is never triggered."""

    def __init__(self, ids, images, masks, train):
        self.ids, self.images, self.masks, self.train = ids, images, masks, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        label = self.masks[idx]
        if self.train:  # RandomGenerator.__call__
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
        image = torch.from_numpy(np.ascontiguousarray(image).astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray(label).astype(np.float32)).long()
        return image, label, self.ids[idx]


def build_loaders(cfg, dataset, seed, subset_train=None, subset_val=None):
    """Loaders as in trainer.py (shuffle, 8 workers, pin_memory, worker seed = seed + id)."""
    tr_ids, tr_img, tr_msk = data.load_split_arrays(dataset, "train")
    va_ids, va_img, va_msk = data.load_split_arrays(dataset, "val")
    if subset_train:
        tr_ids, tr_img, tr_msk = tr_ids[:subset_train], tr_img[:subset_train], tr_msk[:subset_train]
    if subset_val:
        va_ids, va_img, va_msk = va_ids[:subset_val], va_img[:subset_val], va_msk[:subset_val]

    def worker_init_fn(worker_id):  # trainer.py
        random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(TransUNetDataset(tr_ids, tr_img, tr_msk, train=True), batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"], pin_memory=True,
                              worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(TransUNetDataset(va_ids, va_img, va_msk, train=False),
                            batch_size=cfg["val_batch_size"], shuffle=False, num_workers=cfg["num_workers"],
                            pin_memory=True)
    return train_loader, val_loader


def build_model(cfg, load_pretrained=True):
    """train.py: R50-ViT-B_16 config, n_classes, n_skip, patch grid img_size/16, ImageNet-21k weights."""
    m = cfg["model"]
    config_vit = CONFIGS_ViT_seg[m["vit_name"]]
    config_vit.n_classes = m["n_classes"]
    config_vit.n_skip = m["n_skip"]
    if m["vit_name"].find("R50") != -1:
        g = int(cfg["input_size"] / m["vit_patches_size"])
        config_vit.patches.grid = (g, g)
    net = ViT_seg(config_vit, img_size=cfg["input_size"], num_classes=config_vit.n_classes)
    if load_pretrained:
        net.load_from(weights=np.load(os.path.expanduser(cfg["pretrained"]["file"])))
    return net


class Criterion:
    """trainer.py: loss = 0.5 * CE(outputs, label) + 0.5 * Dice(softmax(outputs), label)."""

    def __init__(self, cfg):
        self.ce, self.dice = CrossEntropyLoss(), DiceLoss(cfg["model"]["n_classes"])
        self.w_ce, self.w_dice = cfg["loss"]["ce_weight"], cfg["loss"]["dice_weight"]

    def __call__(self, outputs, labels):
        return self.w_ce * self.ce(outputs, labels[:].long()) + self.w_dice * self.dice(outputs, labels, softmax=True)


@torch.no_grad()
def predict(model, loader, device):
    """Foreground (lesion) softmax probabilities (N, 224, 224) and ids, in loader order."""
    model.eval()
    probs, ids = [], []
    for images, _, names in loader:
        out = model(images.to(device, non_blocking=True))
        probs.append(torch.softmax(out, dim=1)[:, 1].float().cpu().numpy())
        ids.extend(names)
    return ids, np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "transunet.yaml"))
    ap.add_argument("--dataset", required=True, choices=data.DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-epochs", type=int, help="override (quick checks only)")
    ap.add_argument("--subset-train", type=int, help="use the first N training cases (quick checks only)")
    ap.add_argument("--subset-val", type=int, help="use the first N validation cases (quick checks only)")
    ap.add_argument("--no-final-eval", action="store_true")
    args = ap.parse_args()

    cfg = runner.load_config(args.config)
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    runner.set_seed(args.seed, cfg["deterministic"])
    log = runner.make_logger(args.out)
    with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "dataset": args.dataset, "seed": args.seed,
                        "subset_train": args.subset_train, "subset_val": args.subset_val}, f, sort_keys=False)

    device = torch.device("cuda")
    train_loader, val_loader = build_loaders(cfg, args.dataset, args.seed, args.subset_train, args.subset_val)
    val_gt = val_loader.dataset.masks
    model = build_model(cfg).to(device)
    criterion = Criterion(cfg)
    o = cfg["optimizer"]
    base_lr = o["lr"]
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=o["momentum"], weight_decay=o["weight_decay"])
    max_epoch = cfg["max_epochs"]
    max_iterations = max_epoch * len(train_loader)  # trainer.py
    log.info(f"TransUNet | {args.dataset} | seed {args.seed} | train {len(train_loader.dataset)} | "
             f"val {len(val_loader.dataset)} | batch {cfg['batch_size']} | {len(train_loader)} it/epoch | "
             f"{max_iterations} iterations | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    elog = runner.EpochLog(args.out, ["epoch", "lr", "train_loss", "val_dice", "best_val_dice", "best_epoch",
                                      "train_seconds", "val_seconds"])
    iter_num, best_dice, best_epoch = 0, -1.0, 0
    for epoch in range(1, max_epoch + 1):
        lr_start = optimizer.param_groups[0]["lr"]
        with runner.Timer() as t_train:
            model.train()
            total, n = 0.0, 0
            for image_batch, label_batch, _ in train_loader:
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                outputs = model(image_batch)
                loss = criterion(outputs, label_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # poly LR, updated after every iteration (trainer.py)
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** cfg["scheduler"]["power"]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_
                iter_num += 1
                total += loss.item() * len(image_batch)
                n += len(image_batch)
        with runner.Timer() as t_val:
            _, probs = predict(model, val_loader, device)
            val_dice = float(metrics.dice_many(probs, val_gt).mean())
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_dice": val_dice},
                       os.path.join(args.out, "best_model.pt"))
        elog.write(epoch=epoch, lr=lr_start, train_loss=total / n, val_dice=val_dice, best_val_dice=best_dice,
                   best_epoch=best_epoch, train_seconds=t_train.seconds, val_seconds=t_val.seconds)
        log.info(f"epoch {epoch:4d} | lr {lr_start:.2e} | loss {total / n:.4f} | val Dice {val_dice:.4f} | "
                 f"best {best_dice:.4f} @ {best_epoch} | {t_train.seconds:.0f}+{t_val.seconds:.0f} s")

    runner.save_json({"best_epoch": best_epoch, "best_val_dice": best_dice, "last_epoch": max_epoch,
                      "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9},
                     os.path.join(args.out, "training_summary.json"))
    if args.no_final_eval:
        return

    # Final evaluation of the best (validation-selected) epoch with the common evaluator
    model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pt"), map_location=device)["state_dict"])
    for split in ["val", "test"]:
        ids, img, msk = data.load_split_arrays(args.dataset, split)
        loader = DataLoader(TransUNetDataset(ids, img, msk, train=False), batch_size=cfg["val_batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"])
        ids, probs = predict(model, loader, device)
        runner.final_evaluation(args.out, args.dataset, split, ids, probs, log)


if __name__ == "__main__":
    main()
