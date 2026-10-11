"""Train the LViT U-Net on one dataset with one seed (Phase 0), then evaluate the best checkpoint.

The model, loss and LR scheduler are imported unchanged from the LViT repository (external/LViT,
commit ba90775). The LViT data loader cannot be imported (it requires bert_embedding for the text
branch), so its image-only part (augmentation RandomGenerator, ValGenerator) is re-implemented below
line by line on the common 224x224 data. Settings come from configs/unet.yaml.

Run directory (`--out`):
    train.log, epochs.csv, config_used.yaml, best_model.pt,
    val_probs.npz / test_probs.npz, eval_val/ and eval_test/ (per_case.csv, summary.json, masks/)

Usage (inside a Slurm job):
    python src/train_unet.py --dataset QaTa-COV19 --seed 1001 --out runs/unet/QaTa-COV19/seed1001
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "external", "LViT"))
from lgms import data, metrics, runner  # noqa: E402
from nets.UNet import UNet  # noqa: E402  (LViT)
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE  # noqa: E402  (LViT utils.py)


# ---------------------------------------------------------------------------------------------
# Augmentation: copied from LViT Load_Dataset.py (random_rot_flip, random_rotate, RandomGenerator)
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


class UNetDataset(Dataset):
    """Common 224x224 data as LViT feeds it to the network.

    Image: grayscale duplicated to 3 channels (LViT reads 3-channel images with cv2.imread),
    scaled to [0, 1] (torchvision to_tensor). Mask: long {0, 1}, shape (224, 224).
    Data are already 224x224, so the zoom of LViT's generators is never triggered.
    """

    def __init__(self, ids, images, masks, train):
        self.ids, self.images, self.masks, self.train = ids, images, masks, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = np.repeat(self.images[idx][..., None], 3, axis=2)  # (H, W, 3) uint8
        label = self.masks[idx]
        if self.train:  # RandomGenerator.__call__
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return image, label, self.ids[idx]


def build_loaders(cfg, dataset, seed, subset_train=None, subset_val=None):
    """Train / val loaders as in LViT train_model.py (shuffle, 8 workers, worker seed = seed + id)."""
    tr_ids, tr_img, tr_msk = data.load_split_arrays(dataset, "train")
    va_ids, va_img, va_msk = data.load_split_arrays(dataset, "val")
    if subset_train:
        tr_ids, tr_img, tr_msk = tr_ids[:subset_train], tr_img[:subset_train], tr_msk[:subset_train]
    if subset_val:
        va_ids, va_img, va_msk = va_ids[:subset_val], va_img[:subset_val], va_msk[:subset_val]

    def worker_init_fn(worker_id):  # LViT train_model.py
        random.seed(seed + worker_id)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(UNetDataset(tr_ids, tr_img, tr_msk, train=True), batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"], pin_memory=True,
                              worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(UNetDataset(va_ids, va_img, va_msk, train=False), batch_size=cfg["val_batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    return train_loader, val_loader


def build_model(cfg):
    return UNet(n_channels=cfg["model"]["n_channels"], n_classes=cfg["model"]["n_classes"])


def build_criterion(cfg):
    return WeightedDiceBCE(dice_weight=cfg["loss"]["dice_weight"], BCE_weight=cfg["loss"]["bce_weight"])


@torch.no_grad()
def predict(model, loader, device):
    """Foreground probabilities (N, 224, 224) float32 and ids, in loader order."""
    model.eval()
    probs, ids = [], []
    for images, _, names in loader:
        probs.append(model(images.to(device, non_blocking=True))[:, 0].float().cpu().numpy())
        ids.extend(names)
    return ids, np.concatenate(probs)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One LViT training epoch (Train_one_epoch.py, training branch). Returns the mean batch loss."""
    model.train()
    total, n = 0.0, 0
    for images, masks, _ in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        preds = model(images)                       # sigmoid probabilities (B, 1, H, W)
        loss = criterion(preds, masks.float())      # WeightedDiceBCE flattens both tensors per sample
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(images)
        n += len(images)
    return total / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "unet.yaml"))
    ap.add_argument("--dataset", required=True, choices=data.DATASETS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-epochs", type=int, help="override (quick checks only)")
    ap.add_argument("--batch-size", type=int, help="override (timing checks only)")
    ap.add_argument("--subset-train", type=int, help="use the first N training cases (quick checks only)")
    ap.add_argument("--subset-val", type=int, help="use the first N validation cases (quick checks only)")
    ap.add_argument("--no-final-eval", action="store_true", help="skip the val/test evaluation at the end")
    args = ap.parse_args()

    cfg = runner.load_config(args.config)
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    lr = runner.per_dataset(cfg["optimizer"]["lr"], args.dataset)
    runner.set_seed(args.seed, cfg["deterministic"])
    log = runner.make_logger(args.out)
    with open(os.path.join(args.out, "config_used.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "dataset": args.dataset, "seed": args.seed, "lr_used": lr,
                        "subset_train": args.subset_train, "subset_val": args.subset_val}, f, sort_keys=False)

    device = torch.device("cuda")
    train_loader, val_loader = build_loaders(cfg, args.dataset, args.seed, args.subset_train, args.subset_val)
    val_gt = val_loader.dataset.masks
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                 weight_decay=cfg["optimizer"]["weight_decay"])
    sc = cfg["scheduler"]
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=sc["T_0"], T_mult=sc["T_mult"], eta_min=sc["eta_min"])
    log.info(f"U-Net | {args.dataset} | seed {args.seed} | train {len(train_loader.dataset)} | "
             f"val {len(val_loader.dataset)} | batch {cfg['batch_size']} | lr {lr} | "
             f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    elog = runner.EpochLog(args.out, ["epoch", "lr", "train_loss", "val_dice", "best_val_dice", "best_epoch",
                                      "train_seconds", "val_seconds"])
    best_dice, best_epoch = 0.0, 1
    for epoch in range(1, cfg["max_epochs"] + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        with runner.Timer() as t_train:
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        with runner.Timer() as t_val:
            _, probs = predict(model, val_loader, device)
            val_dice = float(metrics.dice_many(probs, val_gt).mean())
        scheduler.step()  # LViT: once per epoch, after validation

        # LViT best-model rule: improvement counts only from the 6th epoch on (epoch + 1 > 5 in 0-based code)
        if val_dice > best_dice and epoch >= cfg["first_epoch_for_best"]:
            best_dice, best_epoch = val_dice, epoch
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_dice": val_dice},
                       os.path.join(args.out, "best_model.pt"))
        elog.write(epoch=epoch, lr=lr_now, train_loss=train_loss, val_dice=val_dice, best_val_dice=best_dice,
                   best_epoch=best_epoch, train_seconds=t_train.seconds, val_seconds=t_val.seconds)
        log.info(f"epoch {epoch:4d} | lr {lr_now:.2e} | loss {train_loss:.4f} | val Dice {val_dice:.4f} | "
                 f"best {best_dice:.4f} @ {best_epoch} | {t_train.seconds:.0f}+{t_val.seconds:.0f} s")
        # LViT early stopping: stop when epoch - best_epoch > patience
        if epoch - best_epoch > cfg["early_stopping_patience"]:
            log.info(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    runner.save_json({"best_epoch": best_epoch, "best_val_dice": best_dice, "last_epoch": epoch,
                      "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9},
                     os.path.join(args.out, "training_summary.json"))
    if args.no_final_eval or not os.path.exists(os.path.join(args.out, "best_model.pt")):
        return

    # Final evaluation of the best (validation-selected) checkpoint with the common evaluator
    model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pt"), map_location=device)["state_dict"])
    for split in ["val", "test"]:
        ids, img, msk = data.load_split_arrays(args.dataset, split)
        loader = DataLoader(UNetDataset(ids, img, msk, train=False), batch_size=cfg["val_batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"])
        ids, probs = predict(model, loader, device)
        runner.final_evaluation(args.out, args.dataset, split, ids, probs, log)


if __name__ == "__main__":
    main()
