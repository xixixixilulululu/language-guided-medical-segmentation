"""Task 0.3 quick check, steps 1-4, for one method and dataset:
  1. load one training batch;  2. run one forward pass;  3. check output and mask shapes;
  4. compute the method's loss (and one backward pass).
Step 5 (short small-subset training) is run with the training scripts themselves (see slurm/quick_check_*.slurm).

Writes a JSON record to --out (used for reports/PHASE0_MODEL_SETUP.md).

Usage (inside a Slurm job):
    python src/quick_check.py --method unet --dataset QaTa-COV19 --out runs/quick_check/unet_QaTa-COV19.json
    python src/quick_check.py --method nnunet --dataset MosMedData+ --out ...   (needs nnUNet_* env variables)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from lgms import runner  # noqa: E402


def describe(t):
    """Shape / dtype / value range of a tensor, for the record."""
    t = t.detach()
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
            "min": float(t.min()), "max": float(t.max())}


def check_unet(dataset, device):
    import train_unet as T
    cfg = runner.load_config(os.path.join(REPO, "configs", "unet.yaml"))
    runner.set_seed(1001)
    loader, _ = T.build_loaders(cfg, dataset, 1001)
    images, masks, ids = next(iter(loader))                       # 1. one batch
    model = T.build_model(cfg).to(device)
    out = model(images.to(device))                                # 2. forward (sigmoid probabilities)
    ok_shape = out.shape[0] == masks.shape[0] and tuple(out.shape[2:]) == tuple(masks.shape[1:]) and out.shape[1] == 1
    loss = T.build_criterion(cfg)(out, masks.to(device).float())  # 4. loss
    loss.backward()
    return images, masks, out, ok_shape, loss, model


def check_transunet(dataset, device):
    import train_transunet as T
    cfg = runner.load_config(os.path.join(REPO, "configs", "transunet.yaml"))
    runner.set_seed(1001)
    loader, _ = T.build_loaders(cfg, dataset, 1001)
    images, masks, ids = next(iter(loader))
    model = T.build_model(cfg).to(device)                          # loads the ImageNet-21k weights
    out = model(images.to(device))                                # logits (B, 2, H, W)
    ok_shape = out.shape[0] == masks.shape[0] and tuple(out.shape[2:]) == tuple(masks.shape[1:]) and out.shape[1] == 2
    loss = T.Criterion(cfg)(out, masks.to(device))
    loss.backward()
    return images, masks, out, ok_shape, loss, model


def check_nnunet(dataset, device):
    """Build the nnUNetTrainerLGMS of fold 0 exactly as nnUNetv2_train does, take one batch of its own
    data loader (with its augmentation) and run the network and the deep-supervision loss once."""
    os.environ.setdefault("LGMS_SEED", "1001")
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    from prepare_nnunet import NNUNET_IDS
    name = NNUNET_IDS[dataset][1]
    plans = load_json(join(nnUNet_preprocessed, name, "nnUNetPlans.json"))
    plans["continue_training"] = False  # set by nnUNetv2_train (run_training.get_trainer_from_args)
    dataset_json = load_json(join(nnUNet_preprocessed, name, "dataset.json"))
    cls = recursive_find_trainer_class_by_name("nnUNetTrainerLGMS")
    tr = cls(plans=plans, configuration="2d", fold=0, dataset_json=dataset_json, device=device)
    tr.initialize()
    # the split must be the LViT one
    tr_keys, val_keys = tr.do_split()
    dl_train, dl_val = tr.get_dataloaders()
    batch = next(dl_train)
    images, target = batch["data"].to(device), batch["target"]
    target = [t.to(device) for t in target] if isinstance(target, list) else target.to(device)
    out = tr.network(images)
    out0 = out[0] if isinstance(out, (list, tuple)) else out
    t0 = target[0] if isinstance(target, list) else target
    ok_shape = out0.shape[0] == t0.shape[0] and tuple(out0.shape[2:]) == tuple(t0.shape[2:]) and out0.shape[1] == 2
    loss = tr.loss(out, target)
    loss.backward()
    extra = {"n_train_keys": len(tr_keys), "n_val_keys": len(val_keys),
             "patch_size": list(tr.configuration_manager.patch_size), "batch_size": tr.batch_size,
             "deep_supervision_outputs": [list(o.shape) for o in out] if isinstance(out, (list, tuple)) else None,
             "network": tr.configuration_manager.network_arch_class_name}
    for dl in (dl_train, dl_val):  # stop the augmentation worker processes
        if hasattr(dl, "_finish"):
            dl._finish()
    return images, t0[:, 0], out0, ok_shape, loss, tr.network, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["unet", "transunet", "nnunet"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")

    extra = {}
    if args.method == "unet":
        images, masks, out, ok_shape, loss, model = check_unet(args.dataset, device)
    elif args.method == "transunet":
        images, masks, out, ok_shape, loss, model = check_transunet(args.dataset, device)
    else:
        images, masks, out, ok_shape, loss, model, extra = check_nnunet(args.dataset, device)

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    rec = {
        "method": args.method, "dataset": args.dataset,
        "batch_image": describe(images), "batch_mask": describe(masks),
        "mask_values": sorted(np.unique(masks.cpu().numpy()).tolist()),
        "output": describe(out), "shapes_ok": bool(ok_shape),
        "loss": float(loss), "loss_finite": bool(torch.isfinite(loss)),
        "all_grads_finite": bool(all(g is None or torch.isfinite(g).all() for g in grads)),
        "n_params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, **extra,
    }
    rec["passed"] = rec["shapes_ok"] and rec["loss_finite"] and rec["all_grads_finite"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
