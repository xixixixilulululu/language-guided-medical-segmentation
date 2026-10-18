"""Phase 1 quick-check probes for one method and dataset (Milestone 1.2). Writes one JSON file.

Two modes:
  --mode init     (untrained model)
    1. pairing: for the first 3 ids of every split, the image / mask / xlsx key / text the loader uses, and
       the check that every sample of every split has exactly one text (common.read_texts raises otherwise);
    2. forward pass on one augmented training batch: output shape and value range;
    3. text reaches the language branch:
       a. gradient of the SEGMENTATION loss with respect to the text features entering the network
          (LViT: the 10x768 BERT vectors; RecLMIS: the CLIP text features entering text_module4;
          DD-CMD: the BiomedBERT last hidden state); must be non-zero and finite;
       b. eval-mode output change when every text is replaced by one fixed text;
    4. RecLMIS only: the loss dictionary of one training step (CCL and both reconstruction losses present
       and finite) and model.aux still True.
  --mode trained  (after a subset training run, --ckpt): validation Dice of the trained model with the true
       texts and with every text replaced by one fixed text (the most frequent training text).
The test split is never used.

Usage:
    python src/phase1/quick_check.py --method lvit --dataset QaTa-COV19 --mode init --out runs/.../probe.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import common  # noqa: E402
from phase1.common import data, metrics  # noqa: E402


def fixed_text(dataset):
    """Most frequent training text: used to replace every text in the text-swap probes."""
    return collections.Counter(common.read_texts(dataset, "train").values()).most_common(1)[0][0]


def pairing_report(dataset):
    out = {}
    for split in data.SPLITS:
        ids = data.split_ids(dataset, split)
        texts = common.read_texts(dataset, split, ids)             # raises if any sample lacks a text
        out[split] = {"n_samples": len(ids), "n_texts": len(texts), "examples": [
            {"id": i, "image": os.path.relpath(data.image_path(dataset, split, i), data.common_root()),
             "mask": os.path.relpath(data.mask_path(dataset, split, i), data.common_root()),
             "xlsx_key": common.text_key(dataset, i), "text": texts[i]} for i in ids[:3]]}
    return out


# ---------------------------------------------------------------------------------------------
# Method adapters: build model + loaders, one supervised forward with a text-feature leaf, prediction
# ---------------------------------------------------------------------------------------------
class LViTAdapter:
    def __init__(self, dataset, device, n):
        from phase1 import lvit_text, train_lvit as T
        self.T, self.device = T, device
        self.cache = lvit_text.load_cache()
        self.train = T.make_dataset(dataset, "train", self.cache, True, n)
        self.val = T.make_dataset(dataset, "val", self.cache, False, n)
        self.model = T.build_model().to(device)
        from utils import WeightedDiceBCE
        self.criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
        self.fixed = self.cache[common.lvit_pad(fixed_text(dataset))]

    def seg_loss_and_text_grad(self, batch_size):
        images, masks, text, _ = next(iter(DataLoader(self.train, batch_size=batch_size, shuffle=False)))
        self.model.train()
        leaf = text.to(self.device).requires_grad_()
        preds = self.model(images.to(self.device), leaf)
        loss = self.criterion(preds, masks.to(self.device).float())
        loss.backward()
        return preds, loss, leaf.grad, {}

    def predict_val(self, override=False, ckpt=None):
        if ckpt:
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device)["state_dict"])
        loader = DataLoader(self.val, batch_size=8, shuffle=False)
        _, probs, _ = self.T.predict(self.model, loader, self.device, self.fixed if override else None)
        return probs


class RecLMISAdapter:
    def __init__(self, dataset, device, n):
        from phase1 import train_reclmis as T
        self.T, self.device = T, device
        cfg = common.runner.load_config(os.path.join(common.REPO, "configs", "reclmis.yaml"))
        self.conf = T.official_config(cfg, dataset)
        self.train = T.make_dataset(dataset, "train", self.conf.token_len, True, n)
        self.val = T.make_dataset(dataset, "val", self.conf.token_len, False, n)
        self.model = T.build_model(self.conf).to(device)
        self.criterion = T.WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
        import clip
        self.fixed = clip.tokenize([common.lvit_pad(fixed_text(dataset))], context_length=self.conf.token_len,
                                   truncate=True)

    def seg_loss_and_text_grad(self, batch_size):
        sample, _ = next(iter(DataLoader(self.train, batch_size=batch_size, shuffle=False)))
        self.model.train()
        store = {}

        def pre_hook(module, inputs):  # text features entering the adapter after the frozen CLIP text tower
            store["leaf"] = inputs[0].detach().requires_grad_()
            return (store["leaf"],)

        h = self.model.text_module4.register_forward_pre_hook(pre_hook)
        preds, masks, loss_dict = self.T.forward_batch(self.model, sample, self.device)
        h.remove()
        loss = self.criterion(preds, masks.float()).mean()   # Train_one_epoch.py takes v.mean() of every term
        loss.backward(retain_graph=True)
        losses = {k: float(v.mean().detach()) for k, v in loss_dict.items()}
        losses["loss_criterion"] = float(loss.detach())
        return preds, loss, store["leaf"].grad, {"loss_dict": losses, "aux_active": bool(self.model.aux),
                                                 "aux_keys_present": all(k in loss_dict for k in self.T.AUX_LOSSES)}

    def predict_val(self, override=False, ckpt=None):
        if ckpt:
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device)["state_dict"])
        loader = DataLoader(self.val, batch_size=8, shuffle=False)
        _, probs, _ = self.T.predict(self.model, loader, self.device, self.fixed if override else None)
        return probs


class DDCMDAdapter:
    def __init__(self, dataset, device, n):
        from phase1 import train_ddcmd as T
        from transformers import AutoTokenizer
        self.T, self.device = T, device
        self.cfg = common.runner.load_config(os.path.join(common.REPO, "configs", "ddcmd.yaml"))
        tok = AutoTokenizer.from_pretrained(self.cfg["model"]["bert_type"], trust_remote_code=True)
        self.train = T.make_dataset(dataset, "train", tok, self.cfg, True, n)
        self.val = T.make_dataset(dataset, "val", tok, self.cfg, False, n)
        self.module = T.DDCMDModule(T.module_args(self.cfg)).to(device)
        f = tok(fixed_text(dataset), padding="max_length", max_length=self.cfg["input_text_len"], truncation=True,
                return_tensors="pt")
        self.fixed = dict(f)

    def seg_loss_and_text_grad(self, batch_size):
        (image, text), gt = next(iter(DataLoader(self.train, batch_size=batch_size, shuffle=False)))
        self.module.train()
        store = {}

        def hook(module, inputs, output):  # BiomedBERT hidden states: the last one is used by the decoders
            feats = list(output["feature"])
            store["leaf"] = feats[-1].detach().requires_grad_()
            feats[-1] = store["leaf"]
            return {"feature": tuple(feats)}

        h = self.module.model.text_encoder.register_forward_hook(hook)
        text = {k: v.to(self.device) for k, v in text.items()}
        logits = self.module([image.to(self.device), text])
        h.remove()
        loss = self.module.loss_fn(logits, gt.to(self.device))
        loss.backward()
        return torch.sigmoid(logits), loss, store["leaf"].grad, {}

    def predict_val(self, override=False, ckpt=None):
        if ckpt:
            self.module.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
            self.module.to(self.device)
        loader = DataLoader(self.val, batch_size=8, shuffle=False)
        probs, _ = self.T.predict(self.module, loader, self.device, self.fixed if override else None)
        return probs


ADAPTERS = {"lvit": LViTAdapter, "reclmis": RecLMISAdapter, "ddcmd": DDCMDAdapter}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=ADAPTERS)
    ap.add_argument("--dataset", required=True, choices=data.DATASETS)
    ap.add_argument("--mode", default="init", choices=["init", "trained"])
    ap.add_argument("--ckpt", help="trained mode: best checkpoint of a subset run")
    ap.add_argument("--n", type=int, default=32, help="cases per split used by the probes")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    common.runner.set_seed(1001, True)
    device = torch.device(args.device)
    ad = ADAPTERS[args.method](args.dataset, device, args.n)
    res = {"method": args.method, "dataset": args.dataset, "mode": args.mode, "n_val_cases": len(ad.val),
           "fixed_text": fixed_text(args.dataset)}
    gt = ad.val.masks
    if args.mode == "init":
        res["pairing"] = pairing_report(args.dataset)
        preds, loss, grad, extra = ad.seg_loss_and_text_grad(args.batch_size)
        p = preds.detach()
        res["forward"] = {"output_shape": list(p.shape), "min": float(p.min()), "max": float(p.max()),
                          "seg_loss": float(loss.detach()), "finite": bool(torch.isfinite(p).all())}
        res["text_gradient"] = {"abs_mean": float(grad.abs().mean()), "nonzero_fraction": float((grad != 0).float().mean()),
                                "finite": bool(torch.isfinite(grad).all()), "shape": list(grad.shape)}
        res.update(extra)
        a, b = ad.predict_val(False), ad.predict_val(True)
        res["text_swap_init"] = {"mean_abs_prob_change": float(np.abs(a - b).mean()),
                                 "max_abs_prob_change": float(np.abs(a - b).max())}
        res["passed"] = bool(res["forward"]["finite"] and res["text_gradient"]["finite"]
                             and res["text_gradient"]["abs_mean"] > 0 and res["text_swap_init"]["max_abs_prob_change"] > 0
                             and (args.method != "reclmis" or (extra["aux_active"] and extra["aux_keys_present"]
                                  and all(np.isfinite(v) for v in extra["loss_dict"].values()))))
    else:
        a, b = ad.predict_val(False, args.ckpt), ad.predict_val(True)
        res["trained_val_dice_true_text"] = float(metrics.dice_many(a, gt).mean())
        res["trained_val_dice_fixed_text"] = float(metrics.dice_many(b, gt).mean())
        res["trained_mean_abs_prob_change"] = float(np.abs(a - b).mean())
        res["valid_masks"] = {"prob_min": float(a.min()), "prob_max": float(a.max()),
                              "empty_predictions": int((metrics.binarize(a).sum(axis=(1, 2)) == 0).sum())}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "pairing"}, indent=1, default=str))


if __name__ == "__main__":
    main()
