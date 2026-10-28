"""Phase 2 inference driver (Task 2.3): frozen Phase 1 / sensitivity checkpoints under the nine text conditions.

One process = one method x one dataset x one split (the three official repositories have clashing module
names such as `utils` and `nets`, so the Phase 1 driver of exactly one method is imported). The Phase 1 drivers
(src/phase1/train_*.py) are imported unchanged: their dataset classes, build_model and predict are used, with
the condition texts in place of the original ones. Images, preprocessing, weights, eval() mode, inference batch
size, threshold and evaluator are those of Phase 1; nothing is adapted at inference.

Stages
  textpath   (validation only; the first Phase 2 job, PHASE2_PLAN Task 2.3 "First")
             per checkpoint: C0 on the full validation split and its reconciliation with the archived Phase 1
             validation evaluation (per-case metrics, scored masks); a repeated C0 pass as the numerical-noise
             reference; on a fixed subset (SUBSET_N C6-eligible cases, smallest hash) C1 and C6 (first donor):
             change of model input, probability map, mask and Dice. Plus the effective-input audit of every
             validation condition (text level, the same for all seeds): raw-text change, change of the actual
             model input, and for LViT-T the words actually inside its 10-word window (from the tokenizer).
  benchmark  every condition and donor variant on its eligible cases, all three seeds (resumable: a finished
             condition pass is skipped). The C0 pass of the textpath stage is reused.
Test split: refused (exit code 5) before anything is read unless the Phase 2 lock is open (src/phase2/lock.py);
every test pass is logged in results/phase2_test_eval_log.tsv.

Outputs: runs/phase2/<split>/<method>/<dataset>/seed<k>/
  per_case_<cond>_<variant>.csv   dataset, split, method, config, checkpoint_sha256, seed, sample_id, condition,
                                  variant_id, donor_sample_id, dice, miou, hd95, assd, empty (Dice in [0, 1])
  masks_<cond>_<variant>.npz      scored binary masks, packed bits (np.packbits), key = sample id
  probs_<cond>_<variant>.npz      float32 probabilities, compressed; validation: C0, C1, C4-C6 (D13);
                                  test: every condition and donor (amendment A1)
  textpath/                       reconcile_C0.csv, repeat_C0.csv, subset_changes.csv, summary.json
runs/phase2/<split>/effective_input_<method>_<dataset>.csv

Usage (inside a Slurm job):
    python src/phase2/infer.py --stage textpath --method lvit --dataset QaTa-COV19
    python src/phase2/infer.py --stage benchmark --method lvit --dataset QaTa-COV19 --split val
"""
import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase2 import lock  # noqa: E402  (no data is read by this import)

METHODS = ["lvit", "reclmis", "ddcmd"]
SEEDS = [1001, 1002, 1003]
SUBSET_N = 64                                     # fixed text-path subset size per dataset
SUBSET_SALT = "phase2-textpath-subset-v1"
PROB_CONDITIONS = {"C0", "C1", "C4", "C5", "C6"}  # D13: float32 probabilities kept for these validation conditions
# Amendment A1 (2026-10-28, PHASE2_PROTOCOL.md section 9): on the TEST split the float32 probabilities of EVERY
# condition and donor are kept, because the Phase 3 gates need P_lang of all conditions (Task 3.2).
CKPT_TABLE = os.path.join(REPO, "results", "phase2_checkpoint_sha256.tsv")
OUT_ROOT = os.path.join(REPO, "runs", "phase2")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint(method, dataset, seed):
    """(checkpoint path, config path, recorded sha256) of the Phase 2 baseline (PHASE2_PROTOCOL.md section 1)."""
    if method == "lvit" and dataset == "MosMedData+":            # post-hoc batch-24 sensitivity runs
        run, cfg = os.path.join("runs", "lvit_b24", dataset, f"seed{seed}"), "configs/lvit_sensitivity_b24.yaml"
        key = "lvit_b24"
    else:
        run, cfg, key = os.path.join("runs", method, dataset, f"seed{seed}"), f"configs/{method}.yaml", method
    path = os.path.join(run, "checkpoints", "best.ckpt") if method == "ddcmd" else os.path.join(run, "best_model.pt")
    table = pd.read_csv(CKPT_TABLE, sep="\t", dtype=str)
    rec = table[(table.method == key) & (table.dataset == dataset) & (table.seed == str(seed))]
    if len(rec) != 1:
        raise KeyError(f"{key} {dataset} {seed} not in {CKPT_TABLE}")
    return os.path.join(REPO, path), os.path.join(REPO, cfg), rec.sha256.iloc[0]


# ---------------------------------------------------------------------------------------------
# Method adapters: the Phase 1 driver of one method, used unchanged
# ---------------------------------------------------------------------------------------------
class Adapter:
    """Loads one method's Phase 1 driver; builds datasets from {sample_id: text}; predicts; describes inputs."""

    def __init__(self, method, dataset, config_path):
        import torch
        self.torch, self.method, self.dataset = torch, method, dataset
        self.D = importlib.import_module(f"phase1.train_{method}")      # frozen Phase 1 driver
        self.cfg = self.D.runner.load_config(config_path)
        self.device = torch.device("cuda")
        self.batch = self.cfg["val_batch_size_eval"] if method == "ddcmd" else self.cfg["val_batch_size"]
        if method == "lvit":
            from phase1 import lvit_text
            self.lt = lvit_text
            self.tok, self.bert = lvit_text.load_bert()
            self.vectors = {}                                           # padded text -> (10, 768)
        elif method == "reclmis":
            self.conf = self.D.official_config(self.cfg, dataset)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model"]["bert_type"], trust_remote_code=True)

    # --- model ---
    def load(self, path):
        torch = self.torch
        if self.method == "lvit":
            model = self.D.build_model()
            model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        elif self.method == "reclmis":
            model = self.D.build_model(self.conf)
            model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        else:
            model = self.D.DDCMDModule(self.D.module_args(self.cfg))
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
        return model.to(self.device).eval()

    # --- text -> model input ---
    def lvit_words(self, text):
        """The words of LViT's text path exactly as lvit_text.word_vectors forms them (word pieces merged)."""
        sentence = self.D.common.lvit_pad(text).split("\n")[0]
        enc = self.tok(sentence, max_length=self.lt.MAX_SEQ_LENGTH, truncation=True, padding="max_length")
        words = []
        for tid in enc["input_ids"]:
            if tid == self.tok.pad_token_id:
                break
            if tid in (self.tok.cls_token_id, self.tok.sep_token_id):
                continue
            piece = self.tok.convert_ids_to_tokens(tid)
            if piece.startswith("##") and words:
                words[-1] += piece[2:]
            else:
                words.append(piece)
        return words

    def prepare(self, text_list):
        """Compute what the method needs for these texts (LViT-T: BERT word vectors)."""
        if self.method != "lvit":
            return
        for t in set(text_list):
            p = self.D.common.lvit_pad(t)
            if p not in self.vectors:
                v = self.lt.word_vectors(p, self.tok, self.bert)
                if v.shape[0] < self.lt.N_WORDS:
                    raise ValueError(f"fewer than {self.lt.N_WORDS} word vectors for {p!r}")
                self.vectors[p] = self.torch.from_numpy(v[:self.lt.N_WORDS].copy())

    def model_input(self, text):
        """Hashable description of the actual model input of one text (for the effective-input audit)."""
        if self.method == "lvit":
            v = self.vectors[self.D.common.lvit_pad(text)]
            return hashlib.sha256(v.numpy().tobytes()).hexdigest()
        if self.method == "reclmis":
            tok = self.D.clip.tokenize(self.D.common.lvit_pad(text).split("\n"), context_length=self.conf.token_len,
                                       truncate=True)
            return tuple(tok.squeeze().tolist())
        enc = self.tokenizer(text, padding="max_length", max_length=self.cfg["input_text_len"], truncation=True)
        return tuple(enc["input_ids"]) + tuple(enc["attention_mask"])

    def truncated(self, text):
        """Whether the method's tokenizer cuts this text."""
        if self.method == "lvit":
            return len(self.tok.tokenize(self.D.common.lvit_pad(text).split("\n")[0])) > self.lt.MAX_SEQ_LENGTH - 2
        if self.method == "reclmis":
            if not hasattr(self, "bpe"):                                    # build the BPE tokenizer once
                from clip.simple_tokenizer import SimpleTokenizer
                self.bpe = SimpleTokenizer()
            return len(self.bpe.encode(self.D.common.lvit_pad(text))) > self.conf.token_len - 2
        return len(self.tokenizer.tokenize(text)) > self.cfg["input_text_len"] - 2

    # --- inference ---
    def predict(self, model, ids, images, masks, texts):
        """(ids, probs float32 (N,224,224), targets) in the order of `ids`."""
        from torch.utils.data import DataLoader
        D = self.D
        if self.method == "lvit":
            ds = D.LViTDataset(ids, images, masks, texts, self.vectors, False)
        elif self.method == "reclmis":
            ds = D.RecLMISDataset(ids, images, masks, texts, self.conf.token_len, False)
        else:
            ds = D.DDCMDDataset(ids, images, masks, texts, self.tokenizer, self.cfg, False)
        loader = DataLoader(ds, batch_size=self.batch, shuffle=False, num_workers=self.cfg["num_workers"])
        if self.method == "ddcmd":
            probs, targets = D.predict(model, loader, self.device)
            out_ids = list(ds.ids)
        else:
            out_ids, probs, targets = D.predict(model, loader, self.device)
        if list(out_ids) != list(ids):
            raise RuntimeError("prediction order differs from the requested ids")
        return out_ids, probs.astype(np.float32), targets


# ---------------------------------------------------------------------------------------------
# Scoring and storage
# ---------------------------------------------------------------------------------------------
def common_evaluator():
    """src/evaluate.py, loaded by file path: external/DD-CMD has its own evaluate.py earlier on sys.path."""
    import importlib.util
    name = "lgms_common_evaluate"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "src", "evaluate.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod                                           # needed by the Pool workers (fork)
        spec.loader.exec_module(mod)
    return sys.modules[name]


def score(ids, probs, masks_by_id):
    """Per-case metrics with the common evaluator's case function (src/evaluate.py)."""
    ev = common_evaluator()
    with Pool(8) as pool:
        rows = pool.map(ev._score, [(i, p, masks_by_id[i]) for i, p in zip(ids, probs)], chunksize=32)
    df = pd.DataFrame(rows)
    df["empty"] = [not (p > 0.5).any() for p in probs]
    return df


def write_atomic_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, float_format="%.6f")
    os.replace(tmp, path)


PASS_LOG_COLUMNS = ["slurm_job_id", "time", "stage", "condition", "variant_id", "n_cases", "note"]


def log_pass(out_dir, stage, cond, variant, n, note=""):
    """One line per finished condition pass in <out>/passes.tsv, with the Slurm job id (the C0 pass of the
    text-path job is reused by the benchmark, so paired differences can mix passes of different jobs)."""
    path = os.path.join(out_dir, "passes.tsv")
    new = not os.path.exists(path)
    with open(path, "a") as f:
        w = csv.writer(f, delimiter="\t")
        if new:
            w.writerow(PASS_LOG_COLUMNS)
        w.writerow([os.environ.get("SLURM_JOB_ID", "none"), time.strftime("%Y-%m-%d %H:%M:%S"), stage, cond,
                    variant, n, note])


def save_pass(out_dir, cond, variant, meta, ids, probs, metrics_df, donors, keep_probs):
    """Per-case CSV last (it marks the pass as finished for resuming)."""
    np.savez_compressed(os.path.join(out_dir, f"masks_{cond}_{variant}.npz"),
                        **{i: np.packbits(p > 0.5) for i, p in zip(ids, probs)})
    if keep_probs:
        np.savez_compressed(os.path.join(out_dir, f"probs_{cond}_{variant}.npz"), **dict(zip(ids, probs)))
    df = metrics_df.copy()
    for k, v in reversed(list(meta.items())):
        df.insert(0, k, v)
    df.insert(df.columns.get_loc("sample_id") + 1, "condition", cond)
    df.insert(df.columns.get_loc("condition") + 1, "variant_id", variant)
    df.insert(df.columns.get_loc("variant_id") + 1, "donor_sample_id", [donors.get(i, "") for i in df.sample_id])
    write_atomic_csv(df, os.path.join(out_dir, f"per_case_{cond}_{variant}.csv"))
    return df


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------
def passes(manifest):
    """[(condition, variant, {sample_id: text}, {sample_id: donor})] of the eligible rows, in condition order."""
    from phase2 import conditions as C
    el = manifest[manifest.eligible.astype(str) == "True"]
    out = []
    for cond in C.CONDITIONS:
        for variant, g in el[el.condition == cond].groupby("variant_id", sort=True):
            out.append((cond, variant, dict(zip(g.sample_id, g.counterfactual_text)),
                        {s: d for s, d in zip(g.sample_id, g.donor_sample_id) if d}))
    return out


def effective_input_audit(ad, manifest, path):
    """Text-level audit of every eligible condition row (identical for all seeds)."""
    el = manifest[manifest.eligible.astype(str) == "True"]
    c0 = dict(zip(el[el.condition == "C0"].sample_id, el[el.condition == "C0"].counterfactual_text))
    ad.prepare(list(el.counterfactual_text) + list(c0.values()))
    rows, ref_in, ref_words = [], {}, {}
    for s, t in c0.items():
        ref_in[s] = ad.model_input(t)
        if ad.method == "lvit":
            ref_words[s] = ad.lvit_words(t)[:ad.lt.N_WORDS]
    for r in el.itertuples():
        t = r.counterfactual_text
        row = {"dataset": r.dataset, "split": r.split, "method": ad.method, "sample_id": r.sample_id,
               "condition": r.condition, "variant_id": r.variant_id, "raw_text_changed": t != c0[r.sample_id],
               "model_input_changed": ad.model_input(t) != ref_in[r.sample_id], "truncated": ad.truncated(t)}
        if ad.method == "lvit":
            w = ad.lvit_words(t)[:ad.lt.N_WORDS]
            row.update(lvit_window=" ".join(w), lvit_window_changed=w != ref_words[r.sample_id])
        rows.append(row)
    df = pd.DataFrame(rows)
    write_atomic_csv(df, path)
    return df


def lvit_phase1_check(ad, c0_texts):
    """C0 text features of the Phase 2 path equal the Phase 1 cache (looked up by key only)."""
    cache = ad.lt.load_cache(os.path.expanduser(ad.cfg["text_cache"]))
    diffs = [float((ad.vectors[ad.D.common.lvit_pad(t)] - cache[ad.D.common.lvit_pad(t)]).abs().max())
             for t in set(c0_texts)]
    return max(diffs)


def run(args):
    from lgms import data
    from phase2 import conditions as C
    info = {}
    if args.split == "test":                                     # before anything is read
        info = lock.require_open(fetch_required=False, what="Phase 2 test inference")
    t0 = time.time()
    manifest, _ = C.build(args.dataset, args.split)
    ids_all, images, masks = data.load_split_arrays(args.dataset, args.split)
    index = {s: k for k, s in enumerate(ids_all)}
    masks_by_id = dict(zip(ids_all, masks))
    _, cfg_path, _ = checkpoint(args.method, args.dataset, SEEDS[0])
    ad = Adapter(args.method, args.dataset, cfg_path)
    split_dir = os.path.join(OUT_ROOT, args.split)
    os.makedirs(split_dir, exist_ok=True)
    plan = passes(manifest)
    # effective-input audit of the current manifest (both stages; the benchmark checks every pass against it)
    audit = effective_input_audit(ad, manifest, os.path.join(
        split_dir, f"effective_input_{args.method}_{args.dataset}.csv"))
    ad.prepare([t for _, _, texts, _ in plan for t in texts.values()])
    c0_texts = next(texts for cond, _, texts, _ in plan if cond == "C0")
    f1 = lvit_phase1_check(ad, c0_texts.values()) if args.method == "lvit" else None
    c6_ids = sorted({s for cond, v, texts, _ in plan if cond == "C6" for s in texts},
                    key=lambda s: hashlib.sha256(f"{SUBSET_SALT}|{args.dataset}|{s}".encode()).hexdigest())
    subset = c6_ids[:SUBSET_N]

    for seed in args.seeds:
        ckpt, cfg_path, rec_sha = checkpoint(args.method, args.dataset, seed)
        got = sha_file(ckpt)
        if got != rec_sha:
            raise RuntimeError(f"{ckpt}: sha256 {got} differs from {CKPT_TABLE}")
        out = os.path.join(split_dir, args.method, args.dataset, f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        if args.stage == "textpath" and os.path.exists(os.path.join(out, "textpath", "summary.json")):
            print(f"seed {seed}: text-path check done earlier, skipped", flush=True)
            continue
        meta = {"dataset": args.dataset, "split": args.split, "method": args.method,
                "config": os.path.relpath(cfg_path, REPO), "checkpoint_sha256": rec_sha, "seed": seed}
        model = ad.load(ckpt)

        def infer(texts):
            ids = [s for s in ids_all if s in texts]
            sel = [index[s] for s in ids]
            ad.D.runner.set_seed(seed, ad.cfg["deterministic"])            # same RNG state for every pass
            return ad.predict(model, ids, images[sel], masks[sel], texts)

        todo = plan if args.stage == "benchmark" else [p for p in plan if p[0] == "C0"]
        for cond, variant, texts, donors in todo:
            done = os.path.join(out, f"per_case_{cond}_{variant}.csv")
            if os.path.exists(done):
                print(f"seed {seed} {cond} {variant}: done earlier, skipped", flush=True)
                continue
            if args.split == "test":
                lock.log_test_run("start", info, "infer.py", args.dataset, args.method, seed, f"{cond}/{variant}")
            # model input differs from C0 for how many cases? must equal the audit and the raw-text changes
            n_input = sum(ad.model_input(texts[s]) != ad.model_input(c0_texts[s]) for s in texts)
            n_raw = sum(texts[s] != c0_texts[s] for s in texts)
            a = audit[(audit.condition == cond) & (audit.variant_id.astype(str) == variant)]
            n_audit = int(a.model_input_changed.sum())
            if not (n_input == n_audit == n_raw) or len(a) != len(texts):
                raise RuntimeError(f"{cond} {variant}: model input differs from C0 in {n_input} cases, audit "
                                   f"{n_audit} of {len(a)} rows, raw text changed in {n_raw} of {len(texts)}")
            ids, probs, targets = infer(texts)
            if not all((targets[k] == masks_by_id[s]).all() for k, s in enumerate(ids)):
                raise RuntimeError("targets differ from the common masks (pairing error)")
            m = score(ids, probs, masks_by_id)
            save_pass(out, cond, variant, meta, ids, probs, m, donors,
                      args.split == "test" or (args.split == "val" and cond in PROB_CONDITIONS))
            log_pass(out, args.stage, cond, variant, len(ids), f"model_input_changed={n_input}/{len(ids)}")
            if args.split == "test":
                lock.log_test_run("end", info, "infer.py", args.dataset, args.method, seed, f"{cond}/{variant}")
            print(f"seed {seed} {cond} {variant}: {len(ids)} cases, mean Dice {m.dice.mean():.4f}, "
                  f"{time.time() - t0:.0f} s", flush=True)

        if args.stage == "textpath":
            textpath(ad, model, infer, out, seed, plan, subset, masks_by_id, f1,
                     audit[audit.method == args.method])
        del model
        ad.torch.cuda.empty_cache()
    print(f"finished in {time.time() - t0:.0f} s", flush=True)


def textpath(ad, model, infer, out, seed, plan, subset, masks_by_id, f1, audit):
    """Reconciliation with Phase 1, noise reference, and the change on the fixed subset (validation)."""
    import cv2
    tp = os.path.join(out, "textpath")
    os.makedirs(tp, exist_ok=True)
    c0 = pd.read_csv(os.path.join(out, "per_case_C0_0.csv"))
    c0_texts = next(t for cond, _, t, _ in plan if cond == "C0")
    p0 = np.load(os.path.join(out, "probs_C0_0.npz"))
    # 1. reconciliation with the archived Phase 1 validation evaluation (same checkpoint)
    run_dir = os.path.dirname(os.path.dirname(checkpoint(ad.method, ad.dataset, seed)[0])) if ad.method == "ddcmd" \
        else os.path.dirname(checkpoint(ad.method, ad.dataset, seed)[0])
    arch = pd.read_csv(os.path.join(run_dir, "eval_val", "per_case.csv")).set_index("sample_id")
    rec = []
    for r in c0.itertuples():
        old = cv2.imread(os.path.join(run_dir, "eval_val", "masks", f"{r.sample_id}.png"), cv2.IMREAD_GRAYSCALE) > 0
        new = p0[r.sample_id] > 0.5
        a = arch.loc[r.sample_id]
        rec.append({"sample_id": r.sample_id, "dice_phase2": r.dice, "dice_phase1": a.dice,
                    "abs_dice_diff": abs(r.dice - a.dice), "abs_hd95_diff": abs(r.hd95 - a.hd95),
                    "mask_pixels_differing": int((old != new).sum())})
    rec = pd.DataFrame(rec)
    write_atomic_csv(rec, os.path.join(tp, "reconcile_C0.csv"))
    # 2. repeated C0 pass: numerical-noise reference
    ids, probs, _ = infer(c0_texts)
    rep = pd.DataFrame({"sample_id": ids,
                        "max_abs_prob_diff": [float(np.abs(p - p0[i]).max()) for i, p in zip(ids, probs)],
                        "mask_pixels_differing": [int(((p > 0.5) != (p0[i] > 0.5)).sum()) for i, p in zip(ids, probs)]})
    m = score(ids, probs, masks_by_id).set_index("sample_id")
    rep["abs_dice_diff"] = [abs(m.dice[i] - d) for i, d in zip(c0.sample_id, c0.dice)]
    write_atomic_csv(rep, os.path.join(tp, "repeat_C0.csv"))
    # 3. fixed subset: C1 and the first C6 donor
    ch = []
    c0d = c0.set_index("sample_id").dice
    for cond, variant in (("C1", "0"), ("C6", "d1")):
        texts = next(t for c, v, t, _ in plan if c == cond and v == variant)
        texts = {s: texts[s] for s in subset if s in texts}
        ad.prepare(texts.values())
        ids, probs, _ = infer(texts)
        m = score(ids, probs, masks_by_id).set_index("sample_id")
        for i, p in zip(ids, probs):
            ch.append({"sample_id": i, "condition": cond, "variant_id": variant,
                       "raw_text_changed": texts[i] != c0_texts[i],
                       "model_input_changed": ad.model_input(texts[i]) != ad.model_input(c0_texts[i]),
                       "mean_abs_prob_diff": float(np.abs(p - p0[i]).mean()),
                       "max_abs_prob_diff": float(np.abs(p - p0[i]).max()),
                       "mask_pixels_changed": int(((p > 0.5) != (p0[i] > 0.5)).sum()),
                       "dice_C0": float(c0d[i]), "dice_condition": float(m.dice[i])})
    ch = pd.DataFrame(ch)
    write_atomic_csv(ch, os.path.join(tp, "subset_changes.csv"))
    # 4. summary (numbers only; interpretation in the report)
    summ = {"method": ad.method, "dataset": ad.dataset, "seed": seed, "n_val": int(len(c0)),
            "c0_mean_dice_phase2": float(c0.dice.mean()), "c0_mean_dice_phase1": float(rec.dice_phase1.mean()),
            "reconcile_max_abs_dice_diff": float(rec.abs_dice_diff.max()),
            "reconcile_cases_mask_differs": int((rec.mask_pixels_differing > 0).sum()),
            "reconcile_max_mask_pixels_differing": int(rec.mask_pixels_differing.max()),
            "repeat_max_abs_prob_diff": float(rep.max_abs_prob_diff.max()),
            "repeat_cases_mask_differs": int((rep.mask_pixels_differing > 0).sum()),
            "repeat_max_abs_dice_diff": float(rep.abs_dice_diff.max()),
            "lvit_c0_features_vs_phase1_cache_max_abs_diff": f1}
    for (cond, variant), g in ch.groupby(["condition", "variant_id"]):
        summ[f"subset_{cond}_{variant}"] = {
            "n": int(len(g)), "raw_text_changed": int(g.raw_text_changed.sum()),
            "model_input_changed": int(g.model_input_changed.sum()),
            "cases_mask_changed": int((g.mask_pixels_changed > 0).sum()),
            "mean_abs_prob_diff": float(g.mean_abs_prob_diff.mean()),
            "mean_dice_C0": float(g.dice_C0.mean()), "mean_dice_condition": float(g.dice_condition.mean())}
    a = audit.groupby("condition")
    summ["effective_input"] = {c: {"rows": int(len(g)), "raw_changed": int(g.raw_text_changed.sum()),
                                   "input_changed": int(g.model_input_changed.sum()),
                                   "truncated": int(g.truncated.sum()),
                                   **({"lvit_window_changed": int(g.lvit_window_changed.sum())}
                                      if "lvit_window_changed" in g else {})} for c, g in a}
    with open(os.path.join(tp, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps({k: v for k, v in summ.items() if not isinstance(v, dict)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["textpath", "benchmark"])
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dataset", required=True, choices=["QaTa-COV19", "MosMedData+"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    if args.split == "test":
        if args.stage == "textpath":
            sys.exit("the text-path stage is for the validation split only")
        lock.require_open(fetch_required=False, what="Phase 2 test inference")   # before any import of data
    run(args)


if __name__ == "__main__":
    main()
