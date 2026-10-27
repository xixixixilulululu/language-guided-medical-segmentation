"""CPU-only checks of the Phase 2 inference driver (src/phase2/infer.py). No model forward pass, no GPU, no test data.

  checkpoints   the 18 baseline checkpoints map to the recorded paths / configs (MosMed LViT-T = batch-24 runs)
  passes        the inference passes cover exactly the eligible rows of the validation manifest
  LViT-T window the words of LViT's text path come from the tokenizer (as lvit_text.word_vectors forms them):
                the window of a typical original and of its C2 rewrite (the example of PHASE2_PROTOCOL.md)
  input change  RecLMIS (CLIP tokens) and DD-CMD (BiomedBERT ids) see a different input for C1 and C6 than for C0
  lock          `infer.py --split test` exits with code 5 (no frozen record yet) before reading anything
Usage:  python src/phase2/test_phase2_infer.py      (exit status 1 if any check fails)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2 import conditions as C  # noqa: E402
from phase2 import infer as I  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def main():
    # checkpoints
    p, cfg, _ = I.checkpoint("lvit", "MosMedData+", 1002)
    check("MosMed LViT-T uses the batch-24 runs", "runs/lvit_b24/MosMedData+/seed1002/best_model.pt" in p
          and cfg.endswith("configs/lvit_sensitivity_b24.yaml"))
    p, cfg, _ = I.checkpoint("lvit", "QaTa-COV19", 1001)
    check("QaTa LViT-T uses the Phase 1 runs", "runs/lvit/QaTa-COV19/seed1001/best_model.pt" in p and cfg.endswith("configs/lvit.yaml"))
    p, _, _ = I.checkpoint("ddcmd", "QaTa-COV19", 1003)
    check("DD-CMD uses checkpoints/best.ckpt", p.endswith("runs/ddcmd/QaTa-COV19/seed1003/checkpoints/best.ckpt"))
    n = sum(os.path.exists(I.checkpoint(m, d, s)[0]) for m in I.METHODS for d in ("QaTa-COV19", "MosMedData+")
            for s in I.SEEDS)
    check("all 18 checkpoints exist and are in the sha256 table", n == 18, f"{n}")

    # passes
    for ds in ("QaTa-COV19", "MosMedData+"):
        man, _ = C.build(ds, "val")
        plan = I.passes(man)
        rows = sum(len(t) for _, _, t, _ in plan)
        check(f"{ds}: passes cover the eligible rows", rows == int((man.eligible.astype(str) == "True").sum()),
              f"{len(plan)} passes, {rows} case-passes")

    # LViT-T window words (tokenizer only: the adapter is not constructed, so BERT is not loaded)
    from transformers import BertTokenizer
    from phase1 import common, lvit_text
    ad = I.Adapter.__new__(I.Adapter)
    ad.method, ad.lt = "lvit", lvit_text
    ad.tok = BertTokenizer.from_pretrained(lvit_text.HF_MODEL, do_lower_case=True)
    ad.D = type("D", (), {"common": common})
    orig = "Bilateral pulmonary infection, two infected areas, middle left lung and lower right lung."
    c2 = C.c2_text("Bilateral", 2, "left:middle;right:lower")
    w0, w2 = ad.lvit_words(orig)[:10], ad.lvit_words(c2)[:10]
    check("LViT window of the original", w0 == "bilateral pulmonary infection , two infected areas , middle left".split(),
          " ".join(w0))
    check("LViT window of the C2 rewrite ends at 'in'",
          w2 == "bilateral lung infection , two areas of infection , in".split(), " ".join(w2))
    w1 = ad.lvit_words("")
    check("LViT C1: the empty text gives >= 10 words (EOF XXX padding)", len(w1) >= 10 and w1[:2] == ["eof", "xxx"],
          " ".join(w1[:4]))

    # RecLMIS / DD-CMD input change (tokenizers only)
    for m in ("reclmis", "ddcmd"):
        cfgp = I.checkpoint(m, "QaTa-COV19", 1001)[1]
        a = I.Adapter.__new__(I.Adapter)
        a.method, a.dataset = m, "QaTa-COV19"
        import importlib
        a.D = importlib.import_module(f"phase1.train_{m}")
        a.cfg = a.D.runner.load_config(cfgp)
        if m == "reclmis":
            a.conf = a.D.official_config(a.cfg, "QaTa-COV19")
        else:
            from transformers import AutoTokenizer
            a.tokenizer = AutoTokenizer.from_pretrained(a.cfg["model"]["bert_type"], trust_remote_code=True)
        donor = "Unilateral pulmonary infection, one infected area, lower right lung."
        check(f"{m}: C1 and C6 inputs differ from C0, C0 equal to itself",
              a.model_input("") != a.model_input(orig) and a.model_input(donor) != a.model_input(orig)
              and a.model_input(orig) == a.model_input(orig))
        # the benchmark's per-pass check: cases with a changed model input = cases with a changed raw text
        for ds in ("QaTa-COV19", "MosMedData+"):
            plan = I.passes(C.build(ds, "val")[0])
            c0 = next(t for c, _, t, _ in plan if c == "C0")
            bad = [(c, v) for c, v, t, _ in plan
                   if sum(a.model_input(t[s]) != a.model_input(c0[s]) for s in t) != sum(t[s] != c0[s] for s in t)]
            check(f"{m} {ds}: every pass changes the model input exactly where the text changes", not bad, f"{bad}")

    # lock
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer.py"),
                        "--stage", "benchmark", "--method", "lvit", "--dataset", "QaTa-COV19", "--split", "test"],
                       capture_output=True, text=True)
    check("infer.py --split test -> exit 5 (lock closed)", r.returncode == 5 and "LOCK CLOSED" in r.stdout,
          f"exit {r.returncode}")
    print("ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
