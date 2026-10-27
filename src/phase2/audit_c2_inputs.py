"""Tokenizer-only effective-input audit of C2 (P2-1) on the validation split; no model is run.

For every eligible C2 row (src/phase2/conditions.py, the original with "pulmonary" -> "lung" and
"infected area(s)" -> "infected region(s)"):
  word check     C2 equals C0 except for the two substitutions (conditions.c2_only_substitutions)
  kept content   per method, the number of the report's words (letter runs) whose tokens all lie inside the
                 method's window: LViT-T 23 BERT word pieces (after the official EOF XXX padding, which comes
                 after the report), RecLMIS 16 CLIP BPE tokens (padded text), DD-CMD 22 BiomedBERT word pieces.
                 C0 and C2 have the same words in the same order, so "C2 cuts more report content than C0" =
                 fewer words kept under C2, and the reverse.
  LViT-T window  the 10 words of LViT's text path (word pieces merged, as lvit_text.word_vectors forms them):
                 they must differ from C0 only at the positions of the substituted words.
Also: for C0 alone, how many RecLMIS truncations cut report content and not only the EOF XXX padding.

Usage:  python src/phase2/audit_c2_inputs.py   -> results/phase2_c2_input_audit_val.csv (numbers only)
"""
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase1 import common, lvit_text  # noqa: E402
from phase2 import conditions as C  # noqa: E402

LIMITS = {"lvit": lvit_text.MAX_SEQ_LENGTH - 2, "reclmis": 18 - 2, "ddcmd": 24 - 2}


def tokenizers():
    from clip.simple_tokenizer import SimpleTokenizer
    from transformers import AutoTokenizer, BertTokenizer
    bert = BertTokenizer.from_pretrained(lvit_text.HF_MODEL, do_lower_case=True)
    bio = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    bpe = SimpleTokenizer()
    return {"lvit": lambda t: bert.tokenize(t), "reclmis": lambda t: bpe.encode(t),
            "ddcmd": lambda t: bio.tokenize(t)}, bert


def kept_words(text, method, tok):
    """Number of the report's letter-run words whose tokens all lie inside the method's window."""
    ends = [m.end() for m in re.finditer(r"[A-Za-z]+", text)]
    return sum(len(tok[method](text[:e])) <= LIMITS[method] for e in ends)


def lvit_window(text, bert):
    """The first 10 words of LViT's text path (same loop as lvit_text.word_vectors)."""
    enc = bert(common.lvit_pad(text).split("\n")[0], max_length=lvit_text.MAX_SEQ_LENGTH, truncation=True,
               padding="max_length")
    words = []
    for tid in enc["input_ids"]:
        if tid == bert.pad_token_id:
            break
        if tid in (bert.cls_token_id, bert.sep_token_id):
            continue
        piece = bert.convert_ids_to_tokens(tid)
        if piece.startswith("##") and words:
            words[-1] += piece[2:]
        else:
            words.append(piece)
    return words[:lvit_text.N_WORDS]


def main():
    tok, bert = tokenizers()
    rows, notes = [], []
    for ds in ("QaTa-COV19", "MosMedData+"):
        man, _ = C.build(ds, "val")
        c0 = man[man.condition == "C0"].set_index("sample_id").counterfactual_text
        c2 = man[(man.condition == "C2") & man.eligible].set_index("sample_id")
        n_word_fail = n_one = 0
        win_only = win_other = 0
        for sid, r in c2.iterrows():
            ok, diff = C.c2_only_substitutions(c0[sid], r.counterfactual_text)
            n_word_fail += not ok
            n_one += r.residual_cue.startswith("only one")
            w0, w2 = lvit_window(c0[sid], bert), lvit_window(r.counterfactual_text, bert)
            pos = [i for i, (a, b) in enumerate(zip(w0, w2)) if a != b]
            if len(w0) == len(w2) and all((w0[i], w2[i]) in C.C2_ALLOWED for i in pos):
                win_only += 1
            else:
                win_other += 1
                notes.append((ds, sid, " ".join(w0), " ".join(w2)))
        for m in ("lvit", "reclmis", "ddcmd"):
            more = less = 0
            for sid, r in c2.iterrows():
                k0, k2 = kept_words(c0[sid], m, tok), kept_words(r.counterfactual_text, m, tok)
                more += k2 < k0
                less += k2 > k0
            row = {"dataset": ds, "method": m, "c2_rows": len(c2), "c2_word_check_failures": n_word_fail,
                   "c2_rows_with_one_substitution": n_one, "c2_cuts_more_report_words_than_c0": more,
                   "c2_cuts_fewer_report_words_than_c0": less}
            if m == "lvit":
                row.update(lvit_window_differs_only_at_substituted_words=win_only, lvit_window_other=win_other)
            if m == "reclmis":                                 # C0 alone: padding-only vs report-content cuts
                padded = [len(tok[m](common.lvit_pad(t))) > LIMITS[m] for t in c0]
                content = [len(tok[m](t)) > LIMITS[m] for t in c0]
                row.update(c0_truncated_any=sum(padded), c0_truncated_report_content=sum(content),
                           c0_truncated_padding_only=sum(p and not c for p, c in zip(padded, content)))
            rows.append(row)
    df = pd.DataFrame(rows)
    out = os.path.join(REPO, "results", "phase2_c2_input_audit_val.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(df.to_string(index=False))
    for n in notes:
        print("LViT window differs elsewhere:", n[:2])
    print("written:", os.path.relpath(out, REPO))


if __name__ == "__main__":
    main()
