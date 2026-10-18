"""LViT-T text features without MXNet: a PyTorch re-implementation of `bert_embedding.BertEmbedding()`.

LViT (Load_Dataset.py:156-189) calls `BertEmbedding()` with its defaults on every text and keeps the first
10 word vectors (10 x 768). bert-embedding 1.0.1 (bert.py) with defaults does:
    model 'bert_12_768_12', dataset_name 'book_corpus_wiki_en_uncased'  (Google's BERT-base, uncased)
    BERTTokenizer(lower=True) + BERTSentenceTransform(max_seq_length=25): [CLS] tokens[:23] [SEP] + padding
    sequence output of the last encoder layer (inference mode: no dropout)
    oov(): skip [CLS]/[SEP], stop at [PAD], merge word pieces ('##...') into the previous word by
    averaging their vectors (oov_way='avg').
bert_embedding pins mxnet 1.4.0 / numpy 1.14.6, which do not exist for aarch64 / Python 3.11, and mxnet
1.9.1 needs the ARM Performance Libraries that the cluster lacks, so the official output could not be
computed here. This module applies the same steps to the same weights in PyTorch
(Hugging Face `google-bert/bert-base-uncased`, also a conversion of Google's uncased BERT-base checkpoint).
Expected differences: floating-point rounding between frameworks only.

The vectors depend only on the text, so they are computed once for every distinct (padded) text of both
datasets and stored as a cache; the LViT data loader reads the cache instead of running BERT per sample.

Usage (CPU, about a minute):
    python src/phase1/lvit_text.py --out ~/datasets/phase1/lvit_bert_cache.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import common  # noqa: E402

HF_MODEL = "google-bert/bert-base-uncased"
MAX_SEQ_LENGTH = 25          # bert_embedding default
N_WORDS = 10                 # LViT keeps text[:10] (Load_Dataset.py:188-189); nets/Vit.py needs exactly 10
DEFAULT_CACHE = os.path.expanduser("~/datasets/phase1/lvit_bert_cache.pt")


def load_bert():
    from transformers import BertModel, BertTokenizer
    tok = BertTokenizer.from_pretrained(HF_MODEL, do_lower_case=True)        # basic + word-piece, as GluonNLP
    model = BertModel.from_pretrained(HF_MODEL, add_pooling_layer=False, attn_implementation="eager").eval()
    return tok, model


@torch.no_grad()
def word_vectors(text, tok, model):
    """(n_words, 768) float32: bert_embedding output for `text` (first line only, as LViT uses)."""
    sentence = text.split("\n")[0]                                           # LViT: text.split('\n') -> [0]
    enc = tok(sentence, max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                token_type_ids=torch.zeros_like(enc["input_ids"])).last_hidden_state[0].numpy()
    ids = enc["input_ids"][0].tolist()
    words, pieces = [], []                                                   # pieces = #vectors merged
    for token_id, vec in zip(ids, out):
        if token_id == tok.pad_token_id:                                     # [PAD]: sentence finished
            break
        if token_id in (tok.cls_token_id, tok.sep_token_id):                 # [CLS], [SEP] are skipped
            continue
        if tok.convert_ids_to_tokens(token_id).startswith("##") and words:   # word piece -> previous word
            words[-1] = words[-1] + vec
            pieces[-1] += 1
        else:
            words.append(vec.copy())                                         # float32 sums, as bert_embedding
            pieces.append(1)
    return np.stack([w / n for w, n in zip(words, pieces)]).astype(np.float32)


def all_padded_texts():
    """Every distinct padded text of both datasets and all splits."""
    texts = set()
    for ds in common.data.DATASETS:
        for split in common.data.SPLITS:
            texts.update(common.lvit_pad(t) for t in common.read_texts(ds, split).values())
    return sorted(texts)


def build_cache(path):
    tok, model = load_bert()
    cache = {}
    for t in all_padded_texts():
        v = word_vectors(t, tok, model)
        if v.shape[0] < N_WORDS:
            raise ValueError(f"fewer than {N_WORDS} word vectors ({v.shape[0]}) for: {t!r}")
        cache[t] = torch.from_numpy(v[:N_WORDS].copy())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": HF_MODEL, "max_seq_length": MAX_SEQ_LENGTH, "n_words": N_WORDS, "vectors": cache}, path)
    return cache


def load_cache(path=DEFAULT_CACHE):
    return torch.load(path, map_location="cpu", weights_only=True)["vectors"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_CACHE)
    args = ap.parse_args()
    c = build_cache(args.out)
    shapes = {tuple(v.shape) for v in c.values()}
    print(f"{len(c)} distinct padded texts -> {args.out}; shapes {shapes}")
