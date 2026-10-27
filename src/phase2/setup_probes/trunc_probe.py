"""Tokenizer-only truncation probe on train+val texts (no model forward). Phase 2 setup check; train/val texts only."""
import os, sys, re, collections
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src")
from phase1 import common
from transformers import BertTokenizer, AutoTokenizer
import clip
from clip.simple_tokenizer import SimpleTokenizer
bert = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", do_lower_case=True)
biomed = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
st = SimpleTokenizer()
def loc_start_char(t):
    m = re.search(r"areas?\s*,\s*", t); return m.end() if m else None
for ds in ["QaTa-COV19", "MosMedData+"]:
    texts = [t for sp in ["train", "val"] for t in common.read_texts(ds, sp).values()]
    stats = collections.Counter(); n = len(texts)
    for t in texts:
        p = common.lvit_pad(t)
        # LViT: BERT basic words of the padded first line within 23 word pieces; first 10 words kept
        words = bert.basic_tokenizer.tokenize(p.split("\n")[0])
        wp = bert.tokenize(p.split("\n")[0])
        stats["lvit_wordpieces>23"] += len(wp) > 23
        stats["lvit_words"] += len(words)
        ls = loc_start_char(t)
        if ls is not None:
            n_before = len(bert.basic_tokenizer.tokenize(t[:ls]))      # words before the location clause
            n_loc = len(bert.basic_tokenizer.tokenize(t[ls:]))
            kept = max(0, min(n_loc, 10 - n_before))
            stats["lvit_loc_words_total"] += n_loc; stats["lvit_loc_words_kept"] += kept
            stats["lvit_loc_fully_kept"] += kept == n_loc; stats["lvit_loc_none_kept"] += kept == 0
            stats["has_loc"] += 1
        # RecLMIS: CLIP BPE of padded text, 18 incl. SOT/EOT -> 16 content tokens
        bpe = st.encode(p)
        stats["clip_tokens>16"] += len(bpe) > 16
        if ls is not None:
            nb = len(st.encode(p[:ls])); nl = len(st.encode(t[ls:]))
            k = max(0, min(nl, 16 - nb)); stats["clip_loc_fully_kept"] += k == nl; stats["clip_loc_none_kept"] += k == 0
        # DD-CMD: BiomedBERT word pieces, 24 incl. CLS/SEP -> 22 content (no LViT padding)
        bt = biomed.tokenize(t)
        stats["biomed_tokens>22"] += len(bt) > 22
        if ls is not None:
            nb = len(biomed.tokenize(t[:ls])); nl = len(biomed.tokenize(t[ls:]))
            k = max(0, min(nl, 22 - nb)); stats["biomed_loc_fully_kept"] += k == nl; stats["biomed_loc_none_kept"] += k == 0
    h = stats["has_loc"]
    print(f"== {ds} train+val images {n}, with location clause {h}")
    print(f"   LViT-T: >23 word pieces {stats['lvit_wordpieces>23']}; mean BERT words {stats['lvit_words']/n:.1f}; "
          f"location words inside first 10: {100*stats['lvit_loc_words_kept']/stats['lvit_loc_words_total']:.1f}%; "
          f"location fully inside {100*stats['lvit_loc_fully_kept']/h:.1f}%, none inside {100*stats['lvit_loc_none_kept']/h:.1f}% of images")
    print(f"   RecLMIS: >16 BPE tokens (truncated) {100*stats['clip_tokens>16']/n:.1f}%; location fully kept {100*stats['clip_loc_fully_kept']/h:.1f}%, none kept {100*stats['clip_loc_none_kept']/h:.1f}%")
    print(f"   DD-CMD: >22 word pieces (truncated) {100*stats['biomed_tokens>22']/n:.1f}%; location fully kept {100*stats['biomed_loc_fully_kept']/h:.1f}%, none kept {100*stats['biomed_loc_none_kept']/h:.1f}%")
# C1 empty string per method (tokenizer level)
print("C1 empty string:")
print("  LViT/RecLMIS padding of '' ->", repr(common.lvit_pad("")))
print("  LViT BERT words of padded '':", bert.basic_tokenizer.tokenize(common.lvit_pad("")))
print("  LViT BERT words of raw '' (no padding):", bert.basic_tokenizer.tokenize(""))
print("  CLIP tokens of padded '':", clip.tokenize([common.lvit_pad("")], context_length=18, truncate=True)[0].tolist())
print("  CLIP tokens of raw '':", clip.tokenize([""], context_length=18, truncate=True)[0].tolist())
e = biomed("", padding="max_length", max_length=24, truncation=True)
print("  BiomedBERT '' input_ids:", e["input_ids"], "attention_mask:", e["attention_mask"])
