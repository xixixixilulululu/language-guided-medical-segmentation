"""Which attributes differ, per distance, over (target, distinct donor text) pairs; validation (Phase 2 setup check)."""
import sys, re, collections
sys.path.insert(0, "src"); sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from phase1 import common
from parse_probe2 import parse
NAMES = ("scope", "count", "location")
for ds in ["QaTa-COV19", "MosMedData+"]:
    texts = common.read_texts(ds, "val")
    A = {i: parse(t)[0] for i, t in texts.items() if parse(t)[0]}
    uniq = {}
    for i, a in A.items(): uniq.setdefault(re.sub(r"\s+", " ", texts[i].strip()), a)
    combos = {k: collections.Counter() for k in (1, 2)}
    targets_with = {k: collections.Counter() for k in (1, 2)}
    for i, a in A.items():
        have = {k: set() for k in (1, 2)}
        for t, b in uniq.items():
            diff = tuple(n for n, x, y in zip(NAMES, a, b) if x != y)
            if len(diff) in combos: combos[len(diff)][diff] += 1; have[len(diff)].add(diff)
        for k in (1, 2):
            for d in have[k]: targets_with[k][d] += 1
    print(f"== {ds} (val; pairs = target x distinct donor text, before group exclusion)")
    for k in (1, 2):
        tot = sum(combos[k].values())
        print(f"   distance {k}: " + "; ".join(f"{'+'.join(d)} {100*n/tot:.1f}% of pairs, targets with such a donor {targets_with[k][d]}" for d, n in combos[k].most_common()))
