"""Eligibility probe on the VALIDATION split (Phase 2 setup check). Donors: same split, distinct text, other group."""
import sys, re, collections
sys.path.insert(0, "src"); sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from phase1 import common
from lgms import data
from parse_probe2 import parse, normalise
def group(ds, i):
    if ds == "QaTa-COV19":
        m = re.match(r"(sub-S\d+)_ses-", i); return m.group(1) if m else None
    return None if i.startswith("bjorke_") else re.sub(r"_\d+$", "", i)
def canon(p, all_eq):
    s, c, loc = p
    if all_eq:
        loc = tuple((k, ("all",) if set(v) == {"upper", "middle", "lower"} else v) for k, v in loc)
    return s, c, loc
for all_eq in (False, True):
    print(f"######## 'all' == 'upper middle lower': {all_eq}")
    for ds in ["QaTa-COV19", "MosMedData+"]:
        texts = common.read_texts(ds, "val"); ids = list(texts)
        P = {i: parse(t)[0] for i, t in texts.items()}
        A = {i: canon(p, all_eq) for i, p in P.items() if p}
        n = len(ids); ok = len(A)
        has_loc = sum(1 for i in A if A[i][2])
        print(f"== {ds} val: {n} images; parsed {ok}; C2 (template rewrite) {ok}; C3a scope {ok}; C3b count {ok}; C3c location {has_loc}")
        for unknown_group_policy in ("self-only", "exclude-ungrouped"):
            res = collections.Counter(); dn = collections.defaultdict(list)
            for i in ids:
                if i not in A: continue
                gi = group(ds, i)
                if unknown_group_policy == "exclude-ungrouped" and gi is None: continue
                seen = {k: set() for k in (1, 2, 3)}
                for j in ids:
                    if j == i or j not in A: continue
                    gj = group(ds, j)
                    if gi is not None and gj == gi: continue
                    tj = re.sub(r"\s+", " ", texts[j].strip())
                    d = sum(a != b for a, b in zip(A[i], A[j]))
                    if d in seen: seen[d].add(tj)
                for k in (1, 2, 3):
                    m = len(seen[k]); dn[k].append(m)
                    res[(k, ">=1")] += m >= 1; res[(k, ">=3")] += m >= 3
                res["all3>=1"] += all(len(seen[k]) >= 1 for k in (1, 2, 3))
            print(f"   donors [{unknown_group_policy}]: " + "; ".join(
                f"C{3+k}: >=1 donor {res[(k,'>=1')]}, >=3 donors {res[(k,'>=3')]}, median distinct donor texts {sorted(dn[k])[len(dn[k])//2] if dn[k] else 0}" for k in (1,2,3))
                + f"; C4∩C5∩C6 {res['all3>=1']}")
