"""Probe v2: fixed normalisation + typo table; ambiguous -> unknown (Phase 2 setup check; train/val texts only, never the test texts)."""
import sys, re, collections
sys.path.insert(0, "src")
from phase1 import common
NUM = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
TYPO = {"lowerl":"lower","lowoer":"lower","lwoer":"lower","ower":"lower","lowe":"lower","midldle":"middle",
        "midddle":"middle","middlr":"middle","mddle":"middle","midde":"middle","iddle":"middle","alll":"all",
        "uppper":"upper","uooer":"upper","upperl":"upper","lift":"left","ung":"lung","thrr":"three",
        "upperlower":"upper lower","middlelower":"middle lower","lowerright":"lower right","rightlung":"right lung",
        "leflung":"left lung","oneinfected":"one infected","threeinfected":"three infected","andupper":"and upper"}
ZONES = {"upper","middle","lower","all"}
def normalise(t):
    t = t.strip().rstrip(".").strip()
    t = re.sub(r"\s*,\s*", ", ", t); t = re.sub(r"\s+", " ", t)
    return " ".join(TYPO.get(w, w) for w in t.split(" "))
HEAD = re.compile(r"^(Bilateral|Unilateral|No) pulmonary infection, (\w+) infected areas?(?:, (.+))?$")
def parse(t):
    n = normalise(t); m = HEAD.match(n)
    if not m: return None, "head: " + n
    scope, cw, loc = m.groups()
    if cw not in NUM: return None, f"count word {cw!r}"
    sides = {}
    if loc:
        # elision: "<z> left and <z> right lung" -> "<z> left lung and <z> right lung"
        loc = re.sub(r"\bleft and\b", "left lung and", loc)
        for part in loc.split(" and "):
            w = part.split()
            if len(w) < 2 or w[-1] != "lung" or w[-2] not in ("left", "right"): return None, f"loc {part!r}"
            z = w[:-2]
            if any(x not in ZONES for x in z) or len(set(z)) != len(z): return None, f"zones {part!r}"
            if w[-2] in sides: return None, f"side twice {loc!r}"
            sides[w[-2]] = frozenset(z) if z else frozenset({"unspecified"})
    return (scope, NUM[cw], tuple(sorted((k, tuple(sorted(v))) for k, v in sides.items()))), "ok"
if __name__ == "__main__":
    for ds in ["QaTa-COV19", "MosMedData+"]:
        tot = ok = 0; bad = collections.Counter(); incons = collections.Counter()
        for sp in ["train", "val"]:
            for i, t in common.read_texts(ds, sp).items():
                tot += 1; p, why = parse(t)
                if p: ok += 1
                else: bad[why] += 1
                if p:
                    s, c, loc = p
                    if {"Bilateral": 2, "Unilateral": 1, "No": 0}[s] != len(loc): incons[f"{s} with {len(loc)} side(s)"] += 1
                    if (s == "No") != (c == 0): incons["No/zero"] += 1
        print(f"== {ds}: parsed {ok}/{tot} = {100*ok/tot:.2f}%; unparsed {tot-ok}; scope/side inconsistency {dict(incons)}")
        for w, n in bad.most_common(): print("   ", n, w)
