"""Phase 4, option 1 supplement: do SGSeg's revised report texts explain the reports that disagree with convention B?

Rule: configs/phase4/option1_text_error_rule_v2.json (committed before running; read, never changed). v2 replaces
only the parsing step of v1 (configs/phase4/option1_text_error_rule.json, which stays unchanged and is checked):
each SGSeg caption is stripped and its first character upper-cased before src/phase2/texts.py parse().
Training split only. External file: SGSeg annotation.json (outside the repository); its "test" key is deleted
right after loading and never read; SGSeg items that are not our training ids are dropped unseen.
Self-checks (exit status 9): rule tracked with one commit, unchanged, an ancestor of HEAD; clean working tree;
output absent (temporary file, then rename); external file sha256 as in the rule.
Output: results/phase4_option1_text_error_check.csv (counts only; no text, no per-image rows).
Usage:  python src/phase4/option1_text_error_check.py      Tests: python src/phase4/test_option1_text_error_check.py
"""
import datetime
import hashlib
import json
import os
import re
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase4 import option1_train_check as O  # noqa: E402

RULE = os.path.join("configs", "phase4", "option1_text_error_rule_v2.json")
RULE_V1 = os.path.join("configs", "phase4", "option1_text_error_rule.json")
RULE_V1_SHA = "46a3fb25578603d6d9fe0e86a7af311118672501dac8029248412ced1c9799de"
RULE_VERSION = "v2"
OUT = os.path.join("results", "phase4_option1_text_error_check.csv")
EXTERNAL = os.path.expanduser("~/datasets/external/sgseg/annotation.json")
EXTERNAL_SHA = "6f62c4b40ed460de53a8e4a80cc7633569853ab8fe2f4cca3cfc216af9e59fb0"
STOP_EXIT = 9
N_REPORTS, N_AGREE = 1129, 1085


def stop(msg):
    print(f"TEXT-ERROR CHECK STOPPED: {msg}", file=sys.stderr)
    sys.exit(STOP_EXIT)


def preflight(repo=REPO, external=EXTERNAL, external_sha=EXTERNAL_SHA):
    """(rule commit, rule sha256, external sha256) or stop (exit 9)."""
    try:
        commit, sha = O.preflight(repo=repo, rule=RULE, out=OUT)
    except SystemExit:
        stop("self-check of the rule / working tree / output failed (see above)")
    v1 = os.path.join(repo, RULE_V1)
    if os.path.exists(v1) and hashlib.sha256(open(v1, "rb").read()).hexdigest() != RULE_V1_SHA:
        stop(f"{RULE_V1} changed (v1 must stay unchanged)")
    if not os.path.exists(external):
        stop(f"{external} missing")
    ext = hashlib.sha256(open(external, "rb").read()).hexdigest()
    if ext != external_sha:
        stop(f"external file sha256 {ext} differs from the rule")
    return commit, sha, ext


def load_external(path):
    """SGSeg annotations without the 'test' key (deleted right after loading)."""
    d = json.load(open(path))
    del d["test"]
    return d


def match_training(d, train_ids):
    """{our training id: SGSeg caption}; SGSeg train + valid only, case-insensitive; stop unless 1:1."""
    lower = {}
    for i in train_ids:
        if i.lower() in lower:
            stop("training ids are not unique in lower case")
        lower[i.lower()] = i
    found = {}
    dup = 0
    for part in ("train", "valid"):
        for item in d[part]:
            key = re.sub(r"\.png$", "", re.sub(r"^mask_", "", item["image_path"].lower()))
            if key not in lower:
                continue                                                     # not ours: dropped unseen
            if lower[key] in found:
                dup += 1
            found[lower[key]] = item["caption"]
    missing = len(train_ids) - len(found)
    if missing or dup:
        stop(f"training ids in SGSeg train + valid: {len(found)} of {len(train_ids)} found, {missing} missing, {dup} duplicated")
    return found


def capitalise(caption):
    """Rule v2: strip, then upper-case the first character only."""
    c = str(caption).strip()
    return c[:1].upper() + c[1:]


def parse_v2(parse):
    """parse() applied to the capitalised caption (parse itself unchanged)."""
    return lambda caption: parse(capitalise(caption))


def norm(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def classify(original, revised, original_side, parse):
    """R1-R4 for one report; parse = src/phase2/texts.py parse."""
    if norm(original) == norm(revised):
        return "R1"
    p = parse(revised)
    side = O.one_sided(p["location"]) if p["parse_ok"] else None
    if side is None:
        return "R4"
    return "R2" if side == original_side else "R3"


def interpret(n_r3_dis, n_dis, p):
    share = n_r3_dis / n_dis
    if share >= 0.5 and p < 0.01:
        return "SGSeg's revisions explain most of the disagreement"
    if share < 0.25:
        return "SGSeg's revisions explain little of the disagreement"
    return "SGSeg's revisions explain part of the disagreement"


def analyse(D, parse):
    """D: one row per report with sample_id, group, report_side, image_side, agree_B, original, revised.
    -> (count rows, summary)."""
    from scipy.stats import fisher_exact
    D = D.copy()
    D["cls"] = [classify(o, r, s, parse) for o, r, s in zip(D.original, D.revised, D.report_side)]
    rows = []
    for subset, g in (("all", D), ("sub", D[D.group == "sub"]), ("covid", D[D.group == "covid"])):
        for name, h in (("agree_B", g[g.agree_B]), ("disagree_B", g[~g.agree_B])):
            n = h.cls.value_counts()
            rows.append({"row": "counts", "subset": subset, "option1_group": name, "n": len(h),
                         **{k: int(n.get(k, 0)) for k in ("R1", "R2", "R3", "R4")}})
    rev = []
    for r in D.itertuples():
        p = parse(r.revised)
        s = O.one_sided(p["location"]) if p["parse_ok"] else None
        if s is not None:
            rev.append(O.agrees(s, r.image_side, "B"))
    ag, dis = D[D.agree_B], D[~D.agree_B]
    a3, d3 = int((ag.cls == "R3").sum()), int((dis.cls == "R3").sum())
    _, p = fisher_exact([[d3, len(dis) - d3], [a3, len(ag) - a3]], alternative="two-sided")
    summary = {"row": "summary", "n_reports": len(D), "n_agree_B": len(ag), "n_disagree_B": len(dis),
               "r3_disagree": d3, "r3_disagree_share": d3 / len(dis) if len(dis) else float("nan"),
               "r3_agree": a3, "r3_agree_share": a3 / len(ag) if len(ag) else float("nan"), "fisher_p_two_sided": p,
               "revised_one_sided_n": len(rev), "revised_agreement_B": sum(rev) / len(rev) if rev else float("nan"),
               "interpretation": interpret(d3, len(dis), p) if len(dis) else "",
               "agree_group_flag": ("SGSeg may use another orientation convention; manual review needed"
                                    if len(ag) and a3 / len(ag) >= 0.05 else "")}
    return rows, summary, D


def main():
    commit, sha, ext = preflight()
    from lgms import data
    from phase2 import texts as T
    from phase4 import guard
    guard.require_dev_split("train")
    train = data.split_ids(O.DATASET, "train")
    val, test = set(data.split_ids(O.DATASET, "val")), set(data.split_ids(O.DATASET, "test"))
    if set(train) & (val | test):
        stop("training ids overlap validation or test ids")
    captions = match_training(load_external(EXTERNAL), train)
    parse = parse_v2(T.parse)
    parsed = [parse(c) for c in captions.values()]
    bad = sum(not p["parse_ok"] for p in parsed) / len(parsed)
    if bad > 0.05:
        import collections
        reasons = collections.Counter(p["parse_note"] for p in parsed if not p["parse_ok"]).most_common(10)
        stop(f"capitalised revised captions not parsed: {bad:.1%} (> 5%); reasons: {reasons}")
    A = pd.read_csv(os.path.join(REPO, O.ATTR), dtype=str, keep_default_na=False)
    A = A[(A.dataset == O.DATASET) & (A.split == "train")].set_index("sample_id")
    items = []
    for sid in train:
        path = data.mask_path(O.DATASET, "train", sid)
        guard.require_dev_path(path)
        items.append((sid, A.parse_ok[sid] == "True", A.location[sid], O.lesion_fraction(data.read_mask(O.DATASET, "train", sid))))
    R = O.records(items)
    R = R[R.excluded == ""].copy()
    R["image_side"] = [O.side_primary(f) for f in R.f]
    R = R[R.image_side.notna()]
    R["agree_B"] = [O.agrees(s, i, "B") for s, i in zip(R.report_side, R.image_side)]
    if len(R) != N_REPORTS or int(R.agree_B.sum()) != N_AGREE:
        stop(f"option 1 recomputation gives {len(R)} reports / {int(R.agree_B.sum())} agreeing (expected 1129 / 1085)")
    R["original"] = R.sample_id.map(A.raw_text)
    R["revised"] = R.sample_id.map(captions)
    rows, summary, _ = analyse(R, parse)
    summary["revised_parse_failure_share_all_training"] = bad
    meta = {"rule_version": RULE_VERSION, "run_time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "rule_commit": commit, "rule_sha256": sha, "external_sha256": ext}
    out = pd.DataFrame(rows + [summary])
    for k, v in meta.items():
        out[k] = v
    path = os.path.join(REPO, OUT)
    if os.path.exists(path):
        stop(f"{OUT} exists; it is never overwritten")
    out.to_csv(path + ".tmp", index=False, float_format="%.6g")
    os.replace(path + ".tmp", path)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
