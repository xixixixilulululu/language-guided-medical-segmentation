"""Phase 3, Task 3.1B: the one fixed group split of the QaTa-COV19 VALIDATION cases into GATE_FIT / GATE_CAL.

Groups are the Phase 2 units (src/phase2/texts.groups): the BIDS subject `sub-S...` for patient images; every
`covid_*` image (no patient id) is its own provisional group (a limitation: images of one unknown patient may
fall into both parts). The split uses only the sample ids, never any result.

Algorithm (fixed before any Phase 3 result):
  1. G = the distinct group ids of the QaTa validation images, sorted as Python strings (ascending).
  2. order = numpy.random.default_rng(3001).permutation(len(G)).
  3. n_fit = floor(0.6 * len(G) + 0.5)  (round half up).
  4. Groups G[order[0]], ..., G[order[n_fit - 1]] are GATE_FIT; all other groups are GATE_CAL.
Every image (with all its conditions, donors and seeds) belongs to the part of its group.
The file is written once; an existing file is never overwritten (no other split is tried).

Usage:  python src/phase3/make_partitions.py   -> results/phase3_gate_partitions.csv
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
OUT = os.path.join(REPO, "results", "phase3_gate_partitions.csv")
SEED, FIT_SHARE = 3001, 0.6


def partition(ids, groups_of):
    """DataFrame sample_id, group_id, group_type, partition for the given ids."""
    g = {i: groups_of(i) for i in ids}
    G = sorted({v[0] for v in g.values()})
    order = np.random.default_rng(SEED).permutation(len(G))
    n_fit = int(np.floor(FIT_SHARE * len(G) + 0.5))
    fit = {G[k] for k in order[:n_fit]}
    return pd.DataFrame([{"dataset": "QaTa-COV19", "sample_id": i, "group_id": g[i][0], "group_type": g[i][1],
                          "partition": "GATE_FIT" if g[i][0] in fit else "GATE_CAL"} for i in ids])


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists: the split is generated once and never regenerated")
    from lgms import data
    from phase2 import texts as T
    ids = data.split_ids("QaTa-COV19", "val")
    df = partition(ids, lambda i: T.groups("QaTa-COV19", i)[2:4])
    df.to_csv(OUT, index=False)
    s = df.groupby("partition").agg(groups=("group_id", "nunique"), images=("sample_id", "size"))
    s["patient_groups"] = df[df.group_type == "patient"].groupby("partition").group_id.nunique()
    s["covid_image_groups"] = df[df.group_type != "patient"].groupby("partition").group_id.nunique()
    print(s.to_string())
    print("written:", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
