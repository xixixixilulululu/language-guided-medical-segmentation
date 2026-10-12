"""Describe the sources and image encoding of the MosMedData+ Kaggle mirror (read-only check).

Reads the Kaggle archive (maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset, the mirror linked in
the LViT README) and reports, per source (filename prefix):
  * number of slices and of distinct volumes;
  * image shape, dtype and PNG encoding (bit depth, colour type);
  * intensity statistics of a fixed random sample of slices (seed 0): mean, share of pixels at 0 and
    at 255, number of grey levels, 1st / 99th percentile -- to see whether a common window was applied.
Nothing is written to the datasets.

Usage:
    python src/analysis/inspect_mosmed_sources.py --zip ~/datasets/raw/covid19-ct-scan-lesion-segmentation-dataset.zip
"""
import argparse
import collections
import json
import os
import re
import zipfile

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def source_of(filename):
    """Source group from the frame name: Jun_coronacases / Jun_radiopaedia / Morozov / bjorke."""
    b = os.path.basename(filename)
    return "_".join(b.split("_")[:2]) if b.startswith("Jun_") else b.split("_")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=os.path.expanduser(
        "~/datasets/raw/covid19-ct-scan-lesion-segmentation-dataset.zip"))
    ap.add_argument("--sample", type=int, default=60, help="slices per source for the intensity statistics")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "phase0_mosmed_sources.json"))
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    names = z.namelist()
    frames = sorted(n for n in names if n.startswith("frames/") and n.endswith(".png"))
    by = collections.defaultdict(list)
    for n in frames:
        by[source_of(n)].append(n)
    rng = np.random.default_rng(0)
    report = {"archive_top_level": dict(collections.Counter(n.split("/")[0] for n in names)),
              "non_png_files": [n for n in names if not n.endswith(".png")], "sources": {}}
    for g, ns in sorted(by.items()):
        vols = {re.sub(r"_\d+\.png$", "", os.path.basename(n)) for n in ns}
        st = []
        for n in rng.choice(ns, size=min(args.sample, len(ns)), replace=False):
            raw = z.read(n)
            a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            x = a[..., 0] if a.ndim == 3 else a
            st.append({"shape": list(a.shape), "dtype": str(a.dtype), "png_bit_depth": raw[24],
                       "png_colour_type": raw[25], "mean": float(x.mean()), "zero": float((x == 0).mean()),
                       "sat": float((x == 255).mean()), "levels": int(len(np.unique(x))),
                       "p1": float(np.percentile(x, 1)), "p99": float(np.percentile(x, 99))})
        report["sources"][g] = {
            "slices": len(ns), "volumes": len(vols),
            "shapes": sorted({str(s["shape"]) for s in st}), "dtypes": sorted({s["dtype"] for s in st}),
            "png": sorted({f"{s['png_bit_depth']}-bit, colour type {s['png_colour_type']}" for s in st}),
            "mean_intensity": round(np.mean([s["mean"] for s in st]), 1),
            "pct_pixels_0": round(100 * np.mean([s["zero"] for s in st]), 2),
            "pct_pixels_255": round(100 * np.mean([s["sat"] for s in st]), 2),
            "grey_levels": round(np.mean([s["levels"] for s in st])),
            "p1": round(np.mean([s["p1"] for s in st])), "p99": round(np.mean([s["p99"] for s in st])),
            "sampled": len(st)}
    print(json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
