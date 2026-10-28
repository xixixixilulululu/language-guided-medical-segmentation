"""Phase 3: per-case files of the validation reruns for independent checking (no text).

  results/phase3_img_val_per_case.csv.gz          IMG rerun (2 archs x 2 datasets x 3 seeds):
                                                  dataset, arch, seed, sample_id, dice, miou, hd95, assd, empty
  results/phase3_lang_rerun_val_per_case.csv.gz   LANG C2 / C3a-c rerun, columns of results/phase2_val_per_case.csv.gz:
                                                  dataset, method, seed, sample_id, condition, variant_id,
                                                  donor_sample_id, unit, unit_type, dice, miou, hd95, assd, empty
Usage:  python src/phase3/export_per_case.py
"""
import glob
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
RUN = os.path.join(REPO, "runs", "phase3", "val")


def main():
    from phase2 import texts as T
    img = pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in
                     sorted(glob.glob(os.path.join(RUN, "img", "*", "*", "seed*", "per_case.csv")))], ignore_index=True)
    img = img[["dataset", "arch", "seed", "sample_id", "dice", "miou", "hd95", "assd", "empty"]]
    img.sort_values(["dataset", "arch", "seed", "sample_id"]).to_csv(
        os.path.join(REPO, "results", "phase3_img_val_per_case.csv.gz"), index=False, float_format="%.6f")
    lang = pd.concat([pd.read_csv(f, dtype={"sample_id": str, "variant_id": str}, keep_default_na=False) for f in
                      sorted(glob.glob(os.path.join(RUN, "lang", "*", "*", "seed*", "per_case_*.csv")))], ignore_index=True)
    g = {(d, s): T.groups(d, s)[2:4] for d, s in lang[["dataset", "sample_id"]].drop_duplicates().itertuples(index=False)}
    lang["unit"] = [g[(d, s)][0] for d, s in zip(lang.dataset, lang.sample_id)]
    lang["unit_type"] = [g[(d, s)][1] for d, s in zip(lang.dataset, lang.sample_id)]
    cols = ["dataset", "method", "seed", "sample_id", "condition", "variant_id", "donor_sample_id", "unit", "unit_type",
            "dice", "miou", "hd95", "assd", "empty"]
    lang[cols].sort_values(["dataset", "method", "seed", "condition", "variant_id", "sample_id"]).to_csv(
        os.path.join(REPO, "results", "phase3_lang_rerun_val_per_case.csv.gz"), index=False, float_format="%.6f")
    print(f"IMG rows {len(img)}, LANG rerun rows {len(lang)}")


if __name__ == "__main__":
    main()
