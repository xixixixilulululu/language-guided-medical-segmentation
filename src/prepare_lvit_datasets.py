"""Arrange QaTa-COV19 and MosMedData+ into the LViT folder layout, using the LViT split lists.

The split is NOT re-created here: every image goes to the folder named by the official LViT
xlsx lists (Google Drive folder linked from the LViT README). Images and masks are copied
unchanged from the two public Kaggle archives; resizing to 224x224 happens later (Task 0.2).

Output layout (the one expected by external/LViT/Load_Dataset.py, see LViT README 1.2):

    <out_root>/QaTa-COV19/{Train,Val,Test}_Folder/{img/, labelcol/, <Split>_text.xlsx}
    <out_root>/MosMedData+/{Train,Val,Test}_Folder/{img/, labelcol/, <Split>_text.xlsx}

File naming follows LViT:
  * QaTa: mask `mask_<name>.png` in labelcol/, image `<name>.png` in img/
    (LViT Load_Dataset.py, ImageToImage2D.__getitem__, the commented "Covid19" lines:
    image_filename = mask_filename.replace('mask_', '')).
  * MosMed: image and mask share the same filename.

Usage:
    python src/prepare_lvit_datasets.py --raw ~/datasets/raw --out ~/datasets
"""
import argparse
import os
import zipfile

import pandas as pd

SPLITS = ["Train", "Val", "Test"]

# Source archives (Kaggle) and split lists (LViT Google Drive)
QATA_ZIP = "qatacov19-dataset.zip"                               # kaggle: aysendegerli/qatacov19-dataset
MOSMED_ZIP = "covid19-ct-scan-lesion-segmentation-dataset.zip"   # kaggle: maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset
LISTS = "lvit_split_lists"


def read_lists(list_dir):
    """Return {dataset: {split: DataFrame(Image, Description)}} from the LViT xlsx files."""
    p = lambda f: os.path.join(list_dir, f)
    # QaTa: the train/val ID lists define the split, the text comes from the joint train+val sheet
    qata_text = pd.read_excel(p("Train_Val_text_for_Covid19.xlsx"))
    qata = {
        "Train": qata_text[qata_text.Image.isin(pd.read_excel(p("Train_ID_for_Covid19.xlsx")).Image)],
        "Val": qata_text[qata_text.Image.isin(pd.read_excel(p("Val_ID_for_Covid19.xlsx")).Image)],
        "Test": pd.read_excel(p("Test_text_for_Covid19.xlsx")),
    }
    # MosMed sheets call the text column "text"; LViT's read_text() expects "Description"
    mosmed = {
        s: pd.read_excel(p(f)).rename(columns={"text": "Description"})
        for s, f in [("Train", "Train_text_MosMedData.xlsx"), ("Val", "Val_text_MosMedData.xlsx"),
                     ("Test", "Test_text_MosMedData+.xlsx")]
    }
    for d in (qata, mosmed):
        for s in SPLITS:
            d[s] = d[s][["Image", "Description"]].reset_index(drop=True)
    return {"QaTa-COV19": qata, "MosMedData+": mosmed}


def qata_sources(split, mask_name):
    """Archive member names (image, mask) for one QaTa sample. LViT train+val = Kaggle v2 'Train Set'."""
    part = "Test Set" if split == "Test" else "Train Set"
    root = f"QaTa-COV19/QaTa-COV19-v2/{part}"
    return f"{root}/Images/{mask_name.replace('mask_', '', 1)}", f"{root}/Ground-truths/{mask_name}"


def mosmed_sources(split, name):
    """Archive member names (image, mask) for one MosMed sample; both folders use the same filename."""
    return f"frames/{name}", f"masks/{name}"


def arrange(zip_path, lists, out_dir, sources, image_name):
    """Copy every listed sample from the archive into <out_dir>/<Split>_Folder/{img,labelcol}."""
    z = zipfile.ZipFile(zip_path)
    members = set(z.namelist())
    for split in SPLITS:
        folder = os.path.join(out_dir, f"{split}_Folder")
        os.makedirs(os.path.join(folder, "img"), exist_ok=True)
        os.makedirs(os.path.join(folder, "labelcol"), exist_ok=True)
        df = lists[split]
        for name in df.Image:
            img_src, mask_src = sources(split, name)
            # Stop immediately if the archive does not contain a listed file (never substitute)
            assert img_src in members and mask_src in members, f"missing in archive: {img_src} / {mask_src}"
            for src, sub, dst_name in [(img_src, "img", image_name(name)), (mask_src, "labelcol", name)]:
                with z.open(src) as fin, open(os.path.join(folder, sub, dst_name), "wb") as fout:
                    fout.write(fin.read())
        df.to_excel(os.path.join(folder, f"{split}_text.xlsx"), index=False)
        print(f"{os.path.basename(out_dir)} {split}: {len(df)} samples -> {folder}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.expanduser("~/datasets/raw"))
    ap.add_argument("--out", default=os.path.expanduser("~/datasets"))
    args = ap.parse_args()
    lists = read_lists(os.path.join(args.raw, LISTS))
    arrange(os.path.join(args.raw, QATA_ZIP), lists["QaTa-COV19"], os.path.join(args.out, "QaTa-COV19"),
            qata_sources, image_name=lambda m: m.replace("mask_", "", 1))
    arrange(os.path.join(args.raw, MOSMED_ZIP), lists["MosMedData+"], os.path.join(args.out, "MosMedData+"),
            mosmed_sources, image_name=lambda m: m)


if __name__ == "__main__":
    main()
