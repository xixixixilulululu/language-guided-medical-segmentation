"""Access to the common 224x224 benchmark data written by src/prepare_common_224.py (Task 0.2).

Layout (one folder per dataset, identical for both datasets):

    <common_root>/<dataset>/<split>.txt              one sample id per line (sorted)
    <common_root>/<dataset>/<split>/images/<id>.png  uint8 grayscale, 224x224
    <common_root>/<dataset>/<split>/masks/<id>.png   uint8 {0, 1}, 224x224

<dataset> is "QaTa-COV19" or "MosMedData+", <split> is "train", "val" or "test".
All three methods read exactly these files; model-specific loading (channel duplication,
intensity scaling, augmentation) happens in the method's own training script.
"""
import os

import cv2
import numpy as np

DATASETS = ["QaTa-COV19", "MosMedData+"]
SPLITS = ["train", "val", "test"]
IMG_SIZE = 224

# Default location of the prepared data (never committed to git)
DEFAULT_COMMON_ROOT = os.path.expanduser("~/datasets/common224")


def common_root():
    """Root of the prepared data; can be overridden with the LGMS_COMMON_ROOT environment variable."""
    return os.environ.get("LGMS_COMMON_ROOT", DEFAULT_COMMON_ROOT)


def split_ids(dataset, split, root=None):
    """Sorted list of sample ids of one split."""
    root = root or common_root()
    with open(os.path.join(root, dataset, f"{split}.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def image_path(dataset, split, sample_id, root=None):
    return os.path.join(root or common_root(), dataset, split, "images", f"{sample_id}.png")


def mask_path(dataset, split, sample_id, root=None):
    return os.path.join(root or common_root(), dataset, split, "masks", f"{sample_id}.png")


def read_image(dataset, split, sample_id, root=None):
    """uint8 grayscale image, shape (224, 224)."""
    img = cv2.imread(image_path(dataset, split, sample_id, root), cv2.IMREAD_UNCHANGED)
    assert img is not None and img.shape == (IMG_SIZE, IMG_SIZE), (dataset, split, sample_id)
    return img


def read_mask(dataset, split, sample_id, root=None):
    """uint8 binary mask {0, 1}, shape (224, 224)."""
    m = cv2.imread(mask_path(dataset, split, sample_id, root), cv2.IMREAD_UNCHANGED)
    assert m is not None and m.shape == (IMG_SIZE, IMG_SIZE), (dataset, split, sample_id)
    return m


def load_split_arrays(dataset, split, root=None):
    """Load a whole split into memory: (ids, images uint8 (N,224,224), masks uint8 (N,224,224)).

    The prepared files are small (QaTa train: 5716 x 50 KB), so keeping a split in RAM removes
    disk I/O from the training loop without changing any value.
    """
    ids = split_ids(dataset, split, root)
    images = np.stack([read_image(dataset, split, i, root) for i in ids])
    masks = np.stack([read_mask(dataset, split, i, root) for i in ids])
    return ids, images, masks
