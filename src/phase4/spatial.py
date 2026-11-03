"""Phase 4, Stage 0: the 2 x 3 image-coordinate grid, the report-location lookup and the spatial score.

Grid (PHASE4_PLAN.md Stage 0 step 4; configs/phase4/stage0_spec.json), on the common 224 x 224 array (row 0 = top,
column 0 = image left; LANG, IMG and ground truth share this array layout):
  columns  image-left 0-111, image-right 112-223   (split at the midpoint, 112)
  rows     upper 0-74, middle 75-148, lower 149-223 (boundaries round(224 k / 3) = 75, 149)
Location strings are the Phase 2 parse (src/phase2/texts.py, `location`, D5 typo rules included), e.g.
"left:lower;right:upper+middle"; sides are PATIENT sides. Zones of one side are united; "all" = the three cells.
Orientation (which image half is the patient's left) is a parameter:
  "patient_left_on_image_right"   radiological display convention
  "patient_left_on_image_left"
  "max_of_both"                   both mappings computed, the higher agreement of the row is kept (per row)
Unavailable (never 0, never "incorrect"): lateral view (file name "_vp-ll_"), empty IMG mask, location empty /
not parsed, a side "unspecified", an unknown zone token.
spatial_agreement = IMG lesion pixels in the report-allowed cells / all IMG lesion pixels; mismatch = 1 - agreement.
Tests: src/phase4/test_spatial.py.
"""
import numpy as np

SIZE = 224
COL_SPLIT = 112                                      # image-left: columns [0, 112); image-right: [112, 224)
ROW_EDGES = (0, 75, 149, 224)                        # upper [0, 75), middle [75, 149), lower [149, 224)
ZONES = ("upper", "middle", "lower")
SIDES = ("left", "right")
ORIENTATIONS = ("patient_left_on_image_right", "patient_left_on_image_left", "max_of_both")

# unavailable reasons (row level)
LATERAL, IMG_EMPTY = "lateral_view", "img_empty"
LOC_EMPTY, LOC_UNSPECIFIED, LOC_UNKNOWN = "location_not_parsed", "location_unspecified", "location_unknown_token"


def is_lateral(sample_id):
    """Lateral view by the BIMCV file name (project lead's rule, 2026-11-03)."""
    return "_vp-ll_" in sample_id


def parse_location(loc):
    """(cells, reason): cells = set of (patient side, zone); reason None if mappable, else an unavailable reason."""
    if loc is None or (isinstance(loc, float) and np.isnan(loc)) or str(loc).strip() == "":
        return None, LOC_EMPTY
    cells = set()
    for part in str(loc).split(";"):
        if ":" not in part:
            return None, LOC_UNKNOWN
        side, zones = part.split(":", 1)
        if side not in SIDES:
            return None, LOC_UNKNOWN
        if zones == "unspecified":
            return None, LOC_UNSPECIFIED
        names = ZONES if zones == "all" else zones.split("+")
        if not names or any(z not in ZONES for z in names):
            return None, LOC_UNKNOWN
        cells |= {(side, z) for z in names}
    return (cells, None) if cells else (None, LOC_EMPTY)


def image_half(side, orientation):
    """'image_left' or 'image_right' for a patient side under one fixed orientation."""
    if orientation == "patient_left_on_image_right":
        return "image_right" if side == "left" else "image_left"
    if orientation == "patient_left_on_image_left":
        return "image_left" if side == "left" else "image_right"
    raise ValueError(orientation)


def allowed_region(cells, orientation):
    """Boolean (224, 224) mask of the union of the cells under one fixed orientation."""
    m = np.zeros((SIZE, SIZE), bool)
    for side, zone in cells:
        k = ZONES.index(zone)
        r0, r1 = ROW_EDGES[k], ROW_EDGES[k + 1]
        c0, c1 = (0, COL_SPLIT) if image_half(side, orientation) == "image_left" else (COL_SPLIT, SIZE)
        m[r0:r1, c0:c1] = True
    return m


def availability(sample_id, img_mask, loc):
    """None if a score can be computed for this row, else the first unavailable reason (fixed order)."""
    if is_lateral(sample_id):
        return LATERAL
    if not np.asarray(img_mask, bool).any():
        return IMG_EMPTY
    return parse_location(loc)[1]


def spatial_score(sample_id, img_mask, loc, orientation):
    """dict(agreement, mismatch, reason); agreement / mismatch are None when unavailable."""
    if orientation not in ORIENTATIONS:
        raise ValueError(orientation)
    reason = availability(sample_id, img_mask, loc)
    if reason:
        return {"agreement": None, "mismatch": None, "reason": reason}
    m = np.asarray(img_mask, bool)
    cells, _ = parse_location(loc)
    total = int(m.sum())
    fixed = list(ORIENTATIONS[:2]) if orientation == "max_of_both" else [orientation]
    agree = max(int((m & allowed_region(cells, o)).sum()) / total for o in fixed)
    return {"agreement": agree, "mismatch": 1.0 - agree, "reason": None}
