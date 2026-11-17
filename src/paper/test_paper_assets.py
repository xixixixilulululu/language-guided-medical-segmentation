"""Self-check of the EMBC paper assets (run after src/paper/make_paper_assets.py; reads the committed results only).

  fig1        panel (a) values rounded to 2 decimals equal the outline numbers (C0-C6, C0-C1, C0-C3c)
  fig2        shaded ranges rounded to 3 decimals equal 0.000-0.689, 0.117-0.841, 0.000-0.606
  table2      15 rows; source rows unique; C0 rows 2113 images / 474 groups, C6 rows 2081 / 472
  numbers     every raw_value of paper/paper_numbers.csv is the exact string in its source row and column
  figures     every text >= 7 pt; no Type 3 font in the PDFs; PDF MediaBox 251.064 x 140.4 bp and 251.064 x 144 bp
Usage:  python src/paper/test_paper_assets.py      (exit status 1 if any check fails)
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_paper_assets as A  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


B = A.build()

# ---- Fig. 1 (a) against the outline (order LViT-T / RecLMIS / DD-CMD)
OUTLINE = {"C6": [23.31, 24.40, 23.73], "C1": [20.13, 16.81, 12.35], "C3c": [11.43, 13.19, 21.76]}
for c, exp in OUTLINE.items():
    got = [round(B["fig1_data"][(m, c)]["dice"][0], 2) for m in A.MODELS]
    check(f"fig1a C0-{c} = outline", got == exp, f"{got}")
check("fig1 counts: C0-C1 2113/474, others 2081/472",
      all(B["fig1_data"][(m, c)]["n"] == (("2113", "474") if c == "C1" else ("2081", "472"))
          for m in A.MODELS for c in A.FIG1_CONDS))

# ---- Fig. 2 ranges
EXP = {"LViT-T": (0.000, 0.689), "RecLMIS": (0.117, 0.841), "DD-CMD": (0.000, 0.606)}
for m, e in EXP.items():
    got = tuple(round(x, 3) for x in B["bands"][m])
    check(f"fig2 range {m}", got == e, f"{got}")

# ---- Table II
rows = B["table2_rows"]
check("table2 has 15 rows", len(rows) == 15)
check("table2 order", [(m, s) for m, s, _ in rows] == [(m, s) for m in A.MODELS for s in A.TAB2_STRATS])
src = [r for r in A.read_rows(A.SRC_P5) if r["dataset"] == A.DATASET]
ok_unique, ok_n = True, True
for m in A.MODELS:
    for s in A.TAB2_STRATS:
        for c, n in (("C0", ("2113", "474")), ("C6", ("2081", "472"))):
            hit = [r for r in src if r["method"] == m and r["strategy"] == s and r["condition"] == c]
            ok_unique &= len(hit) == 1
            ok_n &= bool(hit) and (hit[0]["n_images"], hit[0]["n_groups"]) == n
check("table2 source: each method x strategy x condition matches exactly one row", ok_unique)
check("table2 source: C0 2113/474, C6 2081/472", ok_n)
tex = open(os.path.join(A.REPO, A.OUT_TAB2)).read()
body = [ln for ln in tex.splitlines() if ln.endswith(r"\\") and not ln.startswith("Model")]
check("table2 .tex file equals the generated text and has 15 body rows", tex == B["table2_tex"] and len(body) == 15)
check("table2 .tex: tabular only, \\hline only, % escaped",
      tex.startswith(r"\begin{tabular}") and tex.rstrip().endswith(r"\end{tabular}") and "\\begin{table}" not in tex
      and "toprule" not in tex and "multirow" not in tex and "textbf" not in tex and re.search(r"(?<!\\)%", tex) is None)

# ---- paper_numbers.csv: raw strings found in the source row and column
with open(os.path.join(A.REPO, A.OUT_NUM), newline="") as f:
    nums = list(csv.DictReader(f))
check("paper_numbers.csv equals the generated records", nums == B["numbers"], f"{len(nums)} rows")
cache = {}
bad = []
for r in nums:
    sf = r["source_file"]
    cache.setdefault(sf, A.read_rows(sf))
    rs = cache[sf]
    if sf == A.SRC_PAIR:
        hit = [x for x in rs if x["dataset"] == A.DATASET and x["method"] == r["method"]
               and x["comparison"] == r["condition_or_comparison"]]
    elif sf == A.SRC_P5:
        hit = [x for x in rs if x["dataset"] == A.DATASET and x["method"] == r["method"]
               and x["condition"] == r["condition_or_comparison"] and x["strategy"] == r["strategy"]]
    elif sf == A.SRC_MIX:
        hit = [x for x in rs if x["method"] == r["method"] and x["strategy"] == r["strategy"]
               and x["p"] == r["condition_or_comparison"].removeprefix("p=")]
    elif sf == A.SRC_CONS:
        hit = [x for x in rs if x["method"] == r["method"]]
    else:
        hit = []
    if len(hit) != 1 or hit[0][r["source_column"]] != r["raw_value"]:
        bad.append(r)
check("every raw_value is the exact source string (unique source row)", not bad, f"{len(bad)} bad")
cnt = {}
for r in nums:
    cnt[r["item"]] = cnt.get(r["item"], 0) + 1
check("number counts per item", cnt == {"fig1a": 45, "fig1b": 45, "fig2_endpoint": 18, "fig2_range": 6, "table2": 60},
      f"{cnt}")

# ---- figures: font sizes, fonts, page size
for k in ("fig1", "fig2"):
    s = A.text_sizes(B[k])
    check(f"{k}: all text >= 7 pt", min(s) >= 7, f"min {min(s)} over {len(s)} texts")
for stem, h in ((A.OUT_FIG1, 140.4), (A.OUT_FIG2, 144.0)):
    pdf = open(os.path.join(A.REPO, stem + ".pdf"), "rb").read()
    # fonts: no Type 3 anywhere; only font dictionaries (/Type /Font, not /FontDescriptor) are checked for their
    # /Subtype (marker Form XObjects also carry a /Subtype); every font descriptor must embed the font (/FontFile2)
    objs = re.findall(rb"\d+\s+0\s+obj(.*?)endobj", pdf, re.S)
    fonts = [o for o in objs if re.search(rb"/Type\s*/Font\b", o)]
    subtypes = [m.group(1).decode() for o in fonts for m in [re.search(rb"/Subtype\s*/(\w+)", o)] if m]
    descs = [o for o in objs if re.search(rb"/Type\s*/FontDescriptor\b", o)]
    check(f"{os.path.basename(stem)}.pdf: no Type 3; font subtypes in {{Type0, CIDFontType2, TrueType}}; fonts embedded",
          b"/Type3" not in pdf and len(subtypes) == len(fonts) > 0
          and set(subtypes) <= {"Type0", "CIDFontType2", "TrueType"} and {"CIDFontType2", "TrueType"} & set(subtypes)
          and len(descs) > 0 and all(b"/FontFile2" in d for d in descs),
          f"font subtypes {sorted(subtypes)}; {len(descs)} descriptors with /FontFile2: "
          f"{sum(b'/FontFile2' in d for d in descs)}")
    mb = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]", pdf)
    w_, h_ = (float(mb.group(3)) - float(mb.group(1)), float(mb.group(4)) - float(mb.group(2))) if mb else (0, 0)
    check(f"{os.path.basename(stem)}.pdf: MediaBox 251.064 x {h} bp", abs(w_ - 251.064) <= 0.01 and abs(h_ - h) <= 0.01,
          f"{w_} x {h_}")
    check(f"{os.path.basename(stem)}.png exists", os.path.getsize(os.path.join(A.REPO, stem + ".png")) > 0)

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
