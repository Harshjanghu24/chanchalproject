"""Print category labels and missingness for the three new AI datasets.
Usage: python peek_values.py   (reads D:\\chanchal project\\datasets)
Writes peek_report.txt next to this script. Paste it back to Claude."""
import glob
import os
import re
import sys

import pandas as pd

BASE = sys.argv[1] if len(sys.argv) > 1 else r"D:\chanchal project\datasets"
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


def find(pattern):
    hits = glob.glob(os.path.join(BASE, pattern))
    if not hits:
        log(f"!! no file matching {pattern} in {BASE}")
        return None
    return hits[0]


# ---------- Career anxiety ----------
p = find("Career_Anxiety*.xlsx")
if p:
    c = pd.read_excel(p)
    log("=" * 70 + "\nCAREER ANXIETY  rows=%d" % len(c))
    log("missing per column:\n" + c.isna().sum()[c.isna().sum() > 0].to_string())
    for col in ["career_anxiety", "ai_knowledge", "ai_replace_jobs", "ai_takeover_time",
                "ai_future_perspective", "academic_year", "gender"]:
        log(f"\n--- {col}\n" + c[col].value_counts(dropna=False).to_string())
    n_tools = c["ai_tools_used"].fillna("").apply(lambda s: len([t for t in s.split(",") if t.strip()]))
    log("\n--- number of AI tools listed\n" + n_tools.value_counts().sort_index().to_string())

# ---------- Data_set_Ex (cognitive strain / regret) ----------
p = find("Data_set_Ex.csv")
if p:
    d = pd.read_csv(p)
    log("\n" + "=" * 70 + "\nDATA_SET_EX  rows=%d" % len(d))
    for col in d.columns:
        m = re.match(r"\s*(\d+)\.", str(col))
        tag = m.group(1) if m else "-"
        first = str(col).split("\n")[0][:70]
        nm = int(d[col].isna().sum())
        extra = ""
        if pd.api.types.is_numeric_dtype(d[col]) and d[col].notna().any():
            extra = f" mean={d[col].mean():.2f}"
        log(f"item {tag:>2} | missing={nm:<4}{extra} | {first}")
    for col in d.columns[:3]:
        log(f"\n--- {str(col)[:50]}\n" + d[col].value_counts(dropna=False).head(12).to_string())

# ---------- Bangladesh cognitive dependence ----------
p = find("*AI_Cognitive_Dependence*.csv")
if p:
    b = pd.read_csv(p)
    log("\n" + "=" * 70 + "\nCOGNITIVE DEPENDENCE  rows=%d" % len(b))
    likert = [c for c in b.columns if " [" in c]
    levels = pd.unique(b[likert].values.ravel())
    log("Likert labels seen: %s" % sorted(map(str, levels)))
    groups = {}
    for c in likert:
        groups.setdefault(c.split(" [")[0], []).append(c)
    log("groups: " + ", ".join(f"{k} ({len(v)} items)" for k, v in groups.items()))
    for col in ["Gender", "Academic Level", "University Type"]:
        log(f"\n--- {col}\n" + b[col].value_counts(dropna=False).to_string())

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "peek_report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\nSaved peek_report.txt")