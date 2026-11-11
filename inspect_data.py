"""List every CSV/Excel file under a folder and summarise its columns.
Usage:  python inspect_data.py "D:\\chanchal project\\datasets"
Writes inspect_report.txt next to this script (paste it back to Claude)."""
import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "."
out_lines = []


def log(s=""):
    print(s)
    out_lines.append(s)


def read_any(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return {f"sheet:{k}": v for k, v in pd.read_excel(path, sheet_name=None).items()}
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return {"": pd.read_csv(path, encoding=enc)}
        except UnicodeDecodeError:
            continue
        except Exception as e:  # noqa
            return {"": e}
    return {"": RuntimeError("could not decode")}


for dirpath, _, files in os.walk(root):
    if ".venv" in dirpath or "results" in dirpath:
        continue
    for f in sorted(files):
        if not f.lower().endswith((".csv", ".xlsx", ".xls")):
            continue
        path = os.path.join(dirpath, f)
        log("=" * 78)
        log(f"FILE: {path}")
        for tag, df in read_any(path).items():
            if isinstance(df, Exception):
                log(f"  ERROR: {df}")
                continue
            log(f"  {tag} rows={len(df)} cols={df.shape[1]}  missing cells={int(df.isna().sum().sum())}")
            for c in df.columns:
                s = df[c]
                info = f"unique={s.nunique()}"
                if pd.api.types.is_numeric_dtype(s):
                    info += f" min={s.min()} max={s.max()}"
                else:
                    info += f" e.g. {str(s.dropna().iloc[0])[:40]!r}" if s.notna().any() else ""
                log(f"    - {str(c)[:90]} [{s.dtype}] {info}")

path_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspect_report.txt")
with open(path_out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out_lines))
print(f"\nSaved {path_out}")