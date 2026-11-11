"""Audit all CSV datasets in a directory.

Usage: python audit_datasets.py "<directory>"
"""
import sys
import os
import pandas as pd
import numpy as np


def audit(directory):
    lines = []
    files = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
    if not files:
        lines.append("No CSV files found in " + directory)
        return "\n".join(lines)

    for f in sorted(files):
        path = os.path.join(directory, f)
        lines.append("=" * 70)
        lines.append(f"FILE: {f}")
        lines.append("=" * 70)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            lines.append(f"ERROR reading file: {type(e).__name__}: {e}")
            continue

        lines.append(f"Rows: {len(df)}")
        lines.append(f"Columns: {len(df.columns)}")
        lines.append("")
        lines.append("Column list:")
        lines.append("  " + ", ".join(map(str, df.columns)))
        lines.append("")
        lines.append("Dtypes:")
        lines.append(str(df.dtypes))
        lines.append("")
        lines.append("Missingness (isna count / %):")
        miss = df.isna().sum()
        miss_pct = (df.isna().mean() * 100).round(2)
        md = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
        lines.append(str(md))
        lines.append("")
        lines.append("Zero-value counts (relevant: 0 used as N/A code on some scales):")
        zero_counts = (df.select_dtypes(include=[np.number]) == 0).sum()
        lines.append(str(zero_counts))
        lines.append("")
        lines.append("First 5 rows:")
        lines.append(str(df.head(5)))
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    report = audit(directory)
    print(report)
