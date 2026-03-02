#!/usr/bin/env python3
"""
scripts/check_aliases.py

Checks alias CSVs in app/data/*_aliases.csv against app/data/tcodes.csv.
Reports:
 - alias rows that reference tcodes missing from tcodes.csv
 - rows with suspicious formatting (missing columns, empty tcode)
Usage:
    python scripts/check_aliases.py
"""

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "data"

TCODES_FILE = DATA / "tcodes.csv"
ALIAS_GLOB = "*_aliases.csv"

def load_tcodes(tfile):
    if not tfile.exists():
        print(f"ERROR: tcodes.csv not found at {tfile}")
        sys.exit(1)
    with tfile.open(newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        headers = next(reader, None)
        if headers is None:
            print("ERROR: tcodes.csv appears empty")
            sys.exit(1)
        # Try to find a reasonable tcode column name
        header_names = [h.strip().lower() for h in headers]
        candidates = ["tcode", "code", "transaction", "tcodes"]
        if any(c in header_names for c in candidates):
            for c in candidates:
                if c in header_names:
                    idx = header_names.index(c)
                    break
        else:
            idx = 0
        tcodes = set()
        # iterate rest
        for row in reader:
            if not row: continue
            val = row[idx].strip().upper()
            if val:
                tcodes.add(val)
    return tcodes, headers[idx]

def check_alias_file(path, tcodes):
    problems = []
    with path.open(newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        headers = next(reader, None)
        if headers is None:
            problems.append(("file_empty", 0, None))
            return problems
        # choose column indices heuristically
        header_l = [h.strip().lower() for h in headers]
        # alias assumed at 0, tcode at 1, desc at 2 (most typical)
        try:
            tcode_idx = header_l.index("tcode")
        except ValueError:
            # fallback to index 1 if possible
            tcode_idx = 1 if len(headers) > 1 else 0
        rowno = 1
        for row in reader:
            rowno += 1
            if not row or len(row) <= tcode_idx:
                problems.append(("missing_tcode_column", rowno, row))
                continue
            t = row[tcode_idx].strip().upper()
            if not t:
                problems.append(("empty_tcode", rowno, row))
                continue
            if t not in tcodes:
                problems.append(("missing_tcode_in_master", rowno, t))
    return problems

def main():
    tcodes, keycol = load_tcodes(TCODES_FILE)
    print(f"Loaded {len(tcodes)} canonical tcodes (key column: '{keycol}') from {TCODES_FILE.name}\n")

    alias_files = sorted(list(DATA.glob(ALIAS_GLOB)))
    if not alias_files:
        print("No alias files found (pattern *_aliases.csv) under app/data/")
        return

    overall_missing = {}
    for f in alias_files:
        probs = check_alias_file(f, tcodes)
        if probs:
            overall_missing[f.name] = probs

    if not overall_missing:
        print("✅ All alias tcodes exist in tcodes.csv and files look OK.")
        return

    print("❌ Problems found in alias files:\n")
    for fname, probs in overall_missing.items():
        print(f"File: {fname}  (issues: {len(probs)})")
        for kind, rowno, info in probs[:200]:
            if kind == "missing_tcode_in_master":
                print(f"  - Row {rowno}: tcode '{info}' not found in tcodes.csv")
            elif kind == "empty_tcode":
                print(f"  - Row {rowno}: empty tcode field; row contents: {info}")
            elif kind == "missing_tcode_column":
                print(f"  - Row {rowno}: missing tcode column; row contents: {info}")
            elif kind == "file_empty":
                print(f"  - File is empty")
        print()
    print("Fix the tcode values in the alias files or add the tcode to tcodes.csv.")
    print("You can run scripts/canonicalize_aliases.py --preview to automatically wrap/normalize fields (safe).")

if __name__ == "__main__":
    main()