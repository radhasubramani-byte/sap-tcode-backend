#!/usr/bin/env python3
"""
scripts/canonicalize_aliases.py (fixed)

Normalize alias CSVs in app/data:
 - ensure header alias,tcode,canonical_desc (case-insensitive)
 - reorder columns to canonical order if needed
 - trim whitespace
 - writes preview diffs to stdout
 - when --apply is passed: create a .bak and overwrite the CSV with normalized content

Usage:
    python scripts/canonicalize_aliases.py         # preview-only (prints diffs)
    python scripts/canonicalize_aliases.py --apply # actually overwrite files (creates .bak)
"""
import csv
from pathlib import Path
import argparse
import difflib
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "data"
ALIAS_GLOB = "*_aliases.csv"
EXPECTED_HEADERS = ["alias", "tcode", "canonical_desc"]


def read_csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    return rows


def write_csv_rows(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for r in rows:
            writer.writerow(r)


def normalize_rows(rows):
    """
    Normalize CSV rows to header: alias,tcode,canonical_desc
    Returns normalized rows (including header).
    """
    if not rows:
        # create header only
        return [EXPECTED_HEADERS[:]]

    orig_header = rows[0]
    header_l = [h.strip().lower() for h in orig_header]
    # Build mapping: desired -> index in original header if present else fallback positions
    mapping = {}
    for i, want in enumerate(EXPECTED_HEADERS):
        if want in header_l:
            mapping[want] = header_l.index(want)
        else:
            # fallback to positional mapping if header is missing
            mapping[want] = i if i < len(orig_header) else None

    # compute maximum index we will access (ignore None)
    idx_values = [v for v in mapping.values() if isinstance(v, int)]
    max_idx = max(idx_values) if idx_values else 0

    out = [EXPECTED_HEADERS[:]]
    for r in rows[1:]:
        # ensure list long enough to index into
        r_ext = list(r) + [""] * (max(0, max_idx - len(r) + 1))
        alias = r_ext[mapping["alias"]].strip() if mapping["alias"] is not None else ""
        tcode = r_ext[mapping["tcode"]].strip().upper() if mapping["tcode"] is not None else ""
        desc = r_ext[mapping["canonical_desc"]].strip() if mapping["canonical_desc"] is not None else ""
        out.append([alias, tcode, desc])
    return out


def pretty_diff(orig_text, new_text, fname):
    o = orig_text.splitlines(keepends=True)
    n = new_text.splitlines(keepends=True)
    return "".join(difflib.unified_diff(o, n, fromfile=f"{fname}.orig", tofile=f"{fname}.new"))


def process_file(path: Path, apply: bool = False):
    try:
        orig_rows = read_csv_rows(path)
    except Exception as e:
        raise RuntimeError(f"failed to read {path}: {e}")

    new_rows = normalize_rows(orig_rows)

    # produce CSV text for diff
    import io

    buf1 = io.StringIO()
    w1 = csv.writer(buf1)
    for r in orig_rows:
        w1.writerow(r)
    orig_text = buf1.getvalue()

    buf2 = io.StringIO()
    w2 = csv.writer(buf2)
    for r in new_rows:
        w2.writerow(r)
    new_text = buf2.getvalue()

    diff = pretty_diff(orig_text, new_text, path.name)
    if not diff:
        print(f"[OK] {path.name}: no changes needed")
        return True

    print(f"--- Preview changes for {path.name} ---")
    print(diff)
    if apply:
        bak = path.with_suffix(path.suffix + ".bak")
        # create backup (move original)
        path.replace(bak)
        write_csv_rows(path, new_rows)
        print(f"  Applied. Original moved to: {bak.name}")
    else:
        print(f"  (preview only) To apply these changes run with --apply")
    return True


def main():
    parser = argparse.ArgumentParser(description="Normalize alias CSV files under app/data")
    parser.add_argument("--apply", action="store_true", help="Apply changes (write files). Without this it only previews.")
    args = parser.parse_args()

    files = sorted(list(DATA.glob(ALIAS_GLOB)))
    if not files:
        print("No alias CSVs found (pattern *_aliases.csv) under app/data/")
        return

    for f in files:
        try:
            process_file(f, apply=args.apply)
        except Exception as e:
            print(f"ERROR processing {f.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()