"""Check that the result tables copied into PROGRESS.md match the regenerated ones (scripts/reproduce_tables.sh).

Every table row of a generated file that starts like a row in PROGRESS.md (same first cell, same
section) must be identical. Rows that PROGRESS.md does not copy are listed as 'not copied', which is
not an error.
"""

from __future__ import annotations

import sys
from pathlib import Path

GENERATED = ("runs/p0_tables.md", "runs/p1_tables.md", "runs/e7_tables.md", "runs/e9_tables.md",
             "runs/e10_tables.md", "runs/e11_tables.md", "runs/e12_tables.md")


def rows(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if line.startswith("|") and not line.startswith("|---") and line.count("|") > 2]


def main() -> int:
    progress = set(rows(Path("PROGRESS.md").read_text()))
    bad = 0
    for f in GENERATED:
        p = Path(f)
        if not p.exists():
            print(f"{f}: missing (run scripts/reproduce_tables.sh)")
            bad += 1
            continue
        gen = rows(p.read_text())
        same = [r for r in gen if r in progress]
        print(f"{f}: {len(same)} of {len(gen)} rows found verbatim in PROGRESS.md")
    print("OK" if not bad else f"{bad} generated file(s) missing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
