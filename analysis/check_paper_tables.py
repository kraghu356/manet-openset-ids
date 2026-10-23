"""Check that every numeric row of every generated table appears, with the same
numbers in the same order, in the manuscript source.

    python3 analysis/check_paper_tables.py path/to/body.tex
"""
import glob, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
body = open(sys.argv[1]).read().splitlines()
bad = 0
for f in sorted(glob.glob(os.path.join(ROOT, "tables", "*.tex")) + glob.glob(os.path.join(ROOT, "results", "tables", "*.tex"))):
    rows = [l for l in open(f).read().splitlines() if "&" in l and re.search(r"\d+\.\d+", l)]
    miss = [l for l in rows if not any(re.findall(r"\d+\.\d+", b)[-len(re.findall(r"\d+\.\d+", l)):] == re.findall(r"\d+\.\d+", l)
                                       for b in body if "&" in b)]
    bad += len(miss)
    print(f"{os.path.relpath(f, ROOT)}: {len(rows)} rows, {len(miss)} not found")
    for l in miss:
        print("   ", l[:100])
sys.exit(1 if bad else 0)
