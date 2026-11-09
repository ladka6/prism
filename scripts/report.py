"""Summarise rebuttal runs: mean +/- std over seeds, per (dataset, arm).

    python scripts/report.py            # reads logs/prism/<dataset>/...

Reports A_T, A-bar, routing accuracy, accuracy | correct routing, eval ms/sample,
router fit time and router peak memory. Takes the last occurrence of each
marker per log, so reruns appended to the same file are handled.
"""
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict

NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
PAT = {
    "A_T": r"CNN top1 curve: \[(.*?)\]",
    "routing": r"Routing acc curve: \[(.*?)\]",
    "cond": r"Acc \| correct routing curve: \[(.*?)\]",
    "ms": r"Eval ms/sample curve: \[(.*?)\]",
    "fit_s": r"Router fit time \(s\) curve: \[(.*?)\]",
    "peak_mb": r"Router peak mem \(MB\) curve: \[(.*?)\]",
}


def nums(s):
    s = re.sub(r"np\.float64\(([^)]+)\)", r"\1", s)
    return [float(x) for x in re.findall(NUM, s)]


def parse(path):
    text = open(path).read()
    out = {}
    for k, p in PAT.items():
        hits = [nums(h) for h in re.findall(p, text)]
        hits = [h for h in hits if h]
        if hits:
            longest = max(len(h) for h in hits)
            out[k] = [h for h in hits if len(h) == longest][-1]
    return out


def ms(v):
    return (st.mean(v), st.pstdev(v)) if v else (float("nan"), float("nan"))


def main(root="logs/prism"):
    groups = defaultdict(list)
    for p in glob.glob(f"{root}/*/*/*/*.log"):
        ds = p.split("/")[2]
        m = re.match(r"(.*)_(\d{4})_", os.path.basename(p))
        if not m:
            continue
        groups[(ds, m.group(1).strip() or "(default)")].append(parse(p))

    hdr = f'{"dataset":10s} {"arm":10s} {"n":>2s} {"A_T":>13s} {"A_bar":>13s} {"routing":>13s} ' \
          f'{"acc|ok":>13s} {"ms/smp":>11s} {"fit s (max)":>12s} {"peak MB":>9s}'
    print(hdr)
    for (ds, arm), runs in sorted(groups.items()):
        runs = [r for r in runs if "A_T" in r]
        if not runs:
            continue
        col = lambda k, f: [f(r[k]) for r in runs if k in r]
        cells = [
            ms(col("A_T", lambda c: c[-1])),
            ms(col("A_T", lambda c: sum(c) / len(c))),
            ms(col("routing", lambda c: c[-1])),
            ms(col("cond", lambda c: c[-1])),
            ms(col("ms", lambda c: c[-1])),
            ms(col("fit_s", max)),
            ms(col("peak_mb", max)),
        ]
        f = lambda c: f"{c[0]:6.2f}±{c[1]:4.2f}"
        print(f'{ds:10s} {arm:10s} {len(runs):2d} {f(cells[0]):>13s} {f(cells[1]):>13s} '
              f'{f(cells[2]):>13s} {f(cells[3]):>13s} {cells[4][0]:11.2f} {cells[5][0]:12.2f} {cells[6][0]:9.0f}')


if __name__ == "__main__":
    main(*sys.argv[1:])
