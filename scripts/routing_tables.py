"""Per-task routing accuracy from the saved confusion matrices (reviewer 6Kda).

    python scripts/routing_tables.py            # arm "base", every dataset
    python scripts/routing_tables.py entropy    # another arm

Uses the final-stage matrix of each seed: prism/<dataset>__<arm>__seed<s>/
routing_confusion_stage<T-1>.npy (rows: true task, cols: routed task).

Task t means "the t-th task presented". Class order is shuffled per seed, so
the same index is a different class set in each seed; averaging over seeds
therefore measures routing accuracy by arrival position, not by class content.
Also reports where the errors go: to an earlier task or to a later one.
"""
import glob
import os
import re
import sys

import numpy as np


def final_matrix(run_dir):
    stages = glob.glob(os.path.join(run_dir, "routing_confusion_stage*.npy"))
    if not stages:
        return None
    last = max(stages, key=lambda p: int(re.search(r"stage(\d+)", p).group(1)))
    return np.load(last)


def main(arm="base", root="prism"):
    by_ds = {}
    for d in sorted(glob.glob(os.path.join(root, f"*__{arm}__seed*"))):
        ds = os.path.basename(d).split("__")[0]
        m = final_matrix(d)
        if m is not None:
            by_ds.setdefault(ds, []).append(m)
    if not by_ds:
        raise SystemExit(f"no confusion matrices for arm {arm!r} under {root}/")

    for ds, mats in by_ds.items():
        T = min(m.shape[0] for m in mats)
        mats = [m[:T, :T] for m in mats]
        per_task = np.array([m.diagonal() / np.maximum(m.sum(1), 1) * 100 for m in mats])
        total = np.array([m.trace() / m.sum() * 100 for m in mats])
        older = np.array([np.tril(m, -1).sum() for m in mats])  # routed to an earlier task
        newer = np.array([np.triu(m, 1).sum() for m in mats])   # routed to a later task
        err = np.maximum(older + newer, 1)

        print(f"\n== {ds} | arm={arm} | seeds={len(mats)} | tasks={T}")
        print(f"overall routing acc  {total.mean():6.2f} +/- {total.std():.2f}")
        print("routing acc by task position (mean +/- std over seeds):")
        for t in range(T):
            print(f"  T{t + 1:<3d} {per_task[:, t].mean():6.2f} +/- {per_task[:, t].std():5.2f}")
        print(f"errors routed to an earlier task: {np.mean(older / err) * 100:5.1f}%   "
              f"to a later task: {np.mean(newer / err) * 100:5.1f}%")


if __name__ == "__main__":
    main(*sys.argv[1:])
