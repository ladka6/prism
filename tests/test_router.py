"""Numerical check of the order-independence proposition (paper Sec. 4.3).

For a fixed feature map, the ridge solution accumulated task by task equals the
joint solution computed with simultaneous access to all data, for any task
order and any chunking. Run: python tests/test_router.py
"""
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.router import RidgeRouter  # noqa: E402


def joint_solution(phi, labels, n_out, lam):
    phi, onehot = phi.double(), torch.zeros(phi.size(0), n_out, dtype=torch.double)
    onehot[torch.arange(phi.size(0)), labels] = 1.0
    G = phi.t() @ phi
    return torch.linalg.solve(G + lam * torch.eye(G.size(0), dtype=torch.double), phi.t() @ onehot)


def sequential(phi, labels, tasks, order, lam, chunk=None):
    r = RidgeRouter()
    seen = 0
    for t in order:
        idx = (tasks == t).nonzero(as_tuple=True)[0]
        seen = max(seen, int(labels[idx].max()) + 1)
        step = chunk or len(idx)
        for i in range(0, len(idx), step):
            r.accumulate(phi[idx[i : i + step]], labels[idx[i : i + step]], seen)
    return r.weight(lam)


def main():
    torch.manual_seed(0)
    M, n_tasks, per_task, lam = 128, 4, 5, 10.0
    n = 200 * n_tasks
    tasks = torch.arange(n) // 200
    labels = tasks * per_task + torch.randint(0, per_task, (n,))
    phi = torch.relu(torch.randn(n, M))
    n_out = n_tasks * per_task

    # Evaluated on a fixed set of query points, because columns are appended in
    # task order, so only the column *set* is comparable across orders.
    ref = joint_solution(phi, labels, n_out, lam).float()
    worst = 0.0
    for order in itertools.permutations(range(n_tasks)):
        # Re-label so that the order in which columns appear matches the order
        # in which tasks arrive; compare after un-permuting columns.
        remap = torch.empty(n_out, dtype=torch.long)
        for pos, t in enumerate(order):
            remap[t * per_task : (t + 1) * per_task] = torch.arange(pos * per_task, (pos + 1) * per_task)
        W = sequential(phi, remap[labels], tasks, order, lam, chunk=37)
        W_back = W[:, remap]  # column for original class c is column remap[c]
        err = ((W_back - ref).abs().max() / ref.abs().max()).item()
        worst = max(worst, err)
    print(f"max relative deviation over all {len(list(itertools.permutations(range(n_tasks))))} "
          f"task orders (float32 G): {worst:.2e}")
    assert worst < 1e-3, worst
    print("OK: sequential ridge == joint ridge for every task order and chunking")


if __name__ == "__main__":
    main()
