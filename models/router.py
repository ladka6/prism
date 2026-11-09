"""Closed-form ridge readout over accumulated sufficient statistics.

Keeps only G = sum_s phi_s^T phi_s and C = sum_s phi_s^T Y_s. Both are sums
over samples, so the solution does not depend on the order or the chunking in
which samples arrive (see tests/test_router.py).
"""

import torch


class RidgeRouter:
    def __init__(self):
        self.G = None  # [M, M]
        self.C = None  # [M, n_out]
        self._cache = {}

    @property
    def n_out(self):
        return 0 if self.C is None else self.C.size(1)

    def accumulate(self, phi, labels, n_out):
        """Add one chunk of features. `labels` index the output columns; the
        number of columns grows to `n_out` and existing columns are untouched."""
        labels = labels.to(phi.device).long()
        onehot = torch.zeros(phi.size(0), n_out, device=phi.device)
        onehot[torch.arange(phi.size(0), device=phi.device), labels] = 1.0
        G = (phi.t() @ phi).float()
        C = (phi.t() @ onehot).float()
        if self.G is None:
            self.G, self.C = G, C
        else:
            self.G = self.G + G
            grown = torch.zeros(self.C.size(0), n_out, device=self.C.device)
            grown[:, : self.C.size(1)] = self.C
            self.C = grown + C
        self._cache.clear()

    def weight(self, lam):
        key = float(lam)
        W = self._cache.get(key)
        if W is None:
            G = self.G.double()
            C = self.C.double()
            eye = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
            W = torch.linalg.solve(G + key * eye, C).float()
            self._cache[key] = W
        return W
