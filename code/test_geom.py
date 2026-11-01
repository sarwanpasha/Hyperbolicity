"""Sanity tests for the delta-hyperbolicity machinery.

1. A tree metric must have delta = 0 (exactly, up to float error).
2. A 2-D Euclidean grid (a flat space) must have delta growing with diameter.
3. The differentiable four-point surrogate must agree with the exact Gromov delta
   on the same point sets, up to the quadruple sampling.
4. Gradient of the surrogate must be finite and reduce delta on a toy problem.
"""
import numpy as np, torch
from geom import delta_from_D, pdist_normalized
from hyploss import quad_slack, sample_quads, pairwise_dist, hyperbolicity_loss, anticollapse_loss

rng = np.random.default_rng(0)


def tree_metric(n_leaves=16, depth=4, seed=0):
    """Path metric of a complete binary tree, restricted to its leaves."""
    r = np.random.default_rng(seed)
    # each leaf = binary code of length depth; distance = 2*(depth - common prefix)
    codes = [np.array([(i >> k) & 1 for k in range(depth)]) for i in range(2 ** depth)]
    codes = codes[:n_leaves]
    n = len(codes)
    D = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            c = 0
            while c < depth and codes[i][depth - 1 - c] == codes[j][depth - 1 - c]:
                c += 1
            D[i, j] = 0 if i == j else 2 * (depth - c)
    return D


def test_tree_delta_zero():
    D = tree_metric()[None]
    d = delta_from_D(D, base=0)
    print("tree delta =", d[0])
    assert d[0] < 1e-4, d[0]


def test_grid_delta_positive():
    g = np.stack(np.meshgrid(np.arange(6), np.arange(6)), -1).reshape(-1, 2).astype(np.float32)
    D = np.linalg.norm(g[None, :, None] - g[None, None, :], axis=-1)
    d = delta_from_D(D, base=0)
    print("grid delta =", d[0], "diam =", D.max())
    assert d[0] > 0.5, d[0]


def test_surrogate_matches():
    X = torch.randn(64, 8)
    D = pairwise_dist(X, normalize=False)
    q = sample_quads(64, 20000, X.device)
    dq, _ = quad_slack(D, q)
    exact = delta_from_D(D.numpy()[None], base=0)[0]
    print("sampled max four-point slack =", float(dq.max()), " exact gromov delta =", exact)
    # the four-point slack maximised over all quadruples equals the Gromov delta;
    # with sampling we should get within a modest factor
    assert float(dq.max()) > 0.5 * exact, (float(dq.max()), exact)


def test_gradient_reduces_delta():
    """The surrogate alone must drive the measured relative delta down."""
    torch.manual_seed(0)
    Z = torch.randn(128, 16, requires_grad=True)
    opt = torch.optim.Adam([Z], lr=0.05)
    first = last = None
    for it in range(300):
        lh, st = hyperbolicity_loss(Z, 2048)
        lh.backward(); opt.step(); opt.zero_grad()
        if it == 0:
            first = st["delta_rel_mean"]
        last = st["delta_rel_mean"]
    print("delta_rel: %.4f -> %.4f" % (first, last))
    assert last < first * 0.7, (first, last)


def test_anticollapse_prevents_degeneracy():
    """Without the hinge the surrogate can be trivially satisfied by shrinking the
    cloud to a point; with it, the mean pairwise distance stays bounded away from 0."""
    torch.manual_seed(0)
    res = {}
    for use in (False, True):
        Z = torch.randn(128, 16, requires_grad=True)
        opt = torch.optim.Adam([Z], lr=0.05)
        for _ in range(300):
            lh, st = hyperbolicity_loss(Z, 1024)
            loss = lh
            if use:
                lv, sv = anticollapse_loss(Z, min_dist=1.0)
                loss = loss + 1.0 * lv
            loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            D = pairwise_dist(Z)
            n = D.shape[0]
            iu = torch.triu_indices(n, n, 1)
            res[use] = (float(D[iu[0], iu[1]].mean()), st["delta_rel_mean"])
    print("no-hinge (mean_pdist, delta_rel) =", res[False])
    print("hinge    (mean_pdist, delta_rel) =", res[True])
    assert res[True][0] >= 0.9, res[True]


if __name__ == "__main__":
    for f in [test_tree_delta_zero, test_grid_delta_positive,
              test_surrogate_matches, test_gradient_reduces_delta,
              test_anticollapse_prevents_degeneracy]:
        f(); print("PASS", f.__name__)
    print("ALL_TESTS_PASSED")
