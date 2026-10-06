"""
hdmr_test_functions.py

Synthetic test functions for validating HDMR / fANOVA / n-mode expansion codes.

DESIGN IDEA
-----------
Rather than making 1st/2nd order effects merely "dominate" a general function
(which requires you to trust a variance decomposition to check itself), this
builds functions where 1st and 2nd order terms are exactly the entire signal:

    f(x) = f0 + sum_i g_i(x_i) + sum_{(i,j) in E} h_ij(x_i, x_j)

Each component is constructed to vanish at a reference point x0 (default 0):
    g_i(x0) = 0,   h_ij(x0, x_j) = h_ij(x_i, x0) = 0

Consequence: a cut-HDMR decomposition anchored at x0 recovers g_i and h_ij
*exactly*, and every order-3+ cut term is mathematically zero (there's no
"leftover" -- it's not that higher orders are small, they are absent).
This gives you a ground-truth oracle (`g_true`, `h_true`) to check whatever
HDMR/fANOVA code you're testing against, instead of just eyeballing whether
higher-order terms look negligible.

PAIR DENSITY
------------
`pair_fraction` controls what fraction of all C(D,2) possible pairs carry
active 2-mode coupling:
    pair_fraction=None  -> legacy sparse mode: a dimension chain (i,i+1) plus
                            a controllable number of extra random long-range
                            pairs (`extra_pairs_per_dim`). O(D) active pairs.
    pair_fraction=f      -> a uniformly random f-fraction of all C(D,2) pairs
                            are active (f=1.0 -> every pair is coupled).
Per-pair coupling amplitude is drawn from the same distribution regardless of
how many pairs are active (amplitude is NOT renormalized by pair count), so
sweeping pair_fraction from sparse to dense is also a sweep in total
pairwise-interaction energy -- by design, so you can see how a model handles
both "few strong pairs" and "many pairs, same per-pair strength" regimes.

COMPONENT LIBRARY
------------------
1-mode g_i(x), all vanishing at 0:
    type 0: a*sin(w x)
    type 1: a*(1 - cos(w x))
    type 2: a*sin(w x)^3                       (sharper, odd)
    type 3: a*(sin(w x) + 0.4 sin(2 w x))       (two-frequency / harmonic)

2-mode h_ij(x_i, x_j), all vanishing when either argument is 0:
    type 0: b*sin(x_i)*sin(x_j)                          (separable)
    type 1: b*(1-cos(x_i))*(1-cos(x_j))                  (separable)
    type 2: b*sin(x_i)*(1-cos(x_j))                      (separable, asymmetric)
    type 3: b*(sin(x_i+x_j) - sin(x_i) - sin(x_j))       (non-separable)
    type 4: b*sin(x_i * x_j)                             (non-separable)
Types 3 and 4 cannot be written as a product of univariate functions, so they
stress an additive-kernel model's pairwise term more genuinely than a pure
product-kernel shape like type 0 can.

USAGE
-----
    f_sparse = HDMRTestFunction(D=200, seed=0)                    # legacy sparse
    f_dense  = HDMRTestFunction(D=200, seed=0, pair_fraction=1.0) # all pairs active
    f_half   = HDMRTestFunction(D=200, seed=0, pair_fraction=0.5)

    X = np.random.uniform(*f_dense.domain, size=(1000, f_dense.D))
    y = f_dense(X)                                   # (1000,) vectorized evaluation

    cuts1d = f_dense.one_d_cuts(npoints=10)           # all D 1D cuts
    cuts2d = f_dense.two_d_cuts(npoints=10)           # all C(D,2) 2D cuts (chunked)

    # ground truth check
    xs, ys = cuts1d[3]
    assert np.allclose(ys, f_dense.f0 + f_dense.g_true(3, xs))
"""

import numpy as np
import itertools


class HDMRTestFunction:
    N_G_TYPES = 4
    N_H_TYPES = 5

    def __init__(self, D=200, domain=(-2.0, 2.0), pair_fraction=None,
                 extra_pairs_per_dim=1.0, ensure_chain=True,
                 amp1=(0.5, 1.5), amp2_scale=0.2, f0=0.0, seed=0):
        """
        D                   : number of input dimensions
        domain              : (lo, hi), must contain the reference point 0
        pair_fraction       : None -> legacy sparse chain+random construction.
                               float in [0,1] -> that fraction of all C(D,2)
                               pairs is activated (1.0 = every pair coupled).
        extra_pairs_per_dim : (legacy mode only) extra random long-range pairs
                               on top of the chain; total pairs ~ (1+extra)*D
        ensure_chain        : (pair_fraction mode only) always include the
                               dimension chain (i,i+1) among the active pairs,
                               topping up the rest with a random sample, so
                               every dimension keeps at least one coupling
                               even at low fractions
        amp1                : range for 1-mode amplitudes a_i
        amp2_scale          : per-pair 2-mode amplitude is drawn as
                               Uniform(-1,1) * amp2_scale * mean(a_i),
                               independent of how many pairs are active
        f0                  : constant offset
        seed                : RNG seed, for reproducible test cases
        """
        rng = np.random.default_rng(seed)
        assert domain[0] < 0.0 < domain[1], "domain must contain the reference point x0=0"
        self.D = D
        self.domain = domain
        self.x0 = 0.0
        self.f0 = f0
        self.seed = seed
        self.pair_fraction = pair_fraction

        # ---- 1-mode terms ----
        self.g_type = rng.integers(0, self.N_G_TYPES, size=D)
        self.g_amp = rng.uniform(*amp1, size=D)
        self.g_w = rng.uniform(0.5, 2.5, size=D)
        self._g_groups = [np.where(self.g_type == t)[0] for t in range(self.N_G_TYPES)]
        np.savetxt("g_type.dat", self.g_type, fmt="% .16e")
        np.savetxt("g_amp.dat" , self.g_amp , fmt="% .16e")
        np.savetxt("g_w.dat"   , self.g_w   , fmt="% .16e")

        # ---- 2-mode coupling graph ----
        total_possible = D * (D - 1) // 2
        if pair_fraction is None:
            # legacy: chain + a controllable number of extra random pairs -> O(D) pairs
            chain = [(i, i + 1) for i in range(D - 1)]
            n_extra = int(extra_pairs_per_dim * D)
            extra = set()
            guard = 0
            while len(extra) < n_extra and guard < 50 * n_extra + 100:
                i, j = rng.integers(0, D, size=2)
                if i != j:
                    extra.add((min(int(i), int(j)), max(int(i), int(j))))
                guard += 1
            pairs = np.array(list(dict.fromkeys(chain + list(extra))), dtype=int)
        else:
            assert 0.0 <= pair_fraction <= 1.0
            target = int(round(pair_fraction * total_possible))
            all_pairs = np.array(list(itertools.combinations(range(D), 2)), dtype=int)
            if pair_fraction >= 1.0:
                pairs = all_pairs
            elif ensure_chain:
                chain_set = {(i, i + 1) for i in range(D - 1)}
                is_chain = np.array([tuple(p) in chain_set for p in all_pairs])
                chain_idx = np.where(is_chain)[0]
                rest_idx = np.where(~is_chain)[0]
                rng.shuffle(rest_idx)
                n_extra = max(target - len(chain_idx), 0)
                chosen = np.concatenate([chain_idx, rest_idx[:n_extra]])
                pairs = all_pairs[chosen]
            else:
                idx = np.arange(len(all_pairs))
                rng.shuffle(idx)
                pairs = all_pairs[idx[:target]]

        self.pairs = pairs                                # (P, 2)
        P = len(pairs)
        self._I = pairs[:, 0].copy()
        self._J = pairs[:, 1].copy()

        typical_1mode = float(np.mean(self.g_amp))
        self.h_type = rng.integers(0, self.N_H_TYPES, size=P)
        self.h_amp = rng.uniform(-1.0, 1.0, size=P) * amp2_scale * typical_1mode
        self._h_groups = [np.where(self.h_type == t)[0] for t in range(self.N_H_TYPES)]

        np.savetxt("h_type.dat", self.h_type, fmt="% .16e")
        np.savetxt("h_amp.dat" , self.h_amp , fmt="% .16e")

    # ---------------- component formulas ----------------
    @staticmethod
    def _g_formula(t, a, w, x):
        if t == 0:
            return a * np.sin(w * x)
        if t == 1:
            return a * (1.0 - np.cos(w * x))
        if t == 2:
            return a * np.sin(w * x) ** 3
        return a * (np.sin(w * x) + 0.4 * np.sin(2.0 * w * x))

    @staticmethod
    def _h_formula(t, b, xi, xj):
        if t == 0:
            return b * np.sin(xi) * np.sin(xj)
        if t == 1:
            return b * (1.0 - np.cos(xi)) * (1.0 - np.cos(xj))
        if t == 2:
            return b * np.sin(xi) * (1.0 - np.cos(xj))
        if t == 3:
            return b * (np.sin(xi + xj) - np.sin(xi) - np.sin(xj))
        return b * np.sin(xi * xj)

    # ---------------- core model (vectorized, grouped by type) ----------------
    def evaluate(self, X, active_cols=None):
        """
        X: (N, D) or (D,) -> (N,) function values.

        active_cols: optional iterable of column indices. If given, it's a
        promise that every column NOT in active_cols is exactly 0 for this
        whole batch. Since every g_i and h_ij is constructed to vanish when
        its argument(s) are 0, terms outside active_cols are provably zero
        and can be skipped exactly (no approximation) -- this is what makes
        cut evaluation cheap even when a large fraction of all C(D,2) pairs
        are active: a 2D cut only needs the O(D) pairs touching its 2 active
        dimensions, not all O(D^2) pairs. Leave as None for a fully generic
        black-box evaluation (correct for any X, just O(D + P) per point).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        N = X.shape[0]
        f = np.full(N, self.f0, dtype=float)

        # transpose once: row-gather on a contiguous (D, N) array is far
        # faster than repeated column-gather on the original (N, D) array
        XT = X.T  # (D, N)

        if active_cols is not None:
            active_mask = np.zeros(self.D, dtype=bool)
            active_mask[np.asarray(list(active_cols), dtype=int)] = True
        else:
            active_mask = None

        # ---- 1-mode terms, grouped by type ----
        for t, idx in enumerate(self._g_groups):
            if active_mask is not None:
                idx = idx[active_mask[idx]]
            if len(idx) == 0:
                continue
            xg = XT[idx]                              # (n_t, N)
            a = self.g_amp[idx][:, None]
            w = self.g_w[idx][:, None]
            vals = self._g_formula(t, a, w, xg)        # (n_t, N)
            f += vals.sum(axis=0)

        # ---- 2-mode terms, grouped by type ----
        for t, idx in enumerate(self._h_groups):
            if active_mask is not None:
                keep = active_mask[self._I[idx]] | active_mask[self._J[idx]]
                idx = idx[keep]
            if len(idx) == 0:
                continue
            xi = XT[self._I[idx]]                      # (n_t, N)
            xj = XT[self._J[idx]]                       # (n_t, N)
            b = self.h_amp[idx][:, None]
            vals = self._h_formula(t, b, xi, xj)        # (n_t, N)
            f += vals.sum(axis=0)

        return f

    __call__ = evaluate

    # ---------------- ground truth (cut-HDMR components anchored at x0=0) ----------------
    def g_true(self, dim, x):
        x = np.asarray(x, dtype=float)
        t = int(self.g_type[dim])
        return self._g_formula(t, self.g_amp[dim], self.g_w[dim], x)

    def h_true(self, i, j, xi, xj):
        i, j = (i, j) if i < j else (j, i)
        idx = np.where((self.pairs[:, 0] == i) & (self.pairs[:, 1] == j))[0]
        if len(idx) == 0:
            return np.zeros_like(np.asarray(xi, dtype=float) + np.asarray(xj, dtype=float))
        k = idx[0]
        t = int(self.h_type[k])
        return self._h_formula(t, self.h_amp[k], np.asarray(xi, dtype=float), np.asarray(xj, dtype=float))

    def pair_list(self):
        """List of (i, j) pairs that actually carry 2-mode coupling."""
        return [tuple(p) for p in self.pairs]

    # ---------------- cut generation, for feeding to / checking an HDMR routine ----------------
    def one_d_cuts(self, npoints=10):
        """All D single-dimensional cuts through x0. Returns {dim: (xs, f_vals)}."""
        lo, hi = self.domain
        xs = np.linspace(lo, hi, npoints)
        X = np.zeros((npoints * self.D, self.D))
        for d in range(self.D):
            X[d * npoints:(d + 1) * npoints, d] = xs
        f = self.evaluate(X, active_cols=range(self.D))
        return {d: (xs, f[d * npoints:(d + 1) * npoints]) for d in range(self.D)}

    def two_d_cuts(self, npoints=10, pairs=None):
        """
        2D cuts through x0 for `pairs` (default: ALL C(D,2) pairs).
        Evaluated one pair at a time using active_cols={i,j}, so cost per pair
        is O(#pairs touching i or j), not O(|E|) -- this is what keeps dense
        (pair_fraction=1.0) configurations tractable, since it means a 2D cut
        only pays for the ~O(D) pairs relevant to its own two dimensions.
        Returns {(i,j): (xs, ys, f_grid[np,np])}.
        """
        lo, hi = self.domain
        xs = np.linspace(lo, hi, npoints)
        XX, YY = np.meshgrid(xs, xs, indexing='ij')
        xi_flat, yj_flat = XX.ravel(), YY.ravel()
        npts = npoints * npoints

        if pairs is None:
            pairs = list(itertools.combinations(range(self.D), 2))

        results = {}
        X = np.zeros((npts, self.D))
        for (i, j) in pairs:
            X[:, i] = xi_flat
            X[:, j] = yj_flat
            f = self.evaluate(X, active_cols=(i, j))
            results[(i, j)] = (xs, xs, f.reshape(npoints, npoints).copy())
            X[:, i] = 0.0
            X[:, j] = 0.0
        return results


"""
Generate train.dat / test.dat from 1D and 2D grid cuts of an HDMRTestFunction.

Sampling is restricted to the union of all 1D cuts (single active dimension)
and all 2D cuts (every pair of dimensions), never the full D-dimensional
space. Inactive dimensions sit at the reference value 0.0.

Grid (per active dimension): [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  - the 0.0 entry is identical to the global reference point, so it is not
    re-emitted per cut -- it appears exactly once, as the header line.

Train/test split: a grid point is TRAIN iff every one of its active,
nonzero coordinates lies in the sparse grid
  [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]   (i.e. the 7-point grid minus 0.0)
Otherwise it's TEST. For a 2D point this means: if either coordinate is one
of the 4 "dense-only" values [-0.8, -0.4, 0.4, 0.8], the whole point is TEST.

File format:
    line 1:                 D zeros (space separated)   reference_y
    every subsequent line:  dims (space sep) : displacements (space sep) : y
    (":" is the deliberate field separator)
"""

import time

# ---------------- config ----------------
D = 500
PAIR_FRACTION = 1.0
SEED = 0
GRID = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TRAIN_GRID = {-1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0}
CHUNK_PAIRS = 200   # pairs processed per vectorized evaluate() call

NZ = [v for v in GRID if v != 0.0]                 # 10 nonzero grid values
TRAIN_NZ = set(v for v in NZ if v in TRAIN_GRID)    # 6 nonzero train values

f = HDMRTestFunction(D=D, seed=SEED, pair_fraction=PAIR_FRACTION)

t_start = time.time()

train_f = open("train.dat", "w")
test_f = open("test.dat", "w")

# ---- header: reference point, in both files ----
ref_y = float(f.evaluate(np.zeros((1, D)))[0])
header = " ".join(["0.0"] * D) + f" {ref_y:.10g}\n"
train_f.write(header)
test_f.write(header)

n_train, n_test = 0, 0

# ---------------- 1D cuts: all D dimensions ----------------
X = np.zeros((D * len(NZ), D))
for k, d in enumerate(range(D)):
    X[k * len(NZ):(k + 1) * len(NZ), d] = NZ
y = f.evaluate(X)
for k, d in enumerate(range(D)):
    for m, v in enumerate(NZ):
        val = y[k * len(NZ) + m]
        line = f"{d} : {v:.1f} : {val:.10g}\n"
        if v in TRAIN_NZ:
            train_f.write(line); n_train += 1
        else:
            test_f.write(line); n_test += 1

t_1d = time.time()
print(f"1D cuts done in {t_1d - t_start:.2f}s  (train={n_train}, test={n_test})")

# ---------------- 2D cuts: every pair of dimensions ----------------
pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
n_nz = len(NZ)
VV = np.array(NZ)
XX, YY = np.meshgrid(VV, VV, indexing='ij')   # (10,10) each, all-nonzero grid
xi_flat, yj_flat = XX.ravel(), YY.ravel()      # (100,)
npts = n_nz * n_nz
is_train_flat = np.array([(vi in TRAIN_NZ) and (vj in TRAIN_NZ)
                           for vi, vj in zip(xi_flat, yj_flat)])

n2_train, n2_test = 0, 0
for start in range(0, len(pairs), CHUNK_PAIRS):
    batch = np.array(pairs[start:start + CHUNK_PAIRS], dtype=int)
    nb = batch.shape[0]
    X = np.zeros((nb * npts, D))
    rows = np.arange(nb * npts)
    col_i = np.repeat(batch[:, 0], npts)
    col_j = np.repeat(batch[:, 1], npts)
    X[rows, col_i] = np.tile(xi_flat, nb)
    X[rows, col_j] = np.tile(yj_flat, nb)
    y = f.evaluate(X).reshape(nb, npts)

    for k in range(nb):
        i, j = int(batch[k, 0]), int(batch[k, 1])
        yk = y[k]
        for m in range(npts):
            line = f"{i} {j} : {xi_flat[m]:.1f} {yj_flat[m]:.1f} : {yk[m]:.10g}\n"
            if is_train_flat[m]:
                train_f.write(line); n2_train += 1
            else:
                test_f.write(line); n2_test += 1

t_2d = time.time()
print(f"2D cuts done in {t_2d - t_1d:.2f}s  (train={n2_train}, test={n2_test})")

train_f.close()
test_f.close()

print(f"TOTAL train lines (incl. header): {1 + n_train + n2_train}")
print(f"TOTAL test lines (incl. header):  {1 + n_test + n2_test}")
print(f"Total time: {time.time() - t_start:.2f}s")


