#!/usr/bin/env python3
"""Generate scikit-learn parity fixtures for pca-web.

Fits sklearn PCA / IncrementalPCA across a curated matrix of shapes, solvers,
n_components modes, dtypes, whitening and edge cases, and serializes inputs
plus every fitted attribute / method output to compact binary fixtures.

Re-run with:  npm run fixtures   (or  .venv/bin/python python/generate_fixtures.py)

Output layout (per suite: pca, ipca, rng, numeric):
  fixtures/<suite>/manifest.json  — case descriptors; array entries reference data.bin
  fixtures/<suite>/data.bin       — concatenated little-endian raw arrays
"""

import json
import os
import platform
from datetime import date

import numpy as np
import scipy
import sklearn
from scipy import linalg
from sklearn.decomposition import PCA, IncrementalPCA

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures")


class SuiteWriter:
    def __init__(self, name):
        self.name = name
        self.cases = []
        self.blob = bytearray()
        self._dedup = {}

    def array(self, arr):
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float64 or arr.dtype == np.float32:
            pass
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float64)
        else:
            raise TypeError(f"unsupported dtype {arr.dtype}")
        raw = arr.tobytes()  # C-order, little-endian on all supported platforms
        key = (str(arr.dtype), arr.shape, hash(raw))
        off = self._dedup.get(key)
        if off is None or bytes(self.blob[off : off + len(raw)]) != raw:
            off = len(self.blob)
            self.blob += raw
            self._dedup[key] = off
        return {"dtype": str(arr.dtype), "shape": list(arr.shape), "offset": off}

    @staticmethod
    def scalar(x):
        if isinstance(x, (np.floating, float)):
            x = float(x)
            if np.isnan(x):
                return "nan"
            if np.isinf(x):
                return "inf" if x > 0 else "-inf"
            return x
        if isinstance(x, (np.integer, int)):
            return int(x)
        return x

    def case(self, case_id, params, arrays, scalars, flags=None):
        assert all(c["id"] != case_id for c in self.cases), f"duplicate id {case_id}"
        self.cases.append(
            {
                "id": case_id,
                "params": params,
                "arrays": arrays,
                "scalars": scalars,
                "flags": flags or {},
            }
        )

    def write(self):
        suite_dir = os.path.join(OUT_ROOT, self.name)
        os.makedirs(suite_dir, exist_ok=True)
        manifest = {
            "meta": {
                "suite": self.name,
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "python": platform.python_version(),
                "generated": str(date.today()),
                "n_cases": len(self.cases),
            },
            "cases": self.cases,
        }
        with open(os.path.join(suite_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, separators=(",", ":"))
        with open(os.path.join(suite_dir, "data.bin"), "wb") as f:
            f.write(bytes(self.blob))
        print(f"  {self.name}: {len(self.cases)} cases, {len(self.blob) / 1e6:.2f} MB")


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------


def lowrank(seed, m, n, rank, decay=0.7, noise=0.01, mean_scale=3.0):
    """Low-rank + noise data with a decaying spectrum and shifted means."""
    rng = np.random.RandomState(seed)
    u = rng.standard_normal((m, rank))
    v = rng.standard_normal((rank, n))
    sv = 10.0 * decay ** np.arange(rank)
    x = (u * sv) @ v + noise * rng.standard_normal((m, n))
    x += mean_scale * rng.uniform(-1, 1, n)
    return x


def gaussian(seed, m, n, scale=1.0, mean_scale=0.0):
    rng = np.random.RandomState(seed)
    x = scale * rng.standard_normal((m, n))
    if mean_scale:
        x += mean_scale * rng.uniform(-1, 1, n)
    return x


def structured_edge(seed, m, n):
    """Data with a constant column, an all-zero column, and a duplicated column."""
    rng = np.random.RandomState(seed)
    x = rng.standard_normal((m, n))
    x[:, 0] = 3.25          # constant feature
    x[:, 1] = 0.0           # zero feature
    if n >= 4:
        x[:, 3] = x[:, 2]   # duplicated feature (rank deficiency)
    return x


# ---------------------------------------------------------------------------
# PCA suite
# ---------------------------------------------------------------------------


def precision_comparable(est):
    """False when get_precision/score are numerically meaningless: the model
    covariance is singular (or so close that inv() output is roundoff noise
    that no reimplementation could match)."""
    n_features = est.components_.shape[1]
    ev0 = float(est.explained_variance_[0]) if est.n_components_ > 0 else 0.0
    nv = float(est.noise_variance_)
    if nv > 0:
        # Woodbury path divides by nv^2 — meaningless at roundoff scale.
        return nv > 1e-12 * max(ev0, 1e-300)
    if est.n_components_ < n_features:
        return False  # rank(cov) < n_features, inv() is singular
    try:
        with np.errstate(all="ignore"):
            c = np.linalg.cond(est.get_covariance())
        return bool(np.isfinite(c) and c < 1e10)
    except linalg.LinAlgError:
        return False


def run_pca_case(w, case_id, X, X_test, params, flags=None):
    est = PCA(**params)
    Xt_fit_transform = est.fit_transform(X.copy())

    arrays = {
        "X": w.array(X),
        "X_test": w.array(X_test),
        "components": w.array(est.components_),
        "explained_variance": w.array(est.explained_variance_),
        "explained_variance_ratio": w.array(est.explained_variance_ratio_),
        "singular_values": w.array(est.singular_values_),
        "mean": w.array(est.mean_),
        "fit_transform": w.array(Xt_fit_transform),
        "transform_train": w.array(est.transform(X)),
        "transform_test": w.array(est.transform(X_test)),
    }
    if est.n_components_ > 0:
        # sklearn's inverse_transform rejects 0-feature input.
        arrays["inverse_transform_test"] = w.array(est.inverse_transform(est.transform(X_test)))
    flags = dict(flags or {})
    # Record the solver sklearn actually dispatched to (resolves 'auto').
    flags["expected_solver"] = est._fit_svd_solver
    # n_features×n_features outputs get bulky and exercise no new code path
    # on large cases; capture them only for moderate feature counts.
    skip_covariance = X.shape[1] > 150
    flags["skip_covariance"] = skip_covariance
    if not skip_covariance:
        arrays["get_covariance"] = w.array(est.get_covariance())
    skip_precision = skip_covariance or not precision_comparable(est)
    flags["skip_precision"] = skip_precision
    scalars = {
        "n_components_": w.scalar(est.n_components_),
        "n_samples_": w.scalar(est.n_samples_),
        "n_features_in_": w.scalar(est.n_features_in_),
        "noise_variance_": w.scalar(est.noise_variance_),
    }
    if not skip_precision:
        # numpy's slogdet emits spurious overflow warnings while returning a
        # correct finite result; silence them for the capture.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            arrays["get_precision"] = w.array(est.get_precision())
            arrays["score_samples_test"] = w.array(est.score_samples(X_test))
            scalars["score_test"] = w.scalar(est.score(X_test))
    scalars["feature_names_out"] = list(est.get_feature_names_out())

    # JSON-encode params exactly as given (None stays null).
    jparams = {k: w.scalar(v) if not isinstance(v, str) else v for k, v in params.items()}
    w.case(case_id, jparams, arrays, scalars, flags)
    return est


def pca_suite():
    w = SuiteWriter("pca")

    datasets = {
        # id: (X, X_test)
        "tall20x5": (lowrank(1, 20, 5, 4, noise=0.05), lowrank(101, 8, 5, 4, noise=0.05)),
        "wide5x20": (lowrank(2, 5, 20, 4, noise=0.05), lowrank(102, 6, 20, 4, noise=0.05)),
        "square10": (gaussian(3, 10, 10, mean_scale=2.0), gaussian(103, 7, 10)),
        "tall100x30": (lowrank(4, 100, 30, 12, noise=0.1), lowrank(104, 40, 30, 12, noise=0.1)),
        "wide30x100": (lowrank(5, 30, 100, 12, noise=0.1), lowrank(105, 20, 100, 12, noise=0.1)),
        "gauss50x40": (gaussian(6, 50, 40, mean_scale=1.0), gaussian(106, 25, 40)),
        "edge40x8": (structured_edge(7, 40, 8), structured_edge(107, 15, 8)),
        "tiny3x7": (gaussian(8, 3, 7, mean_scale=1.0), gaussian(108, 4, 7)),
        "tiny7x3": (gaussian(9, 7, 3, mean_scale=1.0), gaussian(109, 4, 3)),
        "tiny2x2": (gaussian(10, 2, 2), gaussian(110, 3, 2)),
        "mle200x30": (lowrank(11, 200, 30, 5, decay=0.9, noise=0.02), lowrank(111, 50, 30, 5, noise=0.02)),
        "cov300x25": (lowrank(12, 300, 25, 10, noise=0.05), lowrank(112, 60, 25, 10, noise=0.05)),
        "big500x80": (lowrank(13, 500, 80, 30, decay=0.85, noise=0.1), lowrank(113, 100, 80, 30, noise=0.1)),
        "big80x500": (lowrank(14, 80, 500, 30, decay=0.85, noise=0.1), lowrank(114, 40, 500, 30, noise=0.1)),
        "wide80x600": (lowrank(17, 80, 600, 25, decay=0.85, noise=0.1), lowrank(117, 30, 600, 25, noise=0.1)),
        "rand1500x300": (
            lowrank(15, 1500, 300, 40, decay=0.9, noise=0.05),
            lowrank(115, 100, 300, 40, noise=0.05),
        ),
        "scaled": (1e6 * lowrank(16, 40, 12, 5, noise=0.01), 1e6 * lowrank(116, 15, 12, 5, noise=0.01)),
    }

    # --- full solver ------------------------------------------------------
    for nc in [None, 0, 1, 3, 5]:
        run_pca_case(
            w, f"full_tall20x5_nc{nc}", *datasets["tall20x5"],
            {"n_components": nc, "svd_solver": "full"},
        )
    run_pca_case(w, "full_tall20x5_nc3_whiten", *datasets["tall20x5"],
                 {"n_components": 3, "svd_solver": "full", "whiten": True})
    run_pca_case(w, "full_wide5x20_ncNone", *datasets["wide5x20"],
                 {"n_components": None, "svd_solver": "full"})
    run_pca_case(w, "full_wide5x20_nc5_allcomp", *datasets["wide5x20"],
                 {"n_components": 5, "svd_solver": "full"})
    run_pca_case(w, "full_wide5x20_nc4_whiten", *datasets["wide5x20"],
                 {"n_components": 4, "svd_solver": "full", "whiten": True})
    run_pca_case(w, "full_square10_ncNone", *datasets["square10"],
                 {"n_components": None, "svd_solver": "full"})
    run_pca_case(w, "full_tall100x30_nc10", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "full"})
    run_pca_case(w, "full_tall100x30_nc10_whiten", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "full", "whiten": True})
    run_pca_case(w, "full_wide30x100_nc12", *datasets["wide30x100"],
                 {"n_components": 12, "svd_solver": "full"})
    run_pca_case(w, "full_gauss50x40_nc40_all", *datasets["gauss50x40"],
                 {"n_components": 40, "svd_solver": "full"})
    # variance-ratio (float) n_components
    for frac in [0.5, 0.8, 0.95, 0.999]:
        run_pca_case(w, f"full_tall100x30_frac{frac}", *datasets["tall100x30"],
                     {"n_components": frac, "svd_solver": "full"})
    run_pca_case(w, "full_edge40x8_frac0.99", *datasets["edge40x8"],
                 {"n_components": 0.99, "svd_solver": "full"})
    # Minka MLE
    run_pca_case(w, "full_mle200x30", *datasets["mle200x30"],
                 {"n_components": "mle", "svd_solver": "full"})
    run_pca_case(w, "full_mle_square10", *datasets["square10"],
                 {"n_components": "mle", "svd_solver": "full"})
    run_pca_case(w, "full_mle_tall20x5", *datasets["tall20x5"],
                 {"n_components": "mle", "svd_solver": "full"})
    # edge data
    run_pca_case(w, "full_edge40x8_ncNone", *datasets["edge40x8"],
                 {"n_components": None, "svd_solver": "full"})
    run_pca_case(w, "full_edge40x8_nc8_whiten", *datasets["edge40x8"],
                 {"n_components": 8, "svd_solver": "full", "whiten": True})
    run_pca_case(w, "full_tiny3x7_ncNone", *datasets["tiny3x7"],
                 {"n_components": None, "svd_solver": "full"})
    run_pca_case(w, "full_tiny7x3_ncNone", *datasets["tiny7x3"],
                 {"n_components": None, "svd_solver": "full"})
    run_pca_case(w, "full_tiny2x2_ncNone", *datasets["tiny2x2"],
                 {"n_components": None, "svd_solver": "full"})
    run_pca_case(w, "full_scaled_nc5", *datasets["scaled"],
                 {"n_components": 5, "svd_solver": "full"})

    # --- covariance_eigh ---------------------------------------------------
    run_pca_case(w, "cov_tall20x5_ncNone", *datasets["tall20x5"],
                 {"n_components": None, "svd_solver": "covariance_eigh"})
    run_pca_case(w, "cov_tall100x30_nc10", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "covariance_eigh"})
    run_pca_case(w, "cov_tall100x30_nc10_whiten", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "covariance_eigh", "whiten": True})
    run_pca_case(w, "cov_cov300x25_ncNone", *datasets["cov300x25"],
                 {"n_components": None, "svd_solver": "covariance_eigh"})
    run_pca_case(w, "cov_cov300x25_frac0.9", *datasets["cov300x25"],
                 {"n_components": 0.9, "svd_solver": "covariance_eigh"})
    run_pca_case(w, "cov_cov300x25_mle", *datasets["cov300x25"],
                 {"n_components": "mle", "svd_solver": "covariance_eigh"})
    run_pca_case(w, "cov_edge40x8_ncNone", *datasets["edge40x8"],
                 {"n_components": None, "svd_solver": "covariance_eigh"})
    run_pca_case(w, "cov_big500x80_nc20", *datasets["big500x80"],
                 {"n_components": 20, "svd_solver": "covariance_eigh"})
    run_pca_case(w, "cov_wide5x20_nc3", *datasets["wide5x20"],
                 {"n_components": 3, "svd_solver": "covariance_eigh"})
    run_pca_case(w, "cov_scaled_nc5", *datasets["scaled"],
                 {"n_components": 5, "svd_solver": "covariance_eigh"})

    # --- arpack -------------------------------------------------------------
    run_pca_case(w, "arpack_tall20x5_nc1", *datasets["tall20x5"],
                 {"n_components": 1, "svd_solver": "arpack", "random_state": 42})
    run_pca_case(w, "arpack_tall20x5_nc4", *datasets["tall20x5"],
                 {"n_components": 4, "svd_solver": "arpack", "random_state": 42})
    run_pca_case(w, "arpack_wide5x20_nc4", *datasets["wide5x20"],
                 {"n_components": 4, "svd_solver": "arpack", "random_state": 0})
    run_pca_case(w, "arpack_tall100x30_nc10", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "arpack", "random_state": 42})
    run_pca_case(w, "arpack_tall100x30_nc10_whiten", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "arpack", "whiten": True, "random_state": 42})
    run_pca_case(w, "arpack_wide30x100_nc12", *datasets["wide30x100"],
                 {"n_components": 12, "svd_solver": "arpack", "random_state": 7})
    run_pca_case(w, "arpack_edge40x8_nc5", *datasets["edge40x8"],
                 {"n_components": 5, "svd_solver": "arpack", "random_state": 42},
                 flags={"rank_deficient": True})
    run_pca_case(w, "arpack_big80x500_nc25", *datasets["big80x500"],
                 {"n_components": 25, "svd_solver": "arpack", "random_state": 42})
    run_pca_case(w, "arpack_square10_nc9", *datasets["square10"],
                 {"n_components": 9, "svd_solver": "arpack", "random_state": 42})

    # --- randomized ----------------------------------------------------------
    run_pca_case(w, "rand_tall100x30_nc10_s42", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized", "random_state": 42})
    run_pca_case(w, "rand_tall100x30_nc10_s0", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized", "random_state": 0})
    run_pca_case(w, "rand_tall100x30_nc10_whiten", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized", "whiten": True, "random_state": 42})
    run_pca_case(w, "rand_wide30x100_nc12", *datasets["wide30x100"],
                 {"n_components": 12, "svd_solver": "randomized", "random_state": 42})
    run_pca_case(w, "rand_tall100x30_nc10_power0", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized", "iterated_power": 0,
                  "random_state": 42})
    run_pca_case(w, "rand_tall100x30_nc10_power2", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized", "iterated_power": 2,
                  "random_state": 42})
    run_pca_case(w, "rand_tall100x30_nc10_power7", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized", "iterated_power": 7,
                  "random_state": 42})
    run_pca_case(w, "rand_tall100x30_nc10_QR", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized",
                  "power_iteration_normalizer": "QR", "random_state": 42})
    run_pca_case(w, "rand_tall100x30_nc10_LU", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized",
                  "power_iteration_normalizer": "LU", "random_state": 42})
    run_pca_case(w, "rand_tall100x30_nc10_none", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized",
                  "power_iteration_normalizer": "none", "random_state": 42})
    run_pca_case(w, "rand_tall100x30_nc10_over2", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized", "n_oversamples": 2,
                  "random_state": 42})
    run_pca_case(w, "rand_tall100x30_nc10_over25", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "randomized", "n_oversamples": 25,
                  "random_state": 42})
    run_pca_case(w, "rand_gauss50x40_nc40_all", *datasets["gauss50x40"],
                 {"n_components": 40, "svd_solver": "randomized", "random_state": 42})
    run_pca_case(w, "rand_tall20x5_nc2", *datasets["tall20x5"],
                 {"n_components": 2, "svd_solver": "randomized", "random_state": 1234})
    run_pca_case(w, "rand_big500x80_nc10", *datasets["big500x80"],
                 {"n_components": 10, "svd_solver": "randomized", "random_state": 42})
    run_pca_case(w, "rand_big80x500_nc10", *datasets["big80x500"],
                 {"n_components": 10, "svd_solver": "randomized", "random_state": 42})
    run_pca_case(w, "rand_1500x300_nc20", *datasets["rand1500x300"],
                 {"n_components": 20, "svd_solver": "randomized", "random_state": 42})
    run_pca_case(w, "rand_edge40x8_nc5", *datasets["edge40x8"],
                 {"n_components": 5, "svd_solver": "randomized", "random_state": 42},
                 flags={"rank_deficient": True})

    # --- auto solver selection (each branch) --------------------------------
    # The dispatched solver is recorded from est._fit_svd_solver per case;
    # the comments state which branch each case is meant to exercise.
    # covariance_eigh branch: n_features <= 1000 and n_samples >= 10*n_features
    run_pca_case(w, "auto_cov300x25_nc5", *datasets["cov300x25"],
                 {"n_components": 5, "svd_solver": "auto", "random_state": 42})
    # small full branch: max(shape) <= 500
    run_pca_case(w, "auto_tall100x30_nc10", *datasets["tall100x30"],
                 {"n_components": 10, "svd_solver": "auto", "random_state": 42})
    # randomized branch, tall: big and small k
    run_pca_case(w, "auto_1500x300_nc20", *datasets["rand1500x300"],
                 {"n_components": 20, "svd_solver": "auto", "random_state": 42})
    # boundary: max(shape) == 500 exactly -> still the small-full branch
    run_pca_case(w, "auto_big80x500_nc10", *datasets["big80x500"],
                 {"n_components": 10, "svd_solver": "auto", "random_state": 42})
    # randomized branch, wide: max(shape) > 500 forces past the full branch
    run_pca_case(w, "auto_wide80x600_nc10", *datasets["wide80x600"],
                 {"n_components": 10, "svd_solver": "auto", "random_state": 42})
    # full fallback: big but k >= 0.8*min(shape)
    run_pca_case(w, "auto_big80x500_nc70", *datasets["big80x500"],
                 {"n_components": 70, "svd_solver": "auto", "random_state": 42})
    # float n_components with big data -> full fallback
    run_pca_case(w, "auto_big80x500_frac0.9", *datasets["big80x500"],
                 {"n_components": 0.9, "svd_solver": "auto", "random_state": 42})
    # mle -> full
    run_pca_case(w, "auto_mle200x30", *datasets["mle200x30"],
                 {"n_components": "mle", "svd_solver": "auto"})
    # None with auto on small data
    run_pca_case(w, "auto_tall20x5_ncNone", *datasets["tall20x5"],
                 {"n_components": None, "svd_solver": "auto"})

    # --- float32 -------------------------------------------------------------
    for base_id, params in [
        ("tall20x5", {"n_components": 3, "svd_solver": "full"}),
        ("tall20x5", {"n_components": 3, "svd_solver": "full", "whiten": True}),
        ("tall100x30", {"n_components": 10, "svd_solver": "covariance_eigh"}),
        ("tall100x30", {"n_components": 10, "svd_solver": "arpack", "random_state": 42}),
        ("tall100x30", {"n_components": 10, "svd_solver": "randomized", "random_state": 42}),
        ("wide30x100", {"n_components": 12, "svd_solver": "full"}),
    ]:
        X, X_test = datasets[base_id]
        solver = params["svd_solver"]
        wh = "_whiten" if params.get("whiten") else ""
        run_pca_case(
            w, f"f32_{solver}{wh}_{base_id}",
            X.astype(np.float32), X_test.astype(np.float32), params,
            flags={"dtype": "float32"},
        )

    w.write()


# ---------------------------------------------------------------------------
# IncrementalPCA suite
# ---------------------------------------------------------------------------


def ipca_capture(w, est):
    arrays = {
        "components": w.array(est.components_),
        "explained_variance": w.array(est.explained_variance_),
        "explained_variance_ratio": w.array(est.explained_variance_ratio_),
        "singular_values": w.array(est.singular_values_),
        "mean": w.array(np.asarray(est.mean_, dtype=np.float64)),
        "var": w.array(np.asarray(est.var_, dtype=np.float64)),
    }
    scalars = {
        "n_components_": w.scalar(est.n_components_),
        "n_samples_seen_": w.scalar(int(est.n_samples_seen_)),
        "noise_variance_": w.scalar(est.noise_variance_),
    }
    return arrays, scalars


def run_ipca_fit_case(w, case_id, X, X_test, params, flags=None):
    est = IncrementalPCA(**params)
    est.fit(X.copy())
    arrays, scalars = ipca_capture(w, est)
    arrays["X"] = w.array(X)
    arrays["X_test"] = w.array(X_test)
    arrays["transform_test"] = w.array(est.transform(X_test))
    arrays["inverse_transform_test"] = w.array(est.inverse_transform(est.transform(X_test)))
    arrays["get_covariance"] = w.array(est.get_covariance())
    if precision_comparable(est):
        arrays["get_precision"] = w.array(est.get_precision())
    scalars["batch_size_"] = w.scalar(est.batch_size_)
    scalars["n_features_in_"] = w.scalar(est.n_features_in_)
    scalars["feature_names_out"] = list(est.get_feature_names_out())
    jparams = {k: w.scalar(v) if not isinstance(v, str) else v for k, v in params.items()}
    w.case(case_id, jparams, arrays, scalars, flags)


def run_ipca_partial_case(w, case_id, batches, X_test, params, flags=None):
    """partial_fit sequence; captures the full state after every step."""
    est = IncrementalPCA(**params)
    arrays = {"X_test": w.array(X_test)}
    step_scalars = []
    for i, b in enumerate(batches):
        est.partial_fit(b.copy())
        a, s = ipca_capture(w, est)
        for name, ref in a.items():
            arrays[f"step{i}_{name}"] = ref
        step_scalars.append(s)
    arrays["transform_test"] = w.array(est.transform(X_test))
    arrays["inverse_transform_test"] = w.array(est.inverse_transform(est.transform(X_test)))
    for i, b in enumerate(batches):
        arrays[f"batch{i}"] = w.array(b)
    scalars = {
        "n_batches": len(batches),
        "steps": step_scalars,
        "n_features_in_": w.scalar(est.n_features_in_),
    }
    jparams = {k: w.scalar(v) if not isinstance(v, str) else v for k, v in params.items()}
    w.case(case_id, jparams, arrays, scalars, flags)


def ipca_suite():
    w = SuiteWriter("ipca")

    X100 = lowrank(21, 100, 20, 8, noise=0.1)
    Xt100 = lowrank(121, 30, 20, 8, noise=0.1)
    X107 = lowrank(22, 107, 12, 6, noise=0.05)
    Xt107 = lowrank(122, 25, 12, 6, noise=0.05)
    Xg = gaussian(23, 80, 15, mean_scale=2.0)
    Xtg = gaussian(123, 20, 15)

    # fit() driven batching
    run_ipca_fit_case(w, "fit_100x20_default", X100, Xt100, {})
    run_ipca_fit_case(w, "fit_100x20_nc5_b25", X100, Xt100, {"n_components": 5, "batch_size": 25})
    run_ipca_fit_case(w, "fit_100x20_nc5_b25_whiten", X100, Xt100,
                      {"n_components": 5, "batch_size": 25, "whiten": True})
    run_ipca_fit_case(w, "fit_100x20_nc20_b30", X100, Xt100,
                      {"n_components": 20, "batch_size": 30})
    run_ipca_fit_case(w, "fit_107x12_nc10_b25_tailmerge", X107, Xt107,
                      {"n_components": 10, "batch_size": 25})
    run_ipca_fit_case(w, "fit_107x12_ncNone_b40", X107, Xt107, {"batch_size": 40})
    run_ipca_fit_case(w, "fit_80x15_nc7_b16", Xg, Xtg, {"n_components": 7, "batch_size": 16})
    run_ipca_fit_case(w, "fit_80x15_nc7_b16_whiten", Xg, Xtg,
                      {"n_components": 7, "batch_size": 16, "whiten": True})
    # batch_size larger than n_samples (single batch)
    run_ipca_fit_case(w, "fit_80x15_nc5_b200_single", Xg, Xtg,
                      {"n_components": 5, "batch_size": 200})
    # float32
    run_ipca_fit_case(w, "f32_fit_100x20_nc5_b25", X100.astype(np.float32),
                      Xt100.astype(np.float32), {"n_components": 5, "batch_size": 25},
                      flags={"dtype": "float32"})

    # partial_fit sequences
    rngs = np.random.RandomState(31)
    seq1 = [X100[:40], X100[40:70], X100[70:]]
    run_ipca_partial_case(w, "partial_3steps_nc5", seq1, Xt100, {"n_components": 5})
    run_ipca_partial_case(w, "partial_3steps_nc5_whiten", seq1, Xt100,
                          {"n_components": 5, "whiten": True})
    # uneven batches incl. a batch smaller than n_components (allowed after first)
    seq2 = [X107[:30], X107[30:34], X107[34:90], X107[90:]]
    run_ipca_partial_case(w, "partial_uneven_nc8", seq2, Xt107, {"n_components": 8})
    # n_components None (inferred from first batch)
    seq3 = [Xg[:25], Xg[25:60], Xg[60:]]
    run_ipca_partial_case(w, "partial_ncNone", seq3, Xtg, {})
    # single partial_fit call
    run_ipca_partial_case(w, "partial_single_nc6", [Xg], Xtg, {"n_components": 6})
    # float32 partial
    run_ipca_partial_case(w, "f32_partial_3steps_nc5",
                          [b.astype(np.float32) for b in seq1],
                          Xt100.astype(np.float32), {"n_components": 5},
                          flags={"dtype": "float32"})

    w.write()


# ---------------------------------------------------------------------------
# RNG suite
# ---------------------------------------------------------------------------


def rng_suite():
    w = SuiteWriter("rng")
    for seed in [0, 1, 42, 123456789, 2**32 - 1]:
        rs = np.random.RandomState(seed)
        uni = rs.uniform(-1, 1, 64)
        rs = np.random.RandomState(seed)
        nrm = rs.standard_normal(64)
        rs = np.random.RandomState(seed)
        mix = np.concatenate(
            [rs.standard_normal(5), rs.uniform(0, 1, 3), rs.standard_normal(8),
             rs.uniform(-2, 3, 4), rs.standard_normal(1)]
        )
        rs = np.random.RandomState(seed)
        mat = rs.normal(size=(7, 3)).ravel()  # C-order flattening check
        w.case(
            f"seed{seed}",
            {"seed": seed},
            {
                "uniform_m1_1": w.array(uni),
                "standard_normal": w.array(nrm),
                "mixed": w.array(mix),
                "normal_7x3_flat": w.array(mat),
            },
            {},
        )
    w.write()


# ---------------------------------------------------------------------------
# Numeric suite (SVD / eigh / QR / LU regression fixtures)
# ---------------------------------------------------------------------------


def numeric_suite():
    w = SuiteWriter("numeric")
    specs = [
        ("tall9x5", gaussian(41, 9, 5)),
        ("wide5x9", gaussian(42, 5, 9)),
        ("square8", gaussian(43, 8, 8)),
        ("rankdef10x6", None),
        ("tall40x12", lowrank(44, 40, 12, 5, noise=0.02)),
    ]
    a = gaussian(45, 10, 6)
    a[:, 5] = a[:, 0]
    a[:, 4] = 0.0
    specs[3] = ("rankdef10x6", a)

    for name, mat in specs:
        m, n = mat.shape
        s = linalg.svd(mat, compute_uv=False)
        q, r = linalg.qr(mat, mode="economic")
        pl, _ = linalg.lu(mat, permute_l=True)
        arrays = {
            "A": w.array(mat),
            "svd_s": w.array(s),
            "qr_q": w.array(q),
            "qr_r": w.array(r),
            "lu_pl": w.array(pl),
        }
        scalars = {}
        if m == n:
            sym = (mat + mat.T) / 2
            evals = linalg.eigh(sym, eigvals_only=True)
            arrays["sym"] = w.array(sym)
            arrays["eigh_values"] = w.array(evals)
            inv = linalg.inv(sym + np.eye(n) * 10)
            arrays["inv_shifted"] = w.array(inv)
            sign, ld = np.linalg.slogdet(sym + np.eye(n) * 10)
            scalars["slogdet_sign"] = w.scalar(sign)
            scalars["slogdet_logdet"] = w.scalar(ld)
        w.case(name, {}, arrays, scalars)
    w.write()


if __name__ == "__main__":
    print(f"sklearn {sklearn.__version__}, numpy {np.__version__}, scipy {scipy.__version__}")
    os.makedirs(OUT_ROOT, exist_ok=True)
    pca_suite()
    ipca_suite()
    rng_suite()
    numeric_suite()
    print("done.")
