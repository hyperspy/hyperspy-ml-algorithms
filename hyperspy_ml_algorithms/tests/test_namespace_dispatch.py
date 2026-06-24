# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

"""Tests for ``array_namespace`` dispatch.

Verifies that the ``array_namespace`` wrapper from ``array_api_compat``
correctly dispatches to the appropriate backend (numpy, cupy, torch).

CuPy and PyTorch tests are skipped if the respective libraries are not
installed — they are optional dependencies.
"""

import importlib

import numpy as np
import pytest

from hyperspy_ml_algorithms.utils._array_namespace import array_namespace

# ---------------------------------------------------------------------------
# Optional backend detection
# ---------------------------------------------------------------------------

CUPY_INSTALLED = importlib.util.find_spec("cupy") is not None
TORCH_INSTALLED = importlib.util.find_spec("torch") is not None


# ===================================================================
# numpy dispatch (always available)
# ===================================================================


class TestNumpyNamespace:
    """array_namespace with numpy inputs."""

    def test_numpy_array_returns_numpy_namespace(self):
        x = np.array([1.0, 2.0, 3.0])
        xp = array_namespace(x)
        # array_api_compat wraps numpy as 'array_api_compat.numpy'
        assert "numpy" in xp.__name__

    def test_numpy_2d_array_returns_numpy_namespace(self):
        x = np.random.RandomState(42).random((77, 13))
        xp = array_namespace(x)
        assert "numpy" in xp.__name__

    def test_multiple_numpy_arrays(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        xp = array_namespace(a, b)
        assert "numpy" in xp.__name__

    def test_array_operations_through_namespace(self):
        """Basic array operations work through the returned namespace."""
        x = np.array([1.0, 2.0, 3.0])
        xp = array_namespace(x)

        # Element-wise operations
        result = xp.asarray(x) + xp.asarray(1.0)
        np.testing.assert_allclose(
            np.asarray(result), np.array([2.0, 3.0, 4.0]), atol=1e-10
        )

    def test_linalg_svd_through_namespace(self):
        """SVD works through the namespace for numpy arrays."""
        x = np.random.RandomState(42).random((77, 13))
        xp = array_namespace(x)

        U, S, Vt = xp.linalg.svd(xp.asarray(x), full_matrices=False)
        assert U.shape == (77, 13)
        assert S.shape == (13,)
        assert Vt.shape == (13, 13)

    def test_mean_through_namespace(self):
        """Mean computation works through the namespace."""
        x = np.random.RandomState(42).random((77, 13))
        xp = array_namespace(x)

        mean = xp.mean(xp.asarray(x), axis=0)
        np.testing.assert_allclose(np.asarray(mean), np.mean(x, axis=0), atol=1e-10)

    def test_matmul_through_namespace(self):
        """Matrix multiplication works through the namespace."""
        a = np.random.RandomState(42).random((77, 13))
        b = np.random.RandomState(43).random((13, 5))
        xp = array_namespace(a, b)

        result = xp.matmul(xp.asarray(a), xp.asarray(b))
        np.testing.assert_allclose(np.asarray(result), a @ b, atol=1e-10)

    def test_reshape_through_namespace(self):
        """Reshape works through the namespace."""
        x = np.random.RandomState(42).random((77, 13))
        xp = array_namespace(x)

        reshaped = xp.reshape(xp.asarray(x), (77 * 13,))
        assert reshaped.shape == (77 * 13,)


# ===================================================================
# Estimator integration — numpy path
# ===================================================================


class TestEstimatorNumpyDispatch:
    """All estimators work with numpy arrays (the default path)."""

    @pytest.fixture(scope="class")
    def data(self):
        return np.random.RandomState(42).random((77, 13))

    def test_svdpca_numpy(self, data):
        from hyperspy_ml_algorithms import SVDPCA

        est = SVDPCA(n_components=5).fit(data)
        assert est.components_.shape == (5, 13)
        scores = est.transform(data)
        assert scores.shape == (77, 5)

    def test_mlpca_numpy(self, data):
        from hyperspy_ml_algorithms import MLPCA

        variance = data.copy()
        est = MLPCA(n_components=5).fit(data, variance)
        assert est.components_.shape == (13, 5)

    def test_orpca_numpy(self, data):
        from hyperspy_ml_algorithms import ORPCA

        est = ORPCA(n_components=5, training_samples=10).fit(data)
        assert est.components_.shape == (5, 13)

    def test_rpca_godec_numpy(self, data):
        from hyperspy_ml_algorithms import RPCAGoDec

        est = RPCAGoDec(rank=5).fit(data)
        assert est.components_.shape == (5, 13)

    def test_ornmf_numpy(self, data):
        from hyperspy_ml_algorithms import ORNMF

        est = ORNMF(n_components=5).fit(data)
        assert est.components_.shape == (5, 13)

    def test_incremental_svd_numpy(self, data):
        from hyperspy_ml_algorithms import IncrementalSVD

        est = IncrementalSVD(n_components=5).fit(data)
        assert est.components_.shape == (5, 13)

    def test_orthomax_numpy(self):
        from hyperspy_ml_algorithms import Orthomax

        X = np.random.RandomState(42).random((13, 5))
        est = Orthomax().fit(X)
        assert est.components_.shape == (13, 5)

    def test_whitening_numpy(self, data):
        from hyperspy_ml_algorithms import Whitening

        est = Whitening().fit(data)
        assert est.whitening_matrix_.shape == (13, 13)


# ===================================================================
# CuPy dispatch (optional)
# ===================================================================


@pytest.mark.skipif(not CUPY_INSTALLED, reason="cupy not installed")
class TestCuPyNamespace:
    """array_namespace with cupy inputs (requires cupy)."""

    def test_cupy_array_returns_cupy_namespace(self):
        import cupy as cp

        x = cp.array([1.0, 2.0, 3.0])
        xp = array_namespace(x)
        assert xp.__name__ == "cupy"

    def test_cupy_2d_array_returns_cupy_namespace(self):
        import cupy as cp

        x = cp.random.random((77, 13))
        xp = array_namespace(x)
        assert xp.__name__ == "cupy"

    def test_mixed_numpy_cupy_raises(self):
        """Mixing numpy and cupy arrays should raise."""
        import cupy as cp

        a = np.array([1.0, 2.0])
        b = cp.array([3.0, 4.0])
        with pytest.raises(ValueError):
            array_namespace(a, b)


# ===================================================================
# PyTorch dispatch (optional)
# ===================================================================


@pytest.mark.skipif(not TORCH_INSTALLED, reason="torch not installed")
class TestTorchNamespace:
    """array_namespace with torch inputs (requires torch)."""

    def test_torch_tensor_returns_torch_namespace(self):
        import torch

        x = torch.tensor([1.0, 2.0, 3.0])
        xp = array_namespace(x)
        assert xp.__name__ == "torch"

    def test_torch_2d_tensor_returns_torch_namespace(self):
        import torch

        x = torch.rand(77, 13)
        xp = array_namespace(x)
        assert xp.__name__ == "torch"

    def test_mixed_numpy_torch_raises(self):
        """Mixing numpy and torch tensors should raise."""
        import torch

        a = np.array([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        with pytest.raises(ValueError):
            array_namespace(a, b)
