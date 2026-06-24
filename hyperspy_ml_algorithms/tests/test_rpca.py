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

import numpy as np
import pytest
import scipy.linalg

from hyperspy_ml_algorithms import ORPCA, RPCAGoDec


def compare_norms(a, b, tol=5e-3):
    assert a.shape == b.shape

    m, n = a.shape
    tol *= m * n
    n1 = np.linalg.norm(a)
    n2 = np.linalg.norm(b)

    assert np.linalg.norm((a / n1) - (b / n2)) < tol


class TestRPCA:
    def setup_method(self, method):
        # Define shape etc.
        m = 128  # Dimensionality
        n = 256  # Number of samples
        r = 3
        s = 0.01

        # Low-rank, sparse and noise matrices
        rng = np.random.RandomState(101)
        U = scipy.linalg.orth(rng.randn(m, r))
        V = rng.randn(n, r)
        A = U @ V.T
        E = 10 * rng.binomial(1, s, (m, n))
        G = 0.005 * rng.randn(m, n)
        X = A + E + G

        self.m = m
        self.n = n
        self.rank = r
        self.A = A
        self.X = X

    def test_default(self):
        est = RPCAGoDec(rank=self.rank).fit(self.X)
        compare_norms(est.low_rank_, self.A)

    @pytest.mark.parametrize("power", [0, 1, 2])
    @pytest.mark.parametrize("maxiter", [1e2, 1e3])
    @pytest.mark.parametrize("tol", [1e-1, 1e-4])
    def test_tol_iter(self, power, maxiter, tol):
        est = RPCAGoDec(rank=self.rank, power=power, max_iter=maxiter, tol=tol).fit(
            self.X
        )
        compare_norms(est.low_rank_, self.A)

    def test_regularization(self):
        est = RPCAGoDec(rank=self.rank, lambda1=0.01).fit(self.X)
        compare_norms(est.low_rank_, self.A)

    @pytest.mark.skip(reason="HyperSpy Signal1D not available in standalone package")
    @pytest.mark.parametrize("poisson", [True, False])
    def test_signal(self, poisson):
        # Note that s1.decomposition() operates on the transpose
        # i.e. (n_samples, n_features).
        x = self.X.copy().T.reshape(16, 16, 128)

        if poisson:
            x -= x.min()
            x[x <= 0] = 1e-16

        s1 = Signal1D(x)  # noqa: F821 — skipped test, hyperspy not available

        X, E = s1.decomposition(
            normalize_poissonian_noise=poisson,
            algorithm="RPCA",
            output_dimension=self.rank,
            return_info=True,
        )
        compare_norms(X, self.A.T)


class TestORPCA:
    def setup_method(self, method):
        # Define shape etc.
        m = 128  # Dimensionality
        n = 256  # Number of samples
        r = 3
        s = 0.01

        # Low-rank and sparse error matrices
        rng = np.random.RandomState(101)
        U = scipy.linalg.orth(rng.randn(m, r))
        V = rng.randn(n, r)
        A = U @ V.T
        E = 10 * rng.binomial(1, s, (m, n))
        X = A + E

        self.m = m
        self.n = n
        self.rank = r
        self.A = A
        self.X = X
        self.U = U

    def test_default(self):
        est = ORPCA(n_components=self.rank, store_error=True).fit(self.X)
        compare_norms(est.low_rank_, self.A)

    def test_project(self):
        est = ORPCA(n_components=self.rank).fit(self.X.T)
        L = est.components_.T
        R = est.transform(self.X.T).T

        assert L.shape == (self.m, self.rank)
        assert R.shape == (self.rank, self.n)

    def test_batch_size(self):
        est = ORPCA(n_components=self.rank)
        est.partial_fit(self.X.T, batch_size=2)
        L = est.components_.T
        R = est.transform(self.X.T).T

        assert L.shape == (self.m, self.rank)
        assert R.shape == (self.rank, self.n)

    def test_method_BCD(self):
        est = ORPCA(n_components=self.rank, store_error=True, method="BCD").fit(self.X)
        compare_norms(est.low_rank_, self.A)

    @pytest.mark.parametrize("subspace_learning_rate", [1.0, 1.1])
    def test_method_SGD(self, subspace_learning_rate):
        est = ORPCA(
            n_components=self.rank,
            store_error=True,
            method="SGD",
            subspace_learning_rate=subspace_learning_rate,
        ).fit(self.X)
        compare_norms(est.low_rank_, self.A)

    @pytest.mark.parametrize("subspace_momentum", [0.5, 0.1])
    def test_method_MomentumSGD(self, subspace_momentum):
        est = ORPCA(
            n_components=self.rank,
            store_error=True,
            method="MomentumSGD",
            subspace_learning_rate=1.1,
            subspace_momentum=subspace_momentum,
        ).fit(self.X)
        compare_norms(est.low_rank_, self.A)

        with pytest.raises(ValueError, match="must be a float between 0 and 1"):
            _ = ORPCA(
                n_components=self.rank,
                method="MomentumSGD",
                subspace_momentum=1.9,
            )

    def test_init_rand(self):
        est = ORPCA(n_components=self.rank, store_error=True, init="rand").fit(self.X)
        compare_norms(est.low_rank_, self.A)

    def test_init_mat(self):
        est = ORPCA(n_components=self.rank, store_error=True, init=self.U).fit(self.X.T)
        compare_norms(est.low_rank_.T, self.A)

        with pytest.raises(ValueError, match="must be a 2-D matrix"):
            mat = np.zeros(self.m)
            _ = ORPCA(n_components=self.rank, init=mat).fit(self.X.T)

        with pytest.raises(ValueError, match="must have shape"):
            mat = np.zeros((self.m, self.rank - 1))
            _ = ORPCA(n_components=self.rank, init=mat).fit(self.X.T)

    @pytest.mark.parametrize("rank", [3, 11])
    @pytest.mark.parametrize("training_samples", [16, 32])
    def test_training(self, rank, training_samples):
        est = ORPCA(
            n_components=rank,
            store_error=True,
            init="qr",
            training_samples=training_samples,
        ).fit(self.X)
        compare_norms(est.low_rank_, self.A)

        with pytest.raises(ValueError, match="must be >="):
            _ = ORPCA(n_components=self.rank, init="qr", training_samples=self.rank - 1)

    def test_regularization(self):
        est = ORPCA(
            n_components=self.rank,
            store_error=True,
            lambda1=0.01,
            lambda2=0.02,
        ).fit(self.X)
        compare_norms(est.low_rank_, self.A)

    def test_exception_method(self):
        with pytest.raises(ValueError, match="'method' not recognised"):
            _ = ORPCA(n_components=self.rank, method="uniform")

    def test_exception_init(self):
        with pytest.raises(ValueError, match="'init' not recognised"):
            _ = ORPCA(n_components=self.rank, init="uniform")

    @pytest.mark.skip(reason="HyperSpy Signal1D not available in standalone package")
    @pytest.mark.parametrize("poisson", [True, False])
    def test_signal(self, poisson):
        # Note that s1.decomposition() operates on the transpose
        # i.e. (n_samples, n_features).
        x = self.X.copy().T.reshape(16, 16, 128)

        if poisson:
            x -= x.min()
            x[x <= 0] = 1e-16

        s1 = Signal1D(x)  # noqa: F821 — skipped test, hyperspy not available

        X, E = s1.decomposition(
            normalize_poissonian_noise=poisson,
            algorithm="ORPCA",
            output_dimension=self.rank,
            return_info=True,
        )
        compare_norms(X, self.A.T)


class TestORPCASklearnAPI:
    """Test the sklearn-compatible API (partial_fit / transform / components_)."""

    def setup_method(self, method):
        m = 10
        n = 20
        r = 2
        s = 0.01

        rng = np.random.RandomState(101)
        U = scipy.linalg.orth(rng.randn(m, r))
        V = rng.randn(n, r)
        A = U @ V.T
        E = 10 * rng.binomial(1, s, (m, n))
        X = A + E

        self.m = m
        self.n = n
        self.rank = r
        self.A = A
        # ORPCA / partial_fit expect (n_samples, n_features)
        self.X = X.T  # shape (n, m)

    def test_partial_fit_transform(self):
        obj = ORPCA(self.rank)
        obj.partial_fit(self.X)

        loadings = obj.transform(self.X)
        assert loadings.shape == (self.n, self.rank)

    def test_components_shape(self):
        obj = ORPCA(self.rank)
        obj.partial_fit(self.X)

        assert obj.components_.shape == (self.rank, self.m)

    def test_reconstruction(self):
        obj = ORPCA(self.rank)
        obj.partial_fit(self.X)

        loadings = obj.transform(self.X)
        Xhat = loadings @ obj.components_
        compare_norms(Xhat.T, self.A)

    def test_partial_fit_batch_size(self):
        obj = ORPCA(self.rank)
        obj.partial_fit(self.X, batch_size=4)

        assert obj.components_.shape == (self.rank, self.m)
        assert obj.transform(self.X).shape == (self.n, self.rank)

    @pytest.mark.skip(reason="Deprecated fit() removed in refactor")
    def test_deprecated_fit_warns(self):
        obj = ORPCA(self.rank)
        with pytest.warns(VisibleDeprecationWarning, match="`fit\\(\\)` is deprecated"):  # noqa: F821 — skipped test, old API
            obj.fit(self.X)

    @pytest.mark.skip(reason="Deprecated project() removed in refactor")
    def test_deprecated_project_warns(self):
        obj = ORPCA(self.rank)
        obj.partial_fit(self.X)
        with pytest.warns(
            VisibleDeprecationWarning,  # noqa: F821 — skipped test, old API
            match="`project\\(\\)` is deprecated",
        ):
            R = obj.project(self.X)
        assert R.shape == (self.rank, self.n)

    @pytest.mark.skip(reason="Deprecated finish() removed in refactor")
    def test_deprecated_finish_warns(self):
        obj = ORPCA(self.rank)
        obj.partial_fit(self.X)
        with pytest.warns(
            VisibleDeprecationWarning,  # noqa: F821 — skipped test, old API
            match="`finish\\(\\)` is deprecated",
        ):
            L, R = obj.finish()
        assert L.shape == (self.m, self.rank)
