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

import importlib

import numpy as np
import pytest

from hyperspy_ml_algorithms import SVDPCA

sklearn = importlib.util.find_spec("sklearn")
skip_sklearn = pytest.mark.skipif(sklearn is None, reason="sklearn not installed")


class TestSVDPCA:
    def setup_method(self, method):
        # Define shape etc.
        m = 100  # Dimensionality
        n = 105  # Number of samples
        r = 3

        self.rng = np.random.RandomState(101)
        U = self.rng.randn(m, r)
        V = self.rng.randn(n, r)

        self.m = m
        self.n = n
        self.rank = r
        self.X = U @ V.T

        self.X_mean_0 = self.X.mean(axis=0)[np.newaxis, :]
        self.X_mean_1 = self.X.mean(axis=1)[:, np.newaxis]

        # Test tolerance
        self.tol = 1e-3 * (self.m * self.n)

    @pytest.mark.parametrize("output_dimension", [None, 3])
    @pytest.mark.parametrize("auto_transpose", [True, False])
    @pytest.mark.parametrize("centre", [None, "signal", "features"])
    @pytest.mark.parametrize("u_based_decision", [True, False])
    def test_full(self, output_dimension, auto_transpose, centre, u_based_decision):
        est = SVDPCA(
            n_components=output_dimension,
            svd_solver="full",
            auto_transpose=auto_transpose,
            centre=centre,
            u_based_decision=u_based_decision,
        ).fit(self.X)
        components = est.components_
        scores = est._scores
        explained_variance = est.explained_variance_
        mean = est.mean_
        X = scores @ components
        if mean is not None:
            X = X + mean

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.X)
        assert normX < self.tol

        # Check singular values
        explained_variance_norm = explained_variance / np.sum(explained_variance)
        np.testing.assert_allclose(explained_variance_norm[: self.rank].sum(), 1.0)

        if centre is None:
            assert mean is None
        elif centre == "signal":
            np.testing.assert_allclose(mean, self.X_mean_1)
        elif centre == "features":
            np.testing.assert_allclose(mean, self.X_mean_0)

    @pytest.mark.parametrize("output_dimension", [None, 3])
    @pytest.mark.parametrize("auto_transpose", [True, False])
    @pytest.mark.parametrize("centre", [None, "signal", "features"])
    @pytest.mark.parametrize("u_based_decision", [True, False])
    def test_arpack(self, output_dimension, auto_transpose, centre, u_based_decision):
        est = SVDPCA(
            n_components=output_dimension,
            svd_solver="arpack",
            auto_transpose=auto_transpose,
            centre=centre,
            u_based_decision=u_based_decision,
        ).fit(self.X)
        components = est.components_
        scores = est._scores
        explained_variance = est.explained_variance_
        mean = est.mean_
        X = scores @ components
        if mean is not None:
            X = X + mean

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.X)
        assert normX < self.tol

        # Check singular values
        explained_variance_norm = explained_variance / np.sum(explained_variance)
        np.testing.assert_allclose(explained_variance_norm[: self.rank].sum(), 1.0)

    @skip_sklearn
    @pytest.mark.parametrize("output_dimension", [None, 3])
    @pytest.mark.parametrize("auto_transpose", [True, False])
    @pytest.mark.parametrize("centre", [None, "signal", "features"])
    def test_randomized(self, output_dimension, auto_transpose, centre):
        est = SVDPCA(
            n_components=output_dimension,
            svd_solver="randomized",
            auto_transpose=auto_transpose,
            centre=centre,
        ).fit(self.X)
        components = est.components_
        scores = est._scores
        explained_variance = est.explained_variance_
        mean = est.mean_
        X = scores @ components
        if mean is not None:
            X = X + mean

        # Check the low-rank component MSE
        normX = np.linalg.norm(X - self.X)
        assert normX < self.tol

        # Check singular values
        explained_variance_norm = explained_variance / np.sum(explained_variance)
        np.testing.assert_allclose(explained_variance_norm[: self.rank].sum(), 1.0)

    @skip_sklearn
    def test_solver_auto(self):
        # Uses "full"
        U = self.rng.randn(100, 5)
        V = self.rng.randn(100, 5)
        X = U @ V.T
        est = SVDPCA(n_components=5, svd_solver="auto").fit(X)
        components = est.components_
        scores = est._scores
        Y = scores @ components
        normX = np.linalg.norm(X - Y)
        assert normX < self.tol

        # Uses "randomized"
        U = self.rng.randn(501, 5)
        V = self.rng.randn(100, 5)
        X = U @ V.T
        est = SVDPCA(n_components=5, svd_solver="auto").fit(X)
        components = est.components_
        scores = est._scores
        Y = scores @ components
        normX = np.linalg.norm(X - Y)
        assert normX < self.tol

        # Uses full
        U = self.rng.randn(501, 5)
        V = self.rng.randn(100, 5)
        X = U @ V.T
        est = SVDPCA(n_components=81, svd_solver="auto").fit(X)
        components = est.components_
        scores = est._scores
        Y = scores @ components
        normX = np.linalg.norm(X - Y)
        assert normX < self.tol

    def test_arpack_error(self):
        pytest.importorskip("scipy", minversion="1.4.0")
        with pytest.raises(
            ValueError, match="requires output_dimension to be strictly"
        ):
            _ = SVDPCA(n_components=min(self.X.shape) + 1, svd_solver="arpack").fit(
                self.X
            )

    def test_centre_error(self):
        with pytest.raises(ValueError, match="'centre' must be one of"):
            _ = SVDPCA(centre="random").fit(self.X)
