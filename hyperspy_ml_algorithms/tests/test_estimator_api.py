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

"""Tests for the sklearn-like estimator API (fit/transform/fit_transform).

Verifies that all 8 estimators follow the standard sklearn pattern:
- ``fit()`` stores fitted attributes (``components_``, etc.)
- ``transform()`` produces correct output shape
- ``fit_transform()`` is consistent with separate ``fit()`` + ``transform()``
"""

import numpy as np
import pytest

from hyperspy_ml_algorithms import (
    MLPCA,
    ORNMF,
    ORPCA,
    SVDPCA,
    IncrementalSVD,
    Orthomax,
    RPCAGoDec,
    Whitening,
)

# ---------------------------------------------------------------------------
# Shared test data — ALL-DIFFERENT dimensions per AGENTS.md convention
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    return np.random.RandomState(42)


@pytest.fixture(scope="module")
def data_77_13(rng):
    """Standard data: 77 samples, 13 features."""
    return rng.random((77, 13))


@pytest.fixture(scope="module")
def data_50_30(rng):
    """Poisson-like data for MLPCA: 50 samples, 30 features."""
    return rng.poisson(10, size=(50, 30)).astype(float)


@pytest.fixture(scope="module")
def components_13_5(rng):
    """Component matrix for Orthomax: 13 features, 5 components."""
    return rng.random((13, 5))


# ===================================================================
# SVDPCA
# ===================================================================


class TestSVDPCAEstimatorAPI:
    """fit / transform / fit_transform for SVDPCA."""

    def test_fit_stores_components(self, data_77_13):
        est = SVDPCA(n_components=5)
        est.fit(data_77_13)
        assert est.components_.shape == (5, 13)
        assert est.singular_values_.shape == (5,)
        assert est.explained_variance_.shape == (5,)
        assert est.explained_variance_ratio_.shape == (5,)

    def test_transform_shape(self, data_77_13):
        est = SVDPCA(n_components=5).fit(data_77_13)
        scores = est.transform(data_77_13)
        assert scores.shape == (77, 5)

    def test_fit_transform_shape(self, data_77_13):
        scores = SVDPCA(n_components=5).fit_transform(data_77_13)
        assert scores.shape == (77, 5)

    def test_fit_transform_matches_fit_plus_transform(self, data_77_13):
        est1 = SVDPCA(n_components=5)
        scores1 = est1.fit_transform(data_77_13)

        est2 = SVDPCA(n_components=5)
        est2.fit(data_77_13)
        scores2 = est2.transform(data_77_13)

        np.testing.assert_allclose(scores1, scores2, atol=1e-10)

    def test_centre_features(self, data_77_13):
        """Standard PCA centering (features)."""
        est = SVDPCA(n_components=5, centre="features").fit(data_77_13)
        assert est.mean_ is not None
        # mean_ is broadcastable: shape (1, 13) for features centering
        assert est.mean_.shape[-1] == 13
        assert est.components_.shape == (5, 13)

    def test_centre_signal(self, data_77_13):
        est = SVDPCA(n_components=5, centre="signal").fit(data_77_13)
        assert est.mean_ is not None
        assert est.components_.shape == (5, 13)

    def test_centre_none(self, data_77_13):
        est = SVDPCA(n_components=5, centre=None).fit(data_77_13)
        assert est.mean_ is None
        assert est.components_.shape == (5, 13)

    def test_explained_variance_ratio_sums_to_one(self, data_77_13):
        est = SVDPCA(n_components=5).fit(data_77_13)
        np.testing.assert_allclose(
            np.sum(est.explained_variance_ratio_), 1.0, atol=1e-10
        )


# ===================================================================
# MLPCA
# ===================================================================


class TestMLPCAEstimatorAPI:
    """fit / transform / fit_transform for MLPCA."""

    def test_fit_stores_components(self, data_50_30):
        variance = data_50_30.copy()
        est = MLPCA(n_components=3)
        est.fit(data_50_30, variance)
        # MLPCA stores components_ as (n_components, n_features), like sklearn
        assert est.components_.shape == (3, 30)
        assert est.singular_values_.shape == (3,)
        assert est.mean_ is None

    def test_transform_shape(self, data_50_30):
        variance = data_50_30.copy()
        est = MLPCA(n_components=3).fit(data_50_30, variance)
        scores = est.transform(data_50_30)
        assert scores.shape == (50, 3)

    def test_fit_transform_shape(self, data_50_30):
        variance = data_50_30.copy()
        scores = MLPCA(n_components=3).fit_transform(data_50_30, variance)
        assert scores.shape == (50, 3)

    def test_fit_transform_matches_scores(self, data_50_30):
        """fit_transform returns same scores as fit().scores_."""
        variance = data_50_30.copy()

        est1 = MLPCA(n_components=3)
        scores1 = est1.fit_transform(data_50_30, variance)

        est2 = MLPCA(n_components=3)
        est2.fit(data_50_30, variance)
        scores2 = est2.scores_

        np.testing.assert_allclose(scores1, scores2, atol=1e-10)

    def test_n_components_none(self, data_50_30):
        variance = data_50_30.copy()
        est = MLPCA(n_components=None).fit(data_50_30, variance)
        assert est.components_.shape == (30, 30)

    def test_all_finite(self, data_50_30):
        variance = data_50_30.copy()
        est = MLPCA(n_components=3).fit(data_50_30, variance)
        assert np.all(np.isfinite(est.components_))
        assert np.all(np.isfinite(est.singular_values_))
        assert np.all(np.isfinite(est.scores_))


# ===================================================================
# ORPCA
# ===================================================================


class TestORPCAEstimatorAPI:
    """fit / transform / fit_transform for ORPCA."""

    def test_fit_stores_components(self, data_77_13):
        est = ORPCA(n_components=5, training_samples=10)
        est.fit(data_77_13)
        assert est.components_.shape == (5, 13)
        assert est.low_rank_.shape == (77, 13)

    def test_transform_shape(self, data_77_13):
        est = ORPCA(n_components=5, training_samples=10).fit(data_77_13)
        scores = est.transform(data_77_13)
        assert scores.shape == (77, 5)

    def test_fit_transform_shape(self, data_77_13):
        scores = ORPCA(n_components=5, training_samples=10).fit_transform(data_77_13)
        assert scores.shape == (77, 5)

    def test_fit_transform_matches_fit_plus_transform(self, data_77_13):
        est1 = ORPCA(n_components=5, training_samples=10)
        scores1 = est1.fit_transform(data_77_13)

        est2 = ORPCA(n_components=5, training_samples=10)
        est2.fit(data_77_13)
        scores2 = est2.transform(data_77_13)

        np.testing.assert_allclose(scores1, scores2, atol=1e-10)

    def test_all_finite(self, data_77_13):
        est = ORPCA(n_components=5, training_samples=10).fit(data_77_13)
        assert np.all(np.isfinite(est.components_))
        assert np.all(np.isfinite(est.low_rank_))


# ===================================================================
# RPCAGoDec
# ===================================================================


class TestRPCAGoDecEstimatorAPI:
    """fit / transform / fit_transform for RPCAGoDec."""

    def test_fit_stores_components(self, data_77_13):
        est = RPCAGoDec(rank=5)
        est.fit(data_77_13)
        assert est.components_.shape == (5, 13)
        assert est.low_rank_.shape == (77, 13)
        assert est.sparse_.shape == (77, 13)
        assert est.singular_values_.shape == (5,)

    def test_transform_shape(self, data_77_13):
        est = RPCAGoDec(rank=5).fit(data_77_13)
        scores = est.transform(data_77_13)
        assert scores.shape == (77, 5)

    def test_fit_transform_returns_low_rank(self, data_77_13):
        """RPCAGoDec.fit_transform returns the low-rank reconstruction."""
        low_rank = RPCAGoDec(rank=5).fit_transform(data_77_13)
        assert low_rank.shape == (77, 13)

    def test_fit_transform_matches_fit_low_rank(self, data_77_13):
        """fit_transform returns same low_rank_ as fit().low_rank_."""
        est1 = RPCAGoDec(rank=5, random_state=42)
        low_rank1 = est1.fit_transform(data_77_13)

        est2 = RPCAGoDec(rank=5, random_state=42)
        est2.fit(data_77_13)
        low_rank2 = est2.low_rank_

        np.testing.assert_allclose(low_rank1, low_rank2, atol=1e-10)

    def test_reconstruction_decomposes(self, data_77_13):
        """X ≈ L + S."""
        est = RPCAGoDec(rank=5).fit(data_77_13)
        reconstruction = est.low_rank_ + est.sparse_
        np.testing.assert_allclose(data_77_13, reconstruction, atol=1e-10)

    def test_all_finite(self, data_77_13):
        est = RPCAGoDec(rank=5).fit(data_77_13)
        assert np.all(np.isfinite(est.components_))
        assert np.all(np.isfinite(est.low_rank_))
        assert np.all(np.isfinite(est.sparse_))


# ===================================================================
# ORNMF
# ===================================================================


class TestORNMFEstimatorAPI:
    """fit / transform / fit_transform for ORNMF."""

    def test_fit_stores_components(self, data_77_13):
        est = ORNMF(n_components=5)
        est.fit(data_77_13)
        assert est.components_.shape == (5, 13)
        assert est.scores_.shape == (77, 5)

    def test_components_non_negative(self, data_77_13):
        est = ORNMF(n_components=5).fit(data_77_13)
        assert np.all(est.components_ >= 0)

    def test_scores_non_negative(self, data_77_13):
        est = ORNMF(n_components=5).fit(data_77_13)
        assert np.all(est.scores_ >= 0)

    def test_transform_shape(self, data_77_13):
        est = ORNMF(n_components=5).fit(data_77_13)
        scores = est.transform(data_77_13)
        assert scores.shape == (77, 5)

    def test_transform_non_negative(self, data_77_13):
        est = ORNMF(n_components=5).fit(data_77_13)
        scores = est.transform(data_77_13)
        assert np.all(scores >= 0)

    def test_fit_transform_shape(self, data_77_13):
        scores = ORNMF(n_components=5).fit_transform(data_77_13)
        assert scores.shape == (77, 5)

    def test_fit_transform_matches_fit_plus_transform(self, data_77_13):
        est1 = ORNMF(n_components=5, random_state=42)
        scores1 = est1.fit_transform(data_77_13)

        est2 = ORNMF(n_components=5, random_state=42)
        est2.fit(data_77_13)
        scores2 = est2.transform(data_77_13)

        np.testing.assert_allclose(scores1, scores2, atol=1e-10)

    def test_pre_fit_guard_components(self):
        est = ORNMF(n_components=5)
        with pytest.raises(AttributeError, match="not been fitted"):
            _ = est.components_

    def test_pre_fit_guard_scores(self):
        est = ORNMF(n_components=5)
        with pytest.raises(AttributeError, match="not been fitted"):
            _ = est.scores_


# ===================================================================
# IncrementalSVD
# ===================================================================


class TestIncrementalSVDEstimatorAPI:
    """fit / transform / fit_transform for IncrementalSVD."""

    def test_fit_stores_components(self, data_77_13):
        est = IncrementalSVD(n_components=5)
        est.fit(data_77_13)
        assert est.components_.shape == (5, 13)
        assert est.singular_values_.shape == (5,)
        assert est.explained_variance_.shape == (5,)
        assert est.explained_variance_ratio_.shape == (5,)

    def test_transform_shape(self, data_77_13):
        est = IncrementalSVD(n_components=5).fit(data_77_13)
        scores = est.transform(data_77_13)
        assert scores.shape == (77, 5)

    def test_fit_transform_shape(self, data_77_13):
        scores = IncrementalSVD(n_components=5).fit_transform(data_77_13)
        assert scores.shape == (77, 5)

    def test_fit_transform_matches_fit_plus_transform(self, data_77_13):
        est1 = IncrementalSVD(n_components=5)
        scores1 = est1.fit_transform(data_77_13)

        est2 = IncrementalSVD(n_components=5)
        est2.fit(data_77_13)
        scores2 = est2.transform(data_77_13)

        np.testing.assert_allclose(scores1, scores2, atol=1e-10)

    def test_mean_is_zeros(self, data_77_13):
        est = IncrementalSVD(n_components=5).fit(data_77_13)
        np.testing.assert_allclose(est.mean_, np.zeros(13), atol=1e-10)

    def test_explained_variance_ratio_sums_to_one(self, data_77_13):
        est = IncrementalSVD(n_components=5).fit(data_77_13)
        np.testing.assert_allclose(
            np.sum(est.explained_variance_ratio_), 1.0, atol=1e-10
        )


# ===================================================================
# Orthomax
# ===================================================================


class TestOrthomaxEstimatorAPI:
    """fit / transform / fit_transform for Orthomax."""

    def test_fit_stores_rotation_matrix(self, components_13_5):
        est = Orthomax()
        est.fit(components_13_5)
        assert est.rotation_matrix_.shape == (5, 5)
        # Orthomax stores components_ as (n_features, n_components)
        assert est.components_.shape == (13, 5)

    def test_transform_shape(self, components_13_5):
        est = Orthomax().fit(components_13_5)
        rotated = est.transform(components_13_5)
        assert rotated.shape == (13, 5)

    def test_fit_transform_shape(self, components_13_5):
        rotated = Orthomax().fit_transform(components_13_5)
        assert rotated.shape == (13, 5)

    def test_fit_transform_matches_fit_plus_transform(self, components_13_5):
        """fit_transform returns components_; fit + transform should match."""
        est1 = Orthomax()
        rotated1 = est1.fit_transform(components_13_5)

        est2 = Orthomax()
        est2.fit(components_13_5)
        rotated2 = est2.transform(components_13_5)

        np.testing.assert_allclose(rotated1, rotated2, atol=1e-10)

    def test_rotation_preserves_orthogonality(self, components_13_5):
        """Rotation matrix should be orthogonal: W @ W.T ≈ I."""
        est = Orthomax().fit(components_13_5)
        W = est.rotation_matrix_
        np.testing.assert_allclose(W @ W.T, np.eye(5), atol=1e-10)

    def test_varimax_default(self, components_13_5):
        """Default gamma=1.0 is varimax."""
        est = Orthomax(gamma=1.0).fit(components_13_5)
        assert est.components_.shape == (13, 5)


# ===================================================================
# Whitening
# ===================================================================


class TestWhiteningEstimatorAPI:
    """fit / transform / fit_transform for Whitening."""

    def test_fit_stores_whitening_matrix(self, data_77_13):
        est = Whitening()
        est.fit(data_77_13)
        assert est.whitening_matrix_.shape == (13, 13)
        assert est.mean_.shape == (13,)

    def test_transform_shape(self, data_77_13):
        est = Whitening().fit(data_77_13)
        whitened = est.transform(data_77_13)
        assert whitened.shape == (77, 13)

    def test_fit_transform_shape(self, data_77_13):
        whitened = Whitening().fit_transform(data_77_13)
        assert whitened.shape == (77, 13)

    def test_fit_transform_matches_fit_plus_transform(self, data_77_13):
        est1 = Whitening()
        whitened1 = est1.fit_transform(data_77_13)

        est2 = Whitening()
        est2.fit(data_77_13)
        whitened2 = est2.transform(data_77_13)

        np.testing.assert_allclose(whitened1, whitened2, atol=1e-10)

    def test_pca_whitening_unit_variance(self, data_77_13):
        """PCA-whitened data should have unit diagonal covariance."""
        est = Whitening(method="PCA").fit(data_77_13)
        whitened = est.transform(data_77_13)
        # Whitening uses population covariance (divide by N), so use bias=True
        cov = np.cov(whitened.T, bias=True)
        np.testing.assert_allclose(np.diag(cov), 1.0, atol=1e-10)

    def test_zca_whitening_unit_variance(self, data_77_13):
        """ZCA-whitened data should have unit diagonal covariance."""
        est = Whitening(method="ZCA").fit(data_77_13)
        whitened = est.transform(data_77_13)
        cov = np.cov(whitened.T, bias=True)
        np.testing.assert_allclose(np.diag(cov), 1.0, atol=1e-10)

    def test_no_centre(self, data_77_13):
        est = Whitening(centre=False).fit(data_77_13)
        np.testing.assert_allclose(est.mean_, np.zeros(13), atol=1e-10)

    def test_invalid_method_raises(self, data_77_13):
        est = Whitening(method="INVALID")
        with pytest.raises(ValueError, match="method must be one of"):
            est.fit(data_77_13)
