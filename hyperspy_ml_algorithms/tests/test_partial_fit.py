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

"""Tests for ``partial_fit`` on incremental estimators.

Covers the three estimators that support online / streaming learning:
- ``IncrementalSVD`` — deterministic, partial_fit matches batch fit exactly
- ``ORPCA`` — stochastic, partial_fit converges to similar result
- ``ORNMF`` — stochastic, partial_fit converges to similar result
"""

import numpy as np
import pytest

from hyperspy_ml_algorithms import ORNMF, ORPCA, IncrementalSVD

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
def data_200_25(rng):
    """Larger data for more meaningful partial_fit: 200 samples, 25 features."""
    return rng.random((200, 25))


# ===================================================================
# IncrementalSVD — deterministic partial_fit
# ===================================================================


class TestIncrementalSVDPartialFit:
    """IncrementalSVD partial_fit is deterministic — exact match expected."""

    def test_single_chunk_equals_fit(self, data_77_13):
        """partial_fit with the full dataset equals fit when fit uses 1 chunk."""
        est_fit = IncrementalSVD(n_components=5, num_chunks=1).fit(data_77_13)
        est_partial = IncrementalSVD(n_components=5)
        est_partial.partial_fit(data_77_13)

        np.testing.assert_allclose(
            est_fit.singular_values_,
            est_partial.singular_values_,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            est_fit.components_,
            est_partial.components_,
            atol=1e-10,
        )

    def test_multiple_chunks_match_batch_fit(self, data_200_25):
        """4 chunks via partial_fit match batch fit exactly."""
        chunks = np.array_split(data_200_25, 4)

        est_batch = IncrementalSVD(n_components=5).fit(data_200_25)

        est_partial = IncrementalSVD(n_components=5)
        for chunk in chunks:
            est_partial.partial_fit(chunk)

        np.testing.assert_allclose(
            est_batch.singular_values_,
            est_partial.singular_values_,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            est_batch.components_,
            est_partial.components_,
            atol=1e-10,
        )

    def test_partial_fit_updates_incrementally(self, data_200_25):
        """State changes after each partial_fit call."""
        chunks = np.array_split(data_200_25, 4)

        est = IncrementalSVD(n_components=5)
        est.partial_fit(chunks[0])
        sv_after_1 = est.singular_values_.copy()

        est.partial_fit(chunks[1])
        sv_after_2 = est.singular_values_.copy()

        # Singular values should change after second chunk
        assert not np.allclose(sv_after_1, sv_after_2)

    def test_partial_fit_then_transform(self, data_200_25):
        """transform works after partial_fit."""
        chunks = np.array_split(data_200_25, 4)

        est = IncrementalSVD(n_components=5)
        for chunk in chunks:
            est.partial_fit(chunk)

        scores = est.transform(data_200_25)
        assert scores.shape == (200, 5)

    def test_partial_fit_tracks_samples_seen(self, data_200_25):
        """n_samples_seen_ accumulates across partial_fit calls."""
        chunks = np.array_split(data_200_25, 4)

        est = IncrementalSVD(n_components=5)
        for i, chunk in enumerate(chunks):
            est.partial_fit(chunk)
            expected = sum(c.shape[0] for c in chunks[: i + 1])
            assert est.n_samples_seen_ == expected

    def test_partial_fit_explained_variance_ratio(self, data_200_25):
        """explained_variance_ratio_ sums to 1 after partial_fit."""
        chunks = np.array_split(data_200_25, 4)

        est = IncrementalSVD(n_components=5)
        for chunk in chunks:
            est.partial_fit(chunk)

        np.testing.assert_allclose(
            np.sum(est.explained_variance_ratio_), 1.0, atol=1e-10
        )

    def test_partial_fit_mean_is_zeros(self, data_200_25):
        """mean_ is always zeros (no centering)."""
        chunks = np.array_split(data_200_25, 4)

        est = IncrementalSVD(n_components=5)
        for chunk in chunks:
            est.partial_fit(chunk)

        np.testing.assert_allclose(est.mean_, np.zeros(25), atol=1e-10)


# ===================================================================
# ORPCA — stochastic partial_fit
# ===================================================================


class TestORPCAPartialFit:
    """ORPCA partial_fit is stochastic — approximate match expected."""

    def test_single_chunk_equals_fit(self, data_77_13):
        """partial_fit with the full dataset equals fit (same data, same path)."""
        est_fit = ORPCA(n_components=5, training_samples=10)
        est_fit.fit(data_77_13)

        est_partial = ORPCA(n_components=5, training_samples=10)
        est_partial.partial_fit(data_77_13)

        np.testing.assert_allclose(
            est_fit.components_,
            est_partial.components_,
            atol=1e-10,
        )

    def test_multiple_chunks_produce_valid_components(self, data_200_25):
        """Multiple chunks produce components with correct shape and finite values."""
        chunks = np.array_split(data_200_25, 4)

        est = ORPCA(n_components=5, training_samples=10)
        for chunk in chunks:
            est.partial_fit(chunk)

        assert est.components_.shape == (5, 25)
        assert np.all(np.isfinite(est.components_))
        assert est.low_rank_.shape == (200, 25)
        assert np.all(np.isfinite(est.low_rank_))

    def test_partial_fit_updates_incrementally(self, data_200_25):
        """State changes after each partial_fit call."""
        chunks = np.array_split(data_200_25, 4)

        est = ORPCA(n_components=5, training_samples=10)
        est.partial_fit(chunks[0])
        comp_after_1 = est.components_.copy()

        est.partial_fit(chunks[1])
        comp_after_2 = est.components_.copy()

        # Components should change after second chunk
        assert not np.allclose(comp_after_1, comp_after_2)

    def test_partial_fit_then_transform(self, data_200_25):
        """transform works after partial_fit."""
        chunks = np.array_split(data_200_25, 4)

        est = ORPCA(n_components=5, training_samples=10)
        for chunk in chunks:
            est.partial_fit(chunk)

        scores = est.transform(data_200_25)
        assert scores.shape == (200, 5)
        assert np.all(np.isfinite(scores))

    def test_partial_fit_converges_to_similar_result(self, data_200_25):
        """Multiple partial_fit chunks produce components similar to batch fit."""
        chunks = np.array_split(data_200_25, 4)

        est_batch = ORPCA(n_components=5, training_samples=10)
        est_batch.fit(data_200_25)

        est_partial = ORPCA(n_components=5, training_samples=10)
        for chunk in chunks:
            est_partial.partial_fit(chunk)

        # ORPCA is stochastic, so we use a looser tolerance.
        # The components should be in a similar subspace.
        # Check that the low-rank reconstructions are similar.
        np.testing.assert_allclose(
            est_batch.low_rank_,
            est_partial.low_rank_,
            atol=1e-1,
        )


# ===================================================================
# ORNMF — stochastic partial_fit
# ===================================================================


class TestORNMFPartialFit:
    """ORNMF partial_fit is stochastic — approximate match expected."""

    def test_single_chunk_equals_fit(self, data_77_13):
        """partial_fit with the full dataset equals fit (same random_state)."""
        est_fit = ORNMF(n_components=5, random_state=42)
        est_fit.fit(data_77_13)

        est_partial = ORNMF(n_components=5, random_state=42)
        est_partial.partial_fit(data_77_13)

        np.testing.assert_allclose(
            est_fit.components_,
            est_partial.components_,
            atol=1e-10,
        )

    def test_multiple_chunks_produce_valid_components(self, data_200_25):
        """Multiple chunks produce non-negative components with correct shape."""
        chunks = np.array_split(data_200_25, 4)

        est = ORNMF(n_components=5)
        for chunk in chunks:
            est.partial_fit(chunk)

        assert est.components_.shape == (5, 25)
        assert np.all(est.components_ >= 0)
        assert np.all(np.isfinite(est.components_))

    def test_partial_fit_updates_incrementally(self, data_200_25):
        """State changes after each partial_fit call."""
        chunks = np.array_split(data_200_25, 4)

        est = ORNMF(n_components=5)
        est.partial_fit(chunks[0])
        comp_after_1 = est.components_.copy()

        est.partial_fit(chunks[1])
        comp_after_2 = est.components_.copy()

        # Components should change after second chunk
        assert not np.allclose(comp_after_1, comp_after_2)

    def test_partial_fit_then_transform(self, data_200_25):
        """transform works after partial_fit."""
        chunks = np.array_split(data_200_25, 4)

        est = ORNMF(n_components=5)
        for chunk in chunks:
            est.partial_fit(chunk)

        scores = est.transform(data_200_25)
        assert scores.shape == (200, 5)
        assert np.all(scores >= 0)

    def test_partial_fit_components_non_negative(self, data_200_25):
        """Components remain non-negative after each partial_fit call."""
        chunks = np.array_split(data_200_25, 4)

        est = ORNMF(n_components=5)
        for chunk in chunks:
            est.partial_fit(chunk)
            assert np.all(est.components_ >= 0)

    def test_partial_fit_converges_to_similar_result(self, data_200_25):
        """Multiple partial_fit chunks produce components similar to batch fit."""
        chunks = np.array_split(data_200_25, 4)

        est_batch = ORNMF(n_components=5)
        est_batch.fit(data_200_25)

        est_partial = ORNMF(n_components=5)
        for chunk in chunks:
            est_partial.partial_fit(chunk)

        # ORNMF is stochastic, so we use a looser tolerance.
        # Check that the components are in a similar range.
        # Both should be non-negative and have similar norms.
        batch_norm = np.linalg.norm(est_batch.components_)
        partial_norm = np.linalg.norm(est_partial.components_)
        assert batch_norm > 0
        assert partial_norm > 0
        # Norms should be within an order of magnitude
        assert 0.1 < partial_norm / batch_norm < 10.0
