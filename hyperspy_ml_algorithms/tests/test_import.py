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


class TestImportability:
    """All 8 estimators must be importable via the lazy __getattr__ pattern."""

    @pytest.mark.parametrize(
        "name,cls",
        [
            ("SVDPCA", SVDPCA),
            ("MLPCA", MLPCA),
            ("ORPCA", ORPCA),
            ("RPCAGoDec", RPCAGoDec),
            ("ORNMF", ORNMF),
            ("IncrementalSVD", IncrementalSVD),
            ("Orthomax", Orthomax),
            ("Whitening", Whitening),
        ],
    )
    def test_import(self, name, cls):
        assert cls.__name__ == name, f"Expected {name}, got {cls.__name__}"


class TestEstimatorFit:
    """All 8 estimators must have working fit() methods."""

    @pytest.mark.parametrize(
        "name,cls,kwargs,shape",
        [
            ("SVDPCA", SVDPCA, {"n_components": 2}, (12, 5)),
            ("MLPCA", MLPCA, {"n_components": 2}, (12, 5)),
            ("ORPCA", ORPCA, {"n_components": 2}, (12, 5)),
            ("RPCAGoDec", RPCAGoDec, {"rank": 2, "max_iter": 10}, (12, 5)),
            ("ORNMF", ORNMF, {"n_components": 2}, (12, 5)),
            ("IncrementalSVD", IncrementalSVD, {"n_components": 2}, (12, 5)),
            ("Orthomax", Orthomax, {}, (5, 2)),
            ("Whitening", Whitening, {}, (12, 5)),
        ],
    )
    def test_estimator_fit(self, name, cls, kwargs, shape):
        """fit() should succeed and return self."""
        rng = np.random.RandomState(42)
        if name == "ORNMF":
            X = rng.rand(*shape)
        else:
            X = rng.randn(*shape)
        estimator = cls(**kwargs)
        if name == "MLPCA":
            variance = np.full(shape, 0.01)
            result = estimator.fit(X, variance)
        else:
            result = estimator.fit(X)
        assert result is estimator, f"{name}.fit() should return self"
