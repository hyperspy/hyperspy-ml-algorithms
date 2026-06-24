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


class TestEstimatorStubs:
    """Each estimator stub raises NotImplementedError on fit()."""

    @pytest.mark.parametrize(
        "estimator",
        [
            SVDPCA(n_components=3),
            MLPCA(n_components=3),
            ORPCA(n_components=3),
            RPCAGoDec(rank=3),
            ORNMF(n_components=3),
            IncrementalSVD(n_components=3),
            Orthomax(),
            Whitening(),
        ],
    )
    def test_fit_raises_not_implemented(self, estimator):
        import numpy as np

        data = np.random.random((10, 5))
        with pytest.raises(NotImplementedError, match="Refactor in task"):
            estimator.fit(data)
