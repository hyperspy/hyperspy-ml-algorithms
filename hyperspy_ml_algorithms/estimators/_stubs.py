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
"""
Stub estimator classes for hyperspy-ml-algorithms.

Each class raises ``NotImplementedError`` in ``fit()``.
Real implementations will replace these in Tasks 3--9.
No HyperSpy dependencies — only NumPy used for type hints.
"""

__all__ = [
    "ORPCA",
    "Orthomax",
    "RPCAGoDec",
    "SVDPCA",
]


class SVDPCA:
    """SVD-based PCA estimator — stub for Task 3."""

    def __init__(self, n_components=None, svd_solver="auto", centre=None):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.centre = centre

    def fit(self, X, y=None):
        raise NotImplementedError("Refactor in task 3")

    def transform(self, X):
        raise NotImplementedError("Refactor in task 3")

    def fit_transform(self, X, y=None):
        raise NotImplementedError("Refactor in task 3")


class ORPCA:
    """Online Robust PCA estimator — stub for Task 5."""

    def __init__(self, n_components=None, max_iter=100, batch_size=20):
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size

    def fit(self, X, y=None):
        raise NotImplementedError("Refactor in task 5")

    def transform(self, X):
        raise NotImplementedError("Refactor in task 5")

    def fit_transform(self, X, y=None):
        raise NotImplementedError("Refactor in task 5")

    def partial_fit(self, X_chunk):
        raise NotImplementedError("Refactor in task 5")


class RPCAGoDec:
    """Batch Robust PCA via GoDec — stub for Task 5."""

    def __init__(self, rank=5, max_iter=100):
        self.rank = rank
        self.max_iter = max_iter

    def fit(self, X, y=None):
        raise NotImplementedError("Refactor in task 5")

    def transform(self, X):
        raise NotImplementedError("Refactor in task 5")

    def fit_transform(self, X, y=None):
        raise NotImplementedError("Refactor in task 5")


class Orthomax:
    """Orthomax / varimax factor rotation — stub for Task 8."""

    def __init__(self, gamma=1.0, max_iter=100, tol=1e-3):
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        raise NotImplementedError("Refactor in task 8")

    def transform(self, X):
        raise NotImplementedError("Refactor in task 8")

    def fit_transform(self, X, y=None):
        raise NotImplementedError("Refactor in task 8")
